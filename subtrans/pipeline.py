# -*- coding: utf-8 -*-
"""
subtrans/pipeline.py
Core orchestration pipeline connecting parsing, Stage A cast sheet generation,
translation backends, caching, output generation (SRT single-line default),
and Stage C QA audit reporting.
"""

import time
import logging
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import typer

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from subtrans.config import load_config, get_gemini_api_key
from subtrans.models import Cue, Glossary
from subtrans.parser import parse_and_align_subtitles
from subtrans.cache import TranslationCache, compute_cue_cache_key
from subtrans.gemini_backend import GeminiTranslator
from subtrans.local_backend import OllamaTranslator
from subtrans.review import LocalReviewer
from subtrans.throttle import RateLimiter, call_with_retry, QuotaExhaustedError
from subtrans.writer import write_subtitles
from subtrans.report import generate_qa_report
from subtrans.glossary import load_glossary, merge_glossaries, DEFAULT_BASE_GLOSSARY_PATH
from subtrans.cast_sheet import generate_cast_sheet

logger = logging.getLogger("subtrans.pipeline")
console = Console()


class TranslationPipeline:
    """Manages full anime subtitle translation workflow."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        local_only: bool = False,
        model_override: Optional[str] = None,
        on_quota: str = "local",  # "local" | "stop"
        enable_review: bool = False,
        limit_cues: Optional[int] = None,
        output_dir: Optional[Path] = None,
        output_format: Optional[str] = "srt",
        wrap_chars: Optional[int] = None,
        gloss_skill_names: bool = False,
        require_approved_cast: bool = False,
        review_file: bool = False,
        allow_latest_alias: bool = False,
    ):
        self.cfg = load_config(config_path)
        self.local_only = local_only or (self.cfg.get("privacy", {}).get("mode") == "local_only")
        self.on_quota = on_quota
        self.enable_review = enable_review
        self.limit_cues = limit_cues
        self.output_dir = output_dir or Path("out")
        self.output_format = output_format or "srt"
        self.wrap_chars = wrap_chars
        self.gloss_skill_names = gloss_skill_names
        self.require_approved_cast = require_approved_cast
        self.review_file = review_file
        self.allow_latest_alias = allow_latest_alias

        # Models & batch sizes
        gemini_cfg = self.cfg.get("gemini", {})
        self.gemini_model = model_override or gemini_cfg.get("model", "gemini-3.8-flash")
        self.fallback_models: List[str] = gemini_cfg.get("fallback_models", ["gemini-3.7-flash", "gemini-3.6-flash"])
        self.lite_model = gemini_cfg.get("lite_model", "gemini-3.5-flash-lite")
        self.gemini_rpm = gemini_cfg.get("max_rpm", 5)

        # Protection against mutable alias 'gemini-flash-latest' per ADDENDUM 8
        if "latest" in self.gemini_model.lower() and not self.allow_latest_alias:
            raise ValueError(
                f"Model alias '{self.gemini_model}' is prohibited by default because its target can change silently. "
                f"Please specify an explicit versioned model (e.g. gemini-3.8-flash) or pass --allow-latest-alias."
            )

        ollama_cfg = self.cfg.get("ollama", {})
        self.ollama_endpoint = ollama_cfg.get("endpoint", "http://localhost:11434")
        self.ollama_model = ollama_cfg.get("model", "translategemma:12b")
        self.ollama_review_model = ollama_cfg.get("review_model", "gemma3:12b")

        trans_cfg = self.cfg.get("translation", {})
        self.batch_size = trans_cfg.get("batch_size", 50)
        self.local_batch_size = trans_cfg.get("local_batch_size", 20)
        self.allow_silent_fallback = trans_cfg.get("allow_silent_fallback", False)
        self.validate_batch_pronouns = trans_cfg.get("validate_batch_pronouns", True)

        # Cache & Rate Limiter
        self.cache = TranslationCache()
        self.rate_limiter = RateLimiter(max_rpm=self.gemini_rpm)

        # Backends
        self.gemini_backends: Dict[str, GeminiTranslator] = {}
        self.gemini_backend: Optional[GeminiTranslator] = None
        self.ollama_backend: Optional[OllamaTranslator] = None
        self.reviewer: Optional[LocalReviewer] = None

    def _init_primary_backend(self):
        if not self.local_only:
            api_key = get_gemini_api_key()
            if api_key:
                all_gemini = [self.gemini_model] + [m for m in self.fallback_models if m != self.gemini_model]
                for m_name in all_gemini:
                    try:
                        self.gemini_backends[m_name] = GeminiTranslator(
                            api_key=api_key,
                            model=m_name,
                            gloss_skill_names=self.gloss_skill_names,
                            allow_latest_alias=self.allow_latest_alias,
                        )
                    except Exception as e:
                        logger.warning(f"Could not initialize backend for {m_name}: {e}")
                if self.gemini_model in self.gemini_backends:
                    self.gemini_backend = self.gemini_backends[self.gemini_model]
                elif self.gemini_backends:
                    first_avail = list(self.gemini_backends.keys())[0]
                    self.gemini_backend = self.gemini_backends[first_avail]
                    self.gemini_model = first_avail
                else:
                    self.local_only = True
            else:
                console.print("[yellow]⚠️  GEMINI_API_KEY not found in environment. Running in local-only mode.[/yellow]")
                self.local_only = True

        self.ollama_backend = OllamaTranslator(
            endpoint=self.ollama_endpoint,
            model=self.ollama_model,
        )

        if self.enable_review:
            self.reviewer = LocalReviewer(
                endpoint=self.ollama_endpoint,
                model=self.ollama_review_model,
            )

    def run(
        self,
        ja_path: Path,
        en_path: Optional[Path] = None,
        glossary_path: Optional[Path] = None,
        en_offset_ms: int = 0,
    ) -> Dict[str, Any]:
        """Executes translation pipeline."""
        start_time = time.monotonic()
        self._init_primary_backend()

        console.print(f"\n[bold green]🚀 Subtrans Pipeline Starting (Standard Quality Mode)[/bold green]")
        console.print(f"  • Source (JA): [cyan]{ja_path}[/cyan]")
        if en_path:
            console.print(f"  • Reference (EN): [cyan]{en_path}[/cyan] (Offset: {en_offset_ms}ms)")
        console.print(f"  • Mode: [bold]{'100% Local (Ollama)' if self.local_only else f'Primary: Gemini ({self.gemini_model}) -> Local Fallback'}[/bold]")

        base_glossary = load_glossary(DEFAULT_BASE_GLOSSARY_PATH) if DEFAULT_BASE_GLOSSARY_PATH.exists() else None
        override_glossary = load_glossary(glossary_path) if glossary_path else None
        glossary = merge_glossaries(base=base_glossary, override=override_glossary)

        if glossary:
            src_desc = []
            if base_glossary:
                src_desc.append(f"Base: {len(base_glossary.terms)} terms")
            if override_glossary and glossary_path:
                src_desc.append(f"Override '{glossary_path.name}': {len(override_glossary.terms)} terms")
            desc_str = f" ({', '.join(src_desc)})" if src_desc else ""
            console.print(f"  • Glossary Loaded: [cyan]{len(glossary.terms)} terms, {len(glossary.characters)} characters[/cyan]{desc_str}")

        stage_times = {
            "stage_1_parse": 0.0,
            "stage_a_cast": 0.0,
            "stage_2_translate": 0.0,
            "stage_review": 0.0,
            "stage_4_write": 0.0,
            "stage_c_report": 0.0,
        }

        # Stage 1: Parse and align
        t_s1 = time.monotonic()
        with console.status("[bold blue]Stage 1: Parsing and aligning subtitle streams...[/bold blue]"):
            cues, ja_ssa, en_ssa = parse_and_align_subtitles(
                ja_path=str(ja_path),
                en_path=str(en_path) if en_path else None,
                en_offset_ms=en_offset_ms,
            )
        stage_times["stage_1_parse"] = time.monotonic() - t_s1

        if self.limit_cues:
            cues = cues[:self.limit_cues]
            console.print(f"[yellow]ℹ️  Limited run to first {self.limit_cues} cues.[/yellow]")

        dialogue_cues = [c for c in cues if not c.is_skipped]
        console.print(f"  • Total Cues: {len(cues)} ({len(dialogue_cues)} dialogue lines to translate)")

        # Stage A: Cast Sheet generation (runs automatically once per episode)
        t_sa = time.monotonic()
        base_name = ja_path.stem
        cast_file = self.output_dir / f"{base_name}.cast.yaml"
        if not self.local_only:
            with console.status("[bold blue]Stage A: Analyzing episode cast & character pronoun table...[/bold blue]"):
                cast_glossary = generate_cast_sheet(cues, output_path=cast_file, gemini_model=self.gemini_model)
                if cast_glossary and cast_glossary.characters:
                    console.print(f"  • Stage A Cast Sheet: [cyan]{cast_file}[/cyan] ({len(cast_glossary.characters)} characters identified)")
                    glossary = merge_glossaries(base=glossary, override=cast_glossary)
                    if self.require_approved_cast:
                        confirm = typer.confirm(f"Please inspect {cast_file}. Approve and proceed?", default=True)
                        if not confirm:
                            console.print("[yellow]Stopped by user to edit cast sheet.[/yellow]")
                            return {}
        else:
            # Auto-detect existing Stage A Cast Sheet from candidate paths for fair local evaluation
            candidate_cast_paths = [
                cast_file,
                self.output_dir / f"{base_name}.cast.yaml",
                Path("out") / f"{base_name}.cast.yaml",
                Path("out/gemini") / f"{base_name}.cast.yaml",
                Path("out/gemini_35") / f"{base_name}.cast.yaml",
            ]
            for cp in candidate_cast_paths:
                if cp.exists():
                    cast_glossary = load_glossary(cp)
                    if cast_glossary and cast_glossary.characters:
                        console.print(f"  • Stage A Cast Sheet Loaded for Local: [cyan]{cp}[/cyan] ({len(cast_glossary.characters)} characters)")
                        glossary = merge_glossaries(base=glossary, override=cast_glossary)
                        break
        stage_times["stage_a_cast"] = time.monotonic() - t_sa

        stats = {
            "gemini": 0,
            "local": 0,
            "cache_hits": 0,
            "reviewed": 0,
            "model_counts": {},
        }

        # Stage 2 & 3: Translation loop (Batch size 50, Context +/-10)
        t_s2 = time.monotonic()
        curr_batch_size = self.local_batch_size if self.local_only else self.batch_size
        dialogue_batches = [
            dialogue_cues[i:i + curr_batch_size]
            for i in range(0, len(dialogue_cues), curr_batch_size)
        ]

        active_backend_mode = "local" if self.local_only else "gemini"
        candidate_gemini_models = [self.gemini_model] + [m for m in self.fallback_models if m != self.gemini_model]
        curr_gemini_idx = 0

        # Compute cast sheet hash for cache keying
        cast_sheet_hash = ""
        if cast_file.exists():
            try:
                cast_sheet_hash = hashlib.sha256(cast_file.read_bytes()).hexdigest()[:16]
            except Exception:
                pass
        elif glossary and glossary.characters:
            canonical_chars = json.dumps([c.model_dump() for c in glossary.characters], sort_keys=True)
            cast_sheet_hash = hashlib.sha256(canonical_chars.encode("utf-8")).hexdigest()[:16]
        prompt_version = "v2.0"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Translating dialogue...", total=len(dialogue_cues))

            for batch_idx, batch in enumerate(dialogue_batches):
                # 1. Check cache for cues in batch
                uncached_cues: List[Cue] = []
                for cue in batch:
                    # Model key for cache lookup
                    target_m = self.ollama_model if active_backend_mode == "local" else candidate_gemini_models[curr_gemini_idx]
                    ckey = compute_cue_cache_key(
                        model=target_m,
                        ja=cue.ja,
                        en=cue.en,
                        glossary_hash=f"{len(glossary.terms)}:{len(glossary.characters)}" if glossary else "",
                        cast_sheet_hash=cast_sheet_hash,
                        prompt_version=prompt_version
                    )
                    cached_val = self.cache.get_cue(ckey)
                    if cached_val:
                        cue.vi = cached_val
                        cue.backend_used = "cache"
                        cue.model_used = target_m
                        stats["cache_hits"] += 1
                        stats["model_counts"]["cache"] = stats["model_counts"].get("cache", 0) + 1
                        progress.advance(task, 1)
                    else:
                        uncached_cues.append(cue)

                if not uncached_cues:
                    continue

                # 2. Extract surrounding context symmetrically (+/-10 cues per ADDENDUM 6)
                first_id = uncached_cues[0].id
                last_id = uncached_cues[-1].id
                context = [c for c in cues if (first_id - 10 <= c.id < first_id) or (last_id < c.id <= last_id + 10)]

                translated_texts: List[str] = []
                backend_used = active_backend_mode
                model_used = self.ollama_model

                if active_backend_mode == "gemini":
                    translated = False
                    while curr_gemini_idx < len(candidate_gemini_models):
                        target_model = candidate_gemini_models[curr_gemini_idx]
                        backend = self.gemini_backends.get(target_model)
                        if not backend:
                            curr_gemini_idx += 1
                            continue
                        try:
                            translated_texts = call_with_retry(
                                fn=lambda: backend.translate_batch(uncached_cues, context, glossary),
                                rate_limiter=self.rate_limiter,
                            )
                            backend_used = "gemini"
                            model_used = target_model
                            stats["gemini"] += len(uncached_cues)
                            stats["model_counts"][target_model] = stats["model_counts"].get(target_model, 0) + len(uncached_cues)
                            translated = True
                            break
                        except Exception as e:
                            err_str = str(e).lower()
                            is_quota = isinstance(e, QuotaExhaustedError) or any(k in err_str for k in ["quota", "resource_exhausted", "429"])
                            is_not_found = any(k in err_str for k in ["404", "not_found", "not found", "410", "gone", "unsupported", "deprecated", "retire"])
                            is_unavailable = any(k in err_str for k in ["503", "unavailable", "high demand"])
                            if is_quota:
                                reason = "quota exhausted (429)"
                            elif is_not_found:
                                reason = "deprecated / not found (404/410)"
                            elif is_unavailable:
                                reason = "high demand / service unavailable (503)"
                            else:
                                reason = str(e)

                            # If transient rate limit (429) or temporary demand spike (503),
                            # and silent fallback is prohibited (default), pause and wait instead of downgrading!
                            # ADDENDUM: Add upper bound (max 5 pauses) to prevent infinite loops on daily quota exhaustion.
                            if (is_quota or is_unavailable) and not self.allow_silent_fallback:
                                quota_retries = getattr(self, "_batch_quota_retries", 0) + 1
                                self._batch_quota_retries = quota_retries
                                max_quota_retries = 5
                                if quota_retries > max_quota_retries:
                                    console.print(f"\n[bold red]🛑 Quota ngày hoặc lỗi tài nguyên kéo dài sau {max_quota_retries} lần tạm dừng ({reason}). "
                                                  f"Dừng tiến trình an toàn để bảo toàn trạng thái (tất cả câu trước đã được lưu cache). "
                                                  f"Có thể chạy tiếp bất cứ lúc nào với resume.[/bold red]")
                                    raise QuotaExhaustedError(f"Persistent quota exhaustion ({reason}) after {max_quota_retries} pause-and-retries.")
                                wait_time = 30.0 if is_quota else 12.0
                                console.print(f"\n[yellow]⏳ Model '{target_model}' hit {reason} (Lần {quota_retries}/{max_quota_retries}). "
                                              f"Tạm dừng {wait_time:.0f}s để bảo toàn tính đồng nhất đại từ toàn tập...[/yellow]")
                                time.sleep(wait_time)
                                # Retry on same model
                                continue
                            self._batch_quota_retries = 0

                            console.print(f"\n[yellow]⚠️  Model '{target_model}' error: {reason}.[/yellow]")
                            curr_gemini_idx += 1
                            if curr_gemini_idx < len(candidate_gemini_models):
                                next_model = candidate_gemini_models[curr_gemini_idx]
                                console.print(f"[bold cyan]🔄 Falling back to next configured model: '{next_model}' (Warning: Verify pronoun continuity!)...[/bold cyan]")
                            else:
                                console.print("[bold yellow]🔄 All configured Gemini models exhausted. Permanently switching to local Ollama fallback...[/bold yellow]")
                                active_backend_mode = "local"
                                break

                    if not translated:
                        # Fallback to local
                        translated_texts = self.ollama_backend.translate_batch(uncached_cues, context, glossary)
                        backend_used = "local"
                        model_used = self.ollama_model
                        stats["local"] += len(uncached_cues)
                        stats["model_counts"][self.ollama_model] = stats["model_counts"].get(self.ollama_model, 0) + len(uncached_cues)
                else:
                    translated_texts = self.ollama_backend.translate_batch(uncached_cues, context, glossary)
                    backend_used = "local"
                    model_used = self.ollama_model
                    stats["local"] += len(uncached_cues)
                    stats["model_counts"][self.ollama_model] = stats["model_counts"].get(self.ollama_model, 0) + len(uncached_cues)

                # Store into cues and cache
                for cue, vi in zip(uncached_cues, translated_texts):
                    cue.vi = vi
                    cue.backend_used = backend_used
                    cue.model_used = model_used
                    ckey = compute_cue_cache_key(
                        model=model_used,
                        ja=cue.ja,
                        en=cue.en,
                        glossary_hash=f"{len(glossary.terms)}:{len(glossary.characters)}" if glossary else "",
                        cast_sheet_hash=cast_sheet_hash,
                        prompt_version=prompt_version
                    )
                    self.cache.set_cue(ckey, backend_used, model_used, vi)
                    progress.advance(task, 1)

        stage_times["stage_2_translate"] = time.monotonic() - t_s2

        if self.ollama_backend:
            self.ollama_backend.unload()

        # Optional review
        t_srev = time.monotonic()
        if self.enable_review and self.reviewer:
            console.print("[bold cyan]🔍 Running optional review pass...[/bold cyan]")
            reviewed_cues = [c for c in dialogue_cues if c.vi]
            for i in range(0, len(reviewed_cues), self.local_batch_size):
                chunk = reviewed_cues[i:i + self.local_batch_size]
                refined = self.reviewer.review_batch(chunk, glossary)
                for c, r_text in zip(chunk, refined):
                    if r_text and r_text != c.vi:
                        c.vi = r_text
                        stats["reviewed"] += 1
            self.reviewer.unload()
        stage_times["stage_review"] = time.monotonic() - t_srev

        # Stage 4: Writing output (SRT default, exactly ONE line per cue unless --wrap)
        t_s4 = time.monotonic()
        out_ext = self.output_format.lower()
        out_file = self.output_dir / f"{base_name}.vi.{out_ext}"

        with console.status(f"[bold blue]Stage 4: Writing subtitle file to {out_file}...[/bold blue]"):
            saved_path = write_subtitles(
                cues=cues,
                original_ja_ssa=ja_ssa,
                output_path=out_file,
                output_format=out_ext,
                wrap_chars=self.wrap_chars,
            )
        stage_times["stage_4_write"] = time.monotonic() - t_s4

        elapsed = time.monotonic() - start_time

        # Stage C: Audit Report
        t_sc = time.monotonic()
        audit_file = self.output_dir / f"{base_name}.audit.md"
        with console.status(f"[bold blue]Stage C: Generating QA audit report at {audit_file}...[/bold blue]"):
            generate_qa_report(
                cues=cues,
                ja_path=ja_path,
                en_path=en_path,
                output_subtitle_path=saved_path,
                report_path=audit_file,
                elapsed_time_sec=elapsed,
                stats=stats,
                glossary=glossary,
                review_file=self.review_file,
                stage_times=stage_times,
            )
        stage_times["stage_c_report"] = time.monotonic() - t_sc

        console.print(f"\n[bold green]✅ Translation Completed in {elapsed:.1f}s![/bold green]")
        console.print(
            f"  • Stage Breakdown: Parse: {stage_times['stage_1_parse']:.2f}s | "
            f"Stage A Cast: {stage_times['stage_a_cast']:.2f}s | "
            f"Translate: {stage_times['stage_2_translate']:.2f}s | "
            f"Write: {stage_times['stage_4_write']:.2f}s | "
            f"Audit: {stage_times['stage_c_report']:.2f}s"
        )
        console.print(f"  • Subtitle Output: [cyan]{saved_path}[/cyan]")
        console.print(f"  • Stage C Audit: [cyan]{audit_file}[/cyan]")
        # Release local model from Unified Memory per AGENTS.md invariant
        if self.ollama_backend and (stats["local"] > 0 or self.local_only):
            self.ollama_backend.unload()

        return {
            "output_path": saved_path,
            "report_path": audit_file,
            "elapsed_seconds": elapsed,
            "stats": stats,
            "stage_times": stage_times,
        }
