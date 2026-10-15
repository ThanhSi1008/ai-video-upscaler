# -*- coding: utf-8 -*-
"""
subtrans/pipeline.py
Core orchestration pipeline connecting parsing, Stage A cast sheet generation,
translation backends, caching, output generation (SRT single-line default),
and Stage C QA audit reporting.
"""

import re
import time
import logging
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import typer

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from subtrans.config import load_config, load_models_config, get_gemini_api_key
from subtrans.models import Cue, Glossary
from subtrans.parser import parse_and_align_subtitles, detect_auto_offset
from subtrans.cache import TranslationCache, compute_cue_cache_key
from subtrans.gemini_backend import GeminiTranslator
from subtrans.throttle import RateLimiter, call_with_retry, QuotaExhaustedError, DailyQuotaExhaustedError
from subtrans.writer import write_subtitles
from subtrans.report import generate_qa_report
from subtrans.glossary import load_glossary, merge_glossaries, DEFAULT_BASE_GLOSSARY_PATH, find_series_glossary
from subtrans.cast_sheet import generate_cast_sheet, load_or_generate_series_cast_sheet

logger = logging.getLogger("subtrans.pipeline")
console = Console()


class TranslationPipeline:
    """Manages full anime subtitle translation workflow."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        local_only: bool = False,
        model_override: Optional[str] = None,
        on_quota: str = "stop",  # "stop" (Graceful exit on reset hour) | "local"
        enable_review: bool = False,
        enable_cascade: bool = True,
        enable_rag: bool = True,
        limit_cues: Optional[int] = None,
        output_dir: Optional[Path] = None,
        output_format: Optional[str] = "srt",
        wrap_chars: Optional[int] = None,
        gloss_skill_names: bool = False,
        require_approved_cast: bool = False,
        review_file: bool = False,
        allow_latest_alias: bool = False,
        dry_run: bool = False,
        batch_size: Optional[int] = None,
        subs_quality: str = "normal",  # "normal" (Lite first) | "high" (Flash first)
        cache_db_path: Optional[Path] = None,
        auto_align_whisper_to_en: Optional[bool] = None,
    ):
        self.cfg = load_config(config_path)
        self.models_cfg = load_models_config()
        self.local_only = local_only or (self.cfg.get("privacy", {}).get("mode") == "local_only")
        self.on_quota = on_quota
        self.enable_review = enable_review
        self.enable_cascade = enable_cascade
        self.enable_rag = enable_rag
        self.limit_cues = limit_cues
        self.output_dir = output_dir or Path("out")
        self.output_format = output_format or "srt"
        self.wrap_chars = wrap_chars
        self.gloss_skill_names = gloss_skill_names
        self.require_approved_cast = require_approved_cast
        self.review_file = review_file
        self.allow_latest_alias = allow_latest_alias
        self.dry_run = dry_run
        self.subs_quality = subs_quality
        if auto_align_whisper_to_en is None:
            self.auto_align_whisper_to_en = bool(self.cfg.get("translation", {}).get("auto_align_whisper_to_en", True))
        else:
            self.auto_align_whisper_to_en = auto_align_whisper_to_en

        # Chuỗi model: Nạp từ models.yaml, tuyệt đối không dùng model preview
        # A6: Lite làm model chính theo mặc định. Cờ --subs-quality high đảo thứ tự (Flash trước, Lite sau).
        raw_chain = self.models_cfg.get("chain", [
            "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "local"
        ])
        gemini_chain = [m for m in raw_chain if m != "local" and "preview" not in m.lower()]
        lite_models = [m for m in gemini_chain if "lite" in m.lower()]
        flash_models = [m for m in gemini_chain if "lite" not in m.lower()]

        if self.subs_quality == "high":
            ordered_chain = flash_models + lite_models
        else:
            ordered_chain = lite_models + flash_models

        gemini_cfg = self.cfg.get("gemini", {})
        self.gemini_model = model_override or (ordered_chain[0] if ordered_chain else "gemini-3.5-flash-lite")
        self.fallback_models: List[str] = [m for m in ordered_chain if m != self.gemini_model]
        self.lite_model = gemini_cfg.get("lite_model", "gemini-3.5-flash-lite")
        self.gemini_rpm = gemini_cfg.get("max_rpm", 15 if "lite" in self.gemini_model.lower() else 5)

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
        self.batch_size = batch_size or trans_cfg.get("batch_size", 100)
        self.local_batch_size = trans_cfg.get("local_batch_size", 20)
        self.allow_silent_fallback = trans_cfg.get("allow_silent_fallback", False)
        self.validate_batch_pronouns = trans_cfg.get("validate_batch_pronouns", True)
        self.stage_watchdog_seconds = int(trans_cfg.get("stage_watchdog_seconds", 1500))
        self.max_retries_per_model = int(trans_cfg.get("max_retries_per_model", 3))

        # Request timeouts
        self.request_timeout_ms = int(gemini_cfg.get("request_timeout_ms", 45000))

        # Cache & Rate Limiter
        self.cache = TranslationCache(db_path=cache_db_path)
        self.rate_limiter = RateLimiter(max_rpm=self.gemini_rpm)

        # Backends
        self.gemini_backends: Dict[str, GeminiTranslator] = {}
        self.gemini_backend: Optional[GeminiTranslator] = None
        if enable_cascade is None:
            self.enable_cascade = trans_cfg.get("cascade", True)
        self.series: Optional[str] = None

    def _init_primary_backend(self):
        if not self.local_only and not self.gemini_backends:
            from subtrans.config import get_gemini_api_keys
            api_keys = get_gemini_api_keys()
            primary_key = api_keys[0] if api_keys else None
            if primary_key or self.dry_run:
                all_gemini = [self.gemini_model] + [m for m in self.fallback_models if m != self.gemini_model]
                for m_name in all_gemini:
                    try:
                        self.gemini_backends[m_name] = GeminiTranslator(
                            api_keys=api_keys if api_keys else [primary_key or "DRY_RUN_KEY"],
                            model=m_name,
                            gloss_skill_names=self.gloss_skill_names,
                            allow_latest_alias=self.allow_latest_alias,
                            timeout_ms=self.request_timeout_ms,
                            dry_run=self.dry_run,
                            enable_rag=self.enable_rag,
                            round_robin=True,
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

    def run(
        self,
        ja_path: Path,
        en_path: Optional[Path] = None,
        glossary_path: Optional[Path] = None,
        en_offset_ms: int = 0,
        sub_offset_ms: int = 0,
        output_basename: Optional[str] = None,
        subs_upgrade: bool = False,
        auto_offset: bool = False,
        sound_cues: str = "drop",
    ) -> Dict[str, Any]:
        """Executes translation pipeline."""
        start_time = time.monotonic()
        self._init_primary_backend()

        # Kiểm tra tự động phát hiện lệch thời gian (--auto-offset)
        auto_offset_applied = False
        offset_reason = "Không có phụ đề EN tham chiếu để so sánh"
        det_offset = 0
        if en_path:
            det_offset, can_apply, reason_str, off_stats = detect_auto_offset(str(ja_path), str(en_path))
            if sub_offset_ms != 0:
                offset_reason = f"Đo được {det_offset}ms, nhưng ưu tiên cờ thủ công --sub-offset-ms={sub_offset_ms}ms"
                console.print(f"  • [cyan]Auto-offset:[/cyan] {offset_reason}.")
            elif auto_offset:
                if can_apply:
                    sub_offset_ms = det_offset
                    auto_offset_applied = True
                    offset_reason = f"Tự động áp dụng {det_offset}ms ({reason_str})"
                    console.print(f"  • [bold green]Auto-offset ĐƯỢC ÁP DỤNG:[/bold green] {det_offset}ms ({reason_str})")
                else:
                    offset_reason = f"Không áp dụng ({reason_str})"
                    console.print(f"  • [yellow]Auto-offset KHÔNG ÁP DỤNG:[/yellow] {reason_str}")
            else:
                offset_reason = f"Cờ --auto-offset tắt (Đo được: {det_offset}ms, {reason_str})"
                logger.debug(f"Auto-offset scan (flag OFF): {det_offset}ms (can_apply={can_apply}: {reason_str})")
        elif sub_offset_ms != 0:
            offset_reason = f"Áp dụng cờ thủ công --sub-offset-ms={sub_offset_ms}ms"
        else:
            offset_reason = "Mặc định 0ms (Không có cờ offset hay file EN tham chiếu)"

        console.print(f"\n[bold green]🚀 Subtrans Pipeline Starting (Standard Quality Mode)[/bold green]")
        console.print(f"  • Source (JA): [cyan]{ja_path}[/cyan]{f' (Offset: {sub_offset_ms}ms)' if sub_offset_ms != 0 else ''}")
        if en_path:
            console.print(f"  • Reference (EN): [cyan]{en_path}[/cyan] (Offset: {en_offset_ms}ms)")
        console.print(f"  • Mode: [bold]100% Cloud API: Gemini ({self.gemini_model})[/bold]")
        if subs_upgrade:
            console.print("[bold yellow]⚡ Cờ --subs-upgrade được kích hoạt: Bỏ qua cache các câu Lite để nâng cấp dịch bằng model cao hơn.[/bold yellow]")

        base_name = output_basename or re.sub(r"\.(ja|en|vi|jpn|eng|vie)$", "", ja_path.stem, flags=re.IGNORECASE)

        # Extract series name and propagate to Gemini backends
        series_match = re.search(r"^(.*?)(?:_S\d+|[ _]E\d+|\.\w+$)", base_name, re.IGNORECASE)
        self.series = series_match.group(1).replace("_", " ").strip() if series_match else None

        base_glossary = load_glossary(DEFAULT_BASE_GLOSSARY_PATH) if DEFAULT_BASE_GLOSSARY_PATH.exists() else None
        if not glossary_path:
            auto_g = find_series_glossary(base_name)
            if auto_g:
                glossary_path = auto_g
                resolved_series = auto_g.resolve().name.split(".")[1]
                self.series = resolved_series
                console.print(f"  • [bold green]Tự động nhận diện Glossary riêng của series:[/bold green] [cyan]{auto_g.name}[/cyan] (Series key: [yellow]{resolved_series}[/yellow])")
        override_glossary = load_glossary(glossary_path) if glossary_path else None
        glossary = merge_glossaries(base=base_glossary, override=override_glossary)

        for b in self.gemini_backends.values():
            b.series = self.series
            b.exclude_episode = base_name

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
                ja_offset_ms=sub_offset_ms,
                sound_cues=sound_cues,
                auto_align_whisper_to_en=self.auto_align_whisper_to_en,
            )
        stage_times["stage_1_parse"] = time.monotonic() - t_s1


        if self.limit_cues:
            cues = cues[:self.limit_cues]
            console.print(f"[yellow]ℹ️  Limited run to first {self.limit_cues} cues.[/yellow]")

        dialogue_cues = [c for c in cues if not c.is_skipped]
        console.print(f"  • Total Cues: {len(cues)} ({len(dialogue_cues)} dialogue lines to translate)")

        # Stage A: Cast Sheet generation (Quản lý theo bộ <series>.cast.yaml, chỉ gọi API khi có nhân vật mới)
        t_sa = time.monotonic()
        base_name = output_basename or ja_path.stem
        audit_cast_file = self.output_dir / "audit" / f"{base_name}.cast.yaml"
        cast_file = audit_cast_file if audit_cast_file.exists() else self.output_dir / f"{base_name}.cast.yaml"
        if not self.local_only:
            with console.status("[bold blue]Stage A: Analyzing series cast & character pronoun table...[/bold blue]"):
                cast_glossary = load_or_generate_series_cast_sheet(
                    cues=cues,
                    source_name=base_name,
                    output_dir=self.output_dir,
                    episode_cast_path=cast_file,
                    gemini_model=self.gemini_model,
                    fallback_models=self.fallback_models,
                    timeout_ms=self.request_timeout_ms,
                    dry_run=self.dry_run,
                )
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
                self.output_dir / "audit" / f"{base_name}.cast.yaml",
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
        candidate_gemini_models = getattr(self, "candidate_gemini_models", None) or (
            [self.gemini_model] + [m for m in self.fallback_models if m != self.gemini_model]
        )
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

        # Khởi tạo HysteresisCastManager đóng kín vòng lặp phản hồi quan hệ nhân vật
        from subtrans.dynamic_cast import HysteresisCastManager, CastRelationDelta
        initial_cast_data = {}
        if cast_file.exists():
            try:
                import yaml
                initial_cast_data = yaml.safe_load(cast_file.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
        cast_manager = HysteresisCastManager(initial_cast_data) if initial_cast_data else HysteresisCastManager.from_glossary(glossary)

        stage_start_time = time.monotonic()
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
                # 0. Check stage watchdog
                if (time.monotonic() - stage_start_time) > self.stage_watchdog_seconds:
                    console.print(f"\n[bold red]⏱️ Watchdog toàn stage phụ đề đã chạm hạn mức {self.stage_watchdog_seconds}s ({(self.stage_watchdog_seconds/60):.1f}m). "
                                  f"Dừng dịch an toàn, đánh dấu tất cả các câu chưa dịch là [Chưa dịch] và tiến hành ghi file.[/bold red]")
                    for rem_batch in dialogue_batches[batch_idx:]:
                        for cue in rem_batch:
                            if not cue.vi:
                                cue.vi = "[Chưa dịch]"
                                cue.backend_used = "untranslated"
                                cue.model_used = "none"
                                cue.uncertain = True
                                stats["model_counts"]["untranslated"] = stats["model_counts"].get("untranslated", 0) + 1
                                progress.advance(task, 1)
                    break

                # 1. Compute RAG snapshot hash for this batch (C0/C2 use 'no_rag', C1/C3 use few-shot hash)
                rag_hash = "no_rag"
                if self.enable_rag:
                    try:
                        import hashlib
                        from subtrans.rag import KnowledgeBase
                        kb = KnowledgeBase()
                        fs = kb.retrieve_few_shots(batch, series=self.series, max_examples=3, exclude_episode=base_name)
                        ca = kb.retrieve_cautions(batch, max_cautions=2)
                        if fs or ca:
                            rag_data = {"few_shots": fs, "cautions": ca}
                            rag_hash = hashlib.sha256(json.dumps(rag_data, sort_keys=True).encode("utf-8")).hexdigest()[:16]
                    except Exception:
                        pass

                # Check cache for cues in batch
                uncached_cues: List[Cue] = []
                for cue in batch:
                    target_m = candidate_gemini_models[curr_gemini_idx]
                    ckey = compute_cue_cache_key(
                        model=target_m,
                        ja=cue.ja,
                        en=cue.en,
                        glossary_hash=f"{len(glossary.terms)}:{len(glossary.characters)}" if glossary else "",
                        cast_sheet_hash=cast_sheet_hash,
                        prompt_version=prompt_version,
                        stage="draft",
                        rag_hash=rag_hash,
                    )
                    cached_val = self.cache.get_cue(ckey)
                    is_lower_model = "lite" in target_m
                    if cached_val and not (subs_upgrade and is_lower_model):
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
                model_used = candidate_gemini_models[curr_gemini_idx] if candidate_gemini_models else "none"

                # Bơm ngữ cảnh xưng hô phân tầng động từ HysteresisCastManager vào batch prompt
                if cast_manager and glossary:
                    dyn_cast_ctx = cast_manager.to_prompt_context()
                    if dyn_cast_ctx and dyn_cast_ctx not in glossary.style_rules:
                        glossary.style_rules.append(dyn_cast_ctx)

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
                                max_retries=self.max_retries_per_model,
                                model_name=target_model,
                            )
                            status_str = "dry_run" if self.dry_run else "success"
                            self.cache.log_request(model=target_model, backend="gemini", cues_count=len(uncached_cues), status=status_str)
                            stats["episode_requests"] = stats.get("episode_requests", 0) + 1
                            backend_used = "gemini"
                            model_used = target_model
                            stats["gemini"] += len(uncached_cues)
                            stats["model_counts"][target_model] = stats["model_counts"].get(target_model, 0) + len(uncached_cues)
                            translated = True
                            break
                        except DailyQuotaExhaustedError as dqe:
                            stats["episode_requests"] = stats.get("episode_requests", 0) + 1
                            console.print(f"\n[bold red]🛑 Model '{target_model}' đã cạn kiệt hạn mức ngày (RPD) - Reset lúc {dqe.reset_vn_str}! "
                                          f"CHUYỂN NGAY sang model kế tiếp trong chuỗi mà không thử lại...[/bold red]")
                            curr_gemini_idx += 1
                            if curr_gemini_idx < len(candidate_gemini_models):
                                next_model = candidate_gemini_models[curr_gemini_idx]
                                console.print(f"[bold cyan]🔄 Chuyển sang model kế tiếp trong chuỗi: '{next_model}'...[/bold cyan]")
                            else:
                                console.print("[bold yellow]🔄 Toàn bộ các model Gemini trong chuỗi đã hết hạn mức. Đánh dấu các câu là [Chưa dịch]...[/bold yellow]")
                                break
                        except Exception as e:
                            stats["episode_requests"] = stats.get("episode_requests", 0) + 1
                            logger.warning(f"⚠️ Model '{target_model}' gặp lỗi sau các lượt thử ({e}). Chuyển sang model fallback kế tiếp...")
                            curr_gemini_idx += 1
                            if curr_gemini_idx < len(candidate_gemini_models):
                                next_model = candidate_gemini_models[curr_gemini_idx]
                                logger.info(f"🔄 Chuyển sang model kế tiếp: '{next_model}'...")
                            else:
                                logger.warning("🔄 Toàn bộ các model Gemini trong chuỗi đều thất bại. Đánh dấu các câu là [Chưa dịch]...")
                                break

                    if not translated:
                        translated_texts = ["[Chưa dịch]"] * len(uncached_cues)
                        backend_used = "untranslated"
                        model_used = "none"

                # Store into cues and cache
                for cue, vi in zip(uncached_cues, translated_texts):
                    cue.vi = vi
                    cue.backend_used = backend_used
                    cue.model_used = model_used
                    if backend_used == "untranslated":
                        cue.uncertain = True
                        stats["model_counts"]["untranslated"] = stats["model_counts"].get("untranslated", 0) + 1
                    else:
                        ckey = compute_cue_cache_key(
                            model=model_used,
                            ja=cue.ja,
                            en=cue.en,
                            glossary_hash=f"{len(glossary.terms)}:{len(glossary.characters)}" if glossary else "",
                            cast_sheet_hash=cast_sheet_hash,
                            prompt_version=prompt_version,
                            stage="draft",
                            rag_hash=rag_hash,
                        )
                        self.cache.set_cue(ckey, backend_used, model_used, vi)
                    progress.advance(task, 1)

                # Ghi nhận phản hồi sau batch để cập nhật HysteresisCastManager
                if cast_manager:
                    for cue in uncached_cues:
                        if cue.speaker and cue.speaker != "unknown" and cue.addressee:
                            if cue.uncertain_reason and "pronoun" in cue.uncertain_reason.lower():
                                cast_manager.apply_deltas([
                                    CastRelationDelta(
                                        scope="SITUATIONAL_SHIFT",
                                        speaker=cue.speaker,
                                        listener=cue.addressee,
                                        pronoun_1p="tôi",
                                        pronoun_2p="ngươi",
                                        reason=f"Batch {batch_idx+1} contextual shift",
                                        confidence=0.80
                                    )
                                ])

        stage_times["stage_2_translate"] = time.monotonic() - t_s2

        # Optional Smart Cloud Cascade Repair Pass (100% Cloud API)
        t_srev = time.monotonic()
        if getattr(self, "enable_cascade", True) and not self.local_only and self.gemini_backends:
            repair_backend = (
                self.gemini_backends.get("gemini-3.8-flash")
                or self.gemini_backends.get("gemini-3.7-flash")
                or next(iter(self.gemini_backends.values()), None)
            )
            if repair_backend:
                from subtrans.cascade import run_cascade_repair_pass
                from subtrans.linter import auto_fix_cues
                # Run auto_fix_cues first to fix CPS and NFC deterministically before S(cue)
                dialogue_cues, fixes_count = auto_fix_cues(dialogue_cues)
                console.print(f"[bold cyan]🔍 [CASCADE PASS] Đánh giá và biên tập các câu khó bằng '{repair_backend.model}'...[/bold cyan]")
                repair_stats = run_cascade_repair_pass(
                    dialogue_cues,
                    repair_translator=repair_backend,
                    glossary=glossary,
                    series=self.series,
                    episode_basename=base_name,
                )
                stats["cascade_repaired"] = repair_stats.get("repaired_count", 0)
        stage_times["stage_review"] = time.monotonic() - t_srev

        # Stage 4: Writing output (Item 4b: Chốt chặn kiểm định chất lượng nội dung)
        t_s4 = time.monotonic()
        out_ext = self.output_format.lower()

        active_dialogue_cues = [c for c in cues if not c.is_skipped and not getattr(c, "is_dropped_sound", False)]
        total_active = len(active_dialogue_cues)
        real_translated_cues = [
            c for c in active_dialogue_cues
            if c.vi and not getattr(c, "used_en_fallback", False) and c.vi.strip() not in ["[Chưa dịch]", "(Chưa dịch)", "[Untranslated]"]
        ]
        real_rate = (len(real_translated_cues) / total_active) if total_active > 0 else 1.0
        uncertain_cues = [c for c in active_dialogue_cues if getattr(c, "uncertain", False)]
        uncertain_rate = (len(uncertain_cues) / total_active) if total_active > 0 else 0.0

        is_qualified = (real_rate >= 0.95 and uncertain_rate < 0.10)
        from src.config import add_pending_sub, remove_pending_sub
        import datetime

        if is_qualified:
            out_file = self.output_dir / f"{base_name}.vi.{out_ext}"
            remove_pending_sub(base_name)
        else:
            out_file = self.output_dir / f"{base_name}.vi.partial.{out_ext}"
            add_pending_sub({
                "base_name": base_name,
                "ja_path": str(ja_path),
                "en_path": str(en_path) if en_path else None,
                "output_dir": str(self.output_dir),
                "real_rate": round(real_rate, 4),
                "uncertain_rate": round(uncertain_rate, 4),
                "timestamp": datetime.datetime.now().isoformat()
            })
            console.print(
                f"\n[bold yellow]⚠️ [CHỐT CHẶN PHỤ ĐỀ] Tập '{base_name}' chưa đạt chuẩn nhúng "
                f"(Bản dịch thật: {real_rate*100:.1f}% < 95% hoặc Nghi vấn: {uncertain_rate*100:.1f}% >= 10%).[/bold yellow]\n"
                f"   • Đã ghi file tạm: [yellow]{out_file}[/yellow]\n"
                f"   • Đã thêm vào pending_subs.json (Dùng cờ --subs-pending để chạy lại sau).\n"
            )

        # Stage 4: Linter V1 Quality Verification & Subtitle Writing
        from subtrans.linter import lint_subtitles, auto_fix_cues
        cues, fixes_count = auto_fix_cues(cues)
        lint_violations = lint_subtitles(cues, glossary=glossary)
        if lint_violations:
            lint_errors = [v for v in lint_violations if v.severity == "ERROR"]
            lint_warnings = [v for v in lint_violations if v.severity == "WARNING"]
            console.print(
                f"[yellow]📋 [LINTER V1] Phát hiện {len(lint_violations)} cảnh báo/lỗi "
                f"({len(lint_errors)} lỗi, {len(lint_warnings)} cảnh báo, {fixes_count} auto-fixes).[/yellow]"
            )

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
        total_batches = 0
        split_batches = 0
        for b in self.gemini_backends.values():
            b_stats = getattr(b, "stats", None)
            if isinstance(b_stats, dict):
                try:
                    total_batches += int(b_stats.get("total_batches", 0))
                    split_batches += int(b_stats.get("split_batches", 0))
                except (TypeError, ValueError):
                    pass
        stats["total_batches"] = total_batches
        stats["split_batches"] = split_batches
        stats["split_rate_pct"] = (split_batches / max(1, total_batches)) * 100.0 if total_batches > 0 else 0.0
        stats["requests_this_episode"] = stats.get("episode_requests", 0)
        stats["batch_size_configured"] = self.batch_size
        stats["dry_run"] = self.dry_run
        stats["pacific_time_cues"] = self.cache.get_cues_today_pt()
        stats["pacific_time_requests"] = self.cache.get_requests_today_pt()
        stats["is_qualified"] = is_qualified
        stats["real_rate"] = real_rate
        stats["uncertain_rate"] = uncertain_rate
        stats["key_rotations"] = list(GeminiTranslator.all_rotation_events)
        stats["provenance"] = {
            "rag_enabled": self.enable_rag,
            "cascade_enabled": self.enable_cascade,
            "primary_model": self.gemini_model,
            "subs_quality": self.subs_quality,
            "dry_run": self.dry_run,
        }
        stats["offset_info"] = {
            "applied_offset_ms": sub_offset_ms,
            "detected_offset_ms": det_offset,
            "auto_offset_enabled": auto_offset,
            "auto_offset_applied": auto_offset_applied,
            "reason": offset_reason,
        }
        if cast_manager:
            stats["dynamic_cast_state"] = cast_manager.export_state()
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

        return {
            "output_path": saved_path,
            "subtitle_path": saved_path,
            "report_path": audit_file,
            "elapsed_seconds": elapsed,
            "stats": stats,
            "stage_times": stage_times,
            "is_qualified": is_qualified,
            "is_partial": not is_qualified,
            "real_rate": real_rate,
            "uncertain_rate": uncertain_rate,
        }

