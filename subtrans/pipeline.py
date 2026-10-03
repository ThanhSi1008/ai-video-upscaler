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

        # Models & batch sizes
        gemini_cfg = self.cfg.get("gemini", {})
        self.gemini_model = model_override or gemini_cfg.get("model", "gemini-2.5-flash")
        self.gemini_rpm = gemini_cfg.get("max_rpm", 15)

        ollama_cfg = self.cfg.get("ollama", {})
        self.ollama_endpoint = ollama_cfg.get("endpoint", "http://localhost:11434")
        self.ollama_model = ollama_cfg.get("model", "translategemma:12b")
        self.ollama_review_model = ollama_cfg.get("review_model", "gemma3:12b")

        trans_cfg = self.cfg.get("translation", {})
        self.batch_size = trans_cfg.get("batch_size", 50)
        self.local_batch_size = trans_cfg.get("local_batch_size", 20)

        # Cache & Rate Limiter
        self.cache = TranslationCache()
        self.rate_limiter = RateLimiter(max_rpm=self.gemini_rpm)

        # Backends
        self.gemini_backend: Optional[GeminiTranslator] = None
        self.ollama_backend: Optional[OllamaTranslator] = None
        self.reviewer: Optional[LocalReviewer] = None

    def _init_primary_backend(self):
        if not self.local_only:
            api_key = get_gemini_api_key()
            if api_key:
                try:
                    self.gemini_backend = GeminiTranslator(
                        api_key=api_key,
                        model=self.gemini_model,
                        gloss_skill_names=self.gloss_skill_names,
                    )
                except Exception as e:
                    console.print(f"[yellow]⚠️  Gemini initialization failed: {e}. Falling back to local.[/yellow]")
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

        # Stage 1: Parse and align
        with console.status("[bold blue]Stage 1: Parsing and aligning subtitle streams...[/bold blue]"):
            cues, ja_ssa, en_ssa = parse_and_align_subtitles(
                ja_path=str(ja_path),
                en_path=str(en_path) if en_path else None,
                en_offset_ms=en_offset_ms,
            )

        if self.limit_cues:
            cues = cues[:self.limit_cues]
            console.print(f"[yellow]ℹ️  Limited run to first {self.limit_cues} cues.[/yellow]")

        dialogue_cues = [c for c in cues if not c.is_skipped]
        console.print(f"  • Total Cues: {len(cues)} ({len(dialogue_cues)} dialogue lines to translate)")

        # Stage A: Cast Sheet generation (runs automatically once per episode)
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

        stats = {
            "gemini": 0,
            "local": 0,
            "cache_hits": 0,
            "reviewed": 0,
        }

        # Stage 2 & 3: Translation loop (Batch size 50, Context +/-10)
        curr_batch_size = self.local_batch_size if self.local_only else self.batch_size
        dialogue_batches = [
            dialogue_cues[i:i + curr_batch_size]
            for i in range(0, len(dialogue_cues), curr_batch_size)
        ]

        active_backend_mode = "local" if self.local_only else "gemini"

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
                    ckey = compute_cue_cache_key(
                        model=self.ollama_model if active_backend_mode == "local" else self.gemini_model,
                        ja=cue.ja,
                        en=cue.en,
                        glossary_hash=str(len(glossary.terms)) if glossary else "",
                    )
                    cached_val = self.cache.get_cue(ckey)
                    if cached_val:
                        cue.vi = cached_val
                        cue.backend_used = "cache"
                        stats["cache_hits"] += 1
                        progress.advance(task, 1)
                    else:
                        uncached_cues.append(cue)

                if not uncached_cues:
                    continue

                # 2. Translate uncached cues
                translated_texts: List[str] = []
                backend_used = active_backend_mode

                if active_backend_mode == "gemini" and self.gemini_backend:
                    try:
                        # Extract surrounding context: +/-10 cues per ADDENDUM 6
                        first_id = uncached_cues[0].id
                        last_id = uncached_cues[-1].id
                        context = [c for c in cues if (first_id - 10 <= c.id < first_id) or (last_id < c.id <= last_id + 10)]

                        translated_texts = call_with_retry(
                            fn=lambda: self.gemini_backend.translate_batch(uncached_cues, context, glossary),
                            rate_limiter=self.rate_limiter,
                        )
                        stats["gemini"] += len(uncached_cues)
                    except QuotaExhaustedError as qe:
                        console.print(f"\n[bold red]⚠️  Gemini quota exhausted: {qe}[/bold red]")
                        if self.on_quota == "local":
                            console.print("[bold yellow]🔄 Switching remaining pipeline to local Ollama fallback...[/bold yellow]")
                            active_backend_mode = "local"
                            backend_used = "local"
                            translated_texts = self.ollama_backend.translate_batch(uncached_cues, [], glossary)
                            stats["local"] += len(uncached_cues)
                        else:
                            console.print("[bold red]🛑 Stopping execution as requested by --on-quota stop.[/bold red]")
                            break
                    except Exception as e:
                        logger.error(f"Gemini translation error: {e}. Falling back to local for this batch.")
                        translated_texts = self.ollama_backend.translate_batch(uncached_cues, [], glossary)
                        backend_used = "local"
                        stats["local"] += len(uncached_cues)
                else:
                    translated_texts = self.ollama_backend.translate_batch(uncached_cues, [], glossary)
                    stats["local"] += len(uncached_cues)

                # Store into cues and cache
                for cue, vi in zip(uncached_cues, translated_texts):
                    cue.vi = vi
                    cue.backend_used = backend_used
                    ckey = compute_cue_cache_key(
                        model=self.ollama_model if backend_used == "local" else self.gemini_model,
                        ja=cue.ja,
                        en=cue.en,
                        glossary_hash=str(len(glossary.terms)) if glossary else "",
                    )
                    self.cache.set_cue(ckey, backend_used, self.ollama_model if backend_used == "local" else self.gemini_model, vi)
                    progress.advance(task, 1)

        if self.ollama_backend:
            self.ollama_backend.unload()

        # Optional review
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

        # Stage 4: Writing output (SRT default, exactly ONE line per cue unless --wrap)
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

        elapsed = time.monotonic() - start_time

        # Stage C: Audit Report
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
            )

        console.print(f"\n[bold green]✅ Translation Completed in {elapsed:.1f}s![/bold green]")
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
        }
