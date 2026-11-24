"""Typer CLI commands (run, batch, eval, rag, check, models) for subtitle translation."""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
import requests

# Configure Windows console to UTF-8 to prevent charmap encoding errors
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from subtrans.config import get_gemini_api_key, load_models_config
from subtrans.pipeline import TranslationPipeline

app = typer.Typer(
    name="subtrans",
    help="CLI tool for translating anime subtitles (Japanese + English) into Vietnamese via Cloud Gemini API.",
    add_completion=False,
)
console = Console(force_terminal=True, legacy_windows=False)


@app.command()
def check():
    """Verifies environment, API keys, and dependencies."""
    console.print("\n[bold cyan]🔍 Inspecting subtrans Environment[/bold cyan]\n")
    table = Table(title="Component Health Check")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    # 1. Check Python Dependencies
    try:
        import pysubs2
        import google.genai
        table.add_row("Python Packages", "[green]OK[/green]", "pysubs2, google-genai, typer, rich available")
    except ImportError as e:
        table.add_row("Python Packages", "[red]MISSING[/red]", str(e))

    # 2. Check Gemini API Keys
    from subtrans.config import get_gemini_api_keys
    api_keys = get_gemini_api_keys()
    if api_keys:
        masked_list = [f"{k[:6]}...{k[-4:]}" for k in api_keys]
        try:
            from google import genai
            from subtrans.cache import execute_and_log_api_request
            # Ping thử nghiệm với key đầu tiên
            client = genai.Client(api_key=api_keys[0])
            _ = execute_and_log_api_request(
                lambda: list(client.models.list(config={"page_size": 5})),
                model="gemini-3.8-flash",
                backend="gemini",
                cues_count=0,
                request_type="ping_check"
            )
            desc_keys = f"{len(api_keys)} keys active: {', '.join(masked_list)}"
            table.add_row("Gemini API", "[green]READY[/green]", desc_keys)
        except Exception as e:
            table.add_row("Gemini API", "[yellow]KEY ERROR[/yellow]", f"Invalid key or network failure: {e}")
    else:
        table.add_row("Gemini API", "[yellow]NOT SET[/yellow]", "GEMINI_API_KEY missing in .env")

    console.print(table)
    console.print("")


@app.command()
def models():
    """Lists configured active Gemini model chain from models.yaml and verifies availability."""
    console.print("\n[bold cyan]📋 Subtrans Translation Models Status[/bold cyan]\n")

    models_cfg = load_models_config()
    raw_chain = models_cfg.get("chain", [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
    ])
    active_chain = [m for m in raw_chain if m and m != "local"]
    models_meta = models_cfg.get("models", {})
    configured_primary = active_chain[0] if active_chain else "gemini-3.5-flash-lite"
    configured_fallbacks = active_chain[1:]

    console.print(f"[bold]Active Fallback Chain (subtrans/models.yaml):[/bold] {' -> '.join(active_chain)}")

    # Gemini Models API check
    api_key = get_gemini_api_key()
    console.print("\n[bold]Configured Gemini Models Status (Google GenAI):[/bold]")
    available_api_models = set()
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            for m in client.models.list():
                raw_name = m.name.replace("models/", "")
                available_api_models.add(raw_name)

            # Check Primary
            prim_meta = models_meta.get(configured_primary, {})
            prim_rpd = prim_meta.get("rpd", "?")
            if configured_primary in available_api_models:
                console.print(f"  • Primary: [bold green]AVAILABLE[/bold green] - [cyan]{configured_primary}[/cyan] (RPD: {prim_rpd})")
            else:
                console.print(f"  • Primary: [bold red]MISSING (404/NOT FOUND)[/bold red] - [yellow]{configured_primary}[/yellow] ⚠️ Check deprecations!")

            # Check Fallbacks
            for fb in configured_fallbacks:
                fb_meta = models_meta.get(fb, {})
                fb_rpd = fb_meta.get("rpd", "?")
                if fb in available_api_models:
                    console.print(f"  • Fallback: [bold green]AVAILABLE[/bold green] - [cyan]{fb}[/cyan] (RPD: {fb_rpd})")
                else:
                    console.print(f"  • Fallback: [bold red]MISSING[/bold red] - [yellow]{fb}[/yellow]")

            # Other active flash/pro models on this key
            other_models = sorted([
                m for m in available_api_models
                if ("flash" in m or "pro" in m) and m not in set(active_chain)
            ])
            if other_models:
                console.print("\n[bold dim]Other active Flash/Pro models on your API key:[/bold dim]")
                for om in other_models:
                    console.print(f"  - [dim]{om}[/dim]")

        except Exception as e:
            console.print(f"  [red]Failed to query Gemini API: {e}[/red]")
    else:
        console.print("  [yellow]GEMINI_API_KEY not configured. Set GEMINI_API_KEY in .env[/yellow]")

    console.print("\n[bold]Official Google Gemini Documentation & Reference:[/bold]")
    console.print("  • Latest Models: [link=https://ai.google.dev/gemini-api/docs/models/gemini]https://ai.google.dev/gemini-api/docs/models/gemini[/link]")
    console.print("  • Deprecations Schedule: [link=https://ai.google.dev/gemini-api/docs/deprecations]https://ai.google.dev/gemini-api/docs/deprecations[/link]")
    console.print("")


@app.command()
def run(
    ja_file: Path = typer.Argument(..., help="Path to Japanese subtitle file (.ass or .srt)", exists=True),
    en_file: Optional[Path] = typer.Option(None, "--en", "-e", help="Path to English subtitle file (.ass or .srt)"),
    en_offset_ms: int = typer.Option(0, "--en-offset-ms", help="Offset applied to English timing in milliseconds"),
    sub_offset_ms: int = typer.Option(0, "--sub-offset-ms", help="Offset applied to Japanese/Output subtitle timing in milliseconds (e.g. -11154)"),
    glossary: Optional[Path] = typer.Option(None, "--glossary", "-g", help="Path to glossary YAML/JSON file"),
    local_only: bool = typer.Option(False, "--local", "-l", help="[Removed] Local Ollama/TranslateGemma backend is no longer supported"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override Gemini model name"),
    allow_latest_alias: bool = typer.Option(False, "--allow-latest-alias", help="Allow using mutable model alias like gemini-flash-latest"),
    on_quota: str = typer.Option("stop", "--on-quota", help="Action when Gemini quota exhausted ('stop')"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit translation to first N dialogue cues"),
    out_dir: Path = typer.Option(Path("out"), "--out-dir", "-o", help="Output directory"),
    format_: str = typer.Option("srt", "--format", "-f", help="Output format ('srt' or 'ass')"),
    wrap: Optional[int] = typer.Option(None, "--wrap", "-w", help="Wrap lines exceeding N characters into at most 2 lines (default: smart-wrap > 72 chars)"),
    gloss_skill_names: bool = typer.Option(False, "--gloss-skill-names", help="Append English skill/attack name in parentheses"),
    require_approved_cast: bool = typer.Option(False, "--require-approved-cast", help="Pause to inspect Stage A cast sheet before continuing"),
    review_file: bool = typer.Option(False, "--review-file", help="Generate out/<name>.review.srt containing uncertain lines"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size in cues (100–150 cues, default: 100)"),
    subs_quality: str = typer.Option("normal", "--subs-quality", help="Chất lượng dịch: 'normal' (Lite trước, tối ưu quota) hoặc 'high' (Flash trước)"),
    subs_upgrade: bool = typer.Option(False, "--subs-upgrade", help="Ưu tiên model Flash trước và bỏ qua cache Lite để nâng cấp bản dịch"),
    auto_offset: bool = typer.Option(False, "--auto-offset", help="Tự động phát hiện và áp dụng lệch thời gian JA-EN nếu thỏa mãn tiêu chí (mặc định: tắt)"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Tắt truy hồi RAG few-shot và error bank"),
    no_cascade: bool = typer.Option(False, "--no-cascade", help="Tắt phân hệ Smart Cloud Cascade Repair pass (Stage P2)"),
    cache_db: Optional[Path] = typer.Option(None, "--cache-db", help="File SQLite cache riêng biệt phục vụ thí nghiệm đo lường"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Chạy thử nghiệm toàn bộ pipeline mà KHÔNG gọi API (bảo toàn 100% quota đêm)"),
    auto_align_en: bool = typer.Option(True, "--auto-align-en/--no-auto-align-en", help="Tự động ghim mốc thời gian Whisper theo track tiếng Anh phòng thu (Mặc định: Bật)"),
    smart_wrap: bool = typer.Option(True, "--smart-wrap/--no-smart-wrap", help="Tự động ngắt dòng thông minh khi câu > 72 ký tự (Mặc định: Bật)"),
    video: Optional[Path] = typer.Option(None, "--video", "-v", help="Đường dẫn file video MKV/MP4 gốc để phân tích âm sắc đổi giọng 5.1 FC cho SRT trần"),
    explicit_cache: bool = typer.Option(False, "--explicit-cache", help="Bật khóa cứng Explicit Context Cache (TTL 15 phút) trên máy chủ Gemini"),
):
    """Translates anime subtitles (JA source + optional EN reference) into Vietnamese."""
    if local_only:
        console.print(
            "\n[bold red]❌ Lỗi: Chế độ dịch cục bộ (--local / Ollama / TranslateGemma) đã bị loại bỏ; "
            "Subtrans hiện chỉ sử dụng 100% Cloud Gemini API.[/bold red]"
        )
        raise typer.Exit(code=1)

    pipeline = TranslationPipeline(
        local_only=False,
        model_override=model,
        allow_latest_alias=allow_latest_alias,
        on_quota=on_quota,
        enable_cascade=not no_cascade,
        enable_rag=not no_rag,
        limit_cues=limit,
        output_dir=out_dir,
        output_format=format_,
        wrap_chars=wrap,
        smart_wrap=smart_wrap,
        gloss_skill_names=gloss_skill_names,
        require_approved_cast=require_approved_cast,
        review_file=review_file,
        dry_run=dry_run,
        batch_size=batch_size,
        subs_quality=subs_quality,
        subs_upgrade=subs_upgrade,
        cache_db_path=cache_db,
        auto_align_whisper_to_en=auto_align_en,
        explicit_caching=explicit_cache,
    )

    eff_en_file = en_file
    if eff_en_file is None and not _EN_STEM_RE.search(ja_file.stem):
        eff_en_file = _match_en_subtitle(ja_file, ja_file.parent)

    try:
        pipeline.run(
            ja_path=ja_file,
            en_path=eff_en_file,
            glossary_path=glossary,
            en_offset_ms=en_offset_ms,
            sub_offset_ms=sub_offset_ms,
            subs_upgrade=subs_upgrade,
            auto_offset=auto_offset,
            video_path=video,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Process cancelled by user. Partial cache preserved in SQLite.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]❌ Pipeline Error: {e}[/bold red]")
        sys.exit(1)


@app.command()
def compare(
    file1: Path = typer.Argument(..., help="Path to first subtitle file", exists=True),
    file2: Path = typer.Argument(..., help="Path to second subtitle file", exists=True),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of cues to display"),
):
    """Side-by-side terminal comparison of two subtitle files."""
    import pysubs2
    s1 = pysubs2.load(str(file1))
    s2 = pysubs2.load(str(file2))

    table = Table(title=f"Comparison: {file1.name} vs {file2.name}")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Time", style="dim")
    table.add_column(file1.name, style="white")
    table.add_column(file2.name, style="green")

    count = min(len(s1), len(s2), limit)
    for i in range(count):
        e1 = s1[i]
        e2 = s2[i]
        start_s = e1.start // 1000
        m, s = divmod(start_s, 60)
        time_str = f"{m:02d}:{s:02d}"
        t1 = e1.text.replace(r"\N", " ").replace("\n", " ")
        t2 = e2.text.replace(r"\N", " ").replace("\n", " ")
        table.add_row(str(i + 1), time_str, t1, t2)

    console.print(table)


_SUBTITLE_INPUT_SUFFIXES = (".ass", ".srt", ".ssa", ".vtt")
_OUTPUT_ARTIFACT_SUFFIXES = tuple(
    f"{mid}{ext}"
    for mid in (".vi", ".vi.partial", ".partial", ".review")
    for ext in _SUBTITLE_INPUT_SUFFIXES
)
_JA_STEM_RE = re.compile(r"(?:\[cc\])?(?:[._]whisper)?(?:[._](?:ja|jpn))(?:\[cc\])?$", re.IGNORECASE)
_EN_STEM_RE = re.compile(r"(?:\[cc\])?(?:[._](?:en|eng))(?:\[cc\])?$", re.IGNORECASE)


def _match_en_subtitle(ja_file: Path, en_dir: Path) -> Optional[Path]:
    """Finds the matching English reference subtitle for a Japanese subtitle file."""
    if _EN_STEM_RE.search(ja_file.stem):
        return None
    base_stem = _JA_STEM_RE.sub("", ja_file.stem)
    exts = [ja_file.suffix] + [e for e in _SUBTITLE_INPUT_SUFFIXES if e.lower() != ja_file.suffix.lower()]
    candidates: List[Path] = []
    for ext in exts:
        candidates.extend(
            [
                en_dir / f"{base_stem}.en{ext}",
                en_dir / f"{base_stem}_en{ext}",
                en_dir / f"{base_stem}_eng{ext}",
                en_dir / f"{base_stem}.eng{ext}",
                en_dir / f"{base_stem}.en[cc]{ext}",
            ]
        )
    ja_resolved = ja_file.resolve()
    same_parent = en_dir.resolve() == ja_resolved.parent
    if not same_parent:
        candidates.append(en_dir / ja_file.name)
        for ext in _SUBTITLE_INPUT_SUFFIXES:
            candidates.append(en_dir / f"{base_stem}{ext}")

    for cand in candidates:
        if (
            cand.is_file()
            and not cand.name.lower().endswith(_OUTPUT_ARTIFACT_SUFFIXES)
            and cand.resolve() != ja_resolved
        ):
            return cand
    return None


def _collect_batch_input_files(input_dir: Path, en_dir: Optional[Path] = None) -> List[Path]:
    """Collects batch input subtitle files while skipping output artifacts and paired EN tracks."""
    raw_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in _SUBTITLE_INPUT_SUFFIXES
        and not f.name.lower().endswith(_OUTPUT_ARTIFACT_SUFFIXES)
    )
    non_en_base_stems = {
        _JA_STEM_RE.sub("", f.stem).lower()
        for f in raw_files
        if not _EN_STEM_RE.search(f.stem)
    }

    eff_en_dir = en_dir if en_dir is not None else input_dir
    same_en_dir = eff_en_dir.resolve() == input_dir.resolve()
    matched_en_resolved = set()
    if same_en_dir:
        for f in raw_files:
            if not _EN_STEM_RE.search(f.stem):
                matched = _match_en_subtitle(f, eff_en_dir)
                if matched is not None:
                    matched_en_resolved.add(matched.resolve())

    selected: List[Path] = []
    for f in raw_files:
        if _EN_STEM_RE.search(f.stem):
            en_base = _EN_STEM_RE.sub("", f.stem).lower()
            if en_base in non_en_base_stems or f.resolve() in matched_en_resolved:
                continue
        selected.append(f)
    return selected


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing subtitle files", exists=True),
    files: Optional[List[Path]] = typer.Option(None, "--file", help="Optional explicit subtitle file(s) to translate instead of scanning all files in input_dir"),
    en_dir: Optional[Path] = typer.Option(None, "--en-dir", help="Directory containing matching English subtitles"),
    en_offset_ms: int = typer.Option(0, "--en-offset-ms", help="Offset applied to English timing in milliseconds"),
    sub_offset_ms: int = typer.Option(0, "--sub-offset-ms", help="Offset applied to Japanese/Output subtitle timing in milliseconds (e.g. -11154)"),
    glossary: Optional[Path] = typer.Option(None, "--glossary", "-g", help="Glossary file"),
    local_only: bool = typer.Option(False, "--local", "-l", help="[Removed] Local Ollama/TranslateGemma backend is no longer supported"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override Gemini model name"),
    allow_latest_alias: bool = typer.Option(False, "--allow-latest-alias", help="Allow using mutable model alias like gemini-flash-latest"),
    on_quota: str = typer.Option("stop", "--on-quota", help="Action when Gemini quota exhausted ('stop')"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit translation to first N dialogue cues"),
    out_dir: Path = typer.Option(Path("out"), "--out-dir", "-o", help="Output directory"),
    format_: str = typer.Option("srt", "--format", "-f", help="Output format ('srt' or 'ass')"),
    wrap: Optional[int] = typer.Option(None, "--wrap", "-w", help="Wrap lines exceeding N characters into at most 2 lines (default: smart-wrap > 72 chars)"),
    gloss_skill_names: bool = typer.Option(False, "--gloss-skill-names", help="Append English skill/attack name in parentheses"),
    require_approved_cast: bool = typer.Option(False, "--require-approved-cast", help="Pause to inspect Stage A cast sheet before continuing"),
    review_file: bool = typer.Option(False, "--review-file", help="Generate out/<name>.review.srt containing uncertain lines"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size in cues (100–150 cues, default: 100)"),
    subs_quality: str = typer.Option("normal", "--subs-quality", help="Chất lượng dịch: 'normal' (Lite trước, tối ưu quota) hoặc 'high' (Flash trước)"),
    subs_upgrade: bool = typer.Option(False, "--subs-upgrade", help="Ưu tiên model Flash trước và bỏ qua cache Lite để nâng cấp bản dịch"),
    auto_offset: bool = typer.Option(False, "--auto-offset", help="Tự động phát hiện và áp dụng lệch thời gian JA-EN nếu thỏa mãn tiêu chí (mặc định: tắt)"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Tắt truy hồi RAG few-shot và error bank"),
    no_cascade: bool = typer.Option(False, "--no-cascade", help="Tắt phân hệ Smart Cloud Cascade Repair pass (Stage P2)"),
    cache_db: Optional[Path] = typer.Option(None, "--cache-db", help="File SQLite cache riêng biệt phục vụ thí nghiệm đo lường"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Chạy thử nghiệm mà không gọi API"),
    auto_align_en: bool = typer.Option(True, "--auto-align-en/--no-auto-align-en", help="Tự động ghim mốc thời gian Whisper theo track tiếng Anh phòng thu (Mặc định: Bật)"),
    smart_wrap: bool = typer.Option(True, "--smart-wrap/--no-smart-wrap", help="Tự động ngắt dòng thông minh khi câu > 72 ký tự (Mặc định: Bật)"),
    explicit_cache: bool = typer.Option(False, "--explicit-cache", help="Bật khóa cứng Explicit Context Cache (TTL 15 phút) trên máy chủ Gemini"),
):
    """Batch translates all Japanese subtitle files in a directory."""
    if local_only:
        console.print(
            "\n[bold red]❌ Lỗi: Chế độ dịch cục bộ (--local / Ollama / TranslateGemma) đã bị loại bỏ; "
            "Subtrans hiện chỉ sử dụng 100% Cloud Gemini API.[/bold red]"
        )
        raise typer.Exit(code=1)

    if files:
        seen_explicit: set[Path] = set()
        cand_explicit: List[Path] = []
        for raw_f in files:
            resolved_f = raw_f.resolve()
            if not resolved_f.is_file():
                console.print(f"[bold red]❌ Subtitle file not found: {raw_f}[/bold red]")
                raise typer.Exit(code=1)
            if resolved_f in seen_explicit or resolved_f.name.lower().endswith(_OUTPUT_ARTIFACT_SUFFIXES):
                continue
            seen_explicit.add(resolved_f)
            cand_explicit.append(resolved_f)
        non_en_by_parent = {
            (p.parent, _JA_STEM_RE.sub("", p.stem).lower())
            for p in cand_explicit
            if not _EN_STEM_RE.search(p.stem)
        }
        matched_explicit_en: set[Path] = set()
        for p in cand_explicit:
            if not _EN_STEM_RE.search(p.stem):
                m_en = _match_en_subtitle(p, en_dir if en_dir is not None else p.parent)
                if m_en is not None:
                    matched_explicit_en.add(m_en.resolve())
        sub_files = [
            p
            for p in cand_explicit
            if not (
                _EN_STEM_RE.search(p.stem)
                and (
                    (p.parent, _EN_STEM_RE.sub("", p.stem).lower()) in non_en_by_parent
                    or p in matched_explicit_en
                )
            )
        ]
    else:
        sub_files = _collect_batch_input_files(input_dir, en_dir)
    if not sub_files:
        console.print(f"[yellow]No .ass or .srt files found in {input_dir}[/yellow]")
        return

    console.print(f"[bold green]Found {len(sub_files)} subtitle files in {input_dir}[/bold green]")
    multi_parents = len({p.parent for p in sub_files}) > 1
    out_stem_counts: dict[str, int] = {}
    if multi_parents:
        for p_sf in sub_files:
            c_stem = re.sub(r"\.(ja|en|vi|jpn|eng|vie)$", "", p_sf.stem, flags=re.IGNORECASE).lower()
            out_stem_counts[c_stem] = out_stem_counts.get(c_stem, 0) + 1
    common_parent: Optional[Path] = None
    if multi_parents and any(cnt > 1 for cnt in out_stem_counts.values()):
        try:
            common_parent = Path(os.path.commonpath([str(p.parent) for p in sub_files]))
        except ValueError:
            common_parent = None

    failed_files: List[str] = []
    for idx, f in enumerate(sub_files):
        console.print(f"\n[{idx+1}/{len(sub_files)}] Processing {f.name}...")
        eff_file_en_dir = en_dir if en_dir is not None else f.parent
        en_f = None if _EN_STEM_RE.search(f.stem) else _match_en_subtitle(f, eff_file_en_dir)
        f_stem_key = re.sub(r"\.(ja|en|vi|jpn|eng|vie)$", "", f.stem, flags=re.IGNORECASE).lower()
        eff_out_dir = out_dir
        if multi_parents and out_stem_counts.get(f_stem_key, 0) > 1:
            rel_sub: Optional[Path] = None
            if common_parent is not None:
                try:
                    cand_rel = f.parent.relative_to(common_parent)
                    if str(cand_rel) not in ("", "."):
                        rel_sub = cand_rel
                except ValueError:
                    rel_sub = None
            eff_out_dir = out_dir / (rel_sub or Path(f.parent.name or f"dir_{idx+1}"))

        try:
            pipeline = TranslationPipeline(
                local_only=False,
                model_override=model,
                allow_latest_alias=allow_latest_alias,
                on_quota=on_quota,
                enable_cascade=not no_cascade,
                enable_rag=not no_rag,
                limit_cues=limit,
                output_dir=eff_out_dir,
                output_format=format_,
                wrap_chars=wrap,
                smart_wrap=smart_wrap,
                gloss_skill_names=gloss_skill_names,
                require_approved_cast=require_approved_cast,
                review_file=review_file,
                dry_run=dry_run,
                batch_size=batch_size,
                subs_quality=subs_quality,
                subs_upgrade=subs_upgrade,
                cache_db_path=cache_db,
                auto_align_whisper_to_en=auto_align_en,
                explicit_caching=explicit_cache,
            )
            pipeline.run(
                ja_path=f,
                en_path=en_f,
                glossary_path=glossary,
                en_offset_ms=en_offset_ms,
                sub_offset_ms=sub_offset_ms,
                subs_upgrade=subs_upgrade,
                auto_offset=auto_offset,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Process cancelled by user. Partial cache preserved in SQLite.[/yellow]")
            sys.exit(130)
        except Exception as e:
            console.print(f"\n[bold red]❌ Pipeline Error ({f.name}): {e}[/bold red]")
            failed_files.append(f.name)

    if failed_files:
        console.print(f"\n[bold red]❌ Batch finished with {len(failed_files)} failed file(s): {', '.join(failed_files)}[/bold red]")
        sys.exit(1)


@app.command(name="eval")
def eval_cmd():
    """Runs the quantitative evaluation harness against regression and pronoun benchmark suites."""
    from subtrans.eval.runner import run_evaluation
    run_evaluation()


rag_app = typer.Typer(help="Manage lightweight Subtitle RAG knowledge base and Error Bank.", add_completion=False)
app.add_typer(rag_app, name="rag")


@rag_app.command(name="stats")
def rag_stats():
    """Displays statistics of the RAG Knowledge Base and Error Bank."""
    from subtrans.rag import KnowledgeBase
    kb = KnowledgeBase()
    stats = kb.get_stats()
    table = Table(title="Knowledge Base (RAG & Error Bank) Status")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="bold green")
    table.add_row("Gold Subtitle Units", str(stats["gold_units_count"]))
    table.add_row("Distinct Anime Series", str(stats["distinct_series_count"]))
    table.add_row("Error Bank Rules", str(stats["error_bank_rules_count"]))
    console.print(table)


@rag_app.command(name="seed")
def rag_seed():
    """Seeds built-in fan-sub gold exemplars into the RAG Knowledge Base to eliminate cold-start."""
    from subtrans.rag import KnowledgeBase
    kb = KnowledgeBase()
    added = kb.seed_default_gold_exemplars()
    stats = kb.get_stats()
    console.print(
        f"\n[bold green]✅ Đã nạp thêm {added} câu mẫu chuẩn vào Knowledge Base "
        f"(Tổng hiện có: {stats['gold_units_count']} gold/reviewed units)![/bold green]\n"
    )


@rag_app.command(name="import")
@app.command(name="import-gold")
def rag_import(
    directory: Path = typer.Argument(..., help="Directory containing subtitle files (*.ja.ass, *.en.ass, *.vi.ass)", exists=True),
    series: str = typer.Option(..., "--series", "-s", help="Anime series identifier (e.g. 'tenken', 'futsutsuka')"),
):
    """Imports gold subtitle triplets (*.ja.ass, *.en.ass, *.vi.ass) into knowledge base."""
    from subtrans.rag import KnowledgeBase
    kb = KnowledgeBase()
    files = list(directory.glob("*.*"))
    episodes: dict = {}
    for f in files:
        fname = f.name
        if ".ja." in fname:
            stem = fname.split(".ja.")[0]
            episodes.setdefault(stem, {})["ja"] = f
        elif ".en." in fname:
            stem = fname.split(".en.")[0]
            episodes.setdefault(stem, {})["en"] = f
        elif ".vi." in fname:
            stem = fname.split(".vi.")[0]
            episodes.setdefault(stem, {})["vi"] = f

    total_imported = 0
    console.print(f"\n[bold cyan]📦 Đang nhập phụ đề mẫu vào Gold Units cho bộ '{series}'...[/bold cyan]")
    for stem, triplet in sorted(episodes.items()):
        if "ja" in triplet and "vi" in triplet:
            count = kb.import_subtitles_triplet(
                ja_path=triplet["ja"],
                en_path=triplet.get("en"),
                vi_path=triplet["vi"],
                series=series,
                episode=stem,
            )
            total_imported += count
            console.print(f"  • {stem}: [green]+{count} units[/green]")

    console.print(f"\n[bold green]✅ Đã nạp thành công {total_imported} gold subtitle units vào Knowledge Base![/bold green]\n")


if __name__ == "__main__":
    app()

