# -*- coding: utf-8 -*-
"""
subtrans/cli.py
Command Line Interface for subtrans anime subtitle translation tool.
"""

import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
import requests

from subtrans.config import load_config, get_gemini_api_key
from subtrans.pipeline import TranslationPipeline

app = typer.Typer(
    name="subtrans",
    help="Local-first CLI tool for translating anime subtitles (Japanese + English) into Vietnamese.",
    add_completion=False,
)
console = Console()


@app.command()
def check():
    """Verifies environment, API keys, dependencies, and Ollama status."""
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
        table.add_row("Gemini API", "[yellow]NOT SET[/yellow]", "GEMINI_API_KEY missing in .env (will use local fallback)")

    # 3. Check Ollama
    cfg = load_config()
    ollama_url = cfg.get("ollama", {}).get("endpoint", "http://localhost:11434")
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m.get("name") for m in r.json().get("models", [])]
            m_str = ", ".join(models) if models else "No models pulled yet"
            table.add_row("Ollama Server", "[green]RUNNING[/green]", f"{ollama_url} (Found: {m_str})")
        else:
            table.add_row("Ollama Server", "[yellow]HTTP ERROR[/yellow]", f"Status {r.status_code}")
    except Exception:
        table.add_row("Ollama Server", "[red]OFFLINE[/red]", f"Cannot connect to {ollama_url}")

    console.print(table)
    console.print("")


@app.command()
def models():
    """Lists available Gemini models and locally installed Ollama models, verifying configured models."""
    console.print("\n[bold cyan]📋 Subtrans Translation Models Status[/bold cyan]\n")

    cfg = load_config()
    gemini_cfg = cfg.get("gemini", {})
    configured_primary = gemini_cfg.get("model", "gemini-3.8-flash")
    configured_fallbacks = gemini_cfg.get("fallback_models", ["gemini-3.7-flash", "gemini-3.6-flash"])
    configured_lite = gemini_cfg.get("lite_model", "gemini-3.5-flash-lite")

    # 1. Ollama Models
    ollama_url = cfg.get("ollama", {}).get("endpoint", "http://localhost:11434")
    console.print(f"[bold]Local Ollama Models ({ollama_url}):[/bold]")
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if r.status_code == 200:
            local_models = r.json().get("models", [])
            if local_models:
                for m in local_models:
                    size_gb = m.get("size", 0) / (1024 ** 3)
                    console.print(f"  • [green]{m.get('name')}[/green] ({size_gb:.1f} GB)")
            else:
                console.print("  [yellow]No models found. Run: ollama pull translategemma:12b[/yellow]")
        else:
            console.print(f"  [red]Failed to fetch models from Ollama (status {r.status_code})[/red]")
    except Exception as e:
        console.print(f"  [red]Ollama unreachable: {e}[/red]")

    console.print("")

    # 2. Gemini Models
    api_key = get_gemini_api_key()
    console.print("[bold]Configured Gemini Models Status (Google GenAI):[/bold]")
    available_api_models = set()
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            for m in client.models.list():
                raw_name = m.name.replace("models/", "")
                available_api_models.add(raw_name)

            # Check Primary
            if configured_primary in available_api_models:
                console.print(f"  • Primary: [bold green]AVAILABLE[/bold green] - [cyan]{configured_primary}[/cyan]")
            else:
                console.print(f"  • Primary: [bold red]MISSING (404/NOT FOUND)[/bold red] - [yellow]{configured_primary}[/yellow] ⚠️ Check deprecations!")

            # Check Fallbacks
            for fb in configured_fallbacks:
                if fb in available_api_models:
                    console.print(f"  • Fallback: [bold green]AVAILABLE[/bold green] - [cyan]{fb}[/cyan]")
                else:
                    console.print(f"  • Fallback: [bold red]MISSING[/bold red] - [yellow]{fb}[/yellow]")

            # Check Lite
            if configured_lite in available_api_models:
                console.print(f"  • Lite: [bold green]AVAILABLE[/bold green] - [cyan]{configured_lite}[/cyan]")
            else:
                console.print(f"  • Lite: [bold red]MISSING[/bold red] - [yellow]{configured_lite}[/yellow]")

            # Other active flash/pro models on this key
            other_models = sorted([
                m for m in available_api_models
                if ("flash" in m or "pro" in m) and m not in {configured_primary, configured_lite, *configured_fallbacks}
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
    local_only: bool = typer.Option(False, "--local", "-l", help="Force 100% local translation using Ollama"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override Gemini model name"),
    allow_latest_alias: bool = typer.Option(False, "--allow-latest-alias", help="Allow using mutable model alias like gemini-flash-latest"),
    on_quota: str = typer.Option("local", "--on-quota", help="Action when Gemini quota exhausted ('local' or 'stop')"),
    review: bool = typer.Option(False, "--review", "-r", help="Run local post-editing review pass using gemma3:12b"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit translation to first N dialogue cues"),
    out_dir: Path = typer.Option(Path("out"), "--out-dir", "-o", help="Output directory"),
    format_: str = typer.Option("srt", "--format", "-f", help="Output format ('srt' or 'ass')"),
    wrap: Optional[int] = typer.Option(None, "--wrap", "-w", help="Wrap lines exceeding N characters into at most 2 lines (default: single line)"),
    gloss_skill_names: bool = typer.Option(False, "--gloss-skill-names", help="Append English skill/attack name in parentheses"),
    require_approved_cast: bool = typer.Option(False, "--require-approved-cast", help="Pause to inspect Stage A cast sheet before continuing"),
    review_file: bool = typer.Option(False, "--review-file", help="Generate out/<name>.review.srt containing uncertain lines"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size in cues (100–150 cues, default: 100)"),
    subs_quality: str = typer.Option("normal", "--subs-quality", help="Chất lượng dịch: 'normal' (Lite trước, tối ưu quota) hoặc 'high' (Flash trước)"),
    auto_offset: bool = typer.Option(False, "--auto-offset", help="Tự động phát hiện và áp dụng lệch thời gian JA-EN nếu thỏa mãn tiêu chí (mặc định: tắt)"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Tắt truy hồi RAG few-shot và error bank"),
    no_cascade: bool = typer.Option(False, "--no-cascade", help="Tắt phân hệ Smart Cloud Cascade Repair pass (Stage P2)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Chạy thử nghiệm toàn bộ pipeline mà KHÔNG gọi API (bảo toàn 100% quota đêm)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm cloud privacy consent automatically"),
):
    """Translates anime subtitles (JA source + optional EN reference) into Vietnamese."""
    cfg = load_config()

    # Privacy check
    if not local_only and not yes and not dry_run:
        consent_given = cfg.get("privacy", {}).get("consent_given", False)
        if not consent_given:
            console.print("\n[bold yellow]⚠️  Privacy Notice[/bold yellow]")
            console.print("You are using Google Gemini Cloud API to translate anime subtitles.")
            console.print("Dialogue lines and glossary terms will be sent to Google API.")
            confirm = typer.confirm("Do you agree to send subtitle text to Gemini API?", default=True)
            if not confirm:
                console.print("[bold cyan]Switched to 100% local translation mode.[/bold cyan]")
                local_only = True

    pipeline = TranslationPipeline(
        local_only=local_only,
        model_override=model,
        allow_latest_alias=allow_latest_alias,
        on_quota=on_quota,
        enable_review=review,
        enable_cascade=not no_cascade,
        enable_rag=not no_rag,
        limit_cues=limit,
        output_dir=out_dir,
        output_format=format_,
        wrap_chars=wrap,
        gloss_skill_names=gloss_skill_names,
        require_approved_cast=require_approved_cast,
        review_file=review_file,
        dry_run=dry_run,
        batch_size=batch_size,
        subs_quality=subs_quality,
    )

    try:
        pipeline.run(
            ja_path=ja_file,
            en_path=en_file,
            glossary_path=glossary,
            en_offset_ms=en_offset_ms,
            sub_offset_ms=sub_offset_ms,
            auto_offset=auto_offset,
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


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing subtitle files", exists=True),
    en_dir: Optional[Path] = typer.Option(None, "--en-dir", help="Directory containing matching English subtitles"),
    out_dir: Path = typer.Option(Path("out"), "--out-dir", "-o", help="Output directory"),
    glossary: Optional[Path] = typer.Option(None, "--glossary", "-g", help="Glossary file"),
    local_only: bool = typer.Option(False, "--local", "-l", help="Force local mode"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size in cues (100–150 cues, default: 100)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Chạy thử nghiệm mà không gọi API"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto confirm consent"),
):
    """Batch translates all Japanese subtitle files in a directory."""
    sub_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in (".ass", ".srt")])
    if not sub_files:
        console.print(f"[yellow]No .ass or .srt files found in {input_dir}[/yellow]")
        return

    console.print(f"[bold green]Found {len(sub_files)} subtitle files in {input_dir}[/bold green]")
    for idx, f in enumerate(sub_files):
        console.print(f"\n[{idx+1}/{len(sub_files)}] Processing {f.name}...")
        en_f = None
        if en_dir:
            candidate = en_dir / f.name
            if candidate.exists():
                en_f = candidate

        pipeline = TranslationPipeline(
            local_only=local_only,
            output_dir=out_dir,
            dry_run=dry_run,
            batch_size=batch_size,
        )
        pipeline.run(ja_path=f, en_path=en_f, glossary_path=glossary)


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

