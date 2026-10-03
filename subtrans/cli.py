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

    # 2. Check Gemini API Key
    api_key = get_gemini_api_key()
    if api_key:
        masked_key = f"{api_key[:6]}...{api_key[-4:]}"
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            # Simple ping
            _ = list(client.models.list(config={"page_size": 5}))
            table.add_row("Gemini API", "[green]READY[/green]", f"Key active ({masked_key})")
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
    """Lists available Gemini models and locally installed Ollama models."""
    console.print("\n[bold cyan]📋 Available Translation Models[/bold cyan]\n")

    # Ollama Models
    cfg = load_config()
    ollama_url = cfg.get("ollama", {}).get("endpoint", "http://localhost:11434")
    console.print(f"[bold]Ollama Models ({ollama_url}):[/bold]")
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if r.status_code == 200:
            models = r.json().get("models", [])
            if models:
                for m in models:
                    size_gb = m.get("size", 0) / (1024 ** 3)
                    console.print(f"  • [green]{m.get('name')}[/green] ({size_gb:.1f} GB)")
            else:
                console.print("  [yellow]No models found. Run: ollama pull translategemma:12b[/yellow]")
        else:
            console.print(f"  [red]Failed to fetch models from Ollama (status {r.status_code})[/red]")
    except Exception as e:
        console.print(f"  [red]Ollama unreachable: {e}[/red]")

    console.print("")

    # Gemini Models
    api_key = get_gemini_api_key()
    console.print("[bold]Gemini Cloud Models (Google GenAI):[/bold]")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            count = 0
            for m in client.models.list():
                if "gemini" in m.name.lower() and ("flash" in m.name.lower() or "pro" in m.name.lower()):
                    console.print(f"  • [cyan]{m.name}[/cyan] ({m.display_name or 'Gemini'})")
                    count += 1
            if count == 0:
                console.print("  No flash/pro models listed.")
        except Exception as e:
            console.print(f"  [red]Failed to query Gemini API: {e}[/red]")
    else:
        console.print("  [yellow]GEMINI_API_KEY not configured. Set GEMINI_API_KEY in .env[/yellow]")

    console.print("")


@app.command()
def run(
    ja_file: Path = typer.Argument(..., help="Path to Japanese subtitle file (.ass or .srt)", exists=True),
    en_file: Optional[Path] = typer.Option(None, "--en", "-e", help="Path to English subtitle file (.ass or .srt)"),
    en_offset_ms: int = typer.Option(0, "--en-offset-ms", help="Offset applied to English timing in milliseconds"),
    glossary: Optional[Path] = typer.Option(None, "--glossary", "-g", help="Path to glossary YAML/JSON file"),
    local_only: bool = typer.Option(False, "--local", "-l", help="Force 100% local translation using Ollama"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override Gemini model name"),
    on_quota: str = typer.Option("local", "--on-quota", help="Action when Gemini quota exhausted ('local' or 'stop')"),
    review: bool = typer.Option(False, "--review", "-r", help="Run local post-editing review pass using gemma3:12b"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit translation to first N dialogue cues"),
    out_dir: Path = typer.Option(Path("out"), "--out-dir", "-o", help="Output directory"),
    format_: str = typer.Option("srt", "--format", "-f", help="Output format ('srt' or 'ass')"),
    wrap: Optional[int] = typer.Option(None, "--wrap", "-w", help="Wrap lines exceeding N characters into at most 2 lines (default: single line)"),
    gloss_skill_names: bool = typer.Option(False, "--gloss-skill-names", help="Append English skill/attack name in parentheses"),
    require_approved_cast: bool = typer.Option(False, "--require-approved-cast", help="Pause to inspect Stage A cast sheet before continuing"),
    review_file: bool = typer.Option(False, "--review-file", help="Generate out/<name>.review.srt containing uncertain lines"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm cloud privacy consent automatically"),
):
    """Translates anime subtitles (JA source + optional EN reference) into Vietnamese."""
    cfg = load_config()

    # Privacy check
    if not local_only and not yes:
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
        on_quota=on_quota,
        enable_review=review,
        limit_cues=limit,
        output_dir=out_dir,
        output_format=format_,
        wrap_chars=wrap,
        gloss_skill_names=gloss_skill_names,
        require_approved_cast=require_approved_cast,
        review_file=review_file,
    )

    try:
        pipeline.run(
            ja_path=ja_file,
            en_path=en_file,
            glossary_path=glossary,
            en_offset_ms=en_offset_ms,
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
        )
        pipeline.run(ja_path=f, en_path=en_f, glossary_path=glossary)


if __name__ == "__main__":
    app()
