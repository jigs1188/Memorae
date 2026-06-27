"""
main.py — Entry point for the Memorae personal-memory query engine.

Usage:
    python main.py                          # run all 5 queries, save results
    python main.py --query "your question"  # run a custom query
    python main.py --output results.json    # custom output file
    python main.py --no-llm                 # dry-run: show selected context only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

from core.config import SCENARIO_NOW
from core.event_store import load_events
from core.memory_extractor import MemoryExtractor
from core.memory_store import MemoryStore
from core.project_builder import ProjectBuilder
from core.query_engine import QueryEngine, QueryResult
from llm.llm_client import get_provider_info

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("memorae.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
console = Console()

DATA_PATH = Path(__file__).parent / "memorae_mock_events.json"


# ── Rich display ───────────────────────────────────────────────────────────────

def display_result(result: QueryResult, index: int) -> None:
    console.print()
    console.rule(f"[bold cyan]Query {index}: {result.query}[/bold cyan]")

    # Answer
    console.print(Panel(
        Markdown(result.answer),
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # Model used + token info
    console.print(
        f"  [dim]Model: [cyan]{result.model_used}[/cyan] | "
        f"Context: [yellow]{result.token_estimate} tokens[/yellow] | "
        f"Events used: [yellow]{len(result.selected_context)}[/yellow] | "
        f"Dropped: [yellow]{result.dropped_count}[/yellow][/dim]"
    )

    # Contradiction notes
    if getattr(result, "contradiction_notes", None):
        console.print("\n[bold yellow]! Contradiction / Update Resolution:[/bold yellow]")
        for note in result.contradiction_notes:
            console.print(f"  • {note}")

    # Top 5 selected events
    console.print("\n[bold]Top selected events:[/bold]")
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    tbl.add_column("Timestamp", style="dim", width=22)
    tbl.add_column("Source", width=14)
    tbl.add_column("Score", width=7, justify="right")
    tbl.add_column("Content preview", no_wrap=False)
    for ev in result.selected_context[:8]:
        tbl.add_row(
            ev["timestamp"][:19],
            ev["source"],
            str(ev["relevance_score"]),
            ev["content"][:80],
        )
    console.print(tbl)

    # Reasoning summary
    r = result.reasoning
    console.print(f"\n[dim]Why ignored: {r.get('why_ignored', '')[:150]}[/dim]")
    if r.get("uncertainty"):
        console.print(f"[dim yellow]Uncertainty: {r['uncertainty'][:200]}[/dim yellow]")


# ── Serializer ─────────────────────────────────────────────────────────────────

def result_to_dict(result: QueryResult) -> dict:
    return {
        "query": result.query,
        "answer": result.answer,
        "model_used": result.model_used,
        "context_stats": {
            "token_estimate": result.token_estimate,
            "events_used": len(result.selected_context),
            "events_dropped": result.dropped_count,
        },
        "selected_context": result.selected_context,
        "reasoning": result.reasoning,
        "contradiction_notes": result.contradiction_notes,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Memorae personal-memory query engine")
    parser.add_argument("--query", "-q", type=str, help="Run a single custom query")
    parser.add_argument("--output", "-o", type=str, default="results.json", help="Output JSON file")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM; show context only")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to events JSON")
    args = parser.parse_args()

    # ── Load scenario time
    now = datetime.fromisoformat(SCENARIO_NOW.replace("Z", "+00:00"))
    pinfo = get_provider_info()
    key_ok = "[green]configured[/green]" if pinfo["key_configured"] else "[red]MISSING[/red]"
    console.print(Panel(
        f"[bold]Memorae Personal Memory Engine[/bold]\n"
        f"Scenario time:  [cyan]{now.strftime('%Y-%m-%d %H:%M UTC')}[/cyan]\n"
        f"LLM Provider:   [magenta]{pinfo['provider']}[/magenta]  (key: {key_ok})\n"
        f"Primary model:  [yellow]{pinfo['primary_model']}[/yellow]\n"
        f"Data file:      [yellow]{args.data}[/yellow]",
        border_style="blue",
    ))

    # ── Load events & memories
    console.print(f"\n[bold]Loading events from:[/bold] {args.data}")
    events = load_events(args.data)
    extractor = MemoryExtractor()
    memories = extractor.extract_memories(events)
    store = MemoryStore(memories, now)
    stats = store.stats()
    
    project_builder = ProjectBuilder(now)
    projects = project_builder.build_projects(memories)
    
    console.print(
        f"  Memories extracted: [cyan]{stats['total']}[/cyan] | "
        f"With urgency: [yellow]{stats['with_urgency']}[/yellow] | "
        f"Projects found: [green]{len(projects)}[/green]"
    )

    engine = QueryEngine(store, now, projects=projects)

    if args.query:
        queries = [args.query]
    else:
        # Fallback preset queries if no query is specified
        queries = [
            "What should I focus on today?",
            "What commitments am I at risk of missing?",
            "What have I been procrastinating on?",
            "Summarize everything related to the UIE proposal.",
            "What personal/family tasks need my attention?",
        ]

    # ── Run queries
    all_results: list[QueryResult] = []
    for i, q in enumerate(queries, 1):
        console.print(f"\n[bold cyan]Running query {i}/{len(queries)}...[/bold cyan]")

        if args.no_llm:
            # Dry-run: just show what would be selected
            # Currently disabled for generic queries without LLM, would need refactoring for dry-run
            console.print(f"  Dry-run not fully supported with generic queries yet.")
            continue

        result = engine.run(q)
        all_results.append(result)
        display_result(result, i)

    # ── Save results
    if all_results and not args.no_llm:
        output_path = Path(args.output)
        
        # Convert dataclasses to dicts
        results_dicts = [result_to_dict(r) for r in all_results]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_dicts, f, indent=2, ensure_ascii=False)
            
        console.print(f"\n[bold green]✓ Results JSON saved to {output_path.resolve()}[/bold green]")

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
