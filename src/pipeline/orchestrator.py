"""Full pipeline orchestrator with CLI interface."""
from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from src.config import PROJECT_ROOT

app = typer.Typer(name="acm", help="Asset Cluster Migration pipeline")
console = Console()

# ── Logging setup: Rich console + persistent file logs ────────────────
LOG_DIR = PROJECT_ROOT / "logs" / "pipeline"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_today = datetime.now().strftime("%Y%m%d")
_log_file = LOG_DIR / f"pipeline_{_today}.log"

_file_handler = RotatingFileHandler(
    _log_file, maxBytes=5_000_000, backupCount=10, encoding="utf-8",
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s", datefmt="%H:%M:%S")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True),
        _file_handler,
    ],
)

STEPS = [
    "fetch-data",
    "validate-data",
    "build-features",
    "run-clustering",
    "run-regimes",
    "run-migration",
    "compute-centrality",
    "export-topology",
]


def _log_run_summary(steps_run: list[str], duration_s: float, error: str | None = None) -> None:
    """Append a JSONL run summary to logs/run_summary.jsonl."""
    import json

    summary_path = PROJECT_ROOT / "logs" / "run_summary.jsonl"
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "started_at": datetime.now().isoformat(),
        "steps": steps_run,
        "duration_s": round(duration_s, 1),
        "status": "error" if error else "ok",
        "error": error,
    }
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@app.command()
def run_step(step: str) -> None:
    """Run a single pipeline step."""
    import time

    from src.pipeline.steps import (
        step_fetch_data,
        step_validate_data,
        step_build_features,
        step_run_clustering,
        step_run_regimes,
        step_run_migration,
        step_compute_centrality,
        step_export_topology,
    )

    dispatch = {
        "fetch-data": step_fetch_data,
        "validate-data": step_validate_data,
        "build-features": step_build_features,
        "run-clustering": step_run_clustering,
        "run-regimes": step_run_regimes,
        "run-migration": step_run_migration,
        "compute-centrality": step_compute_centrality,
        "export-topology": step_export_topology,
    }

    if step not in dispatch:
        console.print(f"[red]Unknown step: {step}[/red]")
        console.print(f"Available: {STEPS}")
        raise typer.Exit(1)

    t0 = time.time()
    console.print(f"[bold]Running step: {step}[/bold]")
    error = None
    try:
        dispatch[step]()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        _log_run_summary([step], time.time() - t0, error)
    console.print(f"[green]Step {step} complete[/green]")


@app.command()
def run_all(
    skip_fetch: bool = typer.Option(False, help="Skip data fetching (use cached)"),
    skip_export: bool = typer.Option(False, help="Skip topology export"),
) -> None:
    """Run full pipeline end-to-end."""
    import time

    from src.pipeline.steps import run_full_pipeline

    t0 = time.time()
    steps_run = list(STEPS)
    error = None

    try:
        if skip_fetch:
            console.print("[yellow]Skipping fetch step (using cached data)[/yellow]")
            steps_run = [s for s in STEPS if s != "fetch-data"]
            from src.pipeline.steps import (
                step_validate_data,
                step_build_features,
                step_run_clustering,
                step_run_regimes,
                step_run_migration,
                step_compute_centrality,
                step_export_topology,
            )

            step_validate_data()
            step_build_features()
            cluster_history, graph_history = step_run_clustering()
            step_run_regimes()
            step_run_migration(cluster_history, graph_history)
            step_compute_centrality(graph_history)
            if not skip_export:
                step_export_topology()
            else:
                steps_run = [s for s in steps_run if s != "export-topology"]
        else:
            if skip_export:
                steps_run = [s for s in steps_run if s != "export-topology"]
            run_full_pipeline(export_topology=not skip_export)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        _log_run_summary(steps_run, time.time() - t0, error)


@app.command()
def export_topology(
    target_dir: str = typer.Option(
        None,
        help="Directory to export topology parquets to (default: TOPOLOGY_EXPORT_DIR env var or data/exports/topology)",
    ),
) -> None:
    """Export topology parquets to an external directory (standalone)."""
    from src.pipeline.steps import step_export_topology

    step_export_topology(topology_dir=target_dir)


if __name__ == "__main__":
    app()
