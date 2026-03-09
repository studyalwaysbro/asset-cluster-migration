"""Full pipeline orchestrator with CLI interface."""
from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(name="acm", help="Asset Cluster Migration pipeline")
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

STEPS = [
    "fetch-data",
    "validate-data",
    "build-features",
    "run-baseline",
    "run-clustering",
    "run-regimes",
    "run-event-study",
    "run-migration",
    "generate-figures",
    "generate-report",
]


@app.command()
def run_step(step: str) -> None:
    """Run a single pipeline step."""
    if step not in STEPS:
        console.print(f"[red]Unknown step: {step}[/red]")
        console.print(f"Available: {STEPS}")
        raise typer.Exit(1)
    console.print(f"[bold]Running step: {step}[/bold]")
    # Step dispatch will be implemented as modules are completed
    console.print(f"[green]Step {step} complete[/green]")


@app.command()
def run_all() -> None:
    """Run full pipeline end-to-end."""
    for step in STEPS:
        run_step(step)


if __name__ == "__main__":
    app()
