"""Council & research output logger.

Persists council recommendations, research reports, and status audits
to timestamped markdown files in logs/council/. Also appends a JSONL
index entry for machine-readable querying.

Usage from cron or interactive sessions:

    from src.pipeline.council_logger import log_council_output, log_research_output

    log_council_output("Status report title", body_markdown)
    log_research_output("Experiment name", body_markdown, metadata={...})
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

COUNCIL_DIR = PROJECT_ROOT / "logs" / "council"
RESEARCH_DIR = PROJECT_ROOT / "logs" / "research"
INDEX_PATH = PROJECT_ROOT / "logs" / "council_index.jsonl"


def _ensure_dirs() -> None:
    COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


def log_council_output(
    title: str,
    body: str,
    *,
    tags: list[str] | None = None,
) -> Path:
    """Save a council output to logs/council/ with timestamp.

    Returns the path to the saved file.
    """
    _ensure_dirs()
    now = datetime.now()
    slug = title.lower().replace(" ", "-").replace("/", "-")[:60]
    filename = f"council-{now.strftime('%Y-%m-%d')}-{slug}.md"
    filepath = COUNCIL_DIR / filename

    # If file exists (multiple councils same day same topic), append sequence
    seq = 1
    while filepath.exists():
        seq += 1
        filename = f"council-{now.strftime('%Y-%m-%d')}-{slug}-{seq}.md"
        filepath = COUNCIL_DIR / filename

    content = f"# {title}\n"
    content += f"**Date:** {now.strftime('%Y-%m-%d %H:%M ET')}\n"
    content += f"**Type:** Council Output\n"
    if tags:
        content += f"**Tags:** {', '.join(tags)}\n"
    content += f"\n---\n\n{body}\n"

    filepath.write_text(content, encoding="utf-8")

    # Append to JSONL index
    entry = {
        "ts": now.isoformat(),
        "type": "council",
        "title": title,
        "file": str(filepath.relative_to(PROJECT_ROOT)),
        "tags": tags or [],
    }
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(f"Council output saved: {filepath.name}")
    return filepath


def log_research_output(
    title: str,
    body: str,
    *,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> Path:
    """Save a research report to logs/research/ with timestamp.

    Returns the path to the saved file.
    """
    _ensure_dirs()
    now = datetime.now()
    slug = title.lower().replace(" ", "-").replace("/", "-")[:60]
    filename = f"research-{now.strftime('%Y-%m-%d')}-{slug}.md"
    filepath = RESEARCH_DIR / filename

    seq = 1
    while filepath.exists():
        seq += 1
        filename = f"research-{now.strftime('%Y-%m-%d')}-{slug}-{seq}.md"
        filepath = RESEARCH_DIR / filename

    content = f"# {title}\n"
    content += f"**Date:** {now.strftime('%Y-%m-%d %H:%M ET')}\n"
    content += f"**Type:** Research Output\n"
    if tags:
        content += f"**Tags:** {', '.join(tags)}\n"
    if metadata:
        content += f"\n**Metadata:**\n```json\n{json.dumps(metadata, indent=2)}\n```\n"
    content += f"\n---\n\n{body}\n"

    filepath.write_text(content, encoding="utf-8")

    # Append to JSONL index
    entry = {
        "ts": now.isoformat(),
        "type": "research",
        "title": title,
        "file": str(filepath.relative_to(PROJECT_ROOT)),
        "tags": tags or [],
        "metadata": metadata or {},
    }
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(f"Research output saved: {filepath.name}")
    return filepath


def log_training_run(
    model_name: str,
    metrics: dict,
    *,
    tags: list[str] | None = None,
) -> None:
    """Append a training run entry to logs/training_runs.jsonl.

    For any model training (HMM, clustering parameters, etc).
    """
    training_log = PROJECT_ROOT / "logs" / "training_runs.jsonl"
    entry = {
        "ts": datetime.now().isoformat(),
        "model": model_name,
        "metrics": metrics,
        "tags": tags or [],
    }
    with open(training_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"Training run logged: {model_name}")
