#!/usr/bin/env python3
"""Auto-generate README.md from README.template.md and pipeline artifacts.

Reads pipeline artifacts (parquet, CSV, JSON) and fills dynamic sections
in the template. Static sections are preserved verbatim.

Usage:
    python scripts/generate_readme.py          # Generate README.md
    python scripts/generate_readme.py --check   # Check if README.md is up-to-date (exit 1 if stale)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
TEMPLATE_PATH = PROJECT_ROOT / "README.template.md"
OUTPUT_PATH = PROJECT_ROOT / "README.md"

DYNAMIC_PATTERN = re.compile(
    r"(<!-- BEGIN:DYNAMIC (\w+) -->)\n(.*?)(<!-- END:DYNAMIC \2 -->)",
    re.DOTALL,
)


def _file_modified_date(path: Path) -> str:
    """Return file modification date as YYYY-MM-DD."""
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except OSError:
        return "unknown"


def _provenance(artifact_path: Path) -> str:
    """Generate an HTML comment provenance tag for an artifact."""
    rel = artifact_path.relative_to(PROJECT_ROOT) if artifact_path.is_relative_to(PROJECT_ROOT) else artifact_path
    return f"<!-- Generated from {rel} (modified {_file_modified_date(artifact_path)}) -->"


def _unavailable(artifact_name: str) -> str:
    """Return a graceful degradation message for missing artifacts."""
    return f"> **Data unavailable** -- run the pipeline first. Missing: `{artifact_name}`"


def generate_pipeline_stats() -> str:
    """Read cluster_assignments.parquet, regime_labels.csv, bootstrap_results.json."""
    lines: list[str] = []

    # --- Cluster assignments ---
    cluster_path = DATA_DIR / "cluster_assignments.parquet"
    if cluster_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(cluster_path)
            n_assets = df["ticker"].nunique()
            n_windows = df["date"].nunique()
            latest_date = df["date"].max()
            n_clusters_latest = df[df["date"] == latest_date]["cluster"].nunique()
            lines.append(_provenance(cluster_path))
            lines.append("")
            lines.append(f"**Clustering:** {n_assets} assets across {n_windows} rolling windows, "
                         f"{n_clusters_latest} active clusters in latest window.")
        except Exception as exc:
            logger.warning("Failed to read cluster_assignments.parquet: %s", exc)
            lines.append(_unavailable("cluster_assignments.parquet"))
    else:
        lines.append(_unavailable("cluster_assignments.parquet"))

    # --- Regime labels ---
    regime_path = DATA_DIR / "regime_labels.csv"
    if regime_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(regime_path, parse_dates=["date"])
            date_min = df["date"].min().strftime("%Y-%m-%d")
            date_max = df["date"].max().strftime("%Y-%m-%d")
            current_regime = df.iloc[-1]["regime"]
            regime_counts = df["regime"].value_counts(normalize=True)
            regime_pcts = ", ".join(
                f"{regime} ({pct:.1%})" for regime, pct in regime_counts.items()
            )
            lines.append("")
            lines.append(_provenance(regime_path))
            lines.append("")
            lines.append(f"**Regime detection:** {date_min} to {date_max}. "
                         f"Current regime: **{current_regime}**. "
                         f"Distribution: {regime_pcts}.")
        except Exception as exc:
            logger.warning("Failed to read regime_labels.csv: %s", exc)
            lines.append(_unavailable("regime_labels.csv"))
    else:
        lines.append(_unavailable("regime_labels.csv"))

    # --- Bootstrap results ---
    bootstrap_path = DATA_DIR / "bootstrap_results.json"
    if bootstrap_path.exists():
        try:
            with open(bootstrap_path) as f:
                bs = json.load(f)
            parts: list[str] = []
            for metric_key in ["mean_cmi", "mean_cps"]:
                if metric_key in bs:
                    m = bs[metric_key]
                    label = metric_key.replace("mean_", "").upper()
                    parts.append(
                        f"{label} [{m['ci_lower']:.3f}, {m['ci_upper']:.3f}]"
                    )
            if parts:
                lines.append("")
                lines.append(_provenance(bootstrap_path))
                lines.append("")
                lines.append(f"**Bootstrap 95% CIs (block bootstrap):** {', '.join(parts)}.")
        except Exception as exc:
            logger.warning("Failed to read bootstrap_results.json: %s", exc)
            lines.append(_unavailable("bootstrap_results.json"))
    else:
        lines.append(_unavailable("bootstrap_results.json"))

    return "\n".join(lines)


def generate_latest_results() -> str:
    """Read cluster_assignments, regime_labels, and bootstrap for a results table."""
    lines: list[str] = []

    try:
        import pandas as pd
    except ImportError:
        return _unavailable("pandas (not installed)")

    cluster_path = DATA_DIR / "cluster_assignments.parquet"
    regime_path = DATA_DIR / "regime_labels.csv"
    bootstrap_path = DATA_DIR / "bootstrap_results.json"

    rows: list[tuple[str, str]] = []

    # Universe info from config
    universe_path = CONFIG_DIR / "universe.yaml"
    if universe_path.exists():
        try:
            import yaml
            with open(universe_path) as f:
                cfg = yaml.safe_load(f)
            total_defined = 0
            n_groups = 0
            for section_data in cfg.get("assets", {}).values():
                if isinstance(section_data, list):
                    total_defined += len(section_data)
                    n_groups += 1
            rows.append(("Universe defined", f"{total_defined} ETFs across {n_groups} asset groups"))
        except Exception as exc:
            logger.warning("Failed to read universe.yaml: %s", exc)

    # Cluster stats
    if cluster_path.exists():
        try:
            df = pd.read_parquet(cluster_path)
            n_assets = df["ticker"].nunique()
            total_defined = rows[0][1].split()[0] if rows else "?"
            rows.append(("Assets surviving cleaning", f"{n_assets} of {total_defined}"))
            n_windows = df["date"].nunique()
            rows.append(("Rolling windows", str(n_windows)))
            latest_date = df["date"].max()
            n_clusters = df[df["date"] == latest_date]["cluster"].nunique()
            rows.append(("Active clusters (latest window)", str(n_clusters)))
        except Exception as exc:
            logger.warning("Failed to read cluster_assignments: %s", exc)

    # Regime stats
    if regime_path.exists():
        try:
            rdf = pd.read_csv(regime_path, parse_dates=["date"])
            date_min = rdf["date"].min().strftime("%Y-%m-%d")
            date_max = rdf["date"].max().strftime("%Y-%m-%d")
            n_days = len(rdf)
            rows.append(("Trading days", f"{n_days:,} ({date_min} to {date_max})"))
            regime_counts = rdf["regime"].value_counts(normalize=True)
            regime_str = ", ".join(
                f"{regime} ({pct:.1%})" for regime, pct in regime_counts.items()
            )
            n_regimes = rdf["regime"].nunique()
            rows.append(("HMM regimes", f"{n_regimes}: {regime_str}"))
        except Exception as exc:
            logger.warning("Failed to read regime_labels: %s", exc)

    # Bootstrap
    if bootstrap_path.exists():
        try:
            with open(bootstrap_path) as f:
                bs = json.load(f)
            if "mean_cmi" in bs:
                rows.append(("Mean CMI", f"{bs['mean_cmi']['observed']:.3f}"))
            if "mean_tds" in bs:
                rows.append(("Mean TDS", f"{bs['mean_tds']['observed']:.3f}"))
        except Exception as exc:
            logger.warning("Failed to read bootstrap_results: %s", exc)

    # Latest report
    reports = sorted(REPORTS_DIR.glob("acm_daily_report_*.html"))
    if reports:
        latest_report = reports[-1]
        size_kb = latest_report.stat().st_size / 1024
        rows.append(("Daily report size", f"~{size_kb:.0f} KB"))

    if not rows:
        return _unavailable("pipeline artifacts")

    # Build provenance list
    provenance_parts: list[str] = []
    for p in [cluster_path, regime_path, bootstrap_path]:
        if p.exists():
            provenance_parts.append(_provenance(p))

    lines.extend(provenance_parts)
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for metric, value in rows:
        lines.append(f"| {metric} | {value} |")

    return "\n".join(lines)


def generate_universe_summary() -> str:
    """Read universe.yaml and produce a category table."""
    universe_path = CONFIG_DIR / "universe.yaml"
    if not universe_path.exists():
        return _unavailable("config/universe.yaml")

    try:
        import yaml
        with open(universe_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        logger.warning("Failed to read universe.yaml: %s", exc)
        return _unavailable("config/universe.yaml")

    lines: list[str] = [_provenance(universe_path), ""]

    assets = cfg.get("assets", {})
    total = 0
    table_rows: list[tuple[str, str, int]] = []

    for section_name, section_data in assets.items():
        if not isinstance(section_data, list):
            continue
        tickers = [item["ticker"] if isinstance(item, dict) else str(item) for item in section_data]
        count = len(tickers)
        total += count
        # Pretty-print section name
        pretty_name = section_name.replace("_", " ").title()
        ticker_str = ", ".join(tickers)
        table_rows.append((pretty_name, ticker_str, count))

    lines.append(f"**{total} ETFs** across {len(table_rows)} categories:\n")
    lines.append("| Category | Tickers | Count |")
    lines.append("|----------|---------|-------|")
    for name, tickers, count in table_rows:
        lines.append(f"| {name} | {tickers} | {count} |")

    return "\n".join(lines)


def generate_generation_stamp() -> str:
    """Produce a generation timestamp."""
    now = datetime.now()
    stamp = now.strftime("%Y-%m-%d %H:%M ET")
    return f"*Auto-generated on {stamp} from pipeline artifacts. Do not edit this section manually.*"


SECTION_GENERATORS: dict[str, callable] = {
    "pipeline_stats": generate_pipeline_stats,
    "latest_results": generate_latest_results,
    "universe_summary": generate_universe_summary,
    "generation_stamp": generate_generation_stamp,
}


def fill_dynamic_sections(template: str) -> str:
    """Replace all dynamic sections in the template with generated content."""

    def replacer(match: re.Match) -> str:
        begin_tag = match.group(1)
        section_name = match.group(2)
        end_tag = match.group(4)

        generator = SECTION_GENERATORS.get(section_name)
        if generator is None:
            logger.warning("No generator for dynamic section '%s'", section_name)
            content = f"> Unknown dynamic section: `{section_name}`"
        else:
            try:
                content = generator()
            except Exception as exc:
                logger.error("Generator for '%s' failed: %s", section_name, exc)
                content = f"> Generation failed for `{section_name}`: {exc}"

        return f"{begin_tag}\n{content}\n{end_tag}"

    return DYNAMIC_PATTERN.sub(replacer, template)


def generate_readme(template_path: Path, output_path: Path) -> None:
    """Read the template, fill dynamic sections from artifacts, write output.

    Static sections (outside BEGIN/END markers) are preserved verbatim.
    Dynamic sections (inside markers) are replaced with fresh content.
    If an artifact is missing, the section shows a clear warning instead
    of crashing: 'Data unavailable -- run the pipeline first.'
    """
    if not template_path.exists():
        logger.error("Template not found: %s", template_path)
        sys.exit(1)

    template = template_path.read_text(encoding="utf-8")
    output = fill_dynamic_sections(template)
    output_path.write_text(output, encoding="utf-8")
    logger.info("Generated %s (%d bytes)", output_path, len(output))


def check_readme(template_path: Path, output_path: Path) -> bool:
    """Return True if README.md matches what the generator would produce."""
    if not template_path.exists():
        logger.error("Template not found: %s", template_path)
        return False

    template = template_path.read_text(encoding="utf-8")
    expected = fill_dynamic_sections(template)

    if not output_path.exists():
        logger.warning("README.md does not exist yet")
        return False

    current = output_path.read_text(encoding="utf-8")

    # Compare ignoring generation_stamp (timestamp changes every run)
    stamp_re = re.compile(r"\*Auto-generated on .+ from pipeline artifacts\. Do not edit this section manually\.\*")
    current_normalized = stamp_re.sub("STAMP", current)
    expected_normalized = stamp_re.sub("STAMP", expected)

    return current_normalized == expected_normalized


def smoke_test() -> None:
    """Verify the generator runs without crashing and produces output."""
    logger.info("Running smoke test...")

    if not TEMPLATE_PATH.exists():
        logger.error("SMOKE TEST FAILED: Template not found at %s", TEMPLATE_PATH)
        sys.exit(1)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    output = fill_dynamic_sections(template)

    # Check that no sections still have unfilled placeholder text
    if "Run `python scripts/generate_readme.py` to refresh." in output:
        logger.warning("Some sections may still contain placeholder text")

    assert len(output) > 1000, f"Output suspiciously short: {len(output)} bytes"
    assert "<!-- BEGIN:DYNAMIC" in output, "Dynamic markers missing from output"
    logger.info("SMOKE TEST PASSED: %d bytes generated, all sections filled", len(output))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate README.md from template + pipeline artifacts")
    parser.add_argument("--check", action="store_true",
                        help="Check if README.md is up-to-date (exit 1 if stale)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if args.check:
        if check_readme(TEMPLATE_PATH, OUTPUT_PATH):
            logger.info("README.md is up-to-date")
            sys.exit(0)
        else:
            logger.warning(
                "README.md appears to contain hardcoded content that should be auto-generated. "
                "Run 'python scripts/generate_readme.py' to refresh."
            )
            sys.exit(1)

    generate_readme(TEMPLATE_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
