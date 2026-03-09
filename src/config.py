"""Configuration loader. Loads settings from YAML and secrets from .env."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def get_api_key() -> str:
    """Get FMP API key from environment. Raises if not found."""
    key = os.environ.get("FMP_API_KEY")
    if not key or key == "your_fmp_api_key_here":
        raise EnvironmentError(
            "FMP_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return key


def load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML config file from the config/ directory."""
    import yaml

    config_path = _PROJECT_ROOT / "config" / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_settings() -> dict[str, Any]:
    return load_yaml("settings.yaml")


def get_universe_config() -> dict[str, Any]:
    return load_yaml("universe.yaml")


def get_methodology_config() -> dict[str, Any]:
    return load_yaml("methodology.yaml")


def get_event_windows_config() -> dict[str, Any]:
    return load_yaml("event_windows.yaml")


PROJECT_ROOT = _PROJECT_ROOT
