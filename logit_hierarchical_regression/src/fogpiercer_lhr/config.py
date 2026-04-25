from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_project_env() -> None:
    """Load the project root .env without failing when it is absent."""
    load_dotenv(PROJECT_ROOT / ".env")


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def env_path(var_name: str) -> Path:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_value(var_name: str, required: bool = True) -> str | None:
    value = os.getenv(var_name)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value or None

