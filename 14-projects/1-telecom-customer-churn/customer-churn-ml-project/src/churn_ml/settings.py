"""Serving settings, read from environment variables (or a .env file).

Twelve-factor style: the same image runs in every environment and only the
environment changes, e.g. CHURN_ARTIFACT_DIR=artifacts/runs/<run_id> to roll back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHURN_", env_file=".env", extra="ignore")

    artifact_dir: Path = Path("artifacts")
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    max_batch_size: int = 1000
