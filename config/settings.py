"""STATIS Configuration — centralised settings for the platform."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache, partial

from pydantic_settings import BaseSettings
from pydantic import Field


def _base() -> Path:
    # config/settings.py -> config -> STATIS (project root)
    return Path(__file__).resolve().parent.parent


class StatisSettings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # --- Paths ---
    BASE_DIR: Path = Field(default_factory=_base)
    DATA_DIR: Path = Field(default_factory=lambda: _base() / "data")
    UPLOAD_DIR: Path = Field(default_factory=lambda: _base() / "data" / "uploads")
    WORKING_DIR: Path = Field(default_factory=lambda: _base() / "data" / "working")
    OUTPUT_DIR: Path = Field(default_factory=lambda: _base() / "outputs")
    DB_PATH: Path = Field(default_factory=lambda: _base() / "database" / "statis.db")
    LOG_DIR: Path = Field(default_factory=lambda: _base() / "logs")
    MEMORY_DIR: Path = Field(default_factory=lambda: _base() / "memory")

    # --- Server ---
    HOST: str = "127.0.0.1"
    PORT: int = 8420
    DEBUG: bool = False

    # --- Ollama ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5-coder:7b"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_NUM_CTX: int = 8192

    # --- Security ---
    ALLOWED_EXTENSIONS: set[str] = {".csv", ".xlsx", ".xls", ".tsv", ".parquet"}
    MAX_UPLOAD_SIZE_MB: int = 500

    # --- Reproducibility ---
    RANDOM_SEED: int = 42

    model_config = {
        "env_prefix": "STATIS_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def model_post_init(self, __context) -> None:
        for d in [
            self.DATA_DIR, self.UPLOAD_DIR, self.WORKING_DIR,
            self.OUTPUT_DIR, self.DB_PATH.parent, self.LOG_DIR,
            self.MEMORY_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> StatisSettings:
    """Return cached singleton settings instance."""
    return StatisSettings()
