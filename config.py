"""Application configuration for the Sillie project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SEQUENCE_LENGTH = 30
SAMPLE_RATE_HZ = 30
FEATURES_PER_FRAME = 66
IDLE_LABEL = "idle"


@dataclass
class Settings:
    """Define runtime configuration with environment overrides."""

    secret_key: str = os.getenv("SILLIE_SECRET_KEY", "dev-secret-key")
    debug: bool = os.getenv("SILLIE_DEBUG", "false").lower() == "true"
    host: str = os.getenv("SILLIE_HOST", "127.0.0.1")
    port: int = int(os.getenv("SILLIE_PORT", "5000"))
    max_sign_samples: int = int(os.getenv("SILLIE_MAX_SIGN_SAMPLES", "100"))
    data_dir: Path = Path(
        os.getenv("SILLIE_DATA_DIR", str(Path(__file__).resolve().parent / "data"))
    )
    model_dir: Path = Path(
        os.getenv("SILLIE_MODEL_DIR", str(Path(__file__).resolve().parent / "models"))
    )


config = Settings()
