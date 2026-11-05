"""Application configuration for the Sillie project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

SEQUENCE_LENGTH = 30
SAMPLE_RATE_HZ = 30
RAW_FRAME_SIZE = 129
FEATURES_PER_FRAME = 117
IDLE_LABEL = "idle"
FEATURE_VERSION = "v2"


def _database_url() -> str:
    """Read a SQLAlchemy URI and normalize legacy postgres scheme spelling."""
    database_url = os.getenv(
        "SILLIE_DATABASE_URL",
        f"sqlite:///{(Path(__file__).resolve().parent / 'instance' / 'sillie.sqlite3').as_posix()}",
    )
    if database_url.startswith("postgres://"):
        return "postgresql://" + database_url[len("postgres://"):]
    return database_url


@dataclass
class Settings:
    """Define runtime configuration with environment overrides."""

    secret_key: str | None = os.getenv("SILLIE_SECRET_KEY")
    debug: bool = os.getenv("SILLIE_DEBUG", "false").lower() == "true"
    database_url: str = _database_url()
    turn_url: str | None = os.getenv("SILLIE_TURN_URL")
    turn_username: str | None = os.getenv("SILLIE_TURN_USERNAME")
    turn_credential: str | None = os.getenv("SILLIE_TURN_CREDENTIAL")
    gnani_api_key: str | None = os.getenv("GNANI_API_KEY")
    gnani_org_id: str | None = os.getenv("GNANI_ORGANIZATION_ID")
    gnani_user_id: str | None = os.getenv("GNANI_USER_ID")
    gnani_base_url: str = os.getenv("GNANI_BASE_URL", "https://api.vachana.ai")
    host: str = os.getenv("SILLIE_HOST", "127.0.0.1")
    port: int = int(os.getenv("SILLIE_PORT", "5000"))
    max_sign_samples: int = int(os.getenv("SILLIE_MAX_SIGN_SAMPLES", "100"))
    data_dir: Path = Path(
        os.getenv("SILLIE_DATA_DIR", str(Path(__file__).resolve().parent / "data"))
    )
    datasets_dir: Path = Path(
        os.getenv(
            "SILLIE_DATASETS_DIR",
            str(Path(__file__).resolve().parent / "datasets"),
        )
    )
    model_dir: Path = Path(
        os.getenv("SILLIE_MODEL_DIR", str(Path(__file__).resolve().parent / "models"))
    )


config = Settings()
