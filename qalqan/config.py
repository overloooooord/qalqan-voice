"""Runtime configuration for QALQAN Voice."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    sample_rate: int = 16000
    max_audio_seconds: float = 60.0
    min_audio_seconds: float = 0.25
    threshold_synthetic: float = 0.65
    threshold_suspicious: float = 0.40
    model_path: Path | None = None
    nn_blend_weight: float = 0.65
    default_lang: str = "ru"
    history_limit: int = 50
    host: str = "0.0.0.0"
    port: int = 8000
    telegram_bot_token: str = ""
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        raw_model = os.getenv("QALQAN_MODEL_PATH", "").strip()
        model_path: Path | None = None
        if raw_model:
            model_path = Path(raw_model)
        else:
            default_ckpt = Path(__file__).resolve().parent.parent / "models" / "qalqan_lcnn.pt"
            if default_ckpt.is_file():
                model_path = default_ckpt

        return cls(
            sample_rate=int(os.getenv("QALQAN_SAMPLE_RATE", "16000")),
            max_audio_seconds=float(os.getenv("QALQAN_MAX_AUDIO_SECONDS", "60.0")),
            min_audio_seconds=float(os.getenv("QALQAN_MIN_AUDIO_SECONDS", "0.25")),
            threshold_synthetic=float(os.getenv("QALQAN_THRESHOLD_SYNTHETIC", "0.65")),
            threshold_suspicious=float(os.getenv("QALQAN_THRESHOLD_SUSPICIOUS", "0.40")),
            model_path=model_path,
            nn_blend_weight=float(os.getenv("QALQAN_NN_BLEND_WEIGHT", "0.65")),
            default_lang=os.getenv("QALQAN_DEFAULT_LANG", "ru").lower(),
            history_limit=int(os.getenv("QALQAN_HISTORY_LIMIT", "50")),
            host=os.getenv("QALQAN_HOST", "0.0.0.0"),
            port=int(os.getenv("QALQAN_PORT", "8000")),
            telegram_bot_token=os.getenv("QALQAN_TELEGRAM_TOKEN", "").strip(),
            log_level=os.getenv("QALQAN_LOG_LEVEL", "INFO").upper(),
        )
