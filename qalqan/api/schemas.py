"""Pydantic v2 schemas for the QALQAN Voice HTTP & WebSocket API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AcousticMetricsSchema(BaseModel):
    hf_vocoder_score: float = Field(..., ge=0.0, le=1.0)
    breath_pause_score: float = Field(..., ge=0.0, le=1.0)
    resonance_smoothness_score: float = Field(..., ge=0.0, le=1.0)
    phase_stationarity_score: float = Field(..., ge=0.0, le=1.0)
    snr_estimate_db: float
    speech_ratio: float = Field(..., ge=0.0, le=1.0)
    codec_narrowband: bool
    physics_probability: float = Field(..., ge=0.0, le=1.0)
    neural_probability: float | None = Field(default=None, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    id: str
    timestamp: str
    source: str
    label: Literal["synthetic", "suspicious", "live"]
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    engine_mode: str
    latency_ms: float
    duration_sec: float
    source_format: str
    lang: str
    reasons: list[str]
    metrics: AcousticMetricsSchema
    spectrogram_preview: list[list[float]]
    transcript_hint: str | None = None


class SimulateRequest(BaseModel):
    scenario: Literal["synthetic", "live"] = "synthetic"
    lang: Literal["ru", "kk", "en"] = "ru"


class HistoryItem(BaseModel):
    id: str
    timestamp: str
    source: str
    label: Literal["synthetic", "suspicious", "live"]
    probability: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float
    duration_sec: float
    engine_mode: str


class HealthResponse(BaseModel):
    status: str
    version: str
    engine_mode: str
    model_loaded: bool
    sample_rate: int
    ffmpeg_available: bool
