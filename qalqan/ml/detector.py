"""Hybrid voice deepfake detector combining acoustic-physics forensics with PyTorch LCNN."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from qalqan.audio.features import AcousticFeatures, extract_acoustic_features
from qalqan.audio.loader import AudioLoadError, LoadedAudio, load_audio_bytes, load_audio_pcm16
from qalqan.config import Settings
from qalqan.ml.model import QalqanLCNN, load_checkpoint

logger = logging.getLogger(__name__)


_REASONS: dict[str, dict[str, str]] = {
    "ru": {
        "breath_missing": "Нет микропауз дыхания между фразами",
        "resonance_smooth": "Неестественно ровный резонанс голосового тракта",
        "vocoder_hf": "Следы нейровокодера в диапазоне 6–8 кГц",
        "phase_uniform": "Аномально стабильный спектральный поток гармоник",
        "narrowband_note": "Узкополосный кодек (телефония / мессенджер) — включён адаптивный режим",
        "live_breath": "Естественные микропаузы дыхания и живая динамика фраз",
        "live_resonance": "Органическая микро-вариативность основного тона и формант",
        "live_hf": "Отсутствуют периодические артефакты вокодера в 6–8 кГц",
        "suspicious_general": "Пограничные акустические показатели — рекомендуется повторная проверка",
    },
    "kk": {
        "breath_missing": "Фразалар арасында табиғи тыныс алу микропаузалары жоқ",
        "resonance_smooth": "Дауыс жолының резонансы табиғи емес тым тегіс",
        "vocoder_hf": "6–8 кГц диапазонында нейровокодер іздері анықталды",
        "phase_uniform": "Гармоникалардың спектрлік ағыны тым біркелкі",
        "narrowband_note": "Тар жолақты кодек (телефония / мессенджер) — бейімделгіш режим",
        "live_breath": "Табиғи тыныс алу және сөйлеу ырғағы сақталған",
        "live_resonance": "Негізгі тон мен форманттардың табиғи өзгерісі байқалады",
        "live_hf": "6–8 кГц жолағында вокодер артефактілері табылмады",
        "suspicious_general": "Шекаралық акустикалық көрсеткіштер — қайта тексеру ұсынылады",
    },
    "en": {
        "breath_missing": "Missing inter-phrase micro-breaths and natural inhalation",
        "resonance_smooth": "Unnaturally smooth vocal-tract resonance and low pitch jitter",
        "vocoder_hf": "Neural vocoder artifacts detected in the 6–8 kHz band",
        "phase_uniform": "Overly stationary harmonic and spectral flux profile",
        "narrowband_note": "Narrowband codec detected (telephony / messenger) — adaptive scoring active",
        "live_breath": "Organic inter-phrase micro-breaths and natural pause cadence",
        "live_resonance": "Natural vocal-tract formant drift and physiological pitch jitter",
        "live_hf": "No periodic neural vocoder artifacts in the 6–8 kHz band",
        "suspicious_general": "Borderline acoustic metrics — secondary verification recommended",
    },
}


@dataclass
class DetectionVerdict:
    """Structured deepfake detection verdict returned by API, CLI, and Telegram bot."""

    label: str  # "synthetic" | "suspicious" | "live"
    probability: float  # synthetic voice probability in [0, 1]
    confidence: float  # decision confidence in [0, 1]
    engine_mode: str  # "hybrid-lcnn" | "acoustic-physics"
    latency_ms: float
    duration_sec: float
    source_format: str
    lang: str
    reasons: list[str]
    metrics: dict[str, Any]
    spectrogram_preview: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class VoiceDeepfakeDetector:
    """Real-time acoustic-physics + optional PyTorch LCNN voice spoofing detector."""

    def __init__(self, settings: Settings | None = None, device: str = "cpu"):
        self.settings = settings or Settings.from_env()
        self.device = torch.device(device)
        self.model: QalqanLCNN | None = None
        self.model_meta: dict[str, Any] = {}

        if self.settings.model_path and self.settings.model_path.is_file():
            try:
                self.model, self.model_meta = load_checkpoint(self.settings.model_path, device=self.device)
                logger.info("Loaded QalqanLCNN checkpoint from %s", self.settings.model_path)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Failed to load checkpoint %s (%s); falling back to acoustic-physics mode.",
                               self.settings.model_path, exc)
                self.model = None

    @property
    def engine_mode(self) -> str:
        return "hybrid-lcnn" if self.model is not None else "acoustic-physics"

    def _physics_probability(self, feat: AcousticFeatures) -> float:
        """Compute calibrated synthetic probability from the 4 acoustic-physics metrics."""
        if feat.codec_narrowband:
            # Down-weight 6-8 kHz when stripped by telephony/WhatsApp narrowband codec
            raw = (
                0.15 * feat.hf_vocoder_score
                + 0.35 * feat.breath_pause_score
                + 0.32 * feat.resonance_smoothness_score
                + 0.18 * feat.phase_stationarity_score
            )
        else:
            raw = (
                0.34 * feat.hf_vocoder_score
                + 0.28 * feat.breath_pause_score
                + 0.24 * feat.resonance_smoothness_score
                + 0.14 * feat.phase_stationarity_score
            )
        # Smooth logistic calibration centered at 0.48
        calibrated = _sigmoid((raw - 0.48) * 7.2)
        return float(max(0.01, min(0.99, calibrated)))

    def _neural_probability(self, feat: AcousticFeatures) -> float:
        """Run PyTorch LCNN inference on LFCC + physics vector."""
        assert self.model is not None
        lfcc = feat.lfcc
        # Standardize LFCC per-utterance (CMVN) for channel/microphone invariance
        mean = lfcc.mean(axis=1, keepdims=True)
        std = lfcc.std(axis=1, keepdims=True) + 1e-6
        lfcc_norm = (lfcc - mean) / std

        lfcc_t = torch.from_numpy(lfcc_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        phys_t = torch.from_numpy(feat.physics_vector()).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logit = self.model(lfcc_t, phys_t)
            prob = torch.sigmoid(logit).item()
        return float(max(0.005, min(0.995, prob)))

    def _build_reasons(self, feat: AcousticFeatures, label: str, lang: str) -> list[str]:
        catalog = _REASONS.get(lang, _REASONS["ru"])
        reasons: list[str] = []

        if label in ("synthetic", "suspicious"):
            ranked = [
                (feat.breath_pause_score, "breath_missing"),
                (feat.resonance_smoothness_score, "resonance_smooth"),
                (feat.hf_vocoder_score, "vocoder_hf"),
                (feat.phase_stationarity_score, "phase_uniform"),
            ]
            ranked.sort(key=lambda item: item[0], reverse=True)
            for score, key in ranked:
                if score >= 0.45 and len(reasons) < 3:
                    reasons.append(catalog[key])
            if not reasons:
                reasons.append(catalog["suspicious_general"])
        else:
            if feat.breath_pause_score < 0.45:
                reasons.append(catalog["live_breath"])
            if feat.resonance_smoothness_score < 0.45:
                reasons.append(catalog["live_resonance"])
            if not feat.codec_narrowband and feat.hf_vocoder_score < 0.45:
                reasons.append(catalog["live_hf"])
            if not reasons:
                reasons.append(catalog["live_breath"])

        if feat.codec_narrowband:
            reasons.append(catalog["narrowband_note"])
        return reasons

    def analyze_waveform(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
        source_format: str = "wav",
        lang: str | None = None,
    ) -> DetectionVerdict:
        """Analyze a mono float32 waveform and return a full forensic verdict."""
        t0 = time.perf_counter()
        use_lang = (lang or self.settings.default_lang).lower()
        if use_lang not in _REASONS:
            use_lang = "ru"

        duration_sec = float(len(samples)) / float(sample_rate)
        if duration_sec < self.settings.min_audio_seconds:
            raise AudioLoadError(
                f"Audio is too short ({duration_sec:.2f}s); minimum required is {self.settings.min_audio_seconds:.2f}s."
            )

        feat = extract_acoustic_features(samples, sr=sample_rate)
        phys_prob = self._physics_probability(feat)

        if self.model is not None:
            nn_prob = self._neural_probability(feat)
            w = float(max(0.0, min(1.0, self.settings.nn_blend_weight)))
            prob = w * nn_prob + (1.0 - w) * phys_prob
        else:
            nn_prob = None
            prob = phys_prob

        if prob >= self.settings.threshold_synthetic:
            label = "synthetic"
            confidence = min(0.99, 0.55 + (prob - self.settings.threshold_synthetic) * 1.25)
        elif prob >= self.settings.threshold_suspicious:
            label = "suspicious"
            confidence = 0.55
        else:
            label = "live"
            confidence = min(0.99, 0.55 + (self.settings.threshold_suspicious - prob) * 1.10)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        reasons = self._build_reasons(feat, label=label, lang=use_lang)

        metrics = {
            "hf_vocoder_score": feat.hf_vocoder_score,
            "breath_pause_score": feat.breath_pause_score,
            "resonance_smoothness_score": feat.resonance_smoothness_score,
            "phase_stationarity_score": feat.phase_stationarity_score,
            "snr_estimate_db": feat.snr_estimate_db,
            "speech_ratio": feat.speech_ratio,
            "codec_narrowband": feat.codec_narrowband,
            "physics_probability": round(phys_prob, 4),
            "neural_probability": round(nn_prob, 4) if nn_prob is not None else None,
        }

        return DetectionVerdict(
            label=label,
            probability=round(prob, 4),
            confidence=round(confidence, 4),
            engine_mode=self.engine_mode,
            latency_ms=round(latency_ms, 2),
            duration_sec=round(duration_sec, 2),
            source_format=source_format,
            lang=use_lang,
            reasons=reasons,
            metrics=metrics,
            spectrogram_preview=feat.spectrogram_preview,
        )

    def analyze_loaded(self, audio: LoadedAudio, lang: str | None = None) -> DetectionVerdict:
        return self.analyze_waveform(
            samples=audio.samples,
            sample_rate=audio.sample_rate,
            source_format=audio.source_format,
            lang=lang,
        )

    def analyze_bytes(
        self,
        data: bytes,
        filename: str = "audio.wav",
        lang: str | None = None,
    ) -> DetectionVerdict:
        audio = load_audio_bytes(
            data,
            filename=filename,
            target_sr=self.settings.sample_rate,
            max_seconds=self.settings.max_audio_seconds,
        )
        return self.analyze_loaded(audio, lang=lang)

    def analyze_pcm16(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        lang: str | None = None,
    ) -> DetectionVerdict:
        audio = load_audio_pcm16(
            pcm_bytes,
            sample_rate=sample_rate,
            target_sr=self.settings.sample_rate,
        )
        return self.analyze_loaded(audio, lang=lang)
