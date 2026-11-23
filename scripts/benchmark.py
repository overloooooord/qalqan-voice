"""Reproducible latency and multi-channel acoustic robustness benchmark for QALQAN Voice.

Evaluates the detector across 5 realistic transmission channels:
  1. Clean Wideband (16 kHz WAV)
  2. Telegram Voice Note (.ogg Opus @ 24 kbps via FFmpeg)
  3. WhatsApp Voice Note (.ogg Opus @ 16 kbps via FFmpeg)
  4. Noisy Environment (15 dB SNR additive background noise)
  5. Telephony Narrowband (8 kHz / 300-3400 Hz bandpass)

Usage:
    python -m scripts.benchmark --samples-per-class 20
    python -m scripts.benchmark --samples-per-class 20 --json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass

import numpy as np

from qalqan.audio.loader import load_audio_bytes, save_wav_bytes
from qalqan.audio.synth_samples import (
    bandpass_filter,
    generate_live_human_sample,
    generate_synthetic_voice_sample,
)
from qalqan.config import Settings
from qalqan.ml.detector import VoiceDeepfakeDetector


@dataclass
class ChannelMetrics:
    channel: str
    n_samples: int
    accuracy: float
    fpr: float
    fnr: float
    eer: float
    p50_latency_ms: float
    p95_latency_ms: float
    realtime_factor: float


def _compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    thresholds = np.unique(scores)
    best_diff = 1.0
    eer = 0.5
    pos = labels == 1
    neg = labels == 0
    for thr in thresholds:
        far = float(np.mean(scores[neg] >= thr))
        frr = float(np.mean(scores[pos] < thr))
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer = 0.5 * (far + frr)
    return eer


def _transcode_opus(samples: np.ndarray, sr: int, bitrate: str) -> bytes:
    wav_bytes = save_wav_bytes(samples, sample_rate=sr)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            bitrate,
            "-vbr",
            "on",
            "-f",
            "ogg",
            "pipe:1",
        ],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def _add_noise(samples: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig_power = float(np.mean(samples * samples)) + 1e-8
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=samples.shape).astype(np.float32)
    return np.clip(samples + noise, -1.0, 1.0)


def _evaluate_channel(
    detector: VoiceDeepfakeDetector,
    channel_name: str,
    pairs: list[tuple[np.ndarray, int]],
    mode: str,
    sr: int = 16000,
) -> ChannelMetrics:
    labels: list[int] = []
    scores: list[float] = []
    latencies: list[float] = []
    durations: list[float] = []

    for idx, (waveform, label) in enumerate(pairs):
        if mode == "clean":
            verdict = detector.analyze_waveform(waveform, sample_rate=sr, source_format="wav")
        elif mode == "telegram_opus":
            ogg = _transcode_opus(waveform, sr, bitrate="24k")
            verdict = detector.analyze_bytes(ogg, filename="tg_voice.ogg")
        elif mode == "whatsapp_opus":
            ogg = _transcode_opus(waveform, sr, bitrate="16k")
            verdict = detector.analyze_bytes(ogg, filename="wa_voice.ogg")
        elif mode == "noise_15db":
            noisy = _add_noise(waveform, snr_db=15.0, seed=9000 + idx)
            verdict = detector.analyze_waveform(noisy, sample_rate=sr, source_format="wav")
        elif mode == "telephony_nb":
            nb = bandpass_filter(waveform, sr=sr, low=300.0, high=3400.0)
            wav_8k = save_wav_bytes(nb[::2], sample_rate=8000)
            loaded = load_audio_bytes(wav_8k, filename="sip_call.wav", target_sr=sr)
            verdict = detector.analyze_loaded(loaded)
        else:
            raise ValueError(f"Unknown channel mode: {mode}")

        labels.append(label)
        scores.append(verdict.probability)
        latencies.append(verdict.latency_ms)
        durations.append(verdict.duration_sec)

    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    preds = (s >= 0.50).astype(np.int32)

    acc = float(np.mean(preds == y))
    fpr = float(np.mean(preds[y == 0] == 1))
    fnr = float(np.mean(preds[y == 1] == 0))
    eer = _compute_eer(y, s)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    mean_dur_ms = float(np.mean(durations)) * 1000.0
    rtf = (p50 / mean_dur_ms) if mean_dur_ms > 0 else 0.0

    return ChannelMetrics(
        channel=channel_name,
        n_samples=len(pairs),
        accuracy=round(acc, 4),
        fpr=round(fpr, 4),
        fnr=round(fnr, 4),
        eer=round(eer, 4),
        p50_latency_ms=round(p50, 2),
        p95_latency_ms=round(p95, 2),
        realtime_factor=round(rtf, 4),
    )


def run_benchmark(samples_per_class: int = 20, duration_sec: float = 4.0) -> list[ChannelMetrics]:
    detector = VoiceDeepfakeDetector(Settings(model_path=None))
    pairs: list[tuple[np.ndarray, int]] = []
    for i in range(samples_per_class):
        pairs.append((generate_live_human_sample(duration_sec=duration_sec, seed=1000 + i), 0))
        pairs.append((generate_synthetic_voice_sample(duration_sec=duration_sec, seed=2000 + i), 1))

    channels = [
        ("Clean Wideband (16 kHz WAV)", "clean"),
        ("Noisy Street/Room (15 dB SNR)", "noise_15db"),
        ("Telephony Narrowband (8 kHz SIP)", "telephony_nb"),
    ]
    if shutil.which("ffmpeg"):
        channels.insert(1, ("Telegram Voice (.ogg Opus 24k)", "telegram_opus"))
        channels.insert(2, ("WhatsApp Voice (.ogg Opus 16k)", "whatsapp_opus"))

    return [_evaluate_channel(detector, title, pairs, mode) for title, mode in channels]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QALQAN Voice multi-channel latency & robustness benchmark.")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = run_benchmark(samples_per_class=args.samples_per_class, duration_sec=args.duration)
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return

    print(f"\nQALQAN Voice — Multi-Channel CPU Benchmark ({args.samples_per_class * 2} clips/channel, {args.duration:.1f}s each)")
    print("-" * 102)
    print(f"{'Channel':<34} {'Acc':>7} {'FPR':>7} {'FNR':>7} {'EER':>7} {'p50 ms':>9} {'p95 ms':>9} {'RTF':>8}")
    print("-" * 102)
    for r in results:
        print(
            f"{r.channel:<34} {r.accuracy*100:6.1f}% {r.fpr*100:6.1f}% {r.fnr*100:6.1f}% "
            f"{r.eer*100:6.1f}% {r.p50_latency_ms:8.1f} {r.p95_latency_ms:8.1f} {r.realtime_factor:8.4f}"
        )


if __name__ == "__main__":
    main()
