"""Audio decoding and normalization (WAV native + FFmpeg fallback for OGG/MP3/WebM/M4A)."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


class AudioLoadError(ValueError):
    """Raised when an uploaded or streamed audio payload cannot be decoded."""


@dataclass(frozen=True)
class LoadedAudio:
    """Normalized mono float32 audio payload at target sample rate."""

    samples: np.ndarray
    sample_rate: int
    duration_sec: float
    source_format: str


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = resample_poly(samples.astype(np.float64), up, down)
    return resampled.astype(np.float32)


def _normalize_waveform(samples: np.ndarray, target_sr: int, max_seconds: float) -> np.ndarray:
    if samples.ndim > 1:
        samples = samples.mean(axis=-1)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise AudioLoadError("Audio stream contains zero samples.")

    # Replace NaN/Inf defensively
    if not np.all(np.isfinite(samples)):
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

    # Remove DC bias
    samples = samples - float(np.mean(samples))

    # Truncate to max duration
    max_samples = int(target_sr * max_seconds)
    if max_samples > 0 and samples.size > max_samples:
        samples = samples[:max_samples]

    peak = float(np.max(np.abs(samples)))
    if peak > 1.0:
        samples = samples / peak
    return np.ascontiguousarray(samples, dtype=np.float32)


def _try_decode_wav(data: bytes, target_sr: int, max_seconds: float) -> LoadedAudio | None:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    try:
        sr, raw = wavfile.read(io.BytesIO(data))
    except Exception:
        return None

    if raw.dtype == np.int16:
        samples = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        samples = raw.astype(np.float32) / 2147483648.0
    elif raw.dtype == np.uint8:
        samples = (raw.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(raw.dtype, np.floating):
        samples = raw.astype(np.float32)
    else:
        return None

    if samples.ndim > 1:
        samples = samples.mean(axis=-1)
    samples = _resample(samples, int(sr), target_sr)
    samples = _normalize_waveform(samples, target_sr, max_seconds)
    return LoadedAudio(
        samples=samples,
        sample_rate=target_sr,
        duration_sec=float(samples.size) / float(target_sr),
        source_format="wav",
    )


def _decode_with_ffmpeg(data: bytes, target_sr: int, max_seconds: float, fmt_hint: str) -> LoadedAudio:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise AudioLoadError(
            "Unsupported non-WAV format and 'ffmpeg' is not available on PATH."
        )

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-t",
        f"{max_seconds:.2f}",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=25,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioLoadError("FFmpeg timed out while decoding audio.") from exc

    if proc.returncode != 0 or not proc.stdout:
        stderr_msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioLoadError(f"FFmpeg failed to decode audio ({fmt_hint}): {stderr_msg or 'invalid stream'}")

    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    samples = _normalize_waveform(samples, target_sr, max_seconds)
    return LoadedAudio(
        samples=samples,
        sample_rate=target_sr,
        duration_sec=float(samples.size) / float(target_sr),
        source_format=fmt_hint or "ffmpeg",
    )


def load_audio_bytes(
    data: bytes,
    filename: str = "audio.wav",
    target_sr: int = 16000,
    max_seconds: float = 60.0,
) -> LoadedAudio:
    """Decode WAV/OGG/MP3/WebM/M4A/FLAC bytes into normalized mono 16kHz float32."""
    if not data:
        raise AudioLoadError("Empty audio payload.")

    ext = Path(filename).suffix.lower().lstrip(".")
    decoded = _try_decode_wav(data, target_sr=target_sr, max_seconds=max_seconds)
    if decoded is not None:
        return decoded
    return _decode_with_ffmpeg(data, target_sr=target_sr, max_seconds=max_seconds, fmt_hint=ext or "audio")


def load_audio_file(
    path: str | Path,
    target_sr: int = 16000,
    max_seconds: float = 60.0,
) -> LoadedAudio:
    """Load an audio file from disk."""
    file_path = Path(path)
    if not file_path.is_file():
        raise AudioLoadError(f"Audio file not found: {file_path}")
    return load_audio_bytes(
        file_path.read_bytes(),
        filename=file_path.name,
        target_sr=target_sr,
        max_seconds=max_seconds,
    )


def load_audio_pcm16(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    target_sr: int = 16000,
    max_seconds: float = 15.0,
) -> LoadedAudio:
    """Decode raw signed 16-bit little-endian mono PCM bytes (for WebSocket streaming)."""
    if len(pcm_bytes) < 2:
        raise AudioLoadError("PCM chunk too short.")
    even_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
    samples = np.frombuffer(pcm_bytes[:even_len], dtype="<i2").astype(np.float32) / 32768.0
    samples = _resample(samples, sample_rate, target_sr)
    samples = _normalize_waveform(samples, target_sr, max_seconds)
    return LoadedAudio(
        samples=samples,
        sample_rate=target_sr,
        duration_sec=float(samples.size) / float(target_sr),
        source_format="pcm16",
    )


def save_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Encode mono float32 samples in [-1, 1] to PCM16 WAV bytes."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()
