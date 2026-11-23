"""Audio decoding and normalization (pure-NumPy WAV native + FFmpeg fallback for OGG/MP3/WebM/M4A)."""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
    """Resample mono float32 waveform using pure-NumPy band-limited FFT resampling."""
    if orig_sr == target_sr or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    n_in = samples.size
    n_out = max(1, int(round(n_in * float(target_sr) / float(orig_sr))))
    if n_out == n_in:
        return samples.astype(np.float32, copy=False)

    # Exact FFT-domain resampling (band-limited, zero scipy dependency)
    spec = np.fft.rfft(samples.astype(np.float64))
    n_out_bins = n_out // 2 + 1
    out_spec = np.zeros(n_out_bins, dtype=np.complex128)
    copy_bins = min(spec.size, n_out_bins)
    out_spec[:copy_bins] = spec[:copy_bins]
    resampled = np.fft.irfft(out_spec, n=n_out) * (float(n_out) / float(n_in))
    return resampled.astype(np.float32)


def _normalize_waveform(samples: np.ndarray, target_sr: int, max_seconds: float) -> np.ndarray:
    if samples.ndim > 1:
        samples = samples.mean(axis=-1)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise AudioLoadError("Audio stream contains zero samples.")

    if not np.all(np.isfinite(samples)):
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

    # Remove DC bias
    samples = samples - float(np.mean(samples))

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
        with wave.open(io.BytesIO(data), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            raw_frames = wf.readframes(wf.getnframes())
    except Exception:
        return None

    if not raw_frames or sr <= 0:
        return None

    if sampwidth == 1:
        arr = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw_frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        raw_u8 = np.frombuffer(raw_frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw_u8[:, 0].astype(np.int32)
            | (raw_u8[:, 1].astype(np.int32) << 8)
            | (raw_u8[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        arr = signed.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw_frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        return None

    if n_channels > 1:
        usable = (arr.size // n_channels) * n_channels
        arr = arr[:usable].reshape(-1, n_channels).mean(axis=1)

    samples = _resample(arr, int(sr), target_sr)
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
            "Unsupported non-WAV format and 'ffmpeg' is not available on PATH. "
            "Upload a WAV file or use the Web Dashboard (which decodes OGG/MP3 in-browser)."
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
