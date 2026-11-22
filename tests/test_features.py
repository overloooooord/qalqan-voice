"""Unit tests for audio ingestion, FFmpeg decoding, and acoustic-physics feature extraction."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from qalqan.audio.features import extract_acoustic_features
from qalqan.audio.loader import AudioLoadError, load_audio_bytes, load_audio_pcm16, save_wav_bytes
from qalqan.audio.synth_samples import generate_live_human_sample, generate_synthetic_voice_sample


def test_wav_roundtrip_and_resampling():
    samples = generate_live_human_sample(duration_sec=2.0, sr=8000)
    wav_bytes = save_wav_bytes(samples, sample_rate=8000)
    loaded = load_audio_bytes(wav_bytes, filename="clip.wav", target_sr=16000)

    assert loaded.sample_rate == 16000
    assert loaded.source_format == "wav"
    assert abs(loaded.duration_sec - 2.0) < 0.05
    assert loaded.samples.dtype == np.float32


def test_pcm16_streaming_loader():
    samples = generate_synthetic_voice_sample(duration_sec=1.5, sr=16000)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    loaded = load_audio_pcm16(pcm, sample_rate=16000, target_sr=16000)
    assert loaded.source_format == "pcm16"
    assert abs(loaded.duration_sec - 1.5) < 0.02


def test_empty_audio_raises():
    with pytest.raises(AudioLoadError):
        load_audio_bytes(b"", filename="empty.wav")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ogg_opus_decoding_via_ffmpeg():
    """Verify Telegram/WhatsApp .ogg (Opus) voice note decoding works end-to-end."""
    samples = generate_live_human_sample(duration_sec=2.0, sr=16000)
    wav_bytes = save_wav_bytes(samples, sample_rate=16000)

    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-c:a", "libopus", "-f", "ogg", "pipe:1"],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    loaded = load_audio_bytes(proc.stdout, filename="voice.ogg", target_sr=16000)
    assert loaded.sample_rate == 16000
    assert loaded.duration_sec > 1.8


def test_acoustic_physics_separates_synthetic_vs_live():
    f_synth = extract_acoustic_features(generate_synthetic_voice_sample(duration_sec=4.0))
    f_live = extract_acoustic_features(generate_live_human_sample(duration_sec=4.0))

    assert f_synth.lfcc.shape[0] == 60
    assert f_synth.log_mel.shape[0] == 64
    assert len(f_synth.spectrogram_preview) == 32
    assert np.all(np.isfinite(f_synth.lfcc))
    assert np.all(np.isfinite(f_live.lfcc))

    assert f_synth.hf_vocoder_score > f_live.hf_vocoder_score + 0.35
    assert f_synth.breath_pause_score > f_live.breath_pause_score + 0.35
    assert f_synth.resonance_smoothness_score > f_live.resonance_smoothness_score + 0.25


def test_silent_and_short_edge_cases_are_finite():
    silence = np.zeros(8000, dtype=np.float32)
    feat = extract_acoustic_features(silence, sr=16000)
    assert np.all(np.isfinite(feat.lfcc))
    assert np.all(np.isfinite(feat.physics_vector()))
