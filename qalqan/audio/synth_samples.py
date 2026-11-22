"""Deterministic speech-like acoustic generators for self-tests and live dashboard simulation.

Produces realistic 16 kHz audio waveforms modeling the physical contrasts between:
1. Organic human speech: natural pitch jitter/shimmer, shifting vocal-tract formants,
   aspiration/micro-breath before phrase onsets, irregular pause timing, natural room floor.
2. Neural-vocoder synthetic speech: over-smooth pitch trajectory, ultra-clean digital pauses,
   overly steady formant envelopes, and periodic 6-8 kHz vocoder upsampling artifacts.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def _bandpass(sig: np.ndarray, sr: int, low: float, high: float, order: int = 3) -> np.ndarray:
    nyq = sr / 2.0
    sos = butter(order, [max(low / nyq, 1e-4), min(high / nyq, 0.999)], btype="band", output="sos")
    return sosfilt(sos, sig).astype(np.float32)


def generate_synthetic_voice_sample(duration_sec: float = 4.0, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """Generate a deterministic audio waveform exhibiting neural-vocoder physical signatures."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples, dtype=np.float64) / sr

    # Overly smooth pitch contour (minimal micro-jitter)
    f0 = 148.0 + 14.0 * np.sin(2.0 * np.pi * 0.45 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / sr

    # Rigid harmonic stack (constant vocal tract resonance)
    voice = np.zeros(n_samples, dtype=np.float64)
    formants = [(520.0, 1.0), (1450.0, 0.55), (2650.0, 0.25)]
    for k in range(1, 20):
        freq_k = k * 148.0
        gain = sum(amp * np.exp(-0.5 * ((freq_k - fc) / 220.0) ** 2) for fc, amp in formants)
        voice += (gain / (k ** 0.7)) * np.sin(k * phase)

    # Characteristic 6-8 kHz neural vocoder periodic upsampling ripple / metallic buzz
    vocoder_artifact = (
        0.045 * np.sin(2.0 * np.pi * 6420.0 * t + 0.3 * np.sin(2.0 * np.pi * 12.0 * t))
        + 0.038 * np.sin(2.0 * np.pi * 7180.0 * t)
        + 0.032 * np.sin(2.0 * np.pi * 7640.0 * t)
        + 0.012 * _bandpass(rng.standard_normal(n_samples), sr, 6100.0, 7800.0)
    )

    # Uniform phrase gating with digitally dead-silent pauses (zero micro-breath),
    # scaled proportionally to duration_sec so any length >= 0.5s works cleanly.
    envelope = np.zeros(n_samples, dtype=np.float64)
    scale = duration_sec / 4.0
    phrase_windows = [(0.15 * scale, 1.15 * scale), (1.45 * scale, 2.45 * scale), (2.75 * scale, 3.75 * scale)]
    for start_s, end_s in phrase_windows:
        s_i = max(0, int(start_s * sr))
        e_i = min(n_samples, int(end_s * sr))
        if e_i <= s_i:
            continue
        ramp = min(int(0.015 * sr), (e_i - s_i) // 4)
        env_seg = np.ones(e_i - s_i, dtype=np.float64)
        if ramp > 0:
            env_seg[:ramp] = np.linspace(0.0, 1.0, ramp)
            env_seg[-ramp:] = np.linspace(1.0, 0.0, ramp)
        envelope[s_i:e_i] = env_seg

    sig = (voice + vocoder_artifact) * envelope
    peak = float(np.max(np.abs(sig))) + 1e-8
    return (0.75 * sig / peak).astype(np.float32)


def generate_live_human_sample(duration_sec: float = 4.0, sr: int = 16000, seed: int = 7) -> np.ndarray:
    """Generate a deterministic audio waveform exhibiting organic human vocal-tract physics."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples, dtype=np.float64) / sr

    # Organic pitch contour with physiological jitter & expressive prosody
    step_samples = int(0.01 * sr)  # 10ms perturbation steps
    n_steps = n_samples // step_samples + 2
    jitter_steps = rng.normal(0.0, 4.2, size=n_steps)
    jitter = np.interp(np.linspace(0, n_steps - 1, n_samples), np.arange(n_steps), jitter_steps)
    f0 = 152.0 + 36.0 * np.sin(2.0 * np.pi * 0.85 * t) + 18.0 * np.cos(2.0 * np.pi * 1.9 * t) + jitter
    f0 = np.clip(f0, 85.0, 310.0)
    phase = 2.0 * np.pi * np.cumsum(f0) / sr

    # Time-varying formants (coarticulation across syllables) + shimmer
    shimmer = 1.0 + 0.14 * np.sin(2.0 * np.pi * 7.3 * t) + 0.08 * _bandpass(rng.standard_normal(n_samples), sr, 2.0, 35.0)
    f1_track = 480.0 + 260.0 * np.sin(2.0 * np.pi * 2.4 * t)
    f2_track = 1480.0 + 520.0 * np.cos(2.0 * np.pi * 1.7 * t)

    voice = np.zeros(n_samples, dtype=np.float64)
    for k in range(1, 18):
        inst_freq = k * f0
        gain = (
            1.0 * np.exp(-0.5 * ((inst_freq - f1_track) / 210.0) ** 2)
            + 0.65 * np.exp(-0.5 * ((inst_freq - f2_track) / 310.0) ** 2)
            + 0.25 * np.exp(-0.5 * ((inst_freq - 2750.0) / 420.0) ** 2)
        )
        voice += (gain / (k ** 0.85)) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))

    # Bursting consonant fricatives (s, sh, t, k) -> natural high temporal CV in upper bands
    scale = duration_sec / 4.0
    fricative_noise = _bandpass(rng.standard_normal(n_samples), sr, 3500.0, 7700.0)
    fricative_gate = np.zeros(n_samples, dtype=np.float64)
    for burst_t in [0.42 * scale, 0.88 * scale, 1.72 * scale, 2.05 * scale, 2.95 * scale, 3.48 * scale]:
        c = int(burst_t * sr)
        w = max(16, int(0.055 * sr))
        s_i, e_i = max(0, c - w // 2), min(n_samples, c + w // 2)
        if e_i > s_i:
            fricative_gate[s_i:e_i] = np.hanning(e_i - s_i)

    # Irregular phrase boundaries + natural pre-onset inhalation micro-breaths
    envelope = np.zeros(n_samples, dtype=np.float64)
    breath_env = np.zeros(n_samples, dtype=np.float64)
    phrases = [(0.24 * scale, 1.08 * scale), (1.52 * scale, 2.28 * scale), (2.52 * scale, 3.82 * scale)]
    for start_s, end_s in phrases:
        s_i = max(0, int(start_s * sr))
        e_i = min(n_samples, int(end_s * sr))
        if e_i <= s_i:
            continue
        seg = np.ones(e_i - s_i, dtype=np.float64)
        # Syllable modulation
        seg_t = np.linspace(0.0, end_s - start_s, e_i - s_i)
        seg *= 0.65 + 0.35 * np.abs(np.sin(2.0 * np.pi * 3.1 * seg_t + 0.4))
        ramp = min(int(0.03 * sr), (e_i - s_i) // 4)
        if ramp > 0:
            seg[:ramp] *= np.linspace(0.0, 1.0, ramp)
            seg[-ramp:] *= np.linspace(1.0, 0.0, ramp)
        envelope[s_i:e_i] = seg

        # Pre-onset inhalation breath (160ms before phrase)
        b_start = max(0, s_i - int(0.16 * sr))
        if s_i - b_start > 16:
            breath_env[b_start:s_i] = np.hanning(s_i - b_start)

    breath_sound = 0.09 * _bandpass(rng.standard_normal(n_samples), sr, 1800.0, 4800.0) * breath_env
    # Subtle natural room noise floor (~-36 dB)
    room_noise = 0.012 * _bandpass(rng.standard_normal(n_samples), sr, 120.0, 6500.0)

    sig = (voice * shimmer * envelope) + (0.28 * fricative_noise * fricative_gate) + breath_sound + room_noise
    peak = float(np.max(np.abs(sig))) + 1e-8
    return (0.78 * sig / peak).astype(np.float32)
