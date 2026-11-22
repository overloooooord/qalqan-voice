"""Acoustic-physics forensics & spectral feature extraction (LFCC, Log-Mel, vocoder & vocal-tract metrics).

Design motivation (language-agnostic physics for Kazakh, Russian & code-switching):
- Linear Frequency Cepstral Coefficients (LFCC) preserve high-frequency spectral resolution
  in the 4-8 kHz range where neural vocoders (HiFi-GAN, BigVGAN, WaveNet, VALL-E, RVC)
  leave subtle phase/energy artifacts.
- Micro-breath and pause analysis checks for natural low-level aspiration/inhalation noise
  and organic pause-duration entropy vs. overly gated digital silence between cloned phrases.
- Vocal-tract resonance & pitch micro-perturbation (jitter/formant flux) separates natural
  physiological pitch variation from overly smooth neural pitch trajectories.
- Narrowband codec detection adapts weights when telephony or WhatsApp Opus strips >3.8 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct
from scipy.signal import stft


@dataclass
class AcousticFeatures:
    """Extracted spectral matrices and acoustic-physics forensic metrics."""

    lfcc: np.ndarray  # shape: (60, T) -> 20 static + 20 delta + 20 delta-delta
    log_mel: np.ndarray  # shape: (64, T)
    spectrogram_preview: list[list[float]]  # downsampled (32, <=96) values in [0, 1]
    hf_vocoder_score: float  # [0, 1] higher => stronger 6-8 kHz vocoder artifact signature
    breath_pause_score: float  # [0, 1] higher => missing organic micro-breaths / uniform pauses
    resonance_smoothness_score: float  # [0, 1] higher => unnaturally smooth formant/pitch track
    phase_stationarity_score: float  # [0, 1] higher => overly uniform harmonic/spectral flux
    snr_estimate_db: float
    speech_ratio: float
    codec_narrowband: bool

    def physics_vector(self) -> np.ndarray:
        """Compact 4-dim forensic feature vector for fusion with neural embeddings."""
        return np.array(
            [
                self.hf_vocoder_score,
                self.breath_pause_score,
                self.resonance_smoothness_score,
                self.phase_stationarity_score,
            ],
            dtype=np.float32,
        )


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _build_triangular_filterbank(
    n_fft: int,
    sr: int,
    n_filters: int,
    fmin: float,
    fmax: float,
    scale: str = "linear",
) -> np.ndarray:
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_bins)

    if scale == "mel":
        mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_filters + 2)
        hz_pts = _mel_to_hz(mel_pts)
    else:
        hz_pts = np.linspace(fmin, fmax, n_filters + 2)

    fb = np.zeros((n_filters, n_bins), dtype=np.float32)
    for m in range(n_filters):
        f_left, f_center, f_right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        left_slope = (freqs - f_left) / max(f_center - f_left, 1e-6)
        right_slope = (f_right - freqs) / max(f_right - f_center, 1e-6)
        fb[m] = np.maximum(0.0, np.minimum(left_slope, right_slope))
    return fb


def _compute_deltas(feat: np.ndarray, width: int = 2) -> np.ndarray:
    """Compute finite-difference temporal deltas along axis=1."""
    n_feats, n_frames = feat.shape
    if n_frames < 2:
        return np.zeros_like(feat)
    padded = np.pad(feat, ((0, 0), (width, width)), mode="edge")
    denom = 2.0 * sum(n * n for n in range(1, width + 1))
    delta = np.zeros((n_feats, n_frames), dtype=np.float32)
    for n in range(1, width + 1):
        delta += n * (padded[:, width + n : width + n + n_frames] - padded[:, width - n : width - n + n_frames])
    return delta / denom


def _clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.5
    return float(max(0.0, min(1.0, value)))


def _estimate_pitch_jitter(
    samples: np.ndarray,
    sr: int,
    frame_len: int = 400,
    hop: int = 160,
    voiced_mask: np.ndarray | None = None,
) -> float:
    """Estimate cycle-to-cycle F0 relative jitter on voiced frames via autocorrelation."""
    min_lag = max(1, int(sr / 420.0))  # ~420 Hz upper speech pitch
    max_lag = min(frame_len - 1, int(sr / 75.0))  # ~75 Hz lower speech pitch
    if max_lag <= min_lag or samples.size < frame_len:
        return 0.015

    n_frames = 1 + (samples.size - frame_len) // hop
    f0_values: list[float] = []
    window = np.hanning(frame_len).astype(np.float32)

    for idx in range(n_frames):
        if voiced_mask is not None and idx < voiced_mask.size and not voiced_mask[idx]:
            continue
        frame = samples[idx * hop : idx * hop + frame_len] * window
        energy = float(np.dot(frame, frame))
        if energy < 1e-6:
            continue
        ac = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
        if ac[0] <= 1e-8:
            continue
        ac_slice = ac[min_lag : max_lag + 1]
        best_offset = int(np.argmax(ac_slice))
        peak_ratio = float(ac_slice[best_offset] / ac[0])
        if peak_ratio > 0.35:
            lag = min_lag + best_offset
            # Parabolic refinement around autocorrelation peak
            if 0 < best_offset < ac_slice.size - 1:
                y0, y1, y2 = ac_slice[best_offset - 1], ac_slice[best_offset], ac_slice[best_offset + 1]
                denom = (y0 - 2.0 * y1 + y2)
                if abs(denom) > 1e-9:
                    lag = lag + 0.5 * (y0 - y2) / denom
            f0_values.append(sr / max(lag, 1.0))

    if len(f0_values) < 6:
        return 0.015

    f0_arr = np.asarray(f0_values, dtype=np.float32)
    # Filter large octave jumps between adjacent frames
    rel_diffs = np.abs(np.diff(f0_arr)) / np.maximum(f0_arr[:-1], 50.0)
    local_diffs = rel_diffs[rel_diffs < 0.18]
    if local_diffs.size < 4:
        return 0.015
    return float(np.median(local_diffs))


def _make_spectrogram_preview(log_mel: np.ndarray, freq_bins: int = 32, max_cols: int = 96) -> list[list[float]]:
    """Downsample Log-Mel spectrogram into a compact 0..1 matrix for dashboard rendering."""
    n_mels, n_frames = log_mel.shape
    # Frequency downsampling
    mel_idx = np.linspace(0, n_mels - 1, freq_bins).astype(int)
    reduced = log_mel[mel_idx, :]

    # Time downsampling
    cols = min(max_cols, max(16, n_frames))
    time_idx = np.linspace(0, max(0, n_frames - 1), cols).astype(int)
    reduced = reduced[:, time_idx]

    lo = float(np.percentile(reduced, 5))
    hi = float(np.percentile(reduced, 99))
    norm = (reduced - lo) / max(hi - lo, 1e-5)
    norm = np.clip(norm, 0.0, 1.0)
    return [[round(float(v), 3) for v in row] for row in norm]


def extract_acoustic_features(samples: np.ndarray, sr: int = 16000) -> AcousticFeatures:
    """Compute LFCC, Log-Mel spectrogram, and 4 acoustic-physics forensic metrics."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < 400:
        samples = np.pad(samples, (0, 400 - samples.size))

    n_fft = 512
    win_length = 400  # 25 ms @ 16 kHz
    hop_length = 160  # 10 ms @ 16 kHz

    freqs, _times, zxx = stft(
        samples,
        fs=sr,
        window="hann",
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    mag = np.abs(zxx).astype(np.float32)
    power = mag * mag + 1e-10

    # 1. Log-Mel Spectrogram (64 bands)
    mel_fb = _build_triangular_filterbank(n_fft, sr, n_filters=64, fmin=40.0, fmax=sr / 2.0, scale="mel")
    mel_spec = np.maximum(mel_fb @ power, 1e-10)
    log_mel = np.log10(mel_spec).astype(np.float32)

    # 2. LFCC (20 static + 20 delta + 20 delta-delta = 60 dims)
    lin_fb = _build_triangular_filterbank(n_fft, sr, n_filters=40, fmin=0.0, fmax=sr / 2.0, scale="linear")
    lin_log = np.log(np.maximum(lin_fb @ power, 1e-10))
    lfcc_static = dct(lin_log, type=2, axis=0, norm="ortho")[:20, :].astype(np.float32)
    lfcc_delta = _compute_deltas(lfcc_static, width=2)
    lfcc_dd = _compute_deltas(lfcc_delta, width=2)
    lfcc = np.concatenate([lfcc_static, lfcc_delta, lfcc_dd], axis=0)

    # Frame energy & Voice Activity mask
    frame_energy = np.mean(power, axis=0)
    energy_db = 10.0 * np.log10(np.maximum(frame_energy, 1e-12))
    p90 = float(np.percentile(energy_db, 90))
    p15 = float(np.percentile(energy_db, 15))
    snr_estimate_db = float(np.clip(p90 - p15, 0.0, 80.0))

    speech_thresh = max(p90 - 22.0, -65.0)
    voiced_mask = energy_db > speech_thresh
    speech_ratio = float(np.mean(voiced_mask))
    if not np.any(voiced_mask):
        voiced_mask = np.ones_like(voiced_mask, dtype=bool)

    voiced_power = power[:, voiced_mask]

    # --- Forensic Indicator 1: 6-8 kHz Neural Vocoder Artifact Signature ---
    mid_mask = (freqs >= 300.0) & (freqs <= 3400.0)
    upper_mask = (freqs >= 4000.0) & (freqs < 6000.0)
    hf_mask = (freqs >= 6000.0) & (freqs <= min(7950.0, sr / 2.0))

    mid_energy = float(np.mean(voiced_power[mid_mask, :])) + 1e-12
    upper_energy = float(np.mean(voiced_power[upper_mask, :])) + 1e-12
    hf_energy = float(np.mean(voiced_power[hf_mask, :])) + 1e-12

    # Detect narrowband telephony / aggressive Opus compression (<3.8 kHz cutoff)
    codec_narrowband = bool((upper_energy / mid_energy) < 1e-4 and (hf_energy / mid_energy) < 1e-5)

    if codec_narrowband:
        # Rely on mid-band spectral flatness anomalies when high band was stripped by codec
        mid_log = np.log(voiced_power[mid_mask, :] + 1e-12)
        mid_flatness = float(np.mean(np.exp(np.mean(mid_log, axis=0)) / (np.mean(voiced_power[mid_mask, :], axis=0) + 1e-12)))
        hf_vocoder_score = _clamp01((mid_flatness - 0.12) / 0.35)
    else:
        hf_slice = voiced_power[hf_mask, :]
        # Spectral flatness (Wiener entropy) & temporal uniformity in 6-8 kHz
        hf_geo = np.exp(np.mean(np.log(hf_slice + 1e-12), axis=0))
        hf_arith = np.mean(hf_slice, axis=0) + 1e-12
        hf_flatness = float(np.mean(hf_geo / hf_arith))

        # Check for sharp spectral shelf/banding or synthetic elevated metallic buzz in 6-8 kHz
        hf_log_spectrum = np.mean(np.log10(hf_slice + 1e-12), axis=1)
        hf_band_ripple = float(np.std(np.diff(hf_log_spectrum))) if hf_log_spectrum.size > 3 else 0.0

        # Temporal CV in 6-8 kHz: organic fricatives burst sharply (high CV); vocoder noise is steadier
        hf_frame_env = np.mean(hf_slice, axis=0)
        hf_temporal_cv = float(np.std(hf_frame_env) / (np.mean(hf_frame_env) + 1e-12))

        # Combine: elevated periodic ripple + unnatural flatness/stationarity => higher vocoder score
        ripple_component = _clamp01((hf_band_ripple - 0.045) / 0.14)
        stationarity_component = _clamp01((0.95 - hf_temporal_cv) / 0.75)
        flatness_component = _clamp01((0.55 - hf_flatness) / 0.45) if hf_band_ripple > 0.07 else _clamp01((hf_flatness - 0.45) / 0.40)
        hf_vocoder_score = _clamp01(0.45 * ripple_component + 0.35 * stationarity_component + 0.20 * flatness_component)

    # --- Forensic Indicator 2: Micro-breath & Pause Cadence ---
    # Natural speech pauses have low-level aspiration/room breath energy (1.5-5 kHz) and variable lengths.
    # TTS/clones often produce digitally zeroed/ultra-clean uniform gaps or zero breathing between phrases.
    pause_mask = ~voiced_mask
    breath_band = (freqs >= 1500.0) & (freqs <= 5000.0)
    if np.sum(pause_mask) >= 6:
        pause_energy_db = energy_db[pause_mask]
        # Ultra-clean digital floor (SNR > 48 dB or dead-silent pauses below -60 dB relative to peak)
        dead_silence_ratio = float(np.mean(pause_energy_db < (p90 - 44.0)))

        # Look for aspiration/breath energy in 200ms leading up to speech onsets
        onset_indices = np.where(np.diff(voiced_mask.astype(np.int8)) == 1)[0]
        pre_onset_breath_ratios: list[float] = []
        for onset in onset_indices:
            start = max(0, onset - 18)
            if onset - start >= 4:
                pre_breath = float(np.mean(power[breath_band, start:onset]))
                pre_ratio = pre_breath / max(mid_energy, 1e-12)
                pre_onset_breath_ratios.append(pre_ratio)

        if pre_onset_breath_ratios:
            mean_pre_breath = float(np.mean(pre_onset_breath_ratios))
            # Very low pre-onset breath energy (< 1.5e-4 of voiced mid energy) => missing natural inhalation
            missing_breath = _clamp01((1.8e-3 - mean_pre_breath) / 1.75e-3)
        else:
            missing_breath = 0.55

        # Pause run-length uniformity
        runs: list[int] = []
        cur = 0
        for is_pause in pause_mask:
            if is_pause:
                cur += 1
            elif cur >= 4:
                runs.append(cur)
                cur = 0
        if cur >= 4:
            runs.append(cur)
        if len(runs) >= 2:
            run_cv = float(np.std(runs) / (np.mean(runs) + 1e-6))
            uniform_pause_score = _clamp01((0.35 - run_cv) / 0.35)
        else:
            uniform_pause_score = 0.45

        breath_pause_score = _clamp01(0.45 * dead_silence_ratio + 0.35 * missing_breath + 0.20 * uniform_pause_score)
    else:
        # Continuous speech without natural pauses over >2.5 seconds is itself mildly suspicious
        duration_sec = samples.size / float(sr)
        breath_pause_score = _clamp01(0.42 + (0.28 if duration_sec > 3.0 else 0.0))

    # --- Forensic Indicator 3: Vocal-Tract Resonance Smoothness & Pitch Jitter ---
    jitter = _estimate_pitch_jitter(samples, sr=sr, frame_len=win_length, hop=hop_length, voiced_mask=voiced_mask)
    # Organic human speech usually has ~0.008 - 0.035 frame-to-frame pitch jitter;
    # Neural vocoders frequently produce < 0.0055 ultra-smooth pitch trajectories.
    low_jitter_score = _clamp01((0.0115 - jitter) / 0.0095)

    # Cepstral delta variance on voiced frames (smoothness of formant trajectory)
    voiced_deltas = lfcc_delta[:, voiced_mask]
    delta_std = float(np.mean(np.std(voiced_deltas, axis=1))) if voiced_deltas.shape[1] > 3 else 0.25
    smooth_formant_score = _clamp01((0.28 - delta_std) / 0.22)

    resonance_smoothness_score = _clamp01(0.60 * low_jitter_score + 0.40 * smooth_formant_score)

    # --- Forensic Indicator 4: Spectral Flux / Harmonic Stationarity ---
    voiced_mag = mag[:, voiced_mask]
    if voiced_mag.shape[1] > 4:
        norm_mag = voiced_mag / (np.linalg.norm(voiced_mag, axis=0, keepdims=True) + 1e-8)
        flux = np.linalg.norm(np.diff(norm_mag, axis=1), axis=0)
        flux_cv = float(np.std(flux) / (np.mean(flux) + 1e-8))
        flux_mean = float(np.mean(flux))
        # Over-regularized neural synthesis tends to show lower spectral flux variability
        phase_stationarity_score = _clamp01(
            0.55 * _clamp01((0.22 - flux_mean) / 0.16)
            + 0.45 * _clamp01((0.75 - flux_cv) / 0.55)
        )
    else:
        phase_stationarity_score = 0.5

    preview = _make_spectrogram_preview(log_mel, freq_bins=32, max_cols=96)

    return AcousticFeatures(
        lfcc=lfcc,
        log_mel=log_mel,
        spectrogram_preview=preview,
        hf_vocoder_score=round(hf_vocoder_score, 4),
        breath_pause_score=round(breath_pause_score, 4),
        resonance_smoothness_score=round(resonance_smoothness_score, 4),
        phase_stationarity_score=round(phase_stationarity_score, 4),
        snr_estimate_db=round(snr_estimate_db, 2),
        speech_ratio=round(speech_ratio, 4),
        codec_narrowband=codec_narrowband,
    )
