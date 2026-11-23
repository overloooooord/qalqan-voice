# QALQAN Voice — Reproducible CPU Latency & Multi-Channel Robustness Benchmark

Reproduce locally with:

```bash
python -m scripts.benchmark --samples-per-class 25
```

## 1. Multi-Channel Acoustic-Physics Pipeline Benchmark (CPU-Only)

Measured on a single CPU core (`Python 3.14`, pure-NumPy DSP engine, **no GPU**, **zero external API calls**), evaluating **250 audio clips** (50 clips/channel, 4.0 seconds each) across 5 realistic transmission conditions including real FFmpeg Opus transcodes (Telegram 24 kbps, WhatsApp 16 kbps), 15 dB SNR additive noise, and 8 kHz SIP narrowband telephony:

| Channel | Clips | Accuracy | FPR | FNR | EER | p50 Latency | p95 Latency | Real-Time Factor (RTF) | Speed vs. Real-Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Clean Wideband (16 kHz WAV)** | 50 | 100.0% | 0.0% | 0.0% | 0.0% | **24.1 ms** | 31.4 ms | `0.0060` | **166×** |
| **Telegram Voice (`.ogg` Opus 24 kbps)** | 50 | 100.0% | 0.0% | 0.0% | 0.0% | **30.2 ms** | 49.8 ms | `0.0076` | **131×** |
| **WhatsApp Voice (`.ogg` Opus 16 kbps)** | 50 | 100.0% | 0.0% | 0.0% | 0.0% | **27.0 ms** | 63.0 ms | `0.0067` | **149×** |
| **Noisy Street / Room (15 dB SNR)** | 50 | 100.0% | 0.0% | 0.0% | 0.0% | **30.9 ms** | 48.0 ms | `0.0077` | **130×** |
| **Telephony Narrowband (8 kHz SIP)** | 50 | 100.0% | 0.0% | 0.0% | 0.0% | **19.1 ms** | 23.0 ms | `0.0048` | **208×** |

### Key Engineering Takeaways
- **Sub-35 ms inline latency (`p50`) on CPU:** Enables real-time call-center and SIP-trunk screening inside a 1.5–2.0 second detection window without GPU servers.
- **Codec & Narrowband Resilience:** Automatic `codec_narrowband` detection prevents false alarms when telephony or WhatsApp compression strips frequencies above 3.8 kHz.
- **Serverless & On-Premise Footprint:** Pure-NumPy core runtime is **~78 MB unzipped** (deploys on serverless edge or inside air-gapped on-premise containers with zero outbound traffic and zero audio persistence).

> **Scope Note (Methodology Transparency):** The table above benchmarks the deterministic acoustic-physics DSP pipeline and FFmpeg/Opus/narrowband channel adapters on controlled synthetic-vs-organic waveforms (`qalqan/audio/synth_samples.py`). Real-world production EER/FPR on human Kazakh/Russian voices requires training and blinded evaluation of `QalqanLCNN` on consented natural + cloned speech (Protocol in Section 2).

## 2. Blinded Kazakh / Russian / Code-Switching Validation Protocol (Pilot Standard)

To establish institutional-grade detection benchmarks for banks, telecoms, and government pilots, QALQAN Voice provides an open, reproducible training and evaluation pipeline (`python -m qalqan.ml.train`):

1. **Bona Fide Corpus:** Consented Kazakh (`kk`), Russian (`ru`), and bilingual code-switched recordings (Mozilla Common Voice `kk`/`ru` + regional accents across Almaty, Astana, West/South KZ).
2. **Spoofing Corpus:** Multi-generator clones generated from 3–30s reference clips across commercial & open-source neural vocoders (`ElevenLabs`, `XTTS-v2`, `RVC v2`, `OpenVoice`, `VALL-E` variants).
3. **Channel Stress Matrix:** Every test utterance evaluated across clean wideband, Telegram `.ogg` Opus 24k, WhatsApp `.ogg` Opus 16k, GSM/SIP 8 kHz narrowband, and 10–20 dB SNR street/office noise.
4. **Target Pilot KPIs:**
   - Equal Error Rate (EER): **< 3.0%** on Kazakh & Kazakh–Russian code-switched speech
   - Operating False Positive Rate (FPR) at bank fraud-review threshold: **< 1.0%**
   - End-to-end streaming inference latency: **< 50 ms** on CPU (ONNX Runtime)
   - Data privacy: **0 bytes** of raw customer audio retained on disk (RAM-only processing)
