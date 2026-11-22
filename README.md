# 🛡️ QALQAN Voice

**Real-time AI voice deepfake detector for Kazakh, Russian, and code-switched speech.**  
*Детектор поддельных ИИ-голосов на казахском и русском — проверяет звонки и голосовые сообщения в реальном времени по физике звука.*

[![CI](https://github.com/overloooooord/qalqan-voice/actions/workflows/ci.yml/badge.svg)](https://github.com/overloooooord/qalqan-voice/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-4f46e5)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20%2B%20WebSocket-059669)
![PyTorch](https://img.shields.io/badge/PyTorch-LCNN%20%2B%20LFCC-ee4c2c)
![License: MIT](https://img.shields.io/badge/license-MIT-slate)

---

## Why QALQAN Voice?

Modern voice cloning models require as little as **3 seconds of audio** and cost **0 ₸** with open-source tools. Meanwhile, Western anti-spoofing solutions are trained mostly on clean English speech and degrade on **Kazakh phonetics**, **Russian–Kazakh code-switching** (*«Сәлем, срочно переведи, мен кейін бәрін түсіндіремін»*), and compressed messenger audio.

**QALQAN Voice listens to the physics of sound, not just words:**
- **60-dim LFCC + Log-Mel Spectrograms** preserving high-frequency resolution (4–8 kHz) where neural vocoders (ElevenLabs, RVC, VALL-E, HiFi-GAN) leave artifacts.
- **4 Acoustic-Physics Forensics:** 6–8 kHz vocoder banding, inter-phrase micro-breath & pause entropy, vocal-tract pitch jitter / formant smoothness, and spectral flux stationarity — with automatic narrowband codec adaptation for WhatsApp/telephony.
- **Compact PyTorch `QalqanLCNN` (Light CNN + MFM activations)** designed for low-latency CPU inference and **air-gapped on-premise** deployment (with ONNX export).

> 📄 **Pitch Deck:** [`docs/QALQAN_pitch.pdf`](docs/QALQAN_pitch.pdf) · **Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · **Demo Script:** [`docs/PITCH_SCRIPT.md`](docs/PITCH_SCRIPT.md)

---

## Key Components

| Module | Path | Description |
|---|---|---|
| **Web Dashboard** | `qalqan/web/static/` | Interactive monitor with live canvas spectrogram, circular risk gauge, RU/KZ/EN UI, drag-drop upload, live microphone streaming (`M`), and one-click AI/Live simulation |
| **REST & WebSocket API** | `qalqan/api/app.py` | FastAPI backend (`POST /api/v1/analyze`, `POST /api/v1/simulate`, `GET /api/v1/history`, `WS /ws/stream`) with OpenAPI docs at `/docs` |
| **Audio Forensics Engine** | `qalqan/audio/` | Native WAV + FFmpeg decoding (`.ogg` Opus, `.mp3`, `.webm`, `.m4a`, `.flac`) + LFCC, Log-Mel, and 4 acoustic-physics indicators |
| **PyTorch LCNN & Trainer** | `qalqan/ml/` | `QalqanLCNN` binary classifier with physics fusion head, EER evaluation, and telephony/noise channel augmentation training CLI |
| **Telegram Bot** | `qalqan/bot/telegram_bot.py` | `aiogram 3` bot that verifies forwarded voice notes, video notes, and audio files in RU / KZ / EN |
| **CLI & ONNX Export** | `scripts/` | Local terminal analyzer (`demo_cli.py`) and ONNX exporter (`export_onnx.py`) for C++/SIP gateway integration |

---

## Quick Start

### 1. Install dependencies

Requires **Python 3.11+** and **FFmpeg** (for `.ogg` / `.mp3` / `.webm` / `.m4a` voice messages; `.wav` works natively without FFmpeg).

```bash
git clone https://github.com/overloooooord/qalqan-voice.git
cd qalqan-voice
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard & API

```bash
uvicorn qalqan.api.app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser:
- Click **«Симуляция: ИИ»** or **«Симуляция: Живой»** to run real acoustic waveforms through the detection pipeline.
- Press **`M`** (or click **«Микрофон»**) to stream live 16 kHz PCM audio over WebSocket `/ws/stream`.
- Drag and drop any `.ogg`, `.wav`, `.mp3`, or `.m4a` voice message to inspect it immediately.
- Interactive Swagger / OpenAPI docs: **http://localhost:8000/docs**.

### 3. Run the CLI Analyzer

```bash
# Built-in synthetic vs live demo
python -m scripts.demo_cli --demo --lang ru

# Inspect any local audio file (supports RU / KZ / EN output)
python -m scripts.demo_cli path/to/voice_note.ogg --lang kk
```

### 4. Run the Telegram Bot

```bash
cp .env.example .env
# Add your BotFather token to QALQAN_TELEGRAM_TOKEN in .env
export QALQAN_TELEGRAM_TOKEN="123456:ABC..."
python -m qalqan.bot.telegram_bot
```

Forward any Telegram voice message or audio file to the bot to receive a formatted risk card with synthetic probability, latency, and acoustic reasons. Switch languages with `/lang`.

---

## API Reference

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Analyze an uploaded voice file (lang: ru | kk | en)
curl -X POST "http://localhost:8000/api/v1/analyze?lang=ru" \
  -F "file=@sample.ogg"

# Run a simulated detection scenario
curl -X POST "http://localhost:8000/api/v1/simulate" \
  -H "Content-Type: application/json" \
  -d '{"scenario": "synthetic", "lang": "kk"}'
```

Example response (truncated spectrogram preview):

```json
{
  "id": "a9f31c0e4b",
  "timestamp": "19:09:57",
  "source": "file:sample.ogg",
  "label": "synthetic",
  "probability": 0.9421,
  "confidence": 0.9151,
  "engine_mode": "acoustic-physics",
  "latency_ms": 28.4,
  "duration_sec": 4.0,
  "lang": "ru",
  "reasons": [
    "Следы нейровокодера в диапазоне 6–8 кГц",
    "Нет микропауз дыхания между фразами",
    "Неестественно ровный резонанс голосового тракта"
  ],
  "metrics": {
    "hf_vocoder_score": 0.98,
    "breath_pause_score": 0.87,
    "resonance_smoothness_score": 0.62,
    "phase_stationarity_score": 0.55,
    "snr_estimate_db": 64.2,
    "speech_ratio": 0.76,
    "codec_narrowband": false
  }
}
```

---

## Training `QalqanLCNN` on Your Own RU / KZ Dataset

Out of the box (without a trained checkpoint in `models/qalqan_lcnn.pt`), QALQAN runs in **`acoustic-physics`** mode. To enable **`hybrid-lcnn`** mode, train on a directory of bona fide (`real/`) and spoofed (`fake/`) recordings:

```text
dataset/
├── real/   # Genuine Kazakh, Russian & code-switched recordings (.wav/.ogg/.mp3)
└── fake/   # Cloned/TTS recordings (ElevenLabs, RVC, XTTS, VALL-E, etc.)
```

```bash
# Train with telephony/WhatsApp bandpass & noise augmentations
python -m qalqan.ml.train --data-dir ./dataset --epochs 20 --out models/qalqan_lcnn.pt

# Or run the pipeline smoke-test trainer on synthetic acoustic fixtures
python -m qalqan.ml.train --bootstrap-demo --epochs 5 --out models/qalqan_lcnn.pt

# Export trained model to ONNX for C++ / telecom gateway inference
python -m scripts.export_onnx --checkpoint models/qalqan_lcnn.pt --out models/qalqan_lcnn.onnx
```

Once `models/qalqan_lcnn.pt` is present (or configured via `QALQAN_MODEL_PATH`), the API and Telegram bot automatically activate `hybrid-lcnn` inference.

---

## Docker / On-Premise Deployment

```bash
# Build & run REST API + Web Dashboard (air-gapped ready, zero external calls)
docker compose up --build

# Also start the Telegram bot container
docker compose --profile bot up --build
```

---

## Testing

```bash
pytest -v
```

---

## Roadmap

- [x] Real-time Web Dashboard with live spectrogram, WebSocket microphone streaming, and RU/KZ/EN localization
- [x] Acoustic-physics forensics engine (60-dim LFCC, 6–8 kHz vocoder analysis, micro-breath & pitch jitter)
- [x] Compact PyTorch `QalqanLCNN` architecture, augmentation trainer, and ONNX export
- [x] FastAPI REST/WebSocket service + `aiogram 3` Telegram bot + Docker on-premise bundle
- [ ] Large-scale Kazakh / Russian / code-switched spoofing benchmark dataset collection & pretrained weights release
- [ ] SIP / RTP gateway adapter for telecom operators and call-center antifraud systems

## License

MIT — see [`LICENSE`](LICENSE).
