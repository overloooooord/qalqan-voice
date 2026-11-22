"""Integration tests for the FastAPI REST and WebSocket endpoints."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from qalqan.api.app import create_app
from qalqan.audio.loader import save_wav_bytes
from qalqan.audio.synth_samples import generate_live_human_sample, generate_synthetic_voice_sample
from qalqan.config import Settings


def _client() -> TestClient:
    return TestClient(create_app(Settings(model_path=None)))


def test_dashboard_and_health():
    client = _client()
    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "QALQAN" in r_home.text

    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200
    body = r_health.json()
    assert body["status"] == "ok"
    assert body["sample_rate"] == 16000


def test_simulate_endpoints_and_history():
    client = _client()

    r_synth = client.post("/api/v1/simulate", json={"scenario": "synthetic", "lang": "ru"})
    assert r_synth.status_code == 200
    synth_data = r_synth.json()
    assert synth_data["label"] == "synthetic"
    assert synth_data["probability"] >= 0.70
    assert len(synth_data["spectrogram_preview"]) == 32

    r_live = client.post("/api/v1/simulate", json={"scenario": "live", "lang": "kk"})
    assert r_live.status_code == 200
    live_data = r_live.json()
    assert live_data["label"] == "live"
    assert live_data["probability"] <= 0.30
    assert live_data["lang"] == "kk"

    r_hist = client.get("/api/v1/history")
    assert r_hist.status_code == 200
    assert len(r_hist.json()) == 2

    r_clear = client.delete("/api/v1/history")
    assert r_clear.status_code == 200
    assert client.get("/api/v1/history").json() == []


def test_analyze_wav_upload_and_invalid_input():
    client = _client()
    wav_bytes = save_wav_bytes(generate_synthetic_voice_sample(duration_sec=3.0), sample_rate=16000)

    r_ok = client.post(
        "/api/v1/analyze?lang=en",
        files={"file": ("suspect_call.wav", wav_bytes, "audio/wav")},
    )
    assert r_ok.status_code == 200
    data = r_ok.json()
    assert data["label"] == "synthetic"
    assert data["source"] == "file:suspect_call.wav"
    assert data["lang"] == "en"

    r_empty = client.post(
        "/api/v1/analyze",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert r_empty.status_code == 400


def test_websocket_realtime_stream():
    client = _client()
    waveform = generate_synthetic_voice_sample(duration_sec=2.2, sr=16000)
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    with client.websocket_connect("/ws/stream?lang=ru&sample_rate=16000") as ws:
        # Send first 0.5s -> buffering ack
        chunk_half_sec = 16000  # 8000 samples * 2 bytes
        ws.send_bytes(pcm[:chunk_half_sec])
        msg1 = ws.receive_json()
        assert msg1["type"] == "buffering"

        # Send remaining 1.7s -> triggers live verdict
        ws.send_bytes(pcm[chunk_half_sec:])
        msg2 = ws.receive_json()
        assert msg2["type"] == "verdict"
        assert msg2["label"] == "synthetic"
        assert msg2["probability"] >= 0.65
