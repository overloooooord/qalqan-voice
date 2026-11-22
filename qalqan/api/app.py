"""FastAPI application for QALQAN Voice — REST API, WebSocket real-time streaming, and Web Dashboard."""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from qalqan import __version__
from qalqan.api.schemas import (
    AcousticMetricsSchema,
    AnalyzeResponse,
    HealthResponse,
    HistoryItem,
    SimulateRequest,
)
from qalqan.api.store import HistoryStore
from qalqan.audio.loader import AudioLoadError
from qalqan.audio.synth_samples import generate_live_human_sample, generate_synthetic_voice_sample
from qalqan.config import Settings
from qalqan.ml.detector import DetectionVerdict, VoiceDeepfakeDetector

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"

_DEMO_HINTS: dict[str, dict[str, str]] = {
    "synthetic": {
        "ru": "«Срочно переведите средства на резервный счёт. Реквизиты отправляю следующим сообщением.»",
        "kk": "«Шұғыл түрде қаражатты резервтік шотқа аударыңыз. Деректемелерді келесі хабарламада жіберемін.»",
        "en": "“Transfer the funds to the reserve account immediately. Sending details in the next message.”",
    },
    "live": {
        "ru": "«Сәлем, всё в порядке, созвонимся через десять минут и обсудим детали по проекту.»",
        "kk": "«Сәлем, бәрі дұрыс, он минуттан кейін хабарласып, жоба бойынша сөйлесейік.»",
        "en": "“Hi, everything looks good — let's hop on a quick call in ten minutes to review the plan.”",
    },
}


def _to_response(
    verdict: DetectionVerdict,
    record_id: str,
    timestamp: str,
    source: str,
    transcript_hint: str | None = None,
) -> AnalyzeResponse:
    return AnalyzeResponse(
        id=record_id,
        timestamp=timestamp,
        source=source,
        label=verdict.label,  # type: ignore[arg-type]
        probability=verdict.probability,
        confidence=verdict.confidence,
        engine_mode=verdict.engine_mode,
        latency_ms=verdict.latency_ms,
        duration_sec=verdict.duration_sec,
        source_format=verdict.source_format,
        lang=verdict.lang,
        reasons=verdict.reasons,
        metrics=AcousticMetricsSchema(**verdict.metrics),
        spectrogram_preview=verdict.spectrogram_preview,
        transcript_hint=transcript_hint,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    detector = VoiceDeepfakeDetector(settings=cfg)
    history = HistoryStore(maxlen=cfg.history_limit)

    application = FastAPI(
        title="QALQAN Voice API",
        description="Real-time voice deepfake detection for Kazakh, Russian, and code-switched speech.",
        version=__version__,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.state.settings = cfg
    application.state.detector = detector
    application.state.history = history

    if STATIC_DIR.is_dir():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="Dashboard assets not found.")
        return FileResponse(index_path)

    @application.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            engine_mode=detector.engine_mode,
            model_loaded=detector.model is not None,
            sample_rate=cfg.sample_rate,
            ffmpeg_available=shutil.which("ffmpeg") is not None,
        )

    @application.post("/api/v1/analyze", response_model=AnalyzeResponse)
    async def analyze_audio(
        file: UploadFile = File(...),
        lang: str = Query(default=cfg.default_lang, pattern="^(ru|kk|en)$"),
    ) -> AnalyzeResponse:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio file exceeds 25 MB limit.")

        filename = file.filename or "audio.wav"
        try:
            verdict = await run_in_threadpool(detector.analyze_bytes, payload, filename, lang)
        except AudioLoadError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        record_id, ts = history.record(verdict, source=f"file:{filename}")
        return _to_response(verdict, record_id=record_id, timestamp=ts, source=f"file:{filename}")

    @application.post("/api/v1/simulate", response_model=AnalyzeResponse)
    async def simulate(req: SimulateRequest) -> AnalyzeResponse:
        if req.scenario == "synthetic":
            waveform = generate_synthetic_voice_sample(duration_sec=4.0, sr=cfg.sample_rate)
            src = "sim:synthetic"
        else:
            waveform = generate_live_human_sample(duration_sec=4.0, sr=cfg.sample_rate)
            src = "sim:live"

        verdict = await run_in_threadpool(
            detector.analyze_waveform,
            waveform,
            cfg.sample_rate,
            "wav",
            req.lang,
        )
        record_id, ts = history.record(verdict, source=src)
        hint = _DEMO_HINTS[req.scenario].get(req.lang, _DEMO_HINTS[req.scenario]["ru"])
        return _to_response(verdict, record_id=record_id, timestamp=ts, source=src, transcript_hint=hint)

    @application.get("/api/v1/history", response_model=list[HistoryItem])
    async def get_history(limit: int = Query(default=20, ge=1, le=50)) -> list[HistoryItem]:
        return history.list_items(limit=limit)

    @application.delete("/api/v1/history")
    async def clear_history() -> dict[str, str]:
        history.clear()
        return {"status": "cleared"}

    @application.websocket("/ws/stream")
    async def stream_audio(websocket: WebSocket) -> None:
        """Real-time audio stream verification over WebSocket.

        Accepts either:
        - Binary Int16 little-endian mono PCM chunks at 16 kHz, OR
        - JSON frames: {"pcm16_b64": "...", "sample_rate": 16000, "lang": "ru", "reset": false}
        Maintains a rolling 3.5-second window and emits live detection verdicts.
        """
        await websocket.accept()
        lang = websocket.query_params.get("lang", cfg.default_lang)
        sr = int(websocket.query_params.get("sample_rate", str(cfg.sample_rate)))
        window_samples = int(cfg.sample_rate * 3.5)
        min_eval_samples = int(cfg.sample_rate * 1.0)
        hop_eval_samples = int(cfg.sample_rate * 0.65)

        ring = np.zeros(0, dtype=np.float32)
        since_last_eval = 0

        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                pcm_bytes = b""
                if msg.get("bytes"):
                    pcm_bytes = msg["bytes"]
                elif msg.get("text"):
                    data = json.loads(msg["text"])
                    if data.get("reset"):
                        ring = np.zeros(0, dtype=np.float32)
                        since_last_eval = 0
                        await websocket.send_json({"type": "reset", "buffered_sec": 0.0})
                        continue
                    if "lang" in data:
                        lang = str(data["lang"])
                    if "sample_rate" in data:
                        sr = int(data["sample_rate"])
                    if data.get("pcm16_b64"):
                        pcm_bytes = base64.b64decode(data["pcm16_b64"])

                if len(pcm_bytes) < 2:
                    continue

                even_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
                chunk = np.frombuffer(pcm_bytes[:even_len], dtype="<i2").astype(np.float32) / 32768.0
                if sr != cfg.sample_rate and chunk.size > 0:
                    # Lightweight linear resample for real-time chunks
                    ratio = cfg.sample_rate / float(sr)
                    new_len = max(1, int(round(chunk.size * ratio)))
                    chunk = np.interp(
                        np.linspace(0, chunk.size - 1, new_len),
                        np.arange(chunk.size),
                        chunk,
                    ).astype(np.float32)

                ring = np.concatenate([ring, chunk])
                if ring.size > window_samples:
                    ring = ring[-window_samples:]
                since_last_eval += chunk.size

                buffered_sec = round(ring.size / float(cfg.sample_rate), 2)
                if ring.size < min_eval_samples or since_last_eval < hop_eval_samples:
                    await websocket.send_json({"type": "buffering", "buffered_sec": buffered_sec})
                    continue

                since_last_eval = 0
                try:
                    verdict = await run_in_threadpool(
                        detector.analyze_waveform,
                        ring.copy(),
                        cfg.sample_rate,
                        "pcm16",
                        lang,
                    )
                    record_id, ts = history.record(verdict, source="ws:microphone")
                    resp = _to_response(verdict, record_id=record_id, timestamp=ts, source="ws:microphone")
                    await websocket.send_json({"type": "verdict", **resp.model_dump()})
                except AudioLoadError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
        except WebSocketDisconnect:
            return

    return application


app = create_app()


def main() -> None:
    import uvicorn

    cfg = Settings.from_env()
    uvicorn.run("qalqan.api.app:app", host=cfg.host, port=cfg.port, reload=False)


if __name__ == "__main__":
    main()
