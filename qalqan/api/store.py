"""Thread-safe in-memory ring buffer storing recent verification records."""

from __future__ import annotations

import threading
import uuid
from collections import deque
from datetime import datetime, timezone

from qalqan.api.schemas import HistoryItem
from qalqan.ml.detector import DetectionVerdict


class HistoryStore:
    """Bounded thread-safe log of recent voice verification verdicts."""

    def __init__(self, maxlen: int = 50):
        self._items: deque[HistoryItem] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, verdict: DetectionVerdict, source: str) -> tuple[str, str]:
        record_id = uuid.uuid4().hex[:10]
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        item = HistoryItem(
            id=record_id,
            timestamp=timestamp,
            source=source,
            label=verdict.label,  # type: ignore[arg-type]
            probability=verdict.probability,
            latency_ms=verdict.latency_ms,
            duration_sec=verdict.duration_sec,
            engine_mode=verdict.engine_mode,
        )
        with self._lock:
            self._items.appendleft(item)
        return record_id, timestamp

    def list_items(self, limit: int = 20) -> list[HistoryItem]:
        with self._lock:
            return list(self._items)[: max(1, limit)]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
