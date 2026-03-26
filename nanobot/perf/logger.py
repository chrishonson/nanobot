"""Append-only JSONL performance logger."""

from __future__ import annotations

import json
import time
from pathlib import Path


class PerfLogger:
    """Writes one JSON object per line to a perf log file."""

    def __init__(self, path: Path, max_bytes: int = 10 * 1024 * 1024) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict) -> None:
        record["ts"] = record.get("ts") or time.time()
        if self._path.exists() and self._path.stat().st_size >= self._max_bytes:
            self._path.rename(self._path.with_suffix(f".{int(time.time())}.jsonl"))
        with self._path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_llm_call(
        self,
        *,
        model: str = "",
        provider: str = "",
        latency_ms: float = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        finish_reason: str = "",
        streaming: bool = False,
    ) -> None:
        self._write(
            {
                "event": "llm_call",
                "model": model,
                "provider": provider,
                "latency_ms": round(latency_ms),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "finish_reason": finish_reason,
                "streaming": streaming,
            }
        )
