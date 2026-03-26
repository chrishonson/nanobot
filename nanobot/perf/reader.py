"""Read and aggregate JSONL perf logs."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path


class PerfReader:
    """Reads and aggregates perf.jsonl records."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def read(self, *, since_seconds: float | None = None, event: str | None = None) -> list[dict]:
        if not self._path.exists():
            return []
        cutoff = (time.time() - since_seconds) if since_seconds else 0
        results = []
        for line in self._path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ts", 0) < cutoff:
                continue
            if event and r.get("event") != event:
                continue
            results.append(r)
        return results

    def summarize_llm(self, since_seconds: float | None = None) -> dict:
        records = self.read(since_seconds=since_seconds, event="llm_call")
        if not records:
            return {
                "total_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "avg_latency_ms": 0,
                "by_model": {},
            }

        by_model: dict[str, list] = defaultdict(list)
        total_lat = total_pt = total_ct = 0
        for r in records:
            lat = r.get("latency_ms", 0)
            pt = r.get("prompt_tokens", 0)
            ct = r.get("completion_tokens", 0)
            total_lat += lat
            total_pt += pt
            total_ct += ct
            by_model[r.get("model", "?")].append(r)

        n = len(records)
        model_summary = {}
        for model, rs in by_model.items():
            model_summary[model] = {
                "calls": len(rs),
                "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rs),
                "completion_tokens": sum(r.get("completion_tokens", 0) for r in rs),
                "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in rs) / len(rs)),
            }

        return {
            "total_calls": n,
            "total_prompt_tokens": total_pt,
            "total_completion_tokens": total_ct,
            "total_tokens": total_pt + total_ct,
            "avg_latency_ms": round(total_lat / n),
            "by_model": model_summary,
        }
