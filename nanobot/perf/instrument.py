"""Monkey-patch a provider instance to log perf data on every LLM call."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

from nanobot.perf.logger import PerfLogger


def instrument_provider(provider: LLMProvider, perf_path: Path | None = None) -> None:
    """Wrap provider._safe_chat and _safe_chat_stream with timing.

    This is non-invasive: it patches the instance only, not the class.
    If perf_path is None, uses the default logs dir.
    """
    if perf_path is None:
        from nanobot.config.paths import get_logs_dir

        perf_path = get_logs_dir() / "perf.jsonl"

    perf = PerfLogger(perf_path)
    provider_name = type(provider).__name__

    original_chat = provider._safe_chat
    original_stream = provider._safe_chat_stream

    async def _timed_chat(**kwargs):
        t0 = time.monotonic()
        finish = "error"
        usage = {}
        try:
            response = await original_chat(**kwargs)
            finish = response.finish_reason
            usage = response.usage or {}
            return response
        finally:
            try:
                perf.log_llm_call(
                    model=kwargs.get("model") or "",
                    provider=provider_name,
                    latency_ms=(time.monotonic() - t0) * 1000,
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    finish_reason=finish,
                    streaming=False,
                )
            except Exception:
                pass

    async def _timed_stream(**kwargs):
        t0 = time.monotonic()
        finish = "error"
        usage = {}
        try:
            response = await original_stream(**kwargs)
            finish = response.finish_reason
            usage = response.usage or {}
            return response
        finally:
            try:
                perf.log_llm_call(
                    model=kwargs.get("model") or "",
                    provider=provider_name,
                    latency_ms=(time.monotonic() - t0) * 1000,
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    finish_reason=finish,
                    streaming=True,
                )
            except Exception:
                pass

    provider._safe_chat = _timed_chat
    provider._safe_chat_stream = _timed_stream
