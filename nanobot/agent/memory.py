"""Memory backends and consolidation policy for long-term agent memory."""

from __future__ import annotations

import asyncio
import json
import weakref
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.agent.tools.mcp import connect_mcp_servers
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import MCPServerConfig, MemoryConfig
from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


_MEMORY_TYPES = ("decision", "requirement", "pattern", "spec", "preference", "general")

_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]

_ARCHIVE_SUMMARY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_archive_summary",
            "description": "Return a concise archival summary for a compacted conversation chunk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Concise summary of the archived chunk, suitable for later retrieval.",
                    },
                },
                "required": ["summary"],
            },
        },
    }
]

_EXTRACT_DURABLE_MEMORIES_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_durable_memories",
            "description": "Extract durable memories from a conversation turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "description": "Durable memories worth writing to long-term storage.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Standalone memory text to store.",
                                },
                                "topic": {
                                    "type": "string",
                                    "description": "Short subsystem label or topic, such as auth or ui-settings.",
                                },
                                "memory_type": {
                                    "type": "string",
                                    "enum": list(_MEMORY_TYPES),
                                    "description": "Durable memory category.",
                                },
                            },
                            "required": ["content", "topic", "memory_type"],
                        },
                    }
                },
                "required": ["memories"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize values to text for prompt use and storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_tool_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None


def _content_to_text(content: Any) -> str:
    """Convert message content blocks into a readable text form."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in ("text", "input_text", "output_text") and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item_type == "image_url":
                    parts.append("[image]")
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _format_messages(messages: list[dict[str, Any]]) -> str:
    """Format messages for extraction prompts."""
    lines = []
    for message in messages:
        text = _content_to_text(message.get("content")).strip()
        if not text:
            continue
        tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
        lines.append(
            f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {text}"
        )
    return "\n".join(lines)


def _format_recall_item(item: dict[str, Any]) -> str:
    """Render one fetched or summary-only recall item for the LLM."""
    meta = item.get("metadata") or {}
    lines = []
    if item.get("id"):
        lines.append(f"id={item['id']}")
    if isinstance(meta.get("module_name"), str) and meta["module_name"]:
        lines.append(f"module_name={meta['module_name']}")
    if isinstance(meta.get("memory_type"), str) and meta["memory_type"]:
        lines.append(f"memory_type={meta['memory_type']}")
    if isinstance(meta.get("branch_state"), str) and meta["branch_state"]:
        lines.append(f"branch_state={meta['branch_state']}")
    if isinstance(meta.get("modality"), str) and meta["modality"]:
        lines.append(f"modality={meta['modality']}")
    if meta.get("artifact_refs"):
        lines.append(f"artifact_refs={', '.join(str(ref) for ref in meta['artifact_refs'])}")
    if isinstance(item.get("summary"), str) and item["summary"]:
        lines.append(f"summary={item['summary']}")
    if isinstance(item.get("content"), str) and item["content"]:
        lines.append(f"content={item['content']}")
    elif isinstance(item.get("content_preview"), str) and item["content_preview"]:
        lines.append(f"content_preview={item['content_preview']}")
    return "\n".join(lines)


def _recent_user_messages(history: list[dict[str, Any]], current_message: str, query_turns: int) -> list[str]:
    """Collect recent user messages for recall query construction."""
    prior = [
        _content_to_text(message.get("content")).strip()
        for message in history
        if message.get("role") == "user" and _content_to_text(message.get("content")).strip()
    ]
    recent = prior[-query_turns:] if query_turns > 0 else []
    current = current_message.strip()
    return [current, *recent] if current else recent


def _normalize_memory_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    memory_type = value.strip().lower()
    return memory_type if memory_type in _MEMORY_TYPES else None


def _normalize_durable_memories_payload(args: Any, max_items: int) -> list[dict[str, str]]:
    payload = _normalize_tool_args(args)
    if payload is None:
        return []
    memories = payload.get("memories")
    if not isinstance(memories, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in memories:
        if not isinstance(item, dict):
            continue
        content = _ensure_text(item.get("content", "")).strip()
        topic = _ensure_text(item.get("topic", "")).strip()
        memory_type = _normalize_memory_type(item.get("memory_type")) or "general"
        if not content or not topic:
            continue
        normalized.append({
            "content": content,
            "topic": topic,
            "memory_type": memory_type,
        })
        if len(normalized) >= max_items:
            break
    return normalized


_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self._consecutive_failures = 0

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def consolidate(
        self,
        messages: list[dict[str, Any]],
        provider: LLMProvider,
        model: str,
    ) -> bool:
        """Consolidate the provided message chunk into MEMORY.md + HISTORY.md."""
        if not messages:
            return True

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{_format_messages(messages)}"""

        chat_messages = [
            {
                "role": "system",
                "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(response.content):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages)

            args = _normalize_tool_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return self._fail_or_raw_archive(messages)

            if "history_entry" not in args or "memory_update" not in args:
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                return self._fail_or_raw_archive(messages)

            entry = args["history_entry"]
            update = args["memory_update"]

            if entry is None or update is None:
                logger.warning("Memory consolidation: save_memory payload contains null required fields")
                return self._fail_or_raw_archive(messages)

            entry = _ensure_text(entry).strip()
            if not entry:
                logger.warning("Memory consolidation: history_entry is empty after normalization")
                return self._fail_or_raw_archive(messages)

            self.append_history(entry)
            update = _ensure_text(update)
            if update != current_memory:
                self.write_long_term(update)

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages)

    def _fail_or_raw_archive(self, messages: list[dict[str, Any]]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict[str, Any]]) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_history(
            f"[{ts}] [RAW] {len(messages)} messages\n"
            f"{_format_messages(messages)}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )


class MemoryBackend(ABC):
    """Abstract long-term memory backend."""

    mode: str = "file"

    def get_memory_context(self) -> str:
        return ""

    def get_identity_memory_lines(self, workspace_path: str) -> list[str]:
        return []

    def should_include_skill(self, name: str) -> bool:
        return True

    @abstractmethod
    async def build_recall_message(
        self,
        history: list[dict[str, Any]],
        current_message: str,
    ) -> dict[str, Any] | None:
        """Build a transient recall message for the current turn."""

    @abstractmethod
    async def store_durable_memories(
        self,
        turn_messages: list[dict[str, Any]],
        final_content: str | None,
    ) -> None:
        """Persist durable memories extracted from the current turn."""

    @abstractmethod
    async def archive_compacted_chunk(self, session_key: str, messages: list[dict[str, Any]]) -> bool:
        """Archive a chunk selected during token compaction."""

    @abstractmethod
    async def archive_session_tail(self, session_key: str, messages: list[dict[str, Any]]) -> bool:
        """Archive the full unconsolidated session tail for /new rollover."""

    async def close(self) -> None:
        """Close any backend resources."""


class FileMemoryBackend(MemoryBackend):
    """Adapter that preserves the current file-based memory behavior."""

    mode = "file"

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ):
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model

    def get_memory_context(self) -> str:
        return self.store.get_memory_context()

    def get_identity_memory_lines(self, workspace_path: str) -> list[str]:
        return [
            f"- Long-term memory: {workspace_path}/memory/MEMORY.md (write important facts here)",
            f"- History log: {workspace_path}/memory/HISTORY.md (grep-searchable). Each entry starts with [YYYY-MM-DD HH:MM].",
        ]

    async def build_recall_message(
        self,
        history: list[dict[str, Any]],
        current_message: str,
    ) -> dict[str, Any] | None:
        return None

    async def store_durable_memories(
        self,
        turn_messages: list[dict[str, Any]],
        final_content: str | None,
    ) -> None:
        return None

    async def archive_compacted_chunk(self, session_key: str, messages: list[dict[str, Any]]) -> bool:
        if not self.provider or not self.model:
            logger.warning("File memory backend missing provider/model for archive")
            return False
        return await self.store.consolidate(messages, self.provider, self.model)

    async def archive_session_tail(self, session_key: str, messages: list[dict[str, Any]]) -> bool:
        return await self.archive_compacted_chunk(session_key, messages)


class McpMemoryBackend(MemoryBackend):
    """Internal MCP-backed long-term memory integration."""

    mode = "mcp"
    _SERVER_NAME = "memory"

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        config: MemoryConfig,
    ):
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.config = config.mcp
        self._stack: AsyncExitStack | None = None
        self._registry: ToolRegistry | None = None
        self._connect_lock = asyncio.Lock()

    def get_identity_memory_lines(self, workspace_path: str) -> list[str]:
        return ["- Long-term memory: managed by the configured internal MCP memory backend."]

    def should_include_skill(self, name: str) -> bool:
        return name != "memory"

    def _tool_name(self, raw_name: str) -> str:
        return f"mcp_{self._SERVER_NAME}_{raw_name}"

    def _server_config(self) -> MCPServerConfig:
        return MCPServerConfig(
            type=self.config.type,
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
            url=self.config.url,
            headers=self.config.headers,
            tool_timeout=self.config.tool_timeout,
            enabled_tools=["search_context", "fetch_context", "remember_context"],
        )

    async def _ensure_connected(self) -> bool:
        async with self._connect_lock:
            if self._registry and all(
                self._registry.has(self._tool_name(name))
                for name in ("search_context", "fetch_context", "remember_context")
            ):
                return True

            if self._stack:
                await self.close()

            if not (self.config.command or self.config.url):
                logger.error("MCP memory backend is enabled but no command or url is configured")
                return False

            stack = AsyncExitStack()
            await stack.__aenter__()
            registry = ToolRegistry()
            try:
                await connect_mcp_servers(
                    {self._SERVER_NAME: self._server_config()},
                    registry,
                    stack,
                )
                missing = [
                    name for name in ("search_context", "fetch_context", "remember_context")
                    if not registry.has(self._tool_name(name))
                ]
                if missing:
                    logger.error("MCP memory backend missing required tools: {}", ", ".join(missing))
                    await stack.aclose()
                    return False
            except Exception as exc:
                logger.error("Failed to connect MCP memory backend: {}", exc)
                await stack.aclose()
                return False

            self._stack = stack
            self._registry = registry
            return True

    async def _call_json_tool(self, raw_name: str, params: dict[str, Any]) -> dict[str, Any]:
        if not await self._ensure_connected():
            raise RuntimeError("MCP memory backend unavailable")

        from mcp import types

        assert self._registry is not None
        tool = self._registry.get(self._tool_name(raw_name))
        if tool is None:
            raise RuntimeError(f"Missing MCP memory tool '{raw_name}'")

        session = getattr(tool, "_session", None)
        original_name = getattr(tool, "_original_name", None)
        if session is None or not isinstance(original_name, str):
            raise RuntimeError(f"MCP memory tool '{raw_name}' is missing its raw session binding")

        try:
            result = await asyncio.wait_for(
                session.call_tool(original_name, arguments=params),
                timeout=self.config.tool_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"MCP memory tool '{raw_name}' timed out after {self.config.tool_timeout}s"
            ) from exc
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            raise RuntimeError(f"MCP memory tool '{raw_name}' was cancelled")

        blocks = getattr(result, "content", None)
        if not isinstance(blocks, list) or len(blocks) != 1 or not isinstance(blocks[0], types.TextContent):
            raise ValueError(
                f"MCP memory tool '{raw_name}' must return exactly one TextContent block"
            )

        payload = json.loads(blocks[0].text)
        if not isinstance(payload, dict):
            raise ValueError(f"MCP memory tool '{raw_name}' returned non-object JSON")
        return payload

    async def _call_structured_extractor(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        tool_name: str,
    ) -> dict[str, Any] | None:
        messages = [
            {
                "role": "system",
                "content": f"You are a memory extraction agent. Call the {tool_name} tool exactly once.",
            },
            {"role": "user", "content": prompt},
        ]

        forced = {"type": "function", "function": {"name": tool_name}}
        response = await self.provider.chat_with_retry(
            messages=messages,
            tools=tools,
            model=self.model,
            tool_choice=forced,
        )
        if response.finish_reason == "error" and _is_tool_choice_unsupported(response.content):
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tools,
                model=self.model,
                tool_choice="auto",
            )

        if not response.has_tool_calls:
            logger.warning(
                "Memory backend extractor '{}' did not call tool (finish_reason={})",
                tool_name,
                response.finish_reason,
            )
            return None
        return _normalize_tool_args(response.tool_calls[0].arguments)

    async def _summarize_archive(self, session_key: str, messages: list[dict[str, Any]]) -> str | None:
        prompt = f"""Summarize this archived conversation chunk for later retrieval.

Requirements:
- Preserve stable facts, decisions, requirements, and active work context.
- Do not include routine tool logs unless they change meaning.
- Keep it concise and self-contained.
- This summary will be stored as a draft archive under topic session:{session_key}.

## Conversation Chunk
{_format_messages(messages)}"""
        payload = await self._call_structured_extractor(
            prompt,
            _ARCHIVE_SUMMARY_TOOL,
            "save_archive_summary",
        )
        if payload is None:
            return None
        summary = _ensure_text(payload.get("summary", "")).strip()
        return summary or None

    async def _extract_durable_memories(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        prompt = f"""Extract at most {self.config.max_durable_writes_per_turn} durable memories from this turn.

Include only:
- accepted decisions
- stable requirements
- durable preferences
- canonical specs
- reusable patterns

Exclude:
- transient chat
- tool logs
- open questions
- speculative discussion
- routine progress notes

Use short subsystem topics like auth, billing, or ui-settings. Use memory_type from:
{", ".join(_MEMORY_TYPES)}

## Conversation Turn
{_format_messages(messages)}"""
        payload = await self._call_structured_extractor(
            prompt,
            _EXTRACT_DURABLE_MEMORIES_TOOL,
            "save_durable_memories",
        )
        if payload is None:
            return []
        return _normalize_durable_memories_payload(
            payload,
            self.config.max_durable_writes_per_turn,
        )

    async def build_recall_message(
        self,
        history: list[dict[str, Any]],
        current_message: str,
    ) -> dict[str, Any] | None:
        query_parts = _recent_user_messages(history, current_message, self.config.query_turns)
        if not query_parts:
            return None

        try:
            search = await self._call_json_tool(
                "search_context",
                {
                    "query": "\n".join(query_parts),
                    "filter_state": "active",
                    "limit": self.config.max_results,
                },
            )
        except Exception as exc:
            logger.warning("MCP memory recall search failed: {}", exc)
            return None

        matches = search.get("matches")
        if not isinstance(matches, list) or not matches:
            return None

        fetched_items: dict[str, dict[str, Any]] = {}
        for match in matches[: self.config.fetch_top_k]:
            if not isinstance(match, dict) or not isinstance(match.get("id"), str):
                continue
            try:
                fetched = await self._call_json_tool(
                    "fetch_context",
                    {"document_id": match["id"]},
                )
            except Exception as exc:
                logger.warning("MCP memory fetch failed for {}: {}", match.get("id"), exc)
                continue
            item = fetched.get("item")
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                fetched_items[item["id"]] = item

        lines = ["[Memory Recall]", "Use this only if it is relevant to the current request."]
        for match in matches:
            if not isinstance(match, dict):
                continue
            item = fetched_items.get(match.get("id")) or match
            rendered = _format_recall_item(item)
            if rendered:
                lines.append(rendered)

        if len(lines) <= 2:
            return None
        return {"role": "assistant", "content": "\n\n".join(lines)}

    async def store_durable_memories(
        self,
        turn_messages: list[dict[str, Any]],
        final_content: str | None,
    ) -> None:
        if not turn_messages:
            return
        try:
            memories = await self._extract_durable_memories(turn_messages)
        except Exception as exc:
            logger.warning("Durable memory extraction failed: {}", exc)
            return

        for memory in memories:
            try:
                await self._call_json_tool(
                    "remember_context",
                    {
                        "content": memory["content"],
                        "topic": memory["topic"],
                        "memory_type": memory["memory_type"],
                        "draft": False,
                    },
                )
            except Exception as exc:
                logger.warning("Durable memory write failed: {}", exc)

    async def archive_compacted_chunk(self, session_key: str, messages: list[dict[str, Any]]) -> bool:
        if not messages:
            return True
        try:
            summary = await self._summarize_archive(session_key, messages)
        except Exception as exc:
            logger.warning("Archive summary extraction failed: {}", exc)
            return False
        if not summary:
            return False

        try:
            await self._call_json_tool(
                "remember_context",
                {
                    "content": summary,
                    "topic": f"session:{session_key}",
                    "memory_type": "general",
                    "draft": True,
                },
            )
            return True
        except Exception as exc:
            logger.warning("Draft archive write failed: {}", exc)
            return False

    async def archive_session_tail(self, session_key: str, messages: list[dict[str, Any]]) -> bool:
        if not await self.archive_compacted_chunk(session_key, messages):
            return False
        await self.store_durable_memories(messages, None)
        return True

    async def close(self) -> None:
        if self._stack:
            try:
                await self._stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass
            self._stack = None
        self._registry = None


def create_memory_backend(
    workspace: Path,
    provider: LLMProvider | None,
    model: str | None,
    config: MemoryConfig | None = None,
) -> MemoryBackend:
    """Build the configured memory backend."""
    if config and config.backend == "mcp":
        if provider is None or model is None:
            raise ValueError("MCP memory backend requires provider and model")
        return McpMemoryBackend(workspace, provider, model, config)
    return FileMemoryBackend(workspace, provider, model)


class MemoryConsolidator:
    """Owns token-based compaction policy and delegates persistence to a backend."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    def __init__(
        self,
        memory_backend: MemoryBackend,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
    ):
        self.memory_backend = memory_backend
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(
        self,
        messages: list[dict[str, object]],
        session_key: str | None = None,
    ) -> bool:
        """Archive a selected message chunk into persistent memory."""
        return await self.memory_backend.archive_compacted_chunk(session_key or "", list(messages))

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_unconsolidated(self, session: Session) -> bool:
        """Archive the full unconsolidated tail for /new-style session rollover."""
        lock = self.get_lock(session.key)
        async with lock:
            snapshot = session.messages[session.last_consolidated:]
            if not snapshot:
                return True
            return await self.memory_backend.archive_session_tail(session.key, snapshot)

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within half the context window."""
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            target = self.context_window_tokens // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < self.context_window_tokens:
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.consolidate_messages(chunk, session_key=session.key):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
