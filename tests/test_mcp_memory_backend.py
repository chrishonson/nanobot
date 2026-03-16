from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import McpMemoryBackend
from nanobot.config.schema import MemoryConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.utils.helpers import sync_workspace_templates


class _FakeTextContent:
    def __init__(self, text: str) -> None:
        self.text = text


@pytest.fixture
def fake_mcp_runtime() -> dict[str, object | None]:
    return {"session": None}


@pytest.fixture(autouse=True)
def _fake_mcp_module(
    monkeypatch: pytest.MonkeyPatch, fake_mcp_runtime: dict[str, object | None]
) -> None:
    mod = ModuleType("mcp")
    mod.types = SimpleNamespace(TextContent=_FakeTextContent)

    class _FakeStdioServerParameters:
        def __init__(self, command: str, args: list[str], env: dict | None = None) -> None:
            self.command = command
            self.args = args
            self.env = env

    class _FakeClientSession:
        def __init__(self, _read: object, _write: object) -> None:
            self._session = fake_mcp_runtime["session"]

        async def __aenter__(self) -> object:
            return self._session

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    @asynccontextmanager
    async def _fake_stdio_client(_params: object):
        yield object(), object()

    @asynccontextmanager
    async def _fake_sse_client(_url: str, httpx_client_factory=None):
        yield object(), object()

    @asynccontextmanager
    async def _fake_streamable_http_client(_url: str, http_client=None):
        yield object(), object(), object()

    mod.ClientSession = _FakeClientSession
    mod.StdioServerParameters = _FakeStdioServerParameters
    monkeypatch.setitem(sys.modules, "mcp", mod)

    client_mod = ModuleType("mcp.client")
    stdio_mod = ModuleType("mcp.client.stdio")
    stdio_mod.stdio_client = _fake_stdio_client
    sse_mod = ModuleType("mcp.client.sse")
    sse_mod.sse_client = _fake_sse_client
    streamable_http_mod = ModuleType("mcp.client.streamable_http")
    streamable_http_mod.streamable_http_client = _fake_streamable_http_client

    monkeypatch.setitem(sys.modules, "mcp.client", client_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.sse", sse_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", streamable_http_mod)


def _tool_def(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"{name} tool",
        inputSchema={"type": "object", "properties": {}},
    )


def _make_session(tool_names: list[str], handler) -> SimpleNamespace:
    async def initialize() -> None:
        return None

    async def list_tools() -> SimpleNamespace:
        return SimpleNamespace(tools=[_tool_def(name) for name in tool_names])

    async def call_tool(name: str, arguments: dict) -> SimpleNamespace:
        return await handler(name, arguments)

    return SimpleNamespace(
        initialize=initialize,
        list_tools=list_tools,
        call_tool=call_tool,
    )


def _memory_config() -> MemoryConfig:
    cfg = MemoryConfig(backend="mcp")
    cfg.mcp.command = "fake"
    return cfg


def test_sync_workspace_templates_skips_memory_files(tmp_path: Path) -> None:
    sync_workspace_templates(tmp_path, include_memory_files=False)

    assert not (tmp_path / "memory" / "MEMORY.md").exists()
    assert not (tmp_path / "memory" / "HISTORY.md").exists()


def test_context_builder_mcp_mode_suppresses_file_memory_prompt(tmp_path: Path) -> None:
    class _DummyMemory:
        def get_memory_context(self) -> str:
            return ""

        def get_identity_memory_lines(self, _workspace_path: str) -> list[str]:
            return ["- Long-term memory: managed by the configured internal MCP memory backend."]

        def should_include_skill(self, name: str) -> bool:
            return name != "memory"

    builder = ContextBuilder(tmp_path, memory_backend=_DummyMemory())
    prompt = builder.build_system_prompt()

    assert "MEMORY.md" not in prompt
    assert "HISTORY.md" not in prompt
    assert "Skill: memory" not in prompt


def test_mcp_memory_backend_builds_recall_message(fake_mcp_runtime, tmp_path: Path) -> None:
    async def handler(name: str, arguments: dict) -> SimpleNamespace:
        if name == "search_context":
            assert arguments["filter_state"] == "active"
            return SimpleNamespace(content=[_FakeTextContent(json.dumps({
                "matches": [
                    {
                        "id": "mem-1",
                        "summary": "API requires OAuth2.",
                        "metadata": {
                            "branch_state": "active",
                            "module_name": "auth",
                            "memory_type": "requirement",
                            "modality": "text",
                            "created_at": "2026-01-01T00:00:00Z",
                            "updated_at": "2026-01-01T00:00:00Z",
                        },
                    }
                ],
                "applied_filters": {"filter_module": None, "filter_state": "active"},
            }))])
        if name == "fetch_context":
            return SimpleNamespace(content=[_FakeTextContent(json.dumps({
                "item": {
                    "id": "mem-1",
                    "content": "The API uses OAuth2 for all access tokens.",
                    "retrieval_text": "OAuth2 access tokens",
                    "metadata": {
                        "branch_state": "active",
                        "module_name": "auth",
                        "memory_type": "requirement",
                        "modality": "text",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:00:00Z",
                    },
                }
            }))])
        raise AssertionError(name)

    fake_mcp_runtime["session"] = _make_session(
        ["search_context", "fetch_context", "remember_context"],
        handler,
    )
    provider = SimpleNamespace(chat_with_retry=AsyncMock())
    backend = McpMemoryBackend(tmp_path, provider, "test-model", _memory_config())

    recall = asyncio.run(backend.build_recall_message(
        history=[{"role": "user", "content": "How does auth work?"}],
        current_message="Remind me what auth requires",
    ))
    asyncio.run(backend.close())

    assert recall is not None
    assert recall["role"] == "assistant"
    assert "[Memory Recall]" in recall["content"]
    assert "OAuth2" in recall["content"]
    assert "module_name=auth" in recall["content"]


def test_mcp_memory_backend_archives_with_draft_true(fake_mcp_runtime, tmp_path: Path) -> None:
    remembered: list[dict] = []

    async def handler(name: str, arguments: dict) -> SimpleNamespace:
        if name == "remember_context":
            remembered.append(arguments)
            return SimpleNamespace(content=[_FakeTextContent(json.dumps({
                "item": {
                    "id": "draft-1",
                    "content": arguments["content"],
                    "metadata": {
                        "branch_state": "wip",
                        "module_name": arguments["topic"],
                        "memory_type": arguments["memory_type"],
                        "modality": "text",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:00:00Z",
                    },
                },
                "write_status": "created",
            }))])
        raise AssertionError(name)

    fake_mcp_runtime["session"] = _make_session(
        ["search_context", "fetch_context", "remember_context"],
        handler,
    )
    provider = SimpleNamespace(
        chat_with_retry=AsyncMock(return_value=LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(
                id="call_1",
                name="save_archive_summary",
                arguments={"summary": "Archived summary"},
            )],
        )),
    )
    backend = McpMemoryBackend(tmp_path, provider, "test-model", _memory_config())

    ok = asyncio.run(backend.archive_compacted_chunk(
        "cli:direct",
        [{"role": "user", "content": "auth uses oauth2", "timestamp": "2026-01-01T00:00:00"}],
    ))
    asyncio.run(backend.close())

    assert ok is True
    assert remembered[0]["draft"] is True
    assert remembered[0]["topic"] == "session:cli:direct"


def test_mcp_memory_backend_missing_required_tools_degrades_cleanly(
    fake_mcp_runtime,
    tmp_path: Path,
) -> None:
    async def handler(name: str, arguments: dict) -> SimpleNamespace:
        raise AssertionError((name, arguments))

    fake_mcp_runtime["session"] = _make_session(["search_context"], handler)
    provider = SimpleNamespace(chat_with_retry=AsyncMock())
    backend = McpMemoryBackend(tmp_path, provider, "test-model", _memory_config())

    recall = asyncio.run(backend.build_recall_message(
        history=[{"role": "user", "content": "How does auth work?"}],
        current_message="Remind me what auth requires",
    ))
    ok = asyncio.run(backend.archive_compacted_chunk(
        "cli:direct",
        [{"role": "user", "content": "auth uses oauth2", "timestamp": "2026-01-01T00:00:00"}],
    ))
    asyncio.run(backend.close())

    assert recall is None
    assert ok is False


def test_mcp_memory_backend_treats_duplicate_write_as_success(fake_mcp_runtime, tmp_path: Path) -> None:
    remembered: list[dict] = []

    async def handler(name: str, arguments: dict) -> SimpleNamespace:
        if name == "remember_context":
            remembered.append(arguments)
            return SimpleNamespace(content=[_FakeTextContent(json.dumps({
                "item": {
                    "id": "dup-1",
                    "content": arguments["content"],
                    "metadata": {
                        "branch_state": "active",
                        "module_name": arguments["topic"],
                        "memory_type": arguments["memory_type"],
                        "modality": "text",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:00:00Z",
                    },
                },
                "write_status": "duplicate",
            }))])
        raise AssertionError(name)

    fake_mcp_runtime["session"] = _make_session(
        ["search_context", "fetch_context", "remember_context"],
        handler,
    )
    provider = SimpleNamespace(
        chat_with_retry=AsyncMock(return_value=LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(
                id="call_1",
                name="save_durable_memories",
                arguments={
                    "memories": [
                        {
                            "content": "The API requires OAuth2.",
                            "topic": "auth",
                            "memory_type": "requirement",
                        }
                    ]
                },
            )],
        )),
    )
    backend = McpMemoryBackend(tmp_path, provider, "test-model", _memory_config())

    asyncio.run(backend.store_durable_memories(
        [{"role": "user", "content": "remember auth requirements"}],
        "ok",
    ))
    asyncio.run(backend.close())

    assert len(remembered) == 1
    assert remembered[0]["draft"] is False
    assert remembered[0]["memory_type"] == "requirement"


def test_agent_loop_keeps_memory_mcp_tools_out_of_public_registry(
    fake_mcp_runtime,
    tmp_path: Path,
) -> None:
    async def handler(name: str, arguments: dict) -> SimpleNamespace:
        if name == "search_context":
            return SimpleNamespace(content=[_FakeTextContent(json.dumps({
                "matches": [],
                "applied_filters": {"filter_module": None, "filter_state": "active"},
            }))])
        raise AssertionError((name, arguments))

    fake_mcp_runtime["session"] = _make_session(
        ["search_context", "fetch_context", "remember_context"],
        handler,
    )
    provider = SimpleNamespace(
        chat_with_retry=AsyncMock(),
        get_default_model=lambda: "test-model",
    )
    loop = AgentLoop(
        bus=SimpleNamespace(publish_outbound=AsyncMock()),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        memory_config=_memory_config(),
    )

    history = asyncio.run(loop._prepare_history_with_recall([], "What do we know about auth?"))
    asyncio.run(loop.close_mcp())

    assert history == []
    assert not any(name.startswith("mcp_memory_") for name in loop.tools.tool_names)
