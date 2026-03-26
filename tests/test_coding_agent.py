from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.config.schema import AgentsConfig


@pytest.mark.asyncio
async def test_run_coding_task_prefers_named_coder_route_and_edits_file(tmp_path) -> None:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.base import LLMResponse, ToolCallRequest

    workspace_file = tmp_path / "app.txt"
    workspace_file.write_text("old value\n", encoding="utf-8")

    bus = MessageBus()
    default_provider = MagicMock()
    default_provider.get_default_model.return_value = "ollama/qwen3.5:32b"

    coding_provider = MagicMock()
    captured_calls: list[dict] = []

    async def scripted_chat_with_retry(*, messages, tools, model, **kwargs):
        captured_calls.append(
            {
                "messages": messages,
                "model": model,
                "tool_names": [tool["function"]["name"] for tool in tools],
            }
        )
        if len(captured_calls) == 1:
            return LLMResponse(
                content="Inspecting and patching the workspace file.",
                tool_calls=[
                    ToolCallRequest(
                        id="call_read",
                        name="read_file",
                        arguments={"path": "app.txt"},
                    ),
                    ToolCallRequest(
                        id="call_edit",
                        name="edit_file",
                        arguments={
                            "path": "app.txt",
                            "old_text": "old value",
                            "new_text": "new value",
                        },
                    ),
                ],
            )
        return LLMResponse(
            content="Updated app.txt, changing `old value` to `new value`. Verification was not run.",
            tool_calls=[],
        )

    coding_provider.chat_with_retry = scripted_chat_with_retry

    provider_pool = MagicMock()
    provider_pool.get.return_value = coding_provider
    agents_config = AgentsConfig(
        coder={
            "provider": "openai_codex",
            "model": "openai-codex/gpt-5.1-codex",
            "aliases": ["codex"],
        }
    )

    mgr = SubagentManager(
        provider=default_provider,
        workspace=tmp_path,
        bus=bus,
        agents_config=agents_config,
        provider_pool=provider_pool,
    )

    result = await mgr.run_coding_task("Update the sample file.")

    assert workspace_file.read_text(encoding="utf-8") == "new value\n"
    assert "Updated app.txt" in result
    provider_pool.get.assert_called_once_with("openai_codex", "openai-codex/gpt-5.1-codex")
    assert captured_calls[0]["model"] == "openai-codex/gpt-5.1-codex"
    assert "read_file" in captured_calls[0]["tool_names"]
    assert "edit_file" in captured_calls[0]["tool_names"]
    assert "web_search" not in captured_calls[0]["tool_names"]


@pytest.mark.asyncio
async def test_coding_agent_tool_delegates_to_manager() -> None:
    from nanobot.agent.tools.coding import CodingAgentTool

    manager = MagicMock()
    manager.run_coding_task = AsyncMock(return_value="summary")
    tool = CodingAgentTool(manager=manager)

    result = await tool.execute(task="Fix the failing test.", route="coder")

    assert result == "summary"
    manager.run_coding_task.assert_awaited_once_with(
        task="Fix the failing test.",
        route="coder",
    )


def test_agent_loop_registers_coding_agent_tool(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(max_tokens=4096)

    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
    )

    assert loop.tools.get("coding_agent") is not None


def test_agent_loop_allowed_paths_enable_repo_writes(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(max_tokens=4096)

    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=workspace,
        restrict_to_workspace=True,
        allowed_paths=[repo],
    )

    result = asyncio.run(loop.tools.execute("write_file", {
        "path": str(repo / "note.txt"),
        "content": "hello",
    }))

    assert "Successfully wrote" in result
    assert (repo / "note.txt").read_text(encoding="utf-8") == "hello"
