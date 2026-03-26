"""Tool for delegating coding tasks to the dedicated worker."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class CodingAgentTool(Tool):
    """Delegate code work to the dedicated coding worker."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @property
    def name(self) -> str:
        return "coding_agent"

    @property
    def description(self) -> str:
        return (
            "Delegate a code task to the dedicated coding worker. "
            "Use this for tasks that require inspecting workspace files, editing code, "
            "and optionally running local verification commands. "
            "The worker returns a concise summary of changes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The coding task to complete in the current workspace.",
                },
                "route": {
                    "type": "string",
                    "description": (
                        "Optional named agent route to prefer for this task, "
                        "for example 'coder' or 'codex'."
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, route: str | None = None, **kwargs: Any) -> str:
        """Run the coding worker and return its summary."""
        return await self._manager.run_coding_task(task=task, route=route)
