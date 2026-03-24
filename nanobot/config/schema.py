"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ChannelsConfig(Base):
    """Configuration for chat channels.

    Built-in and plugin channel configs are stored as extra fields (dicts).
    Each channel parses its own config in __init__.
    Per-channel "streaming": true enables streaming output (requires send_delta impl).
    """

    model_config = ConfigDict(extra="allow")

    send_progress: bool = True  # stream agent's text progress to the channel
    send_tool_hints: bool = False  # stream tool-call hints (e.g. read_file("…"))


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    provider: str = "anthropic"  # Explicit provider reference (no model-based inference)
    max_tokens: int = 8192
    context_window_tokens: int = 65_536
    temperature: float = 0.1
    max_tool_iterations: int = 40
    reasoning_effort: str | None = None  # low / medium / high - enables LLM thinking mode


class AgentModel(Base):
    """Named agent with a specific model (used for @prefix routing)."""

    model: str
    provider: str
    aliases: list[str] = Field(default_factory=list)


class AgentsConfig(Base):
    """Agent configuration.

    The ``defaults`` key holds base settings.  Additional keys define named
    agents that users can invoke via ``@name`` or ``@alias`` message prefixes
    to route a single request to a different model.
    """

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)
    models: dict[str, AgentModel] = Field(default_factory=dict)

    def __init__(self, **data):
        # Pull any extra keys (haiku, opus, …) out of *data* and park them in
        # ``models`` so the Pydantic schema stays strict while the JSON config
        # stays flat.
        known = {"defaults", "models"}
        extra = {k: v for k, v in data.items() if k not in known and isinstance(v, dict)}
        for k in extra:
            data.pop(k)
        super().__init__(**data)
        for name, cfg in extra.items():
            self.models[name] = AgentModel(**cfg)

    def resolve_agent(self, alias: str) -> AgentModel | None:
        """Return the AgentModel for *alias*, or ``None``."""
        alias = alias.lower()
        if alias in self.models and self.models[alias].model:
            return self.models[alias]
        for agent in self.models.values():
            if alias in (a.lower() for a in agent.aliases):
                return agent if agent.model else None
        return None


class ProviderConfig(Base):
    """LLM provider configuration."""

    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None  # Custom headers (e.g. APP-Code for AiHubMix)


class ProviderInstanceConfig(ProviderConfig):
    """Named provider instance config (for custom provider references)."""

    type: str


class ResolvedProvider(NamedTuple):
    """Resolved provider reference."""

    ref: str
    type_name: str
    config: ProviderConfig


def _normalize_provider_token(value: str) -> str:
    """Normalize provider token for registry lookups."""
    return value.strip().replace("-", "_")


class ProvidersConfig(Base):
    """Configuration for LLM providers."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")

    custom: ProviderConfig = Field(default_factory=ProviderConfig)  # Any OpenAI-compatible endpoint
    azure_openai: ProviderConfig = Field(default_factory=ProviderConfig)  # Azure OpenAI (model = deployment name)
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    ollama: ProviderConfig = Field(default_factory=ProviderConfig)  # Ollama local models
    ovms: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenVINO Model Server (OVMS)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    mistral: ProviderConfig = Field(default_factory=ProviderConfig)
    aihubmix: ProviderConfig = Field(default_factory=ProviderConfig)  # AiHubMix API gateway
    siliconflow: ProviderConfig = Field(default_factory=ProviderConfig)  # SiliconFlow (硅基流动)
    volcengine: ProviderConfig = Field(default_factory=ProviderConfig)  # VolcEngine (火山引擎)
    volcengine_coding_plan: ProviderConfig = Field(default_factory=ProviderConfig)  # VolcEngine Coding Plan
    byteplus: ProviderConfig = Field(default_factory=ProviderConfig)  # BytePlus (VolcEngine international)
    byteplus_coding_plan: ProviderConfig = Field(default_factory=ProviderConfig)  # BytePlus Coding Plan
    openai_codex: ProviderConfig = Field(default_factory=ProviderConfig, exclude=True)  # OpenAI Codex (OAuth)
    github_copilot: ProviderConfig = Field(default_factory=ProviderConfig, exclude=True)  # Github Copilot (OAuth)

    def _get_dynamic_raw(self, provider_ref: str) -> dict | None:
        extra = self.model_extra or {}
        if provider_ref in extra and isinstance(extra[provider_ref], dict):
            return extra[provider_ref]
        normalized = _normalize_provider_token(provider_ref)
        if normalized in extra and isinstance(extra[normalized], dict):
            return extra[normalized]
        for raw_name, raw_value in extra.items():
            if not isinstance(raw_value, dict):
                continue
            if _normalize_provider_token(raw_name) == normalized:
                return raw_value
        return None

    def dynamic_instances(self) -> dict[str, ProviderInstanceConfig]:
        """Return custom named provider instances from extra config keys."""
        instances: dict[str, ProviderInstanceConfig] = {}
        for name, raw in (self.model_extra or {}).items():
            if not isinstance(raw, dict):
                continue
            if "type" not in raw:
                continue
            instances[name] = ProviderInstanceConfig.model_validate(raw)
        return instances

    def resolve(self, provider_ref: str) -> ResolvedProvider | None:
        """Resolve a provider reference to concrete provider type + config."""
        if not provider_ref:
            return None

        dynamic_raw = self._get_dynamic_raw(provider_ref)
        if dynamic_raw is not None:
            inst = ProviderInstanceConfig.model_validate(dynamic_raw)
            type_name = _normalize_provider_token(inst.type)
            cfg = ProviderConfig(
                api_key=inst.api_key,
                api_base=inst.api_base,
                extra_headers=inst.extra_headers,
            )
            return ResolvedProvider(provider_ref, type_name, cfg)

        attr_name = _normalize_provider_token(provider_ref)
        cfg = getattr(self, attr_name, None)
        if isinstance(cfg, ProviderConfig):
            return ResolvedProvider(attr_name, attr_name, cfg)
        return None

    def find_first_by_type(self, provider_type: str) -> ResolvedProvider | None:
        """Find the first configured provider matching a concrete provider type."""
        target = _normalize_provider_token(provider_type)

        for name, inst in self.dynamic_instances().items():
            if _normalize_provider_token(inst.type) == target:
                cfg = ProviderConfig(
                    api_key=inst.api_key,
                    api_base=inst.api_base,
                    extra_headers=inst.extra_headers,
                )
                return ResolvedProvider(name, target, cfg)

        cfg = getattr(self, target, None)
        if isinstance(cfg, ProviderConfig):
            return ResolvedProvider(target, target, cfg)
        return None


class HeartbeatConfig(Base):
    """Heartbeat service configuration."""

    enabled: bool = True
    interval_s: int = 30 * 60  # 30 minutes
    keep_recent_messages: int = 8


class GatewayConfig(Base):
    """Gateway/server configuration."""

    host: str = "0.0.0.0"
    port: int = 18790
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)


class WebSearchConfig(Base):
    """Web search tool configuration."""

    provider: str = "brave"  # brave, tavily, duckduckgo, searxng, jina
    api_key: str = ""
    base_url: str = ""  # SearXNG base URL
    max_results: int = 5


class WebToolsConfig(Base):
    """Web tools configuration."""

    proxy: str | None = (
        None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"
    )
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    enable: bool = True
    timeout: int = 60
    path_append: str = ""


class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # auto-detected if omitted
    command: str = ""  # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)  # Stdio: command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Stdio: extra env vars
    url: str = ""  # HTTP/SSE: endpoint URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE: custom headers
    tool_timeout: int = 30  # seconds before a tool call is cancelled
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])  # Only register these tools; accepts raw MCP names or wrapped mcp_<server>_<tool> names; ["*"] = all tools; [] = no tools


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # If true, restrict all tool access to workspace directory
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class Config(BaseSettings):
    """Root configuration for nanobot."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def _match_provider(
        self, provider_ref: str | None = None
    ) -> tuple["ProviderConfig | None", str | None]:
        """Match explicit provider ref to config and provider type."""
        ref = provider_ref or self.agents.defaults.provider
        if not ref:
            return None, None
        resolved = self.providers.resolve(ref)
        if not resolved:
            return None, None
        return resolved.config, resolved.type_name

    def get_provider(self, provider_ref: str | None = None) -> ProviderConfig | None:
        """Get explicit provider config (api_key, api_base, extra_headers)."""
        p, _ = self._match_provider(provider_ref)
        return p

    def get_provider_name(self, provider_ref: str | None = None) -> str | None:
        """Get the resolved provider type (e.g. "deepseek", "openrouter")."""
        _, name = self._match_provider(provider_ref)
        return name

    def get_api_key(self, provider_ref: str | None = None) -> str | None:
        """Get API key for an explicit provider reference."""
        p = self.get_provider(provider_ref)
        return p.api_key if p else None

    def get_api_base(self, provider_ref: str | None = None) -> str | None:
        """Get API base URL for an explicit provider reference."""
        from nanobot.providers.registry import find_by_name

        p, name = self._match_provider(provider_ref)
        if p and p.api_base:
            return p.api_base
        if name:
            spec = find_by_name(name)
            if spec and (spec.is_gateway or spec.is_local) and spec.default_api_base:
                return spec.default_api_base
        return None

    def validate_runtime_routes(self) -> None:
        """Validate explicit provider/model routing for defaults and @agents."""
        from nanobot.providers.registry import find_by_name

        errors: list[str] = []

        for ref_name, inst in self.providers.dynamic_instances().items():
            provider_type = _normalize_provider_token(inst.type)
            if not find_by_name(provider_type):
                errors.append(
                    f"providers.{ref_name}.type '{inst.type}' is unknown"
                )

        def _validate_target(label: str, provider_ref: str, model: str) -> None:
            if not provider_ref or not provider_ref.strip():
                errors.append(f"{label}.provider is required")
                return
            if not model or not model.strip():
                errors.append(f"{label}.model is required")
                return
            resolved = self.providers.resolve(provider_ref)
            if resolved is None:
                errors.append(f"{label}.provider '{provider_ref}' is not defined")
                return
            if not find_by_name(resolved.type_name):
                errors.append(
                    f"{label}.provider '{provider_ref}' resolves to unknown type '{resolved.type_name}'"
                )

        _validate_target(
            "agents.defaults",
            self.agents.defaults.provider,
            self.agents.defaults.model,
        )
        for name, route in self.agents.models.items():
            _validate_target(
                f"agents.models.{name}",
                route.provider,
                route.model,
            )

        if errors:
            raise ValueError("Invalid runtime config:\n- " + "\n- ".join(errors))

    model_config = ConfigDict(env_prefix="NANOBOT_", env_nested_delimiter="__")
