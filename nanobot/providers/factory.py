"""Factory for instantiating LLM providers."""

from __future__ import annotations

from typing import TYPE_CHECKING
import typer
from rich.console import Console

if TYPE_CHECKING:
    from nanobot.config.schema import Config
    from nanobot.providers.base import LLMProvider

console = Console()


def create_provider(
    config: Config,
    provider_ref: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    """Create the configured LLM provider from explicit provider/model routes."""
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider
    from nanobot.providers.azure_openai_provider import AzureOpenAIProvider
    from nanobot.providers.base import GenerationSettings
    from nanobot.providers.registry import find_by_name

    provider_ref = provider_ref or config.agents.defaults.provider
    model = model or config.agents.defaults.model
    resolved = config.providers.resolve(provider_ref)
    if resolved is None:
        console.print(f"[red]Error: Unknown provider reference '{provider_ref}'.[/red]")
        console.print("Set agents.defaults.provider or agents.models.<name>.provider to a defined provider key.")
        raise typer.Exit(1)

    provider_type = resolved.type_name
    p = resolved.config

    # OpenAI Codex (OAuth)
    if provider_type == "openai_codex":
        provider = OpenAICodexProvider(default_model=model)

    # Custom: direct OpenAI-compatible endpoint, bypasses LiteLLM
    elif provider_type == "custom":
        from nanobot.providers.custom_provider import CustomProvider

        provider = CustomProvider(
            api_key=p.api_key if p else "no-key",
            api_base=config.get_api_base(provider_ref) or "http://localhost:8000/v1",
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )

    # Azure OpenAI: direct Azure OpenAI endpoint with deployment name
    elif provider_type == "azure_openai":
        if not p or not p.api_key or not p.api_base:
            console.print("[red]Error: Azure OpenAI requires api_key and api_base.[/red]")
            console.print("Set them in ~/.nanobot/config.json under providers.azure_openai section")
            console.print("Use the model field to specify the deployment name.")
            raise typer.Exit(1)

        provider = AzureOpenAIProvider(
            api_key=p.api_key,
            api_base=p.api_base,
            default_model=model,
        )

    # OpenVINO Model Server: direct OpenAI-compatible endpoint at /v3
    elif provider_type == "ovms":
        from nanobot.providers.custom_provider import CustomProvider

        provider = CustomProvider(
            api_key=p.api_key if p else "no-key",
            api_base=config.get_api_base(provider_ref) or "http://localhost:8000/v3",
            default_model=model,
        )

    else:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        spec = find_by_name(provider_type)
        if spec is None:
            console.print(
                f"[red]Error: Provider reference '{provider_ref}' resolves to unknown type '{provider_type}'.[/red]"
            )
            raise typer.Exit(1)
        if not model.startswith("bedrock/") and not (p and p.api_key) and not (spec.is_oauth or spec.is_local):
            console.print(f"[red]Error: No API key configured for {provider_ref} ({provider_type}).[/red]")
            console.print("Set one in ~/.nanobot/config.json under providers section")
            raise typer.Exit(1)

        provider = LiteLLMProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(provider_ref),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            provider_name=provider_type,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider
