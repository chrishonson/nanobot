"""Reusable provider pool for explicit provider/model routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nanobot.providers.factory import create_provider

if TYPE_CHECKING:
    from nanobot.config.schema import Config
    from nanobot.providers.base import LLMProvider


class ProviderPool:
    """Lazily create and cache providers keyed by (provider_ref, model)."""

    def __init__(self, config: Config):
        self._config = config
        self._cache: dict[tuple[str, str], LLMProvider] = {}

    def get(self, provider_ref: str, model: str) -> LLMProvider:
        """Get or create the provider instance for the explicit route."""
        key = (provider_ref, model)
        provider = self._cache.get(key)
        if provider is None:
            provider = create_provider(self._config, provider_ref=provider_ref, model=model)
            self._cache[key] = provider
        return provider

    def get_default(self) -> LLMProvider:
        """Get or create the default route provider."""
        defaults = self._config.agents.defaults
        return self.get(defaults.provider, defaults.model)
