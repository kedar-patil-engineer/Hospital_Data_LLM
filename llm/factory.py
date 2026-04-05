# llm/factory.py
from __future__ import annotations

import os
from typing import Optional

from llm.providers import LLMProvider, MockLocalProvider, OllamaProvider, OpenAIProvider


def get_llm_provider(
    provider_name: str,
    *,
    openai_model: Optional[str] = None,
    local_model: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
) -> LLMProvider:
    """Factory returning a provider instance by name.

    provider_name:
      - "openai"
      - "ollama"
      - "local" (alias for ollama)
      - "mock"
    """
    name = (provider_name or "").strip().lower()

    if name in {"openai", "oa"}:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = (openai_model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        return OpenAIProvider(api_key=api_key, default_model=model)

    if name in {"ollama", "local"}:
        base_url = (
            ollama_base_url
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).strip()
        model = (local_model or os.getenv("OLLAMA_MODEL") or "llama3.1").strip()
        return OllamaProvider(base_url=base_url, default_model=model)

    if name in {"mock", "mock_local"}:
        return MockLocalProvider()

    # Default behavior (industry-friendly):
    # Prefer OpenAI if key exists, else Ollama if configured, else mock.
    if os.getenv("OPENAI_API_KEY"):
        return get_llm_provider("openai", openai_model=openai_model)
    if os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_MODEL"):
        return get_llm_provider(
            "ollama",
            local_model=local_model,
            ollama_base_url=ollama_base_url,
        )
    return MockLocalProvider()
