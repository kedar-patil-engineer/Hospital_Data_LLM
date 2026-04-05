# llm/providers.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol


class LLMProvider(Protocol):
    """Minimal provider interface used by the project (OpenAI / Local)."""

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        ...


@dataclass(frozen=True)
class OpenAIProvider:
    """OpenAI provider (official OpenAI Python SDK)."""

    api_key: str
    default_model: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Set it in your environment or .env file."
            )

    def complete(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            model: Optional[str] = None,
            temperature: float = 0.2,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=model or self.default_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(temperature),
        )
        return (resp.choices[0].message.content or "").strip()

    def complete_with_meta(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> tuple[str, dict]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=model or self.default_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(temperature),
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        meta = {
            "tokens_in": getattr(usage, "prompt_tokens", None),
            "tokens_out": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        return text, meta

@dataclass(frozen=True)
class OllamaProvider:
    """Local provider via Ollama HTTP API.

    Requirements:
      - Ollama installed and running locally
      - Default endpoint: http://localhost:11434
    """

    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.1"

    def complete(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            model: Optional[str] = None,
            temperature: float = 0.2,
    ) -> str:
        text, _meta = self.complete_with_meta(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
        )
        return text

    def complete_with_meta(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            model: Optional[str] = None,
            temperature: float = 0.2,
    ) -> tuple[str, dict]:
        import requests

        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model or self.default_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": float(temperature),
                "num_predict": 512,
            },
        }

        try:
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to call Ollama at {url}. Details: {e}") from e
        except ValueError as e:
            raise RuntimeError(f"Ollama returned non-JSON response. Details: {e}") from e

        msg = data.get("message") or {}
        text = (msg.get("content") or "").strip()

        meta = {
            "tokens_in": data.get("prompt_eval_count"),
            "tokens_out": data.get("eval_count"),
            "total_duration_s": (data.get("total_duration") or 0) / 1e9 if data.get("total_duration") else None,
            "load_duration_s": (data.get("load_duration") or 0) / 1e9 if data.get("load_duration") else None,
        }
        return text, meta


@dataclass(frozen=True)
class MockLocalProvider:
    """Fallback provider for demo/architecture mode (no external runtime)."""

    default_model: str = "mock-local"

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        return (
            "LOCAL_MODE (mock)\n"
            "This is a model-agnostic architecture placeholder.\n"
            "Switch Provider to OpenAI or Ollama for real responses."
        )
