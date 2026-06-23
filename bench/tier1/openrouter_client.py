"""OpenRouter client for isolated TEXT-arm baselines (no PRI / no KV inject).

Set credentials via environment only — never commit API keys:

    export OPENROUTER_API_KEY=sk-or-v1-...
    export OPENROUTER_MODEL=qwen/qwen3.5-35b-a3b   # optional override
"""

from __future__ import annotations

import os
from typing import Any

import requests

OPENROUTER_CHAT_URL = os.environ.get(
    "OPENROUTER_API_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)
DEFAULT_OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL",
    "qwen/qwen3.5-35b-a3b",
)
OPENROUTER_HTTP_REFERER = os.environ.get("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_APP_TITLE = os.environ.get("OPENROUTER_APP_TITLE", "Punk Records Inference Bench")


def is_configured() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


def resolve_model(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return DEFAULT_OPENROUTER_MODEL


def chat(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    max_tokens: int = 200,
    temperature: float = 0.0,
    user_id: str = "bench",
    timeout: int = 180,
) -> tuple[str, dict[str, Any] | None]:
    """OpenAI-compatible chat completion via OpenRouter (plain text, no kvp)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set — export it in the shell or .env"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_TITLE:
        headers["X-Title"] = OPENROUTER_APP_TITLE

    body = {
        "model": resolve_model(model),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "user": user_id,
    }

    response = requests.post(
        OPENROUTER_CHAT_URL,
        headers=headers,
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    usage = data.get("usage")
    return content, usage
