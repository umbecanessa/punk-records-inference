"""Model-native chat template boundaries via vLLM ``/tokenize``.

All capture-boundary and inline-history token counts go through the loaded
model's chat template (Gemma ``<start_of_turn>``, Qwen ``<|im_start|>``,
Llama, etc.). Callers pass ``api_root`` + ``model`` from the live server —
no hard-coded family strings or Qwen render shortcuts in the hot path.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import requests

logger = logging.getLogger("pri.chat_template")

# Stable probe for capture_start delta; must not appear in real user content.
CAPTURE_PROBE_USER = "x"

DEFAULT_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}


def _parse_tokenize_count(data: dict[str, Any]) -> int:
    count = data.get("count")
    if count is None and isinstance(data.get("tokens"), list):
        count = len(data["tokens"])
    return int(count or 0)


@lru_cache(maxsize=128)
def _tokenize_messages_cached(
    api_root: str,
    model: str,
    payload_key: str,
) -> int:
    if not payload_key:
        return 0
    payload = json.loads(payload_key)
    try:
        response = requests.post(
            f"{api_root.rstrip('/')}/tokenize",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return _parse_tokenize_count(response.json())
    except Exception as exc:
        logger.debug("tokenize(messages) failed: %s", exc)
        return 0


def _tokenize_payload(
    api_root: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> int:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "chat_template_kwargs": chat_template_kwargs or DEFAULT_CHAT_TEMPLATE_KWARGS,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        body["tools"] = tools
    key = json.dumps(body, separators=(",", ":"), sort_keys=True)
    return _tokenize_messages_cached(api_root, model, key)


def token_count_prompt(api_root: str, model: str, prompt: str) -> int:
    """Count tokens for a raw prompt string (fallback / diagnostics only)."""
    if not prompt:
        return 0
    try:
        response = requests.post(
            f"{api_root.rstrip('/')}/tokenize",
            json={"model": model, "prompt": prompt},
            timeout=30,
        )
        response.raise_for_status()
        return _parse_tokenize_count(response.json())
    except Exception as exc:
        logger.debug("tokenize(prompt) failed: %s", exc)
        return max(int(len(prompt) / 3.5), 0)


def token_count_messages(
    api_root: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> int:
    """Count tokens for ``messages`` using the loaded model's chat template."""
    return _tokenize_payload(
        api_root,
        model,
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        chat_template_kwargs=chat_template_kwargs,
    )


def compute_capture_start(
    system_prompt: str,
    *,
    api_root: str,
    model: str,
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Token offset where user content begins (after system and optional tools).

    Computed as ``tokenize([system, probe_user]) - tokenize([probe_user])`` so
    any model family resolves through vLLM's template for the mounted checkpoint.
    """
    if not system_prompt.strip():
        return 0
    if not api_root.strip() or not model.strip():
        return 0

    prefix_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": CAPTURE_PROBE_USER},
    ]
    probe_only: list[dict[str, Any]] = [
        {"role": "user", "content": CAPTURE_PROBE_USER},
    ]
    with_prefix = token_count_messages(
        api_root, model, prefix_messages, tools=tools, add_generation_prompt=False,
    )
    user_only = token_count_messages(
        api_root, model, probe_only, add_generation_prompt=False,
    )
    if with_prefix > 0:
        return max(0, with_prefix - user_only)
    return 0


def build_inline_history_messages(
    system_prompt: str,
    turns: list[tuple[str, str]],
    question: str,
    *,
    empty_assistant_placeholder: str = "Acknowledged.",
    include_assistant: bool = True,
) -> list[dict[str, str]]:
    """Messages list for TEXT-style inline recall."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for user_text, asst_text in turns:
        messages.append({"role": "user", "content": user_text})
        if include_assistant:
            messages.append({
                "role": "assistant",
                "content": asst_text or empty_assistant_placeholder,
            })
    messages.append({"role": "user", "content": question})
    return messages


def token_count_inline_history(
    api_root: str,
    model: str,
    *,
    system_prompt: str,
    turns: list[tuple[str, str]],
    question: str,
    include_assistant: bool = True,
    empty_assistant_placeholder: str = "Acknowledged.",
) -> int:
    """Token count for inline TEXT recall using the model-native template."""
    messages = build_inline_history_messages(
        system_prompt,
        turns,
        question,
        empty_assistant_placeholder=empty_assistant_placeholder,
        include_assistant=include_assistant,
    )
    return token_count_messages(
        api_root,
        model,
        messages,
        add_generation_prompt=True,
    )


def run_capture_start_self_check(
    *,
    api_root: str,
    model: str,
    system_prompt: str,
) -> dict[str, Any]:
    """Log-friendly probe: capture_start for a system prompt on the live model."""
    capture_start = compute_capture_start(
        system_prompt, api_root=api_root, model=model,
    )
    ok = capture_start > 0
    report = {
        "ok": ok,
        "capture_start": capture_start,
        "model": model,
        "system_prompt_chars": len(system_prompt),
    }
    if ok:
        logger.info(
            "NLS chat-template self-check: model=%s capture_start=%d",
            model,
            capture_start,
        )
    else:
        logger.warning(
            "NLS chat-template self-check FAILED: model=%s capture_start=0 "
            "(check /tokenize messages support)",
            model,
        )
    return report
