"""KL #648 wire helpers — system-prompt accounting for kv_transfer_params.

Production (punk-records openai.service.ts) sends ``memory_capture_start``
and ``memory_sys_prompt_hash`` so captures slice KV at the system/user
boundary (``rope_start > 0`` in the .nls manifest). That skips the brittle
inject-time ``NLS_STRIP_INJECT_SYS_BLOCK_LEN`` path, which amputates user
facts when the static strip length does not match the live prompt.

Capture boundaries are resolved through ``pri.chat_template`` (vLLM
``/tokenize`` + ``messages``) so any mounted model family works without
hard-coded Qwen delimiters.
"""

from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import urlparse

from pri.chat_template import (
    build_inline_history_messages,
    compute_capture_start,
    token_count_inline_history,
    token_count_messages,
)

# Re-export for bench code that builds Qwen-style strings for diagnostics only.
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def sys_prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def render_system_turn_like_qwen(system_content: str) -> str:
    """Legacy Qwen render string — do not use for capture_start on new paths."""
    return f"{_IM_START}system\n{system_content}{_IM_END}\n"


def api_root_from_chat_url(api: str) -> str:
    """``http://host:8000/v1/chat/completions`` -> ``http://host:8000``."""
    parsed = urlparse(api)
    if not parsed.scheme or not parsed.netloc:
        return api.rstrip("/")
    path = parsed.path or ""
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/") or api.rstrip("/")


from pri.chat_template import token_count_prompt


def _token_count_cached(api_root: str, model: str, rendered: str) -> int:
    """Legacy prompt-string tokenize — prefer ``token_count_messages``."""
    return token_count_prompt(api_root, model, rendered)


def enrich_kv_params(
    kvp: dict[str, Any],
    system_prompt: str,
    *,
    api_root: str,
    model: str,
) -> dict[str, Any]:
    """Add KL #648 capture boundary fields when a system prompt is present."""
    if not system_prompt.strip():
        return kvp
    capture_start = compute_capture_start(
        system_prompt, api_root=api_root, model=model,
    )
    if capture_start > 0:
        kvp = dict(kvp)
        kvp["memory_capture_start"] = str(capture_start)
        kvp["memory_sys_prompt_hash"] = sys_prompt_hash(system_prompt)
    return kvp


def enrich_prefilled_capture_kv_params(
    kvp: dict[str, Any],
    system_prompt: str,
    *,
    api_root: str,
    model: str,
) -> dict[str, Any]:
    """KVP for 3-message prefilled-assistant turn capture.

    Keeps ``memory_capture_start`` + ``memory_sys_prompt_hash`` so resume inject
    can prepend the system block and strip the live system prefix, and sets
    ``memory_prefilled_capture`` so the connector slices KV from post-strip
    position 0 (manifest ``rope_start`` stays at ``capture_start``).
    """
    kvp = enrich_kv_params(kvp, system_prompt, api_root=api_root, model=model)
    kvp = dict(kvp)
    kvp["memory_prefilled_capture"] = "1"
    return kvp


__all__ = [
    "api_root_from_chat_url",
    "build_inline_history_messages",
    "compute_capture_start",
    "enrich_kv_params",
    "enrich_prefilled_capture_kv_params",
    "render_system_turn_like_qwen",
    "sys_prompt_hash",
    "token_count_inline_history",
    "token_count_messages",
    "_token_count_cached",
]
