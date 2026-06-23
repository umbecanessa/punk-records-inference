"""KL #648 wire helpers — system-prompt accounting for kv_transfer_params.

Production (punk-records openai.service.ts) sends ``memory_capture_start``
and ``memory_sys_prompt_hash`` so captures slice KV at the system/user
boundary (``rope_start > 0`` in the .nls manifest). That skips the brittle
inject-time ``NLS_STRIP_INJECT_SYS_BLOCK_LEN`` path, which amputates user
facts when the static strip length does not match the live prompt.

Test harnesses and ``prod_conversation_test.py`` must emit the same fields.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import requests

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def sys_prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def render_system_turn_like_qwen(system_content: str) -> str:
    """Match punk-records ``renderSystemTurnLikeQwen``."""
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


@lru_cache(maxsize=32)
def _token_count_cached(api_root: str, model: str, rendered: str) -> int:
    if not rendered:
        return 0
    try:
        r = requests.post(
            f"{api_root}/tokenize",
            json={"model": model, "prompt": rendered},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        count = data.get("count")
        if count is None and isinstance(data.get("tokens"), list):
            count = len(data["tokens"])
        return int(count or 0)
    except Exception:
        return max(int(len(rendered) / 3.5), 0)


def compute_capture_start(
    system_prompt: str,
    *,
    api_root: str,
    model: str,
) -> int:
    if not system_prompt.strip():
        return 0
    rendered = render_system_turn_like_qwen(system_prompt)
    return _token_count_cached(api_root, model, rendered)


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
        system_prompt, api_root=api_root, model=model
    )
    if capture_start > 0:
        kvp = dict(kvp)
        kvp["memory_capture_start"] = str(capture_start)
        kvp["memory_sys_prompt_hash"] = sys_prompt_hash(system_prompt)
    return kvp
