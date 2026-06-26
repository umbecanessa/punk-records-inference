"""Agent middleware — transcript strip and ``kv_transfer_params`` for tool-calling clients.

Port of hosted Punk Records ``openai.service.ts`` logic for standalone vLLM.
When ``NLS_AGENT_SHIM=1`` (default), OpenCode-style chat completions get:

  - **Strip** — remove prior assistant noise from the resend transcript on turn ≥ 2
  - **capture_start** — token offset after system/tools (``memory_capture_start``,
    ``memory_sys_prompt_hash``) so ``.nls`` manifests have correct ``rope_start``
  - **Chain metadata** — ``memory_base_session``, ``memory_turn_index``,
    ``memory_prev_hash``, ``memory_silo`` on turn 1, ``memory_inject_mode``

Clients can instead send full ``kv_transfer_params`` explicitly — both paths work.
See ``docs/guides/integrating-opencode.md`` and ``docs/CLIENT_CONTRACT.md``.

Registered in ``docker/start.sh``::

    --middleware pri.middleware.agent_shim.AgentShimMiddleware
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import urllib.request
from typing import Any

from pri.chat_template import compute_capture_start

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("pri.agent_shim")

AGENT_SHIM_ENABLED = os.environ.get("NLS_AGENT_SHIM", "1") == "1"
CHAIN_CAPTURE_MODE = os.environ.get("NLS_CHAIN_CAPTURE_MODE", "turn").strip().lower()
DEFAULT_INJECT_MODE = os.environ.get("NLS_API_INJECT_MODE", "resume_overflow").strip()

OPENCODE_COMPACTION_MARKER = "Provide a detailed but concise summary"

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"

_sys_prompt_token_cache: dict[str, int] = {}


def _sys_prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _block_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _render_system_turn_like_qwen(system_content: str) -> str:
    return f"{_IM_START}system\n{system_content}{_IM_END}\n"


def _is_agent_mode(body: dict[str, Any]) -> bool:
    if body.get("nls_mode") == "agent":
        return True
    if body.get("tools"):
        return True
    for msg in body.get("messages") or []:
        if msg.get("role") == "tool":
            return True
        if msg.get("tool_calls"):
            return True
        if msg.get("tool_call_id"):
            return True
    return False


def _is_compaction_generation(body: dict[str, Any]) -> bool:
    if body.get("tools"):
        return False
    last_user = None
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") == "user":
            last_user = msg
            break
    if not last_user:
        return False
    text = (last_user.get("content") or "").strip()
    if not text:
        return False
    return (
        OPENCODE_COMPACTION_MARKER in text
        or ("<template>" in text and "## Goal" in text)
    )


def _strip_agent_messages_for_resume(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system = next((m for m in messages if m.get("role") == "system"), None)
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx < 0:
        tail = [m for m in messages if m.get("role") != "system"]
        return ([system, *tail] if system else tail)
    tail = messages[last_user_idx:]
    return ([system, *tail] if system else tail)


def _find_tool_call_for_result(
    messages: list[dict[str, Any]],
    tool_result_msg: dict[str, Any],
) -> dict[str, Any] | None:
    call_id = tool_result_msg.get("tool_call_id")
    if not call_id:
        return None
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("id") == call_id:
                    return msg
    return None


def _first_user_message_text(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _normalize_chain_base_session(raw: str) -> str:
    trimmed = raw.strip()
    if not trimmed:
        return ""
    if trimmed.startswith("chain_") or trimmed.startswith("oc_"):
        return f"oc_{_block_hash(trimmed)}" if len(trimmed) > 64 else trimmed
    return f"oc_{_block_hash(trimmed)}"


def _resolve_chain_base_session(body: dict[str, Any], user_id: str) -> str:
    meta = body.get("metadata") or {}
    explicit = (
        (body.get("nls_chain_id") or "").strip()
        or str(meta.get("session_id") or meta.get("sessionID") or "").strip()
    )
    if explicit:
        return _normalize_chain_base_session(explicit)
    system_msg = next((m for m in body.get("messages") or [] if m.get("role") == "system"), None)
    system_content = system_msg.get("content", "") if system_msg else ""
    if not isinstance(system_content, str):
        system_content = ""
    first_user = _first_user_message_text(body.get("messages") or [])
    sys_hash = _sys_prompt_hash(system_content) if system_content else ""
    fingerprint = _block_hash(f"{user_id}|{sys_hash}|{first_user}")
    return f"chain_{fingerprint}"


def _agent_turn_index(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


async def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    def _do() -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    return await asyncio.to_thread(_do)


async def _token_count_for(
    request: Request,
    text: str,
    *,
    model: str | None = None,
) -> int:
    if not text:
        return 0
    cache_key = _sys_prompt_hash(text)
    cached = _sys_prompt_token_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        base = str(request.base_url).rstrip("/")
        payload: dict[str, Any] = {"prompt": text}
        if model:
            payload["model"] = model
        data = await _post_json(f"{base}/tokenize", payload)
        count = data.get("count")
        if count is None and isinstance(data.get("tokens"), list):
            count = len(data["tokens"])
        count = int(count or 0)
        _sys_prompt_token_cache[cache_key] = count
        return count
    except Exception as exc:
        logger.warning("tokenize fallback (char-ratio): %s", exc)
        return max(1, int(len(text) / 3.5))


async def _system_and_tools_token_count(
    request: Request,
    system_content: str,
    tools: list[dict[str, Any]],
    *,
    model: str | None = None,
) -> int:
    if not system_content:
        return 0

    tools_json = json.dumps(tools, separators=(",", ":")) if tools else ""
    cache_key = _sys_prompt_hash(
        system_content + ("\u0000" + tools_json if tools_json else ""),
    )
    cached = _sys_prompt_token_cache.get(cache_key)
    if cached is not None:
        return cached

    base = str(request.base_url).rstrip("/")

    def _compute() -> int:
        return compute_capture_start(
            system_content,
            api_root=base,
            model=model or "",
            tools=tools or None,
        )

    try:
        count = await asyncio.to_thread(_compute)
        if count > 0:
            _sys_prompt_token_cache[cache_key] = count
            return count
    except Exception as exc:
        logger.warning("capture_start via chat_template failed: %s", exc)

    return await _token_count_for(
        request, _render_system_turn_like_qwen(system_content), model=model
    )


def _ensure_kv_transfer_params(body: dict[str, Any]) -> dict[str, str]:
    raw = body.get("kv_transfer_params")
    if not isinstance(raw, dict):
        raw = {}
    kvp = {str(k): str(v) for k, v in raw.items()}
    body["kv_transfer_params"] = kvp
    return kvp


async def _apply_agent_shim(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    if not _is_agent_mode(body):
        return body

    if _is_compaction_generation(body):
        logger.info("Agent shim: compaction generation — memory_off")
        kvp = _ensure_kv_transfer_params(body)
        kvp["memory_off"] = "1"
        return body

    user_id = str(body.get("user") or "default")
    client_messages = list(body.get("messages") or [])
    agent_turn_index = _agent_turn_index(client_messages)
    nls_agent_chain = True

    messages = client_messages
    if agent_turn_index > 1:
        before = len(messages)
        messages = _strip_agent_messages_for_resume(messages)
        logger.info(
            "Agent shim resume strip: %d -> %d messages (turn %d)",
            before, len(messages), agent_turn_index,
        )
    body["messages"] = messages

    last_msg = client_messages[-1] if client_messages else {}
    block_role = "tool" if last_msg.get("role") == "tool" else "user"

    last_user_msg = ""
    for msg in reversed(client_messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            last_user_msg = content if isinstance(content, str) else ""
            break

    memory_text = last_user_msg
    if block_role == "tool" and last_msg:
        tool_call_msg = _find_tool_call_for_result(client_messages, last_msg)
        call_text = (
            f"[tool_call] {json.dumps(tool_call_msg.get('tool_calls'))}"
            if tool_call_msg
            else ""
        )
        response_text = f"[tool_response] {last_msg.get('content') or ''}"
        memory_text = f"{call_text}\n{response_text}".strip()

    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    system_content = system_msg.get("content", "") if system_msg else ""
    if not isinstance(system_content, str):
        system_content = ""

    model = body.get("model")
    capture_start = 0
    if system_content:
        if body.get("tools"):
            capture_start = await _system_and_tools_token_count(
                request, system_content, body["tools"], model=model
            )
        else:
            capture_start = await _system_and_tools_token_count(
                request, system_content, [], model=model
            )

    sys_prompt_hash = _sys_prompt_hash(system_content) if system_content else ""
    chain_base = _resolve_chain_base_session(body, user_id)
    kvp = _ensure_kv_transfer_params(body)

    kvp.setdefault("memory_user", user_id)
    kvp.setdefault("memory_ring", "general")
    kvp["memory_block_role"] = block_role
    kvp["memory_base_session"] = chain_base
    kvp["memory_text"] = memory_text

    if capture_start > 0:
        kvp["memory_capture_start"] = str(capture_start)
    if sys_prompt_hash:
        kvp["memory_sys_prompt_hash"] = sys_prompt_hash

    turn_capture_mode = CHAIN_CAPTURE_MODE == "turn"
    if nls_agent_chain and not turn_capture_mode and capture_start > 0 and memory_text.strip():
        leg_tokens = await _token_count_for(request, memory_text, model=model)
        if leg_tokens > 0:
            kvp["memory_capture_end"] = str(capture_start + leg_tokens + 24)

    if nls_agent_chain:
        turn_session = f"{chain_base}_t{agent_turn_index}_{block_role}"
        asst_session = f"{chain_base}_t{agent_turn_index}_asst"
        prev_hash = (
            ""
            if agent_turn_index <= 1
            else _block_hash(f"{chain_base}_t{agent_turn_index - 1}_user")
        )
        kvp["memory_session"] = turn_session
        kvp["memory_turn_index"] = str(agent_turn_index)
        kvp["memory_prev_hash"] = prev_hash
        kvp["memory_asst_session"] = asst_session
        if agent_turn_index == 1:
            kvp["memory_silo"] = "1"
        if agent_turn_index > 1 and block_role == "user":
            kvp["memory_deltanet_init_session"] = (
                f"{chain_base}_t{agent_turn_index - 1}_user"
            )
        if agent_turn_index > 1:
            kvp["memory_inject_mode"] = DEFAULT_INJECT_MODE
            resume_max_blocks = os.environ.get("NLS_RESUME_MAX_BLOCKS")
            resume_max_tokens = os.environ.get("NLS_RESUME_MAX_TOKENS")
            swiss_max_tokens = os.environ.get("NLS_RESUME_SWISS_MAX_TOKENS")
            if resume_max_blocks:
                kvp["memory_resume_max_blocks"] = resume_max_blocks
            if resume_max_tokens:
                kvp["memory_resume_max_tokens"] = resume_max_tokens
            if swiss_max_tokens and DEFAULT_INJECT_MODE == "resume_overflow":
                kvp["memory_resume_swiss_max_tokens"] = swiss_max_tokens
    elif agent_turn_index == 1 and _is_agent_mode(body):
        kvp["memory_silo"] = "1"

    if not body.get("cache_salt"):
        body["cache_salt"] = f"nls_user_{user_id}"

    return body


class AgentShimMiddleware(BaseHTTPMiddleware):
    """Strip agent transcript + inject chain kv_transfer_params before vLLM."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if not AGENT_SHIM_ENABLED:
            return await call_next(request)

        if request.method != "POST" or not request.url.path.endswith(
            "/v1/chat/completions"
        ):
            return await call_next(request)

        try:
            raw = await request.body()
            body = json.loads(raw)
        except Exception:
            return await call_next(request)

        try:
            body = await _apply_agent_shim(request, body)
        except Exception as exc:
            logger.warning("Agent shim failed (pass-through): %s", exc, exc_info=True)
            return await call_next(request)

        kvp = body.get("kv_transfer_params") or {}
        if isinstance(kvp, dict) and kvp.get("memory_session"):
            logger.info(
                "Agent shim kvp: session=%s turn=%s capture_start=%s",
                kvp.get("memory_session"),
                kvp.get("memory_turn_index"),
                kvp.get("memory_capture_start", "0"),
            )

        new_body = json.dumps(body).encode("utf-8")

        async def receive():
            return {"type": "http.request", "body": new_body, "more_body": False}

        modified = Request(request.scope, receive)
        modified._body = new_body  # noqa: SLF001 — starlette cache hint
        return await call_next(modified)
