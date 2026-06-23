"""Shared helpers for OpenCode-style Punk Records API harnesses."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pri.text_quality import is_garbled_response

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _api_root_from_chat_url(api: str) -> str:
    parsed = urlparse(api)
    if not parsed.scheme or not parsed.netloc:
        return api.rstrip("/")
    path = parsed.path or ""
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/") or api.rstrip("/")


def _resolve_harness_config() -> tuple[str, str, bool]:
    """Return (chat_completions_url, api_key, direct_vllm)."""
    chat_url = (
        os.environ.get("PRI_API", "").strip()
        or os.environ.get("NLS_API", "").strip()
    )
    base = os.environ.get("PRI_BASE_URL", "").strip().rstrip("/")
    if not chat_url and base:
        chat_url = f"{base}/v1/chat/completions"

    hosted_default = "https://api.punkrecords.live"
    direct_vllm = bool(base or (chat_url and hosted_default not in chat_url))

    if not chat_url:
        punk_base = os.environ.get(
            "PUNK_API_BASE", "https://api.punkrecords.live/v1"
        ).rstrip("/")
        chat_url = f"{punk_base}/chat/completions"
        direct_vllm = False

    api_key = os.environ.get("PUNK_API_KEY", "").strip()
    if direct_vllm and not api_key:
        api_key = ""

    return chat_url, api_key, direct_vllm


_config: tuple[str, str, bool] | None = None
_resolved_model: str | None = None


def reset_harness_config() -> None:
    global _config, _resolved_model
    _config = None
    _resolved_model = None


def get_harness_config() -> tuple[str, str, bool]:
    global _config
    if _config is None:
        _config = _resolve_harness_config()
    return _config


def get_api_base() -> str:
    chat_url, _, _ = get_harness_config()
    return _api_root_from_chat_url(chat_url)


def resolve_model() -> str:
    global _resolved_model
    if _resolved_model:
        return _resolved_model
    env_model = os.environ.get("PRI_MODEL", "").strip()
    if env_model:
        _resolved_model = env_model
        return _resolved_model
    models_url = f"{get_api_base().rstrip('/')}/v1/models"
    r = requests.get(models_url, timeout=15)
    r.raise_for_status()
    models = r.json().get("data") or []
    if not models:
        raise RuntimeError(f"no models from {models_url}")
    _resolved_model = models[0]["id"]
    return _resolved_model


SYSTEM_PROMPT = (
    "You are OpenCode, an autonomous coding agent operating inside the user's "
    "terminal. Operational decisions MUST be stated explicitly as "
    "'DECISION: <fact>' so they persist across context compaction via NLS.\n"
    "Today's task lives at docs/PRD.md (read on demand).\n"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
]

BANNED_PORTS = frozenset({3000, 3001, 8000, 8080, 5000, 5173, 4000, 4200, 3306, 5432})


@dataclass(frozen=True)
class SessionFacts:
    """Non-obvious planted values for one harness run."""

    backend_port: int
    frontend_port: int
    metrics_port: int
    db_name: str
    redis_port: int
    api_prefix: str

    def as_decision_block(self) -> str:
        return (
            f"DECISION: monorepo manager = pnpm workspaces\n"
            f"DECISION: backend port = {self.backend_port}\n"
            f"DECISION: frontend port = {self.frontend_port}\n"
            f"DECISION: metrics port = {self.metrics_port}\n"
            f"DECISION: dev database = {self.db_name}\n"
            f"DECISION: redis port = {self.redis_port}\n"
            f"DECISION: api path prefix = {self.api_prefix}"
        )


def generate_session_facts(seed: int) -> SessionFacts:
    """Pick ports/names unlikely to be model defaults."""
    rng = random.Random(seed)

    def pick_port() -> int:
        while True:
            p = rng.randint(10240, 60999)
            if p not in BANNED_PORTS:
                return p

    backend = pick_port()
    frontend = pick_port()
    while frontend == backend:
        frontend = pick_port()
    metrics = pick_port()
    while metrics in (backend, frontend):
        metrics = pick_port()
    redis = pick_port()
    while redis in (backend, frontend, metrics):
        redis = pick_port()

    suffix = rng.randint(1000, 9999)
    db_name = f"icf_eval_x{suffix}"
    api_prefix = f"/v{rng.randint(2, 9)}/icf-{rng.randint(10, 99)}"

    return SessionFacts(
        backend_port=backend,
        frontend_port=frontend,
        metrics_port=metrics,
        db_name=db_name,
        redis_port=redis,
        api_prefix=api_prefix,
    )


def safe_print(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"), flush=True)


def _request_headers() -> dict[str, str]:
    _, api_key, direct_vllm = get_harness_config()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if not direct_vllm:
        headers["Accept"] = "text/event-stream"
    return headers


def _parse_hosted_sse(
    r: requests.Response,
) -> tuple[str, dict[str, Any] | None]:
    deltas: list[str] = []
    nls: dict[str, Any] | None = None
    for raw in r.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "):
            continue
        data = raw[6:].strip()
        if data == "[DONE]":
            break
        try:
            evt = json.loads(data)
        except json.JSONDecodeError:
            continue
        if evt.get("nls"):
            nls = evt["nls"]
        choices = evt.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            deltas.append(delta["content"])
    return "".join(deltas).strip(), nls


def _parse_direct_json(data: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    choices = data.get("choices") or []
    if not choices:
        return "", None
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    return str(content).strip(), data.get("nls")


def stream_chat(
    messages: list[dict[str, Any]],
    *,
    label: str,
    max_tokens: int = 400,
    with_tools: bool = True,
    agent_mode: bool | None = None,
    chain_id: str | None = None,
    tool_choice: str | dict[str, Any] = "none",
    temperature: float = 0.0,
    user_id: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    include_tools = with_tools and tool_choice != "none"
    model = resolve_model()
    chat_url, _, direct_vllm = get_harness_config()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if chain_id:
        payload["nls_chain_id"] = chain_id
    use_agent = agent_mode if agent_mode is not None else include_tools
    if use_agent:
        payload["nls_mode"] = "agent"
    if include_tools:
        payload["tools"] = TOOLS
        payload["tool_choice"] = tool_choice

    effective_user = user_id or os.environ.get("PRI_USER") or f"opencode_{chain_id or 'bench'}"
    payload["user"] = effective_user

    if direct_vllm:
        payload["stream"] = False
    else:
        payload["stream"] = True
        payload["nls_metadata"] = "summary"

    headers = _request_headers()
    t0 = time.perf_counter()
    r = requests.post(
        chat_url,
        headers=headers,
        json=payload,
        stream=not direct_vllm,
        timeout=300,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"{label} HTTP {r.status_code}: {r.text[:800]}")

    if direct_vllm:
        data = r.json()
        text, nls = _parse_direct_json(data)
        if nls is None:
            safe_print(f"  [{label}] direct vLLM: no nls metadata in response (expected)")
    else:
        text, nls = _parse_hosted_sse(r)

    elapsed = time.perf_counter() - t0
    mem = (nls or {}).get("memory") or {}
    inj = mem.get("injected_tokens", 0)
    saved = mem.get("saved_tokens", 0)
    chain_hint = ""
    if nls:
        cid = nls.get("chain_id")
        tidx = nls.get("turn_index")
        compact = nls.get("compaction_detected")
        bits = []
        if cid:
            bits.append(f"chain={cid}")
        if tidx is not None:
            bits.append(f"turn={tidx}")
        if compact:
            bits.append("compaction=1")
        if bits:
            chain_hint = " " + " ".join(bits)
    safe_print(
        f"  [{label}] {elapsed:.1f}s inj={inj} saved={saved}{chain_hint} "
        f"| {text[:100]}{'...' if len(text) > 100 else ''}"
    )
    return text, nls


def score_recall(answer: str, must: list[str], must_not: list[str]) -> bool:
    low = answer.lower()
    if any(bad.lower() in low for bad in must_not):
        return False
    return all(need.lower() in low for need in must)


def agent_turn(
    transcript: list[dict[str, Any]],
    *,
    label: str,
    user_text: str,
    chain_id: str,
    max_tokens: int = 400,
    pause_s: float = 1.0,
    temperature: float = 0.0,
    user_id: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    transcript.append({"role": "user", "content": user_text})
    text, nls = stream_chat(
        transcript,
        label=label,
        max_tokens=max_tokens,
        chain_id=chain_id,
        temperature=temperature,
        agent_mode=True,
        tool_choice="none",
        user_id=user_id,
    )
    clean = text.strip()
    if is_garbled_response(clean):
        safe_print(f"  [{label}] garbled assistant stripped from transcript")
        clean = ""
    transcript.append({"role": "assistant", "content": clean})
    if pause_s > 0:
        time.sleep(pause_s)
    return clean, nls
