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

import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nls_vllm_plugin.text_quality import is_garbled_response

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

API_BASE = os.environ.get("PUNK_API_BASE", "https://api.punkrecords.live/v1").rstrip("/")
API_KEY = os.environ.get("PUNK_API_KEY", "").strip()

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

# Obvious defaults — used as must_not in recall scoring.
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
) -> tuple[str, dict[str, Any] | None]:
    include_tools = with_tools and tool_choice != "none"
    payload: dict[str, Any] = {
        "model": "nls-qwen3.5-moe",
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "nls_metadata": "summary",
    }
    if chain_id:
        payload["nls_chain_id"] = chain_id
    use_agent = agent_mode if agent_mode is not None else include_tools
    if use_agent:
        payload["nls_mode"] = "agent"
    if include_tools:
        payload["tools"] = TOOLS
        payload["tool_choice"] = tool_choice
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    t0 = time.perf_counter()
    r = requests.post(
        f"{API_BASE}/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=300,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"{label} HTTP {r.status_code}: {r.text[:800]}")

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

    text = "".join(deltas).strip()
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
) -> tuple[str, dict[str, Any] | None]:
    transcript.append({"role": "user", "content": user_text})
    text, nls = stream_chat(
        transcript,
        label=label,
        max_tokens=max_tokens,
        chain_id=chain_id,
        temperature=temperature,
        agent_mode=True,
    )
    clean = text.strip()
    if is_garbled_response(clean):
        safe_print(f"  [{label}] garbled assistant stripped from transcript")
        clean = ""
    transcript.append({"role": "assistant", "content": clean})
    if pause_s > 0:
        time.sleep(pause_s)
    return clean, nls
