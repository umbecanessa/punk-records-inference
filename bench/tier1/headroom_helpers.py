"""Headroom compression helpers for PRI turn-sweep experiments."""

from __future__ import annotations

import json
from typing import Any

try:
    from headroom import compress
    from headroom.compress import CompressConfig
except ImportError:  # pragma: no cover - bench optional dep
    compress = None  # type: ignore[assignment]
    CompressConfig = None  # type: ignore[assignment,misc]

DEFAULT_COMPRESS_CONFIG = None


def _default_config() -> Any:
    global DEFAULT_COMPRESS_CONFIG
    if DEFAULT_COMPRESS_CONFIG is None and CompressConfig is not None:
        DEFAULT_COMPRESS_CONFIG = CompressConfig(
            protect_recent=0,
            target_ratio=0.15,
            compress_user_messages=True,
            compress_system_messages=False,
        )
    return DEFAULT_COMPRESS_CONFIG


def headroom_available() -> bool:
    return compress is not None


def compress_messages(
    messages: list[dict[str, Any]],
    *,
    model: str = "gpt-4o",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run Headroom compress(); return (messages, stats dict)."""
    if compress is None:
        return messages, {
            "tokens_before": 0,
            "tokens_after": 0,
            "tokens_saved": 0,
            "compression_ratio": 0.0,
            "transforms_applied": [],
            "skipped": "headroom_not_installed",
        }

    cfg = _default_config()
    result = compress(messages, model=model, config=cfg)
    stats = {
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "tokens_saved": result.tokens_saved,
        "compression_ratio": result.compression_ratio,
        "transforms_applied": list(result.transforms_applied or []),
        "skipped": None,
    }
    return list(result.messages), stats


def compress_assistant_for_capture(
    system_prompt: str,
    user_msg: str,
    assistant_text: str,
    *,
    model: str = "gpt-4o",
) -> tuple[str, dict[str, Any]]:
    """Compress a planted turn's assistant text for smaller KV capture."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_text},
    ]
    compressed, stats = compress_messages(messages, model=model)
    if not compressed:
        return assistant_text, stats
    for msg in reversed(compressed):
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"]), stats
    return assistant_text, stats


def agent_noise_user_message(index: int, *, payload_rows: int = 200) -> str:
    """Synthetic agent turn: large tool JSON + short instruction (Headroom-friendly)."""
    rows = [
        {
            "id": i,
            "status": "ok" if i % 11 else "error",
            "latency_ms": 40 + (i % 17),
            "message": f"event-{index}-{i}: subsystem check detail payload",
        }
        for i in range(payload_rows)
    ]
    payload = json.dumps({"tool": "telemetry_scan", "turn": index + 1, "rows": rows})
    return (
        f"Tool `telemetry_scan` returned {len(rows)} rows. "
        f"Summarize anomalies in two short paragraphs.\n\n```json\n{payload}\n```"
    )


def _split_agent_json_block(user_msg: str) -> tuple[str, str] | None:
    """Split ``prefix + ```json ... ``` `` agent tool user messages."""
    marker = "```json\n"
    if marker not in user_msg:
        return None
    prefix, rest = user_msg.split(marker, 1)
    if "```" not in rest:
        return None
    body, _ = rest.split("```", 1)
    prefix = prefix.strip()
    body = body.strip()
    if not prefix or not body:
        return None
    return prefix, body


def _tool_name_from_prefix(prefix: str) -> str:
    if "`" in prefix:
        inner = prefix.split("`", 2)
        if len(inner) >= 2 and inner[1].strip():
            return inner[1].strip()
    return "tool_output"


def warmup_headroom(*, model: str = "gpt-4o") -> None:
    """Preload Headroom models before the first plant turn (avoids N7 race after N6)."""
    if compress is None:
        return
    compress_messages([{"role": "user", "content": "warmup"}], model=model)
    compress_user_payload(agent_noise_user_message(0, payload_rows=8), model=model)


def compress_user_payload(user_msg: str, *, model: str = "gpt-4o") -> tuple[str, dict[str, Any]]:
    """Compress a user message; agent tool JSON uses Headroom tool-role SmartCrusher."""
    split = _split_agent_json_block(user_msg)
    if split is None:
        compressed, stats = compress_messages([{"role": "user", "content": user_msg}], model=model)
        content = compressed[0].get("content") if compressed else user_msg
        return str(content or user_msg), stats

    prefix, json_body = split
    tool_name = _tool_name_from_prefix(prefix)
    tool_messages: list[dict[str, Any]] = [
        {"role": "user", "content": prefix},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "tc_agent_compress",
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "tc_agent_compress", "content": json_body},
    ]
    compressed, stats = compress_messages(tool_messages, model=model)
    compressed_json = json_body
    for msg in compressed:
        if msg.get("role") == "tool" and msg.get("content"):
            compressed_json = str(msg["content"])
            break
    # CCR hash stubs are for provider-side retrieve — keep inline JSON for self-hosted vLLM.
    if "Retrieve more: hash=" in compressed_json and "rows" not in compressed_json[:200]:
        compressed_json = json_body
        stats = {**stats, "skipped": "ccr_stub_fallback"}
    out = f"{prefix}\n\n```json\n{compressed_json}\n```"
    return out, stats


def is_agent_noise_turn(index: int, *, every: int = 4) -> bool:
    """Every Nth noise turn (1-based cycle) simulates a tool-output agent step."""
    return every > 0 and (index + 1) % every == 0


def mixed_noise_user_message(
    index: int,
    classic_prompt: str,
    *,
    agent_every: int = 4,
    payload_rows: int = 120,
) -> tuple[str, str]:
    """Return (user_message, kind) where kind is ``classic`` or ``agent``."""
    if is_agent_noise_turn(index, every=agent_every):
        return agent_noise_user_message(index, payload_rows=payload_rows), "agent"
    return classic_prompt, "classic"
