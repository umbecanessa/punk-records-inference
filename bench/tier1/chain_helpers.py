"""Admin API + index.jsonl helpers for chain-of-latest bench scripts."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import requests


def fresh_chain_ids(prefix: str = "bench") -> tuple[str, str]:
    """Return a new ``(memory_user, memory_base_session)`` pair for one chain run.

    Each bench invocation should use a fresh pair so turn captures never stack on
    an old chain for the same user/session keys.
    """
    run_id = uuid.uuid4().hex[:10]
    return f"{prefix}_{run_id}", f"chain_thread_{uuid.uuid4().hex[:12]}"

DEFAULT_CHAIN_ROLES = frozenset({"user", "tool"})
TURN_ROLES = frozenset({"turn", "tool"})


def _default_index_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("NLS_MEMORY_INDEX", "").strip()
    if env:
        paths.append(Path(env))
    mem_dir = os.environ.get("NLS_MEMORY_DIR", "").strip()
    if mem_dir:
        paths.append(Path(mem_dir) / "index.jsonl")
    paths.extend([
        Path("/data/pri/index.jsonl"),
        Path("/data/pri/snapshot/index.jsonl"),
    ])
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _memories_from_index(user_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
    index_path: Path | None = None
    for candidate in _default_index_paths():
        if candidate.is_file():
            index_path = candidate
            break
    if index_path is None:
        return []

    rows: list[dict[str, Any]] = []
    with open(index_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if raw.get("user_id") != user_id:
                continue
            rows.append({
                "id": raw.get("id", ""),
                "sessionId": raw.get("session_id", ""),
                "role": raw.get("role") or "user",
                "timestamp": float(raw.get("timestamp") or 0),
                "preview": (raw.get("description") or "")[:120],
                "baseSessionId": raw.get("base_session_id") or "",
                "turnIndex": int(raw.get("turn_index", -1)),
                "prevHash": raw.get("prev_hash") or "",
                "kvPath": raw.get("kv_path") or "",
                "numTokens": int(raw.get("num_tokens") or 0),
            })

    rows.sort(key=lambda m: (int(m.get("turnIndex") or -1), float(m.get("timestamp") or 0)))
    if limit > 0:
        rows = rows[-limit:]
    return rows


def fetch_user_memories(
    api_root: str,
    user_id: str,
    *,
    include_kv: bool = True,
    limit: int = 500,
) -> list[dict[str, Any]]:
    params: dict[str, str] = {"user_id": user_id, "limit": str(limit)}
    if include_kv:
        params["include_kv"] = "1"
    try:
        response = requests.get(
            f"{api_root.rstrip('/')}/admin/memory/user-memories",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        memories = list(response.json().get("memories") or [])
    except Exception:
        memories = []

    if include_kv and (not memories or not any(m.get("kvPath") for m in memories)):
        indexed = _memories_from_index(user_id, limit=limit)
        if indexed:
            return indexed
    return memories


def select_chain_latest(
    memories: list[dict[str, Any]],
    base_session_id: str,
    *,
    k: int = 8,
    roles: frozenset[str] = DEFAULT_CHAIN_ROLES,
    max_tokens: int = 0,
) -> list[dict[str, Any]]:
    blocks = [
        m
        for m in memories
        if m.get("baseSessionId") == base_session_id
        and (m.get("role") or "user") in roles
        and m.get("kvPath")
        and int(m.get("numTokens") or 0) > 0
    ]
    if not blocks:
        return []

    latest_by_slot: dict[tuple[int, str], dict[str, Any]] = {}
    for block in blocks:
        turn = int(block.get("turnIndex") or -1)
        role = str(block.get("role") or "user")
        ts = float(block.get("timestamp") or 0)
        slot = (turn, role)
        prev = latest_by_slot.get(slot)
        if prev is None or ts >= float(prev.get("timestamp") or 0):
            latest_by_slot[slot] = block
    blocks = list(latest_by_slot.values())
    blocks.sort(key=lambda m: (int(m.get("turnIndex") or -1), float(m.get("timestamp") or 0)))

    if max_tokens <= 0:
        return blocks[-k:] if k > 0 else blocks

    selected: list[dict[str, Any]] = []
    budget = max_tokens
    for block in reversed(blocks):
        nt = int(block.get("numTokens") or 0)
        if nt <= 0:
            continue
        if selected and budget - nt < 0:
            break
        selected.append(block)
        budget -= nt
        if len(selected) >= k:
            break
    selected.reverse()
    return selected
