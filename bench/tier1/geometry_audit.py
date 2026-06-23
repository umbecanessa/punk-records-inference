#!/usr/bin/env python3
"""Offline inject-geometry audit for a resume chain thread.

Checks RoPE pack deltas, chain rope continuity, sys_prompt_hash consistency,
Mamba resume mode, and capture provenance for turn-level blocks.

Usage:
    python bench/tier1/geometry_audit.py \\
        --user-id turn_sweep_abc --base-session chain_thread_xyz \\
        --base-url http://127.0.0.1:8000

    python bench/tier1/geometry_audit.py \\
        --from-sweep bench/results/turn_sweep_cp20_80.json \\
        --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TIER1 = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_TIER1) not in sys.path:
    sys.path.insert(0, str(_TIER1))
if str(_REPO / "bench" / "opencode") not in sys.path:
    sys.path.insert(0, str(_REPO / "bench" / "opencode"))

from chain_helpers import fetch_user_memories, select_chain_latest, TURN_ROLES  # noqa: E402
from nls_kvp_helpers import api_root_from_chat_url, sys_prompt_hash  # noqa: E402
from pri.inject_geometry_audit import summarize_geometry_audit  # noqa: E402
from pri.resume import (  # noqa: E402
    ChainBlock,
    build_resume_inject_config,
    collect_chain_blocks,
    trim_chain_blocks,
)
from pri.store import MemoryStore  # noqa: E402
from sweep_lib import SYSTEM_PROMPT  # noqa: E402


def _load_store() -> MemoryStore | None:
    mem_dir = os.environ.get("NLS_MEMORY_DIR", "").strip()
    if not mem_dir:
        for candidate in (Path("/data/pri"), Path(os.environ.get("PRI_MEMORY_DIR", ""))):
            if candidate.is_dir():
                mem_dir = str(candidate)
                break
    if not mem_dir:
        return None
    try:
        return MemoryStore(mem_dir, readonly=True)
    except Exception as exc:
        print(f"MemoryStore unavailable: {exc}", file=sys.stderr)
        return None


def _blocks_from_admin(api_root: str, user_id: str, base_session: str) -> list[ChainBlock]:
    memories = fetch_user_memories(api_root, user_id, include_kv=True)
    raw = select_chain_latest(
        memories, base_session, k=10**9, max_tokens=0, roles=TURN_ROLES,
    )
    if not raw:
        raw = select_chain_latest(memories, base_session, k=10**9, max_tokens=0)
    return [
        ChainBlock(
            kv_path=block.get("kvPath") or "",
            num_tokens=int(block.get("numTokens") or 0),
            rope_start=0,
            turn_index=int(block.get("turnIndex") or -1),
            role=str(block.get("role") or "user"),
            session_id=str(block.get("sessionId") or ""),
        )
        for block in raw
    ]


def _snapshots_from_blocks(blocks: list[ChainBlock]) -> list[dict]:
    cfg = build_resume_inject_config(blocks)
    if not cfg:
        return []
    return list(cfg.get("snapshots") or [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", default="")
    parser.add_argument("--base-session", default="")
    parser.add_argument("--from-sweep", type=Path, default=None)
    parser.add_argument("--base-url", default=os.environ.get("PRI_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    user_id = args.user_id
    base_session = args.base_session
    if args.from_sweep:
        sweep = json.loads(args.from_sweep.read_text(encoding="utf-8"))
        user_id = user_id or sweep["user_id"]
        base_session = base_session or sweep["base_session"]

    if not user_id or not base_session:
        print("ERROR: --user-id and --base-session required (or --from-sweep)", file=sys.stderr)
        return 1

    api_root = api_root_from_chat_url(args.base_url)
    live_hash = sys_prompt_hash(SYSTEM_PROMPT)

    store = _load_store()
    blocks: list[ChainBlock] = []
    if store is not None:
        blocks = collect_chain_blocks(store, user_id, base_session, roles=TURN_ROLES)
        if not blocks:
            blocks = collect_chain_blocks(store, user_id, base_session)
    if not blocks:
        blocks = _blocks_from_admin(api_root, user_id, base_session)

    # Enrich admin-sourced blocks with manifest rope_start when store unavailable.
    if blocks and blocks[0].rope_start == 0:
        from pri.format import read_manifest

        enriched: list[ChainBlock] = []
        for block in blocks:
            manifest = read_manifest(block.kv_path) if block.kv_path else None
            rope_start = int((manifest or {}).get("rope_start") or 0)
            enriched.append(
                ChainBlock(
                    kv_path=block.kv_path,
                    num_tokens=block.num_tokens,
                    rope_start=rope_start,
                    turn_index=block.turn_index,
                    role=block.role,
                    session_id=block.session_id,
                    ring_type=block.ring_type,
                    meta_score=block.meta_score,
                ),
            )
        blocks = enriched

    full_blocks = len(blocks)
    full_tokens = sum(block.num_tokens for block in blocks)
    blocks = trim_chain_blocks(
        blocks, max_blocks=args.max_blocks, max_tokens=args.max_tokens,
    )
    snapshots = _snapshots_from_blocks(blocks)
    if not snapshots:
        print("No chain blocks / snapshots to audit", file=sys.stderr)
        return 1

    cfg = build_resume_inject_config(blocks) or {}
    summary = summarize_geometry_audit(
        snapshots,
        rope_offset=0,
        resume_mode=True,
        mamba_delta_sum=int(cfg.get("mamba_delta_sum", 3)),
        live_sys_hash=live_hash,
    )
    summary["user_id"] = user_id
    summary["base_session"] = base_session
    summary["trim"] = {
        "full_blocks": full_blocks,
        "audited_blocks": len(blocks),
        "full_tokens": full_tokens,
        "audited_tokens": summary["total_inject_tokens"],
    }

    out_path = args.out or Path("bench/results") / f"geometry_audit_{user_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "verdict": summary["verdict"],
        "findings": summary["findings"],
        "blocks": summary["block_count"],
        "tokens": summary["total_inject_tokens"],
        "saved": str(out_path),
    }, indent=2))

    return 0 if summary["verdict"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
