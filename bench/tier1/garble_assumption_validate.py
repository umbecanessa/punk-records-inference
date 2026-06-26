#!/usr/bin/env python3
"""Validate garble hypotheses on an existing chain (no replant).

Discriminates:
  A) Inject path vs cold (memory_off) on the same question
  B) Inject scope: max_blocks ladder (facts-only vs tail vs full)
  C) Inject budget: max_tokens ladder (3k / 8k / 17k / 28k / unlimited)
  D) Chain profile: block count, neutral stubs, large agent blocks

Usage:
    python bench/tier1/garble_assumption_validate.py \\
        --user-id hr_sweep_8a9d65f547 \\
        --base-session chain_thread_dae1b8c9f4df \\
        --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_TIER1 = Path(__file__).resolve().parent
_REPO = _TIER1.parents[1]
for path in (_REPO, _REPO / "bench" / "opencode", _TIER1):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import requests

from chain_helpers import fetch_user_memories, select_chain_latest, TURN_ROLES  # noqa: E402
from nls_kvp_helpers import api_root_from_chat_url  # noqa: E402
from pri.text_quality import is_garbled_response  # noqa: E402
from sweep_lib import (  # noqa: E402
    RECALL,
    SYSTEM_PROMPT,
    apply_resume_inject_caps,
    inline_messages,
    resolve_model,
    score_recall_clean,
    strip_think,
)

NEUTRAL_MARKERS = (
    "trouble generating a response",
    "technical glitch",
    "testing the system",
)
LARGE_BLOCK_TOKENS = 1000
SMALL_BLOCK_TOKENS = 120


def _probe(
    api: str,
    model: str,
    *,
    arm: str,
    user_id: str,
    base_session: str,
    question: str,
    expected: list[str],
    max_blocks: int = 0,
    max_inject_tokens: int = 0,
    inline_turns: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    kv: dict[str, str] = {
        "memory_user": user_id,
        "memory_ring": "general",
        "memory_no_capture": "1",
    }
    if arm == "cold":
        kv["memory_off"] = "1"
    elif arm == "text_inline" and inline_turns is not None:
        kv["memory_off"] = "1"
        messages = inline_messages(inline_turns, question)
    elif arm == "arm_d":
        kv["memory_inject_mode"] = "resume_overflow"
        kv["memory_base_session"] = base_session
    else:
        kv["memory_inject_mode"] = "resume"
        kv["memory_base_session"] = base_session

    if max_blocks > 0:
        kv["memory_resume_max_blocks"] = str(max_blocks)
    if max_inject_tokens > 0:
        kv["memory_resume_max_tokens"] = str(max_inject_tokens)

    if arm in ("resume", "arm_d") and max_inject_tokens <= 0:
        kv = apply_resume_inject_caps(kv)

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.0,
        "kv_transfer_params": kv,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"garble_val_{arm}_{uuid.uuid4().hex[:8]}",
    }
    t0 = time.perf_counter()
    try:
        response = requests.post(api, json=body, timeout=300)
        if response.status_code >= 400:
            return {
                "arm": arm,
                "pass_clean": False,
                "garbled": False,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        data = response.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()
        scored = score_recall_clean(content, expected)
        return {
            "arm": arm,
            "question": question,
            **scored,
            "answer_preview": strip_think(content)[:160],
            "usage": data.get("usage") or {},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "max_blocks": max_blocks,
            "max_inject_tokens": max_inject_tokens,
        }
    except Exception as exc:
        return {
            "arm": arm,
            "pass_clean": False,
            "garbled": False,
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "max_blocks": max_blocks,
            "max_inject_tokens": max_inject_tokens,
        }


def _chain_profile(api_root: str, user_id: str, base_session: str) -> dict[str, Any]:
    mems = fetch_user_memories(api_root, user_id, include_kv=True, limit=500)
    blocks = select_chain_latest(mems, base_session, k=10**9, max_tokens=0, roles=TURN_ROLES)
    turns = sorted(int(b.get("turnIndex") or -1) for b in blocks)
    token_sizes = [int(b.get("numTokens") or 0) for b in blocks]
    neutral_turns: list[int] = []
    large_turns: list[int] = []
    stub_turns: list[int] = []
    garbled_previews: list[int] = []
    by_turn: list[dict[str, Any]] = []

    for block in blocks:
        turn = int(block.get("turnIndex") or -1)
        preview = (block.get("preview") or "").strip()
        ntok = int(block.get("numTokens") or 0)
        row = {
            "turn_index": turn,
            "num_tokens": ntok,
            "preview_head": preview[:80],
        }
        by_turn.append(row)
        low = preview.lower()
        if any(m in low for m in NEUTRAL_MARKERS):
            neutral_turns.append(turn)
        if ntok >= LARGE_BLOCK_TOKENS:
            large_turns.append(turn)
        if ntok <= SMALL_BLOCK_TOKENS:
            stub_turns.append(turn)
        if preview and is_garbled_response(preview):
            garbled_previews.append(turn)

    return {
        "block_count": len(blocks),
        "total_tokens": sum(token_sizes),
        "turn_min": turns[0] if turns else None,
        "turn_max": turns[-1] if turns else None,
        "turn_gaps": len(set(turns)) != len(turns),
        "neutral_turns": neutral_turns,
        "large_block_turns": large_turns,
        "stub_turns": stub_turns,
        "garbled_preview_turns": garbled_previews,
        "token_p50": sorted(token_sizes)[len(token_sizes) // 2] if token_sizes else 0,
        "token_max": max(token_sizes) if token_sizes else 0,
        "blocks_by_turn": by_turn,
    }


def _turns_from_chain(api_root: str, user_id: str, base_session: str) -> list[tuple[str, str]]:
    mems = fetch_user_memories(api_root, user_id, include_kv=True, limit=500)
    blocks = select_chain_latest(mems, base_session, k=10**9, max_tokens=0, roles=TURN_ROLES)
    blocks.sort(key=lambda b: int(b.get("turnIndex") or 0))
    turns: list[tuple[str, str]] = []
    for block in blocks:
        preview = (block.get("preview") or "").strip()
        if not preview:
            continue
        parts = preview.split("\n", 1)
        if len(parts) == 2:
            turns.append((parts[0], parts[1]))
        else:
            turns.append((preview, "Acknowledged."))
    return turns


def _interpret(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    profile = report.get("chain_profile") or {}
    cold = report.get("cold_vs_resume") or {}
    blocks_l = report.get("blocks_ladder") or []
    tokens_l = report.get("tokens_ladder") or []

    cold_r = cold.get("cold") or {}
    resume_r = cold.get("resume_full") or {}
    text_r = cold.get("text_inline") or {}

    if text_r.get("pass_clean") and not resume_r.get("pass_clean"):
        lines.append(
            "VERIFIED inject-path: TEXT inline passes, full RESUME fails — not missing model knowledge."
        )
    if text_r.get("pass_clean") and resume_r.get("garbled"):
        lines.append(
            "VERIFIED garble is inject-mediated: TEXT clean, RESUME garbled on same facts."
        )
    if cold_r.get("pass_clean") is False and cold_r.get("garbled"):
        lines.append("Cold (no inject) can garble too — check question difficulty / model state.")
    elif not resume_r.get("garbled") and resume_r.get("pass_clean"):
        lines.append("Full RESUME currently clean on hotel probe — chain may not be broken at recall.")

    facts_rows = [r for r in blocks_l if r.get("max_blocks") == 3]
    full_rows = [r for r in blocks_l if r.get("max_blocks") == 0]
    if facts_rows and full_rows:
        f, full = facts_rows[0], full_rows[0]
        if f.get("pass_clean") and not full.get("pass_clean"):
            lines.append(
                "VERIFIED tail pollution: facts-only (3 blocks) passes, full chain fails."
            )
        elif not f.get("pass_clean") and not full.get("pass_clean"):
            lines.append(
                "NOT tail pollution alone: facts-only (3 blocks) also fails — "
                "look at chain-wide state or probe difficulty, not inject size."
            )
        elif f.get("pass_clean") and full.get("pass_clean"):
            lines.append("Both facts-only and full chain pass — prior plant garble may not affect recall.")

    if len(tokens_l) >= 2:
        small = next((r for r in tokens_l if r.get("max_inject_tokens") == 3000), None)
        large = next((r for r in tokens_l if r.get("max_inject_tokens") == 0), None)
        if small and large:
            if small.get("garbled") == large.get("garbled") and small.get("pass_clean") == large.get("pass_clean"):
                lines.append(
                    "NOT inject token budget: 3k-cap and unlimited inject give same outcome."
                )
            elif small.get("pass_clean") and not large.get("pass_clean"):
                lines.append(
                    "VERIFIED inject size matters: small cap passes, unlimited fails."
                )

    neutrals = profile.get("neutral_turns") or []
    if neutrals and not lines:
        lines.append(
            f"Chain has {len(neutrals)} neutral stub turn(s) {neutrals[:8]} — "
            "correlate with blocks_ladder if tail fails but facts pass."
        )
    if profile.get("garbled_preview_turns"):
        lines.append(
            f"Index previews garbled on turns {profile['garbled_preview_turns'][:8]} "
            "(preview text only; KV may still be usable)."
        )
    if not lines:
        lines.append("Inconclusive — inspect experiment rows in JSON output.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--base-session", required=True)
    parser.add_argument("--base-url", default=os.environ.get("PRI_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    api_root = api_root_from_chat_url(args.base_url)
    api = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    model = os.environ.get("PRI_MODEL") or resolve_model(args.base_url)
    hotel_q, hotel_exp = RECALL[3]
    marco_q, marco_exp = RECALL[0]

    print("=" * 72)
    print("GARBLE ASSUMPTION VALIDATION")
    print(f"user={args.user_id} base={args.base_session}")
    print("=" * 72)

    profile = _chain_profile(api_root, args.user_id, args.base_session)
    print(
        f"\nChain: {profile['block_count']} blocks, {profile['total_tokens']} tok, "
        f"turns {profile['turn_min']}..{profile['turn_max']}",
    )
    print(
        f"  neutral={len(profile['neutral_turns'])} large(>{LARGE_BLOCK_TOKENS})="
        f"{len(profile['large_block_turns'])} stubs(<={SMALL_BLOCK_TOKENS})="
        f"{len(profile['stub_turns'])} garbled_previews={len(profile['garbled_preview_turns'])}",
    )
    if profile["neutral_turns"]:
        print(f"  neutral turns: {profile['neutral_turns']}")

    inline_turns = _turns_from_chain(api_root, args.user_id, args.base_session)
    print(f"  inline turns rebuilt: {len(inline_turns)}")

    print("\n--- A) Cold vs RESUME vs TEXT inline (hotel) ---")
    cold_vs: dict[str, Any] = {
        "cold": _probe(
            api, model, arm="cold", user_id=args.user_id, base_session=args.base_session,
            question=hotel_q, expected=hotel_exp,
        ),
        "resume_full": _probe(
            api, model, arm="resume", user_id=args.user_id, base_session=args.base_session,
            question=hotel_q, expected=hotel_exp,
        ),
        "text_inline": _probe(
            api, model, arm="text_inline", user_id=args.user_id, base_session=args.base_session,
            question=hotel_q, expected=hotel_exp, inline_turns=inline_turns,
        ),
    }
    for label, row in cold_vs.items():
        print(
            f"  {label}: pass_clean={row.get('pass_clean')} garbled={row.get('garbled')} "
            f"preview={row.get('answer_preview', row.get('error', ''))[:60]!r}",
        )
    time.sleep(0.5)

    print("\n--- B) max_blocks ladder (hotel) ---")
    blocks_ladder: list[dict[str, Any]] = []
    for mb in (3, 5, 10, 20, 40, 0):
        row = _probe(
            api, model, arm="resume", user_id=args.user_id, base_session=args.base_session,
            question=hotel_q, expected=hotel_exp, max_blocks=mb,
        )
        blocks_ladder.append(row)
        print(
            f"  max_blocks={mb or 'all'}: pass_clean={row.get('pass_clean')} "
            f"garbled={row.get('garbled')}",
        )
        time.sleep(0.4)

    print("\n--- C) max_inject_tokens ladder (hotel) ---")
    tokens_ladder: list[dict[str, Any]] = []
    for mt in (3000, 8000, 17000, 28000, 0):
        row = _probe(
            api, model, arm="resume", user_id=args.user_id, base_session=args.base_session,
            question=hotel_q, expected=hotel_exp, max_inject_tokens=mt,
        )
        tokens_ladder.append(row)
        print(
            f"  max_tokens={mt or 'unlimited'}: pass_clean={row.get('pass_clean')} "
            f"garbled={row.get('garbled')}",
        )
        time.sleep(0.4)

    print("\n--- D) Full RECALL matrix (resume full chain) ---")
    recall_resume: list[dict[str, Any]] = []
    for q, exp in RECALL:
        row = _probe(
            api, model, arm="resume", user_id=args.user_id, base_session=args.base_session,
            question=q, expected=exp,
        )
        recall_resume.append(row)
        print(
            f"  {q[:45]}... pass_clean={row.get('pass_clean')} garbled={row.get('garbled')}",
        )
        time.sleep(0.3)

    report = {
        "user_id": args.user_id,
        "base_session": args.base_session,
        "model": model,
        "chain_profile": profile,
        "cold_vs_resume": cold_vs,
        "blocks_ladder": blocks_ladder,
        "tokens_ladder": tokens_ladder,
        "recall_resume_full": recall_resume,
    }
    report["interpretation"] = _interpret(report)

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    for line in report["interpretation"]:
        print(f"  • {line}")
    print("=" * 72)

    out = args.out or (
        _TIER1.parent / "results" / f"garble_assumption_{args.user_id}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
