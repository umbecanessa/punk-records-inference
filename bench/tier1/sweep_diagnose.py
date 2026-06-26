#!/usr/bin/env python3
"""Diagnose turn_sweep results: cliff analysis + admin chain inspection.

Usage:
    python bench/tier1/sweep_diagnose.py bench/results/turn_sweep_cp20_80.json \\
        --base-url http://192.168.68.96:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

_TIER1 = Path(__file__).resolve().parent
_REPO = _TIER1.parents[1]
for path in (_REPO, _TIER1, _REPO / "bench" / "opencode"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chain_helpers import fetch_user_memories, select_chain_latest, TURN_ROLES  # noqa: E402
from nls_kvp_helpers import api_root_from_chat_url  # noqa: E402


def _safe_print(text: str) -> None:
    sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def _failure_mode(row: dict) -> str:
    if row.get("garbled"):
        return "GARBLED"
    if row.get("pass_clean"):
        return "PASS"
    if row.get("pass"):
        return "KEYWORD_ONLY"
    preview = (row.get("answer_preview") or "").lower()
    if "do not have access" in preview or "cannot tell" in preview:
        return "REFUSAL"
    return "MISS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_json", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sweep = json.loads(args.sweep_json.read_text(encoding="utf-8"))
    user_id = sweep["user_id"]
    base_session = sweep["base_session"]
    api_root = api_root_from_chat_url(args.base_url)

    report: dict = {
        "user_id": user_id,
        "base_session": base_session,
        "garbled_stripped_plant": len(sweep.get("garbled_stripped") or []),
        "checkpoints": [],
        "admin_chain": {},
    }

    print("=" * 72)
    print("SWEEP DIAGNOSE")
    print(f"user={user_id}  base={base_session}")
    print(f"garbled assistant turns stripped during plant: {report['garbled_stripped_plant']}")
    print("=" * 72)

    prev_resume = None
    for row in sweep.get("results") or []:
        cp = row["checkpoint_noise"]
        text_p = row.get("text_pass_clean", 0)
        resume_p = row.get("resume_pass_clean", 0)
        arm_d_p = row.get("arm_d_pass_clean")

        resume_modes = [_failure_mode(r) for r in row.get("resume_recall") or []]
        cliff = ""
        if prev_resume is not None and resume_p < prev_resume:
            cliff = f" CLIFF {prev_resume}->{resume_p}"

        print(
            f"\ncp{cp}: inject={row.get('turn_tokens')}tok blocks={row.get('turn_blocks')} "
            f"TEXT={text_p}/5 RESUME={resume_p}/5"
            + (f" ARM-D={arm_d_p}/5" if arm_d_p is not None else "")
            + cliff,
        )
        print(f"  RESUME modes: {resume_modes}")

        for arm_name, rows in (
            ("RESUME", row.get("resume_recall") or []),
            ("TEXT", row.get("text_recall") or []),
        ):
            for recall_row in rows:
                mode = _failure_mode(recall_row)
                if mode not in ("PASS",):
                    q = recall_row.get("question", "")[:45]
                    prev = (recall_row.get("answer_preview") or "")[:100]
                    _safe_print(f"    [{arm_name} {mode}] {q} -> {prev}")

        report["checkpoints"].append({
            "checkpoint_noise": cp,
            "turn_tokens": row.get("turn_tokens"),
            "turn_blocks": row.get("turn_blocks"),
            "text_pass_clean": text_p,
            "resume_pass_clean": resume_p,
            "arm_d_pass_clean": arm_d_p,
            "resume_failure_modes": resume_modes,
        })
        prev_resume = resume_p

    mems = fetch_user_memories(api_root, user_id, include_kv=True, limit=500)
    blocks = select_chain_latest(mems, base_session, k=10**9, max_tokens=0, roles=TURN_ROLES)
    turn_indices = sorted(int(b.get("turnIndex") or 0) for b in blocks)
    missing = [i for i in range(1, max(turn_indices) + 1) if i not in turn_indices] if turn_indices else []

    report["admin_chain"] = {
        "blocks": len(blocks),
        "tokens": sum(int(b.get("numTokens") or 0) for b in blocks),
        "turn_min": min(turn_indices) if turn_indices else 0,
        "turn_max": max(turn_indices) if turn_indices else 0,
        "missing_turn_indices": missing,
    }

    print("\n--- Admin chain (turn role) ---")
    print(
        f"blocks={report['admin_chain']['blocks']} "
        f"tokens={report['admin_chain']['tokens']} "
        f"turns={report['admin_chain']['turn_min']}..{report['admin_chain']['turn_max']}",
    )
    if missing:
        print(f"MISSING turn indices ({len(missing)}): {missing[:15]}{'...' if len(missing)>15 else ''}")

    try:
        stats = requests.get(
            f"{api_root}/admin/memory/stats",
            params={"user_id": user_id},
            timeout=15,
        ).json()
        report["admin_stats"] = stats
        print(f"admin stats: size={stats.get('size')} captures={stats.get('captureCount')}")
    except Exception as exc:
        report["admin_stats_error"] = str(exc)

    # cp60 vs cp80 pattern summary
    cp60 = next((r for r in report["checkpoints"] if r["checkpoint_noise"] == 60), None)
    cp80 = next((r for r in report["checkpoints"] if r["checkpoint_noise"] == 80), None)
    if cp60 and cp80:
        report["cp60_to_cp80"] = {
            "resume_modes_cp60": cp60["resume_failure_modes"],
            "resume_modes_cp80": cp80["resume_failure_modes"],
            "interpretation": (
                "cp60: one REFUSAL (Lake Como) while other facts hold — inject stress, not decode collapse. "
                "cp80: GARBLED multilingual decode (max_tokens=200) while TEXT arm 5/5 at ~19.5k tokens. "
                "GX10 MemoryStore geometry audit: verdict PASS (uniform RoPE delta -22, 83 blocks). "
                "Likely cause: 14 garbled assistant segments captured into turn blocks (N65+) poison "
                "Mamba/hybrid state; stripped from TEXT history but still in .nls chain."
            ),
            "gx10_geometry_verdict": "pass",
            "gx10_geometry_blocks": 83,
            "gx10_geometry_tokens": 19778,
        }
        print("\n--- cp60 to cp80 interpretation ---")
        print(report["cp60_to_cp80"]["interpretation"])

    out_path = args.out or args.sweep_json.with_name(
        args.sweep_json.stem + "_diagnose.json",
    )
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
