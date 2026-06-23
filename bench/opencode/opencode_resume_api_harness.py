#!/usr/bin/env python3
"""OpenCode-style multi-turn harness through Punk Records API (NestJS path).

Quick smoke: plant non-default stack facts, one confirm turn, recall on same chain.

Usage:
    set PUNK_API_KEY=nls_live_...
    python -u scripts/opencode_resume_api_harness.py
    python -u scripts/opencode_resume_api_harness.py --seed 42

Env: PUNK_API_BASE, PUNK_API_KEY
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from opencode_harness_lib import (
    API_KEY,
    API_BASE,
    BANNED_PORTS,
    SYSTEM_PROMPT,
    agent_turn,
    generate_session_facts,
    is_garbled_response,
    safe_print,
    score_recall,
    stream_chat,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Planted-facts RNG seed")
    parser.add_argument("--wipe-user", action="store_true", help="hint for GX10 wipe")
    args = parser.parse_args()
    if args.wipe_user:
        safe_print("Note: wipe via scripts/nls_wipe_user_captures.py on GX10 before a clean run")

    if not API_KEY:
        safe_print("ERROR: set PUNK_API_KEY")
        return 2

    facts = generate_session_facts(args.seed)
    banned_str = [str(p) for p in BANNED_PORTS]
    recall_probes = [
        (
            "What exact backend port did we pick? Number only.",
            [str(facts.backend_port)],
            banned_str + [str(facts.frontend_port)],
        ),
        (
            "What exact frontend port did we pick? Number only.",
            [str(facts.frontend_port)],
            banned_str + [str(facts.backend_port)],
        ),
        (
            "What is the dev database name we chose? Name only.",
            [facts.db_name],
            ["icf_coaching_dev", "3000", "3001"],
        ),
    ]

    safe_print("=" * 72)
    safe_print("OpenCode resume API harness (Punk Records -> vLLM)")
    safe_print(f"API: {API_BASE}")
    chain_id = f"harness_{uuid.uuid4().hex[:12]}"
    safe_print(f"chain: {chain_id}  seed: {args.seed}")
    safe_print(
        f"planted: backend={facts.backend_port} frontend={facts.frontend_port} "
        f"db={facts.db_name}"
    )
    safe_print("=" * 72)

    transcript: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    safe_print("\n-- Turn 1: plant DECISIONs --")
    plant_msg = (
        "Phase 1 scaffold for the ICF Coaching Eval Tool. No tool calls yet.\n"
        "Reply in 3 short bullets, then end with EXACTLY these DECISION lines:\n"
        f"{facts.as_decision_block()}"
    )
    agent_turn(transcript, label="turn1", user_text=plant_msg, chain_id=chain_id)

    safe_print("\n-- Turn 2: confirm stack --")
    agent_turn(
        transcript,
        label="turn2",
        user_text=(
            "Confirm backend port, frontend port, and database name in one sentence. "
            "No tool calls."
        ),
        chain_id=chain_id,
    )

    safe_print("\n-- Recall probes (same chain, full resend + resume) --")
    passed = 0
    for i, (q, must, must_not) in enumerate(recall_probes, 1):
        safe_print(f"\n>> Q{i}")
        transcript.append({"role": "user", "content": q})
        ans, nls = stream_chat(
            transcript,
            label=f"Q{i}",
            max_tokens=80,
            chain_id=chain_id,
            agent_mode=True,
        )
        clean = ans.strip()
        if is_garbled_response(clean):
            clean = ""
        transcript.append({"role": "assistant", "content": clean})
        ok = score_recall(clean, must, must_not)
        passed += int(ok)
        mark = "PASS" if ok else "FAIL"
        safe_print(f"  {mark} expected={must}")
        if not ok:
            safe_print(f"       got: {clean[:200]}")

    safe_print("\n" + "=" * 72)
    safe_print(f"RECALL: {passed}/{len(recall_probes)}")
    safe_print("=" * 72)
    return 0 if passed == len(recall_probes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
