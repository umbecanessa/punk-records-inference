#!/usr/bin/env python3
"""Long OpenCode-style agent session through Punk Records API or direct vLLM.

Builds a multi-turn transcript on one chain (full resend each HTTP call),
plants **non-default** stack facts (seeded random ports/names), runs several
work turns, then probes recall. Strong pass = exact weird values recalled
after many turns — not 3000/3001 guesses.

Usage:
    # Direct vLLM (GX10 / local compose):
    python -u bench/opencode/opencode_long_session_harness.py --base-url http://127.0.0.1:8000

    # Hosted Punk Records API:
    set PUNK_API_KEY=nls_live_...
    python -u bench/opencode/opencode_long_session_harness.py

Env: PRI_BASE_URL, PRI_API, PUNK_API_BASE, PUNK_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from opencode_harness_lib import (
    BANNED_PORTS,
    SYSTEM_PROMPT,
    SessionFacts,
    agent_turn,
    generate_session_facts,
    get_api_base,
    get_harness_config,
    is_garbled_response,
    reset_harness_config,
    safe_print,
    score_recall,
    stream_chat,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results"


def build_work_script(facts: SessionFacts) -> list[tuple[str, str]]:
    """Ordered user turns simulating a longer agentic session."""
    d = facts.as_decision_block()
    return [
        (
            "plant",
            "Phase 1 scaffold for the ICF Coaching Eval Tool. No tool calls yet.\n"
            "Reply in 3 short bullets, then end with EXACTLY these DECISION lines "
            "(copy verbatim, do not substitute common ports like 3000 or 3001):\n"
            f"{d}",
        ),
        (
            "confirm",
            "Good. In one sentence, confirm the stack including backend port, "
            "frontend port, metrics port, redis port, database name, and API prefix. "
            "No tool calls.",
        ),
        (
            "api_layout",
            "Outline the backend folder layout under apps/backend/src (routes, services, "
            "db). Mention which port the HTTP server binds to. No tool calls.",
        ),
        (
            "env_draft",
            f"Draft a .env.example with PORT, METRICS_PORT, REDIS_URL, DATABASE_URL "
            f"using our chosen values only. No tool calls.",
        ),
        (
            "tangent_git",
            "Unrelated: should we use squash merge or merge commits for feature branches? "
            "Two sentences max. No tool calls.",
        ),
        (
            "frontend_proxy",
            "How should the frontend dev server proxy API requests given our api path "
            "prefix? One short paragraph. No tool calls.",
        ),
        (
            "docker_compose",
            "Sketch docker-compose service ports for postgres, redis, backend, frontend "
            "using OUR assigned ports from DECISION lines — not defaults. No tool calls.",
        ),
        (
            "pre_recall_check",
            "Before we run tests: list every numeric port and the exact database name "
            "and api prefix from our DECISION lines only. Bullet list, no tool calls.",
        ),
    ]


def build_recall_probes(facts: SessionFacts) -> list[tuple[str, list[str], list[str]]]:
    banned_str = [str(p) for p in BANNED_PORTS]
    return [
        (
            f"What exact backend port did we pick? Reply with the number only.",
            [str(facts.backend_port)],
            banned_str + [str(facts.frontend_port)],
        ),
        (
            f"What exact frontend port did we pick? Reply with the number only.",
            [str(facts.frontend_port)],
            banned_str + [str(facts.backend_port)],
        ),
        (
            f"What is the dev database name we chose? Reply with the name only.",
            [facts.db_name],
            ["icf_coaching_dev", "postgres", "3000", "3001"],
        ),
        (
            f"What metrics port did we assign? Reply with the number only.",
            [str(facts.metrics_port)],
            banned_str,
        ),
        (
            f"What redis port did we assign? Reply with the number only.",
            [str(facts.redis_port)],
            ["6379"],
        ),
        (
            f"What is our API path prefix? Reply with the path only.",
            [facts.api_prefix],
            ["/api", "/v1"],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Long OpenCode NLS session harness")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("PRI_BASE_URL", ""),
        help="Direct vLLM base URL (sets PRI_BASE_URL for harness lib)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(datetime.now(timezone.utc).strftime("%H%M%S")),
        help="RNG seed for non-default planted facts (default: time-based)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds between turns (default 1.0)",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_DIR / "opencode_long_session.json"),
        help="JSON results path",
    )
    args = parser.parse_args()

    if args.base_url:
        os.environ["PRI_BASE_URL"] = args.base_url.rstrip("/")
        reset_harness_config()

    chat_url, api_key, direct_vllm = get_harness_config()
    if not direct_vllm and not api_key:
        safe_print("ERROR: set PUNK_API_KEY for hosted API, or --base-url for direct vLLM")
        return 2

    facts = generate_session_facts(args.seed)
    chain_id = f"long_{uuid.uuid4().hex[:12]}"
    bench_user = f"opencode_{args.seed}"
    work = build_work_script(facts)
    probes = build_recall_probes(facts)

    safe_print("=" * 72)
    safe_print("OpenCode LONG session harness (Punk Records -> vLLM)")
    safe_print(f"API: {chat_url} (direct_vllm={direct_vllm})")
    safe_print(f"chain_id: {chain_id}")
    safe_print(f"seed: {args.seed}")
    safe_print("Planted facts (non-default — recall must match exactly):")
    safe_print(f"  backend={facts.backend_port} frontend={facts.frontend_port} "
               f"metrics={facts.metrics_port} redis={facts.redis_port}")
    safe_print(f"  db={facts.db_name} api_prefix={facts.api_prefix}")
    safe_print("=" * 72)

    transcript: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    turn_log: list[dict[str, Any]] = []

    safe_print(f"\n-- Work phase ({len(work)} turns) --")
    for label, user_text in work:
        safe_print(f"\n>> {label}")
        text, nls = agent_turn(
            transcript,
            label=label,
            user_text=user_text,
            chain_id=chain_id,
            max_tokens=500 if label in ("plant", "env_draft", "docker_compose") else 350,
            pause_s=args.pause,
            user_id=bench_user,
        )
        turn_log.append({
            "phase": "work",
            "label": label,
            "turn_index": (nls or {}).get("turn_index"),
            "chain_id": (nls or {}).get("chain_id"),
            "response_preview": text[:300],
        })

    safe_print(f"\n-- Recall phase ({len(probes)} probes) --")
    recall_results: list[dict[str, Any]] = []
    passed = 0
    for i, (question, must, must_not) in enumerate(probes, 1):
        safe_print(f"\n>> recall_{i}")
        transcript.append({"role": "user", "content": question})
        ans, nls = stream_chat(
            transcript,
            label=f"recall_{i}",
            max_tokens=80,
            chain_id=chain_id,
            agent_mode=True,
            tool_choice="none",
            user_id=bench_user,
        )
        clean = ans.strip()
        if is_garbled_response(clean):
            clean = ""
        transcript.append({"role": "assistant", "content": clean})
        ok = score_recall(clean, must, must_not)
        passed += int(ok)
        mark = "PASS" if ok else "FAIL"
        safe_print(f"  {mark} expected={must} forbidden={must_not[:4]}...")
        if not ok:
            safe_print(f"       got: {clean[:200]}")
        recall_results.append({
            "question": question,
            "must": must,
            "must_not": must_not,
            "answer": clean,
            "pass": ok,
            "turn_index": (nls or {}).get("turn_index"),
        })
        if args.pause > 0:
            time.sleep(args.pause)

    total_probes = len(probes)
    safe_print("\n" + "=" * 72)
    safe_print(f"RECALL: {passed}/{total_probes}")
    safe_print(f"Transcript messages: {len(transcript)}")
    safe_print("=" * 72)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chain_id": chain_id,
        "seed": args.seed,
        "facts": {
            "backend_port": facts.backend_port,
            "frontend_port": facts.frontend_port,
            "metrics_port": facts.metrics_port,
            "redis_port": facts.redis_port,
            "db_name": facts.db_name,
            "api_prefix": facts.api_prefix,
        },
        "work_turns": len(work),
        "recall_passed": passed,
        "recall_total": total_probes,
        "recall_results": recall_results,
        "turn_log": turn_log,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    safe_print(f"Wrote {out_path}")

    return 0 if passed == total_probes else 1


if __name__ == "__main__":
    raise SystemExit(main())
