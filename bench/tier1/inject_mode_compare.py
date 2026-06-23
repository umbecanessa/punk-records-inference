#!/usr/bin/env python3
"""Compare inject modes: TEXT vs RESUME vs RESUME_OVERFLOW.

Plants Marco facts (+ optional noise turns), then scores recall on three
inject arms. Intended for GX10 A/B to pick the default ``NLS_API_INJECT_MODE``.

Usage:
    python bench/tier1/inject_mode_compare.py --base-url http://127.0.0.1:8000
    python bench/tier1/inject_mode_compare.py --noise-turns 12 --resume-max-tokens 4096

For overflow to exercise Swiss backfill, use ``--noise-turns 12`` and/or
``--resume-max-tokens`` so chain trim evicts older blocks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests

_TIER1 = Path(__file__).resolve().parent
if str(_TIER1) not in sys.path:
    sys.path.insert(0, str(_TIER1))

from marco_facts import (  # noqa: E402
    FACTS,
    RECALL,
    SYSTEM_PROMPT,
    chat,
    plant_facts,
    resolve_model,
)
from recall_helpers import score_recall_any  # noqa: E402

from sweep_lib import NOISE_BANK  # noqa: E402
from openrouter_client import chat as openrouter_chat, is_configured, resolve_model as openrouter_model  # noqa: E402


def _resolve_text_backend(explicit: str) -> str:
    mode = (explicit or os.environ.get("BENCH_TEXT_BACKEND", "")).strip().lower()
    if mode in ("openrouter", "or"):
        return "openrouter"
    if mode in ("local", "pri", "vllm"):
        return "local"
    if is_configured():
        return "openrouter"
    return "local"


def _usage_totals(rows: list[dict]) -> dict:
    prompt = completion = total = 0
    n = 0
    for row in rows:
        usage = row.get("usage") or {}
        if not usage:
            continue
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
        n += 1
    return {
        "requests": n,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _arm_kv(
    arm: str,
    *,
    user_id: str,
    base_session: str,
    resume_max_tokens: int,
    resume_max_blocks: int,
    swiss_always: bool,
) -> dict[str, str]:
    kv: dict[str, str] = {
        "memory_user": user_id,
        "memory_no_capture": "1",
    }
    if arm == "text":
        kv["memory_off"] = "1"
        return kv

    kv["memory_base_session"] = base_session
    if arm == "resume_overflow":
        kv["memory_inject_mode"] = "resume_overflow"
        if resume_max_tokens > 0:
            kv["memory_resume_max_tokens"] = str(resume_max_tokens)
        if resume_max_blocks > 0:
            kv["memory_resume_max_blocks"] = str(resume_max_blocks)
        if swiss_always:
            kv["memory_resume_swiss_always"] = "1"
    else:
        kv["memory_inject_mode"] = "resume"
    return kv


def run_recall_arm(
    api: str,
    model: str,
    *,
    arm: str,
    turns: list[tuple[str, str]],
    user_id: str,
    base_session: str,
    resume_max_tokens: int,
    resume_max_blocks: int,
    swiss_always: bool,
    text_backend: str = "local",
    openrouter_model_id: str | None = None,
) -> tuple[list[dict], dict]:
    results: list[dict] = []
    latencies_ms: list[float] = []

    for question, expected in RECALL:
        t0 = time.perf_counter()
        if arm == "text":
            messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            for user_msg, assistant_msg in turns:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": question})
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]

        kv = _arm_kv(
            arm,
            user_id=user_id,
            base_session=base_session,
            resume_max_tokens=resume_max_tokens,
            resume_max_blocks=resume_max_blocks,
            swiss_always=swiss_always,
        )

        try:
            if arm == "text" and text_backend == "openrouter":
                answer, usage = openrouter_chat(
                    messages,
                    model=openrouter_model_id,
                    max_tokens=200,
                    temperature=0.0,
                    user_id=user_id,
                )
            else:
                answer, usage = chat(api, model, messages, user_id=user_id, kv=kv)
            scored = score_recall_any(answer, expected)
            err = None
        except Exception as exc:
            answer = ""
            usage = None
            scored = {"hits": [], "misses": expected, "pass": False}
            err = str(exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        row = {
            "arm": arm,
            "backend": (
                "openrouter" if arm == "text" and text_backend == "openrouter"
                else "pri"
            ),
            "question": question,
            "expected": expected,
            "answer": answer[:500],
            "pass": scored["pass"],
            "hits": scored.get("hits", []),
            "misses": scored.get("misses", []),
            "usage": usage,
            "latency_ms": round(elapsed_ms, 1),
            "error": err,
        }
        results.append(row)
        mark = "PASS" if row["pass"] else "FAIL"
        print(f"  [{arm}] {mark} {question[:40]}... ({elapsed_ms:.0f}ms)")
        time.sleep(0.3)

    stats = {
        "latency_ms_mean": round(sum(latencies_ms) / max(len(latencies_ms), 1), 1),
        "latency_ms_p95": round(sorted(latencies_ms)[int(len(latencies_ms) * 0.95)] if latencies_ms else 0, 1),
        "usage": _usage_totals(results),
    }
    return results, stats


def plant_noise(
    api: str,
    model: str,
    *,
    user_id: str,
    base_session: str,
    start_turn: int,
    count: int,
) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for i in range(count):
        turn_index = start_turn + i
        noise = NOISE_BANK[i % len(NOISE_BANK)]
        kv = {
            "memory_user": user_id,
            "memory_ring": "general",
            "memory_base_session": base_session,
            "memory_session": f"{base_session}_t{turn_index}_user",
            "memory_turn_index": str(turn_index),
            "memory_block_role": "user",
            "memory_text": noise,
            "memory_inject_mode": "resume",
        }
        if turn_index == 1:
            kv["memory_silo"] = "1"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": noise},
        ]
        reply, _ = chat(api, model, messages, user_id=user_id, kv=kv)
        turns.append((noise, reply))
        print(f"  noise turn {turn_index}: {noise[:50]}...")
        time.sleep(0.3)
    return turns


def store_size_bytes(base_url: str, user_id: str) -> dict:
    """Best-effort store stats via admin API."""
    try:
        r = requests.get(
            f"{base_url.rstrip('/')}/admin/memory/stats",
            params={"user_id": user_id},
            timeout=15,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("PRI_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-turns", type=int, default=0)
    parser.add_argument("--resume-max-tokens", type=int, default=0)
    parser.add_argument("--resume-max-blocks", type=int, default=0)
    parser.add_argument("--swiss-always", action="store_true")
    parser.add_argument(
        "--text-backend",
        choices=("auto", "openrouter", "local"),
        default="auto",
        help="TEXT arm backend: openrouter=isolated cloud baseline; local=same vLLM",
    )
    parser.add_argument(
        "--openrouter-model",
        default=os.environ.get("OPENROUTER_MODEL", ""),
        help="OpenRouter model slug (default: qwen/qwen3.5-35b-a3b)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    text_backend = _resolve_text_backend(
        "" if args.text_backend == "auto" else args.text_backend
    )
    or_model = args.openrouter_model.strip() or None

    api = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    model = os.environ.get("PRI_MODEL") or resolve_model(args.base_url)
    user_id = f"bench_mode_cmp_{args.seed}"
    base_session = f"chain_mode_cmp_{args.seed}"

    print("=" * 72)
    print("Inject mode compare (TEXT / RESUME / RESUME_OVERFLOW)")
    print(f"  API:    {api}")
    print(f"  model:  {model}")
    print(f"  user:   {user_id}")
    print(f"  noise:  {args.noise_turns} turns")
    print(f"  TEXT:   {text_backend}" + (
        f" ({openrouter_model(or_model)})" if text_backend == "openrouter" else ""
    ))
    print("=" * 72)

    print("\n-- Planting facts --")
    turns = plant_facts(api, model, user_id, base_session)

    if args.noise_turns > 0:
        print(f"\n-- Planting {args.noise_turns} noise turns --")
        noise_turns = plant_noise(
            api, model,
            user_id=user_id,
            base_session=base_session,
            start_turn=len(FACTS) + 1,
            count=args.noise_turns,
        )
        turns.extend(noise_turns)

    arms = ("text", "resume", "resume_overflow")
    arm_results: dict[str, list[dict]] = {}
    arm_stats: dict[str, dict] = {}

    for arm in arms:
        print(f"\n-- Recall {arm.upper()} arm --")
        rows, stats = run_recall_arm(
            api, model,
            arm=arm,
            turns=turns,
            user_id=user_id,
            base_session=base_session,
            resume_max_tokens=args.resume_max_tokens,
            resume_max_blocks=args.resume_max_blocks,
            swiss_always=args.swiss_always,
            text_backend=text_backend,
            openrouter_model_id=or_model,
        )
        arm_results[arm] = rows
        arm_stats[arm] = stats

    summary = {
        arm: {
            "pass": sum(1 for r in arm_results[arm] if r["pass"]),
            "total": len(arm_results[arm]),
            **arm_stats[arm],
        }
        for arm in arms
    }

    payload = {
        "timestamp": time.time(),
        "seed": args.seed,
        "user_id": user_id,
        "base_session": base_session,
        "model": model,
        "text_backend": text_backend,
        "openrouter_model": openrouter_model(or_model) if text_backend == "openrouter" else None,
        "plant_turns": len(turns),
        "noise_turns": args.noise_turns,
        "resume_max_tokens": args.resume_max_tokens,
        "resume_max_blocks": args.resume_max_blocks,
        "swiss_always": args.swiss_always,
        "summary": summary,
        "results": arm_results,
        "store_stats": store_size_bytes(args.base_url, user_id),
    }

    out_path = args.out or Path("bench/results") / f"inject_mode_compare_{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    for arm in arms:
        s = summary[arm]
        print(
            f"{arm.upper():16} {s['pass']}/{s['total']}  "
            f"tokens={s['usage'].get('total_tokens', 0)}  "
            f"lat_mean={s['latency_ms_mean']}ms"
        )
    print(f"Wrote {out_path}")
    print("=" * 72)

    best = max(
        ("resume", "resume_overflow"),
        key=lambda a: (summary[a]["pass"], -summary[a]["usage"].get("total_tokens", 0)),
    )
    print(f"Suggested default (recall-first): {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
