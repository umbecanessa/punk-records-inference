#!/usr/bin/env python3
"""Tier-1 Marco facts: TEXT vs RESUME recall on a live vLLM instance.

Plants three fact turns, adds optional noise, then probes recall on both arms:
  - text   — full inline message history, memory_off=1
  - resume — latest user message only, memory_inject_mode=resume

Usage:
    python bench/tier1/marco_facts.py --base-url http://127.0.0.1:8000
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

from recall_helpers import score_recall_any

SYSTEM_PROMPT = (
    "You are a personal assistant with persistent memory. "
    "Answer from prior conversation context when available."
)

FACTS = [
    "My name is Marco and I live in Milan, Italy. I work as an architect.",
    "I have a golden retriever named Luna. She's 3 years old and loves swimming.",
    "Last weekend I went to Lake Como with my wife Sofia. We stayed at Hotel Bellagio.",
]

RECALL = [
    ("What's my name and where do I live?", ["Marco", "Milan"]),
    ("What's my dog's name?", ["Luna"]),
    ("Where did I go last weekend and who with?", ["Lake Como", "Sofia"]),
    ("What hotel did I stay at?", ["Bellagio"]),
    ("What do I do for work?", ["architect"]),
]


def chat(
    api: str,
    model: str,
    messages: list[dict],
    *,
    user_id: str,
    kv: dict[str, str],
    max_tokens: int = 200,
) -> tuple[str, dict | None]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "user": user_id,
        "kv_transfer_params": kv,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"pri_bench_{uuid.uuid4().hex[:8]}",
    }
    r = requests.post(api, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"] or ""
    return content, data.get("usage")


def resolve_model(base_url: str) -> str:
    r = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=15)
    r.raise_for_status()
    models = r.json().get("data") or []
    if not models:
        raise RuntimeError("no models from /v1/models")
    return models[0]["id"]


def plant_facts(
    api: str,
    model: str,
    user_id: str,
    base_session: str,
) -> list[tuple[str, str]]:
    """Return list of (user_text, assistant_text) turns."""
    turns: list[tuple[str, str]] = []
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, fact in enumerate(FACTS, start=1):
        messages.append({"role": "user", "content": fact})
        kv = {
            "memory_user": user_id,
            "memory_ring": "general",
            "memory_base_session": base_session,
            "memory_session": f"{base_session}_t{i}_user",
            "memory_turn_index": str(i),
            "memory_block_role": "user",
            "memory_text": fact,
        }
        if i == 1:
            kv["memory_silo"] = "1"
        if i > 1:
            kv["memory_inject_mode"] = "resume"
        reply, _ = chat(api, model, messages, user_id=user_id, kv=kv)
        messages.append({"role": "assistant", "content": reply})
        turns.append((fact, reply))
        print(f"  planted turn {i}: {fact[:60]}...")
        time.sleep(0.5)

    return turns


def run_recall_arm(
    api: str,
    model: str,
    user_id: str,
    base_session: str,
    turns: list[tuple[str, str]],
    arm: str,
) -> list[dict]:
    results: list[dict] = []
    for question, expected in RECALL:
        if arm == "text":
            messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            for u, a in turns:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
            messages.append({"role": "user", "content": question})
            kv = {
                "memory_user": user_id,
                "memory_off": "1",
                "memory_no_capture": "1",
            }
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            kv = {
                "memory_user": user_id,
                "memory_base_session": base_session,
                "memory_inject_mode": "resume",
                "memory_no_capture": "1",
            }

        try:
            answer, usage = chat(api, model, messages, user_id=user_id, kv=kv)
            scored = score_recall_any(answer, expected)
        except Exception as exc:
            answer = ""
            usage = None
            scored = {"hits": [], "misses": expected, "pass": False, "error": str(exc)}

        row = {
            "arm": arm,
            "question": question,
            "expected": expected,
            "answer": answer[:500],
            "pass": scored["pass"],
            "hits": scored.get("hits", []),
            "misses": scored.get("misses", []),
            "usage": usage,
        }
        results.append(row)
        mark = "PASS" if row["pass"] else "FAIL"
        print(f"  [{arm}] {mark} {question[:40]}...")
        time.sleep(0.3)

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("PRI_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    api = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    model = os.environ.get("PRI_MODEL") or resolve_model(args.base_url)
    user_id = f"bench_marco_{args.seed}"
    base_session = f"chain_marco_{args.seed}"

    print("=" * 72)
    print("Tier-1 Marco facts (TEXT vs RESUME)")
    print(f"  API:   {api}")
    print(f"  model: {model}")
    print(f"  user:  {user_id}")
    print("=" * 72)

    print("\n-- Planting facts --")
    turns = plant_facts(api, model, user_id, base_session)

    print("\n-- Recall TEXT arm --")
    text_results = run_recall_arm(api, model, user_id, base_session, turns, "text")

    print("\n-- Recall RESUME arm --")
    resume_results = run_recall_arm(api, model, user_id, base_session, turns, "resume")

    text_pass = sum(1 for r in text_results if r["pass"])
    resume_pass = sum(1 for r in resume_results if r["pass"])
    payload = {
        "timestamp": time.time(),
        "seed": args.seed,
        "user_id": user_id,
        "base_session": base_session,
        "model": model,
        "text_pass": text_pass,
        "text_total": len(text_results),
        "resume_pass": resume_pass,
        "resume_total": len(resume_results),
        "text_results": text_results,
        "resume_results": resume_results,
    }

    out_path = args.out or Path("bench/results") / f"tier1_marco_facts_{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"TEXT   {text_pass}/{len(text_results)}")
    print(f"RESUME {resume_pass}/{len(resume_results)}")
    print(f"Wrote {out_path}")
    print("=" * 72)

    return 0 if resume_pass >= text_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
