#!/usr/bin/env python3
"""Probe OpenRouter request shapes for Qwen3.5 TEXT baseline (no reasoning burn).

Compares payloads against bench defaults to find a variant that returns visible
content without empty answers when max_tokens=200.

Usage (GX10 or any host with bench/.env):
    python bench/tier1/openrouter_probe.py
    python bench/tier1/openrouter_probe.py --variant reasoning_none
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

_TIER1 = Path(__file__).resolve().parent
if str(_TIER1) not in sys.path:
    sys.path.insert(0, str(_TIER1))

from openrouter_client import (  # noqa: E402
    OPENROUTER_APP_TITLE,
    OPENROUTER_CHAT_URL,
    OPENROUTER_HTTP_REFERER,
    resolve_model,
)

SYSTEM_PROMPT = (
    "You are a personal assistant with persistent memory. "
    "Answer from prior conversation context when available."
)

# Short inline history (Marco facts) — similar to inject_mode_compare TEXT arm.
SHORT_MESSAGES: list[dict[str, str]] = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "My name is Marco and I live in Milan, Italy. I work as an architect."},
    {"role": "assistant", "content": "Nice to meet you, Marco! Milan is a beautiful city for architecture."},
    {"role": "user", "content": "I have a golden retriever named Luna. She's 3 years old and loves swimming."},
    {"role": "assistant", "content": "Luna sounds wonderful — golden retrievers usually love the water!"},
    {"role": "user", "content": "What's my dog's name?"},
]

# Long inline history (~4k tokens) — reproduces long-chain TEXT failures.
LONG_MESSAGES: list[dict[str, str]] = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "My name is Marco and I live in Milan, Italy. I work as an architect."},
    {"role": "assistant", "content": "Nice to meet you, Marco!"},
    {"role": "user", "content": "I have a golden retriever named Luna. She's 3 years old and loves swimming."},
    {"role": "assistant", "content": "Luna sounds wonderful!"},
    {"role": "user", "content": "Last weekend I went to Lake Como with my wife Sofia. We stayed at Hotel Bellagio."},
    {"role": "assistant", "content": "Lake Como is gorgeous — Hotel Bellagio is a great choice."},
]
for i in range(4, 16):
    LONG_MESSAGES.extend([
        {
            "role": "user",
            "content": f"Noise turn {i}: explain topic {i} in two sentences for a general audience.",
        },
        {
            "role": "assistant",
            "content": f"Topic {i} is interesting; here is a brief two-sentence summary for turn {i}.",
        },
    ])
LONG_MESSAGES.append({"role": "user", "content": "What's my dog's name?"})


def _headers() -> dict[str, str]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_TITLE:
        headers["X-Title"] = OPENROUTER_APP_TITLE
    return headers


def variant_bodies(model: str, max_tokens: int) -> dict[str, dict[str, Any]]:
    """Named request bodies to compare."""
    base = {
        "model": model,
        "messages": SHORT_MESSAGES,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "user": "openrouter_probe",
    }
    return {
        "bench_current": {
            **base,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "plain_minimal": dict(base),
        "extra_body_ctk": {
            **base,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "reasoning_none": {
            **base,
            "reasoning": {"effort": "none"},
        },
        "reasoning_exclude": {
            **base,
            "reasoning": {"effort": "none", "exclude": True},
        },
        "include_reasoning_false": {
            **base,
            "include_reasoning": False,
        },
        "ctk_and_reasoning_none": {
            **base,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning": {"effort": "none"},
        },
        "extra_body_both": {
            **base,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning": {"effort": "none"},
            },
        },
        "enable_thinking_top": {
            **base,
            "enable_thinking": False,
        },
        "max_tokens_512": {
            **base,
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 512,
        },
        "reasoning_budget_0": {
            **base,
            "reasoning": {"max_tokens": 0},
        },
    }


def call_variant(name: str, body: dict[str, Any], *, timeout: int = 120) -> dict[str, Any]:
    t0 = time.perf_counter()
    response = requests.post(
        OPENROUTER_CHAT_URL,
        headers=_headers(),
        json=body,
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    row: dict[str, Any] = {
        "variant": name,
        "status_code": response.status_code,
        "elapsed_ms": round(elapsed_ms, 1),
    }
    if not response.ok:
        row["error"] = response.text[:500]
        return row

    data = response.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens", 0)

    row.update({
        "content_len": len(content),
        "content_preview": content[:120].replace("\n", " "),
        "reasoning_field_len": len(reasoning) if isinstance(reasoning, str) else 0,
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "has_luna": "luna" in content.lower(),
        "finish_reason": choice.get("finish_reason"),
        "message_keys": sorted(message.keys()),
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe OpenRouter Qwen3.5 request shapes")
    parser.add_argument("--model", default="", help="OpenRouter model slug")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--variant", default="", help="Run one variant only")
    parser.add_argument(
        "--history",
        choices=("short", "long"),
        default="short",
        help="Message history length",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write JSON results")
    args = parser.parse_args()

    model = resolve_model(args.model or None)
    messages = SHORT_MESSAGES if args.history == "short" else LONG_MESSAGES
    bodies = variant_bodies(model, args.max_tokens)
    for body in bodies.values():
        body["messages"] = messages

    names = [args.variant] if args.variant else list(bodies.keys())
    if args.variant and args.variant not in bodies:
        print(f"Unknown variant: {args.variant}", file=sys.stderr)
        return 2

    print(f"OpenRouter probe model={model} history={args.history} max_tokens={args.max_tokens}")
    print(f"messages={len(messages)} turns")
    print("=" * 72)

    results: list[dict[str, Any]] = []
    for name in names:
        print(f"\n--- {name} ---")
        row = call_variant(name, bodies[name])
        results.append(row)
        if row.get("error"):
            print(f"  HTTP {row['status_code']}: {row['error'][:200]}")
            continue
        print(
            f"  content_len={row['content_len']} reasoning_tokens={row['reasoning_tokens']} "
            f"completion={row['completion_tokens']} luna={row['has_luna']} "
            f"({row['elapsed_ms']}ms)"
        )
        print(f"  preview: {row.get('content_preview', '')!r}")
        print(f"  message_keys: {row.get('message_keys')}")
        time.sleep(0.5)

    print("\n" + "=" * 72)
    print("SUMMARY (content_len / reasoning_tokens / has_luna):")
    for row in results:
        if row.get("error"):
            print(f"  {row['variant']:28} ERROR {row['status_code']}")
        else:
            print(
                f"  {row['variant']:28} "
                f"content={row['content_len']:3} "
                f"reasoning_tok={row.get('reasoning_tokens', '?'):>4} "
                f"luna={row['has_luna']}"
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")

    ok = sum(1 for r in results if r.get("has_luna"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
