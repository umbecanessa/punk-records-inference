#!/usr/bin/env python3
"""Discriminate garbled RESUME failures: model stress vs inject / chain pollution.

Runs live A/B probes on an existing turn-sweep chain (must still be on disk).

Usage:
    python bench/tier1/garble_root_cause.py \\
        --from-sweep bench/results/.../turn_sweep_cp60_80_ropefix.json \\
        --base-url http://127.0.0.1:8000 \\
        --checkpoint 60
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
from sweep_lib import RECALL, SYSTEM_PROMPT, score_recall_clean, strip_think  # noqa: E402


def _recall_once(
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
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    kv: dict[str, str] = {
        "memory_user": user_id,
        "memory_ring": "general",
        "memory_no_capture": "1",
        "memory_base_session": base_session,
    }
    if arm == "text_off":
        kv["memory_off"] = "1"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    elif arm == "arm_d":
        kv["memory_inject_mode"] = "resume_overflow"
    else:
        kv["memory_inject_mode"] = "resume"
    if max_blocks > 0:
        kv["memory_resume_max_blocks"] = str(max_blocks)
    if max_inject_tokens > 0:
        kv["memory_resume_max_tokens"] = str(max_inject_tokens)

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.0,
        "kv_transfer_params": kv,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"garble_diag_{arm}_{uuid.uuid4().hex[:8]}",
    }
    t0 = time.perf_counter()
    try:
        response = requests.post(api, json=body, timeout=300)
        if response.status_code >= 400:
            return {
                "arm": arm,
                "pass_clean": False,
                "error": f"HTTP {response.status_code}",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        data = response.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()
        usage = data.get("usage") or {}
        scored = score_recall_clean(content, expected)
        return {
            "arm": arm,
            "question": question,
            **scored,
            "answer_preview": strip_think(content)[:160],
            "usage": usage,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as exc:
        return {
            "arm": arm,
            "pass_clean": False,
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


def _scan_chain_previews(
    api_root: str,
    user_id: str,
    base_session: str,
) -> dict[str, Any]:
    mems = fetch_user_memories(api_root, user_id, include_kv=True, limit=500)
    blocks = select_chain_latest(mems, base_session, k=10**9, max_tokens=0, roles=TURN_ROLES)
    suspicious: list[dict] = []
    meta_neutral: list[dict] = []
    for block in blocks:
        preview = (block.get("preview") or "").strip()
        turn = int(block.get("turnIndex") or -1)
        if not preview:
            continue
        row = {
            "turn_index": turn,
            "session_id": block.get("sessionId"),
            "num_tokens": block.get("numTokens"),
            "preview": preview[:120],
        }
        if is_garbled_response(preview):
            suspicious.append(row)
        low = preview.lower()
        if "technical glitch" in low or "trouble generating" in low or "testing the system" in low:
            meta_neutral.append(row)
    return {
        "blocks": len(blocks),
        "tokens": sum(int(b.get("numTokens") or 0) for b in blocks),
        "suspicious_previews": suspicious,
        "neutral_meta_previews": meta_neutral,
    }


def _interpret(experiments: dict[str, Any], sweep_row: dict | None) -> list[str]:
    lines: list[str] = []
    text_p = (sweep_row or {}).get("text_pass_clean")
    resume_p = (sweep_row or {}).get("resume_pass_clean")

    if text_p == 5 and resume_p is not None and resume_p < 5:
        lines.append(
            "TEXT 5/5 + RESUME <5 on same checkpoint: failure is inject-mediated decode, "
            "not the model lacking inline context."
        )

    iso = experiments.get("isolated_hotel_resume") or {}
    seq = experiments.get("sequential_through_hotel") or {}
    if iso.get("pass_clean") and not seq.get("pass_clean"):
        lines.append(
            "Hotel probe passes in isolation but fails after prior RESUME probes: "
            "suggests vLLM/GPU state or back-to-back inject pressure, not static chain pollution alone."
        )
    elif not iso.get("pass_clean") and not seq.get("pass_clean"):
        lines.append(
            "Hotel probe fails even in isolation: static chain inject state or model "
            "decode under ~17k inject tokens."
        )

    facts = experiments.get("resume_facts_only_3blocks") or {}
    full = experiments.get("resume_full_chain") or {}
    if facts.get("pass_clean") and not full.get("pass_clean"):
        lines.append(
            "RESUME with max_blocks=3 (facts only) passes but full chain fails: "
            "tail blocks (noise/neutral) pollute inject."
        )
    elif not facts.get("pass_clean") and not full.get("pass_clean"):
        lines.append(
            "Even facts-only inject fails: unlikely to be noise-tail pollution alone; "
            "check RoPE/inject geometry or probe difficulty."
        )

    repeats = experiments.get("hotel_resume_repeats") or []
    if len(repeats) >= 2:
        outcomes = [bool(r.get("pass_clean")) for r in repeats]
        if any(outcomes) and not all(outcomes):
            lines.append("Same hotel probe inconsistent across repeats: stochastic decode / GPU state.")
        elif not any(outcomes):
            lines.append("Hotel probe fails all repeats: stable failure mode under current inject.")

    scan = experiments.get("chain_scan") or {}
    if scan.get("suspicious_previews"):
        lines.append(
            f"Index previews flagged {len(scan['suspicious_previews'])} block(s) as garbled-looking "
            "(KV may still be clean if connector stripped decode at capture)."
        )
    if scan.get("neutral_meta_previews"):
        lines.append(
            f"{len(scan['neutral_meta_previews'])} neutral-substitute blocks in chain "
            "(meta 'glitch' text in history; TEXT arm still passes with same content inline)."
        )

    if not lines:
        lines.append("Inconclusive — inspect per-probe experiment rows.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-sweep", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("PRI_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--checkpoint", type=int, default=60)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sweep = json.loads(args.from_sweep.read_text(encoding="utf-8"))
    user_id = sweep["user_id"]
    base_session = sweep["base_session"]
    api_root = api_root_from_chat_url(args.base_url)
    api = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    model = sweep.get("model") or os.environ.get("PRI_MODEL", "/model")

    sweep_row = next(
        (r for r in (sweep.get("results") or []) if r.get("checkpoint_noise") == args.checkpoint),
        None,
    )
    hotel_q, hotel_exp = RECALL[3]
    work_q, work_exp = RECALL[4]

    print("=" * 72)
    print("GARBLE ROOT-CAUSE DIAGNOSTICS")
    print(f"chain user={user_id} base={base_session} cp={args.checkpoint}")
    if sweep_row:
        print(
            f"sweep scores: TEXT={sweep_row.get('text_pass_clean')}/5 "
            f"RESUME={sweep_row.get('resume_pass_clean')}/5 "
            f"inject={sweep_row.get('turn_tokens')}tok blocks={sweep_row.get('turn_blocks')}",
        )
    print("=" * 72)

    experiments: dict[str, Any] = {}

    experiments["chain_scan"] = _scan_chain_previews(api_root, user_id, base_session)
    scan = experiments["chain_scan"]
    print(
        f"\nChain scan: {scan['blocks']} blocks, {scan['tokens']} tok, "
        f"suspicious={len(scan['suspicious_previews'])} neutral_meta={len(scan['neutral_meta_previews'])}",
    )

    print("\n--- Isolated vs sequential (hotel probe) ---")
    experiments["isolated_hotel_resume"] = _recall_once(
        api, model, arm="resume", user_id=user_id, base_session=base_session,
        question=hotel_q, expected=hotel_exp,
    )
    print(f"  isolated RESUME hotel: pass_clean={experiments['isolated_hotel_resume'].get('pass_clean')}")

    # Warm sequential: run probes 1-3 then hotel (mirrors sweep order)
    for q, exp in RECALL[:3]:
        _recall_once(
            api, model, arm="resume", user_id=user_id, base_session=base_session,
            question=q, expected=exp,
        )
        time.sleep(0.3)
    experiments["sequential_through_hotel"] = _recall_once(
        api, model, arm="resume", user_id=user_id, base_session=base_session,
        question=hotel_q, expected=hotel_exp,
    )
    print(
        f"  after probes 1-3 RESUME hotel: "
        f"pass_clean={experiments['sequential_through_hotel'].get('pass_clean')}",
    )

    print("\n--- Inject scope (facts-only vs full) ---")
    experiments["resume_facts_only_3blocks"] = _recall_once(
        api, model, arm="resume", user_id=user_id, base_session=base_session,
        question=hotel_q, expected=hotel_exp, max_blocks=3,
    )
    experiments["resume_full_chain"] = experiments["isolated_hotel_resume"]
    print(
        f"  hotel max_blocks=3: pass_clean={experiments['resume_facts_only_3blocks'].get('pass_clean')}",
    )
    print(
        f"  hotel full chain:   pass_clean={experiments['resume_full_chain'].get('pass_clean')}",
    )

    experiments["work_resume_full"] = _recall_once(
        api, model, arm="resume", user_id=user_id, base_session=base_session,
        question=work_q, expected=work_exp,
    )
    experiments["work_arm_d_full"] = _recall_once(
        api, model, arm="arm_d", user_id=user_id, base_session=base_session,
        question=work_q, expected=work_exp,
    )
    print(
        f"  work RESUME: pass_clean={experiments['work_resume_full'].get('pass_clean')} "
        f"ARM-D: pass_clean={experiments['work_arm_d_full'].get('pass_clean')}",
    )

    print("\n--- Repeat hotel RESUME (3x, fresh cache_salt) ---")
    repeats: list[dict] = []
    for i in range(3):
        row = _recall_once(
            api, model, arm="resume", user_id=user_id, base_session=base_session,
            question=hotel_q, expected=hotel_exp,
        )
        repeats.append(row)
        print(f"  repeat {i + 1}: pass_clean={row.get('pass_clean')} garbled={row.get('garbled')}")
        time.sleep(0.5)
    experiments["hotel_resume_repeats"] = repeats

    interpretation = _interpret(experiments, sweep_row)

    report = {
        "sweep_json": str(args.from_sweep),
        "user_id": user_id,
        "base_session": base_session,
        "checkpoint": args.checkpoint,
        "sweep_row_scores": {
            "text_pass_clean": sweep_row.get("text_pass_clean") if sweep_row else None,
            "resume_pass_clean": sweep_row.get("resume_pass_clean") if sweep_row else None,
            "arm_d_pass_clean": sweep_row.get("arm_d_pass_clean") if sweep_row else None,
        },
        "experiments": experiments,
        "interpretation": interpretation,
    }

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    for line in interpretation:
        print(f"  • {line}")
    print("=" * 72)

    out_path = args.out or args.from_sweep.with_name(
        args.from_sweep.stem + f"_garble_cause_cp{args.checkpoint}.json",
    )
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
