#!/usr/bin/env python3
"""Invalidation tests — Disease A (capture geometry) + Disease B (Mamba modes).

Hypotheses:
  A1  Resume prefilled capture saves full turn KV (not ~240 tok short).
  A2  Healthy N7 block does not poison N8 (0/N garbled on probe).
  B0  Mode 0 (genesis-only Mamba) < mode 1 on classic homogeneous chain.
  B1  Mode 1 best on classic chain @ ~17k inject (historical default).
  B2  Mode 2 (genesis + last delta) ≥ mode 1 on v14 large-block chain.
  B3  Mode 0 or 3 fixes v14 full-pack hotel recall if Mamba is Disease B.

Pass-2 note: turn capture mode skips Pass-2 loopback; stored block SSM is
end-of-request readback. Inject modes 0–3 are *replay* strategies, not Pass-2.

Usage:
  python bench/tier1/mamba_invalidation.py --base-url http://127.0.0.1:8000
  python bench/tier1/mamba_invalidation.py --skip-capture --chains v14
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_TIER1 = Path(__file__).resolve().parent
_REPO = _TIER1.parents[1]
for path in (_REPO, _REPO / "bench" / "opencode", _TIER1):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import requests

from headroom_helpers import mixed_noise_user_message, warmup_headroom
from headroom_sweep_lib import plant_turn_headroom
from nls_kvp_helpers import api_root_from_chat_url
from pri.text_quality import is_garbled_response
from probe_garble_isolate import (
    fetch_chain_blocks,
    plant_prefix,
    probe_decode,
    wait_for_chain_turn_block,
)
from sweep_lib import (
    RECALL,
    SYSTEM_PROMPT,
    apply_resume_inject_caps,
    noise_prompt,
    resolve_model,
    score_recall_clean,
    strip_think,
)

MAMBA_MODES = {
    0: "genesis_only",
    1: "genesis_plus_sum_deltas",
    2: "genesis_plus_last_delta",
    3: "last_block_verbatim",
}

CHAINS = {
    "classic": (
        "hr_sweep_12c6137747",
        "chain_thread_eceebeab8bfc",
        "headroom classic ~17k homogeneous",
    ),
    "v14": (
        "hr_sweep_8a9d65f547",
        "chain_thread_dae1b8c9f4df",
        "mixed sweep w/ ~2.6k agent blocks",
    ),
}

HEALTHY_N7_MIN = 2500  # full-size capture without headroom compression
POISON_N7_MAX = 2450  # v13 double-strip poison ceiling
HEADROOM_N7_MIN = 1200  # compressed agent block still carries assistant KV tail


def _mode_label(mamba_mode: int | None) -> tuple[int, str]:
    if mamba_mode is None:
        return -1, "env_default"
    return mamba_mode, MAMBA_MODES.get(mamba_mode, f"mode_{mamba_mode}")


@dataclass
class RecallRow:
    hypothesis: str
    chain: str
    mamba_mode: int
    mamba_name: str
    question: str
    pass_recall: bool
    garbled: bool
    answer_preview: str
    latency_ms: float
    error: str = ""


def _recall_probe(
    api: str,
    model: str,
    *,
    hypothesis: str,
    chain_key: str,
    user_id: str,
    base_session: str,
    question: str,
    expected: list[str],
    mamba_mode: int | None = None,
) -> RecallRow:
    kv: dict[str, str] = {
        "memory_user": user_id,
        "memory_ring": "general",
        "memory_no_capture": "1",
        "memory_inject_mode": "resume",
        "memory_base_session": base_session,
    }
    kv = apply_resume_inject_caps(kv)
    if mamba_mode is not None:
        kv["memory_mamba_mode"] = str(mamba_mode)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": 160,
        "temperature": 0.0,
        "kv_transfer_params": kv,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"inv_{hypothesis}_{mamba_mode}_{uuid.uuid4().hex[:8]}",
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(api, json=body, timeout=300)
        if r.status_code >= 400:
            mid, mname = _mode_label(mamba_mode)
            return RecallRow(
                hypothesis, chain_key, mid, mname,
                question, False, False, "", 0,
                error=f"HTTP {r.status_code}",
            )
        content = strip_think(r.json()["choices"][0]["message"]["content"] or "")
        scored = score_recall_clean(content, expected)
        mid, mname = _mode_label(mamba_mode)
        return RecallRow(
            hypothesis, chain_key, mid, mname,
            question,
            scored["pass"],
            scored.get("garbled") or is_garbled_response(content),
            content[:140],
            round((time.perf_counter() - t0) * 1000, 1),
        )
    except Exception as exc:
        mid, mname = _mode_label(mamba_mode)
        return RecallRow(
            hypothesis, chain_key, mid, mname,
            question, False, False, "", 0, error=str(exc),
        )


def _disease_a_capture(api: str, model: str, *, trials: int) -> dict[str, Any]:
    """Plant N7 under mixed-sweep recipe; check block size + N8 probe."""
    warmup_headroom()
    n7_msg, _ = mixed_noise_user_message(3, noise_prompt(3), agent_every=4)
    n8_msg = noise_prompt(4)
    api_root = api_root_from_chat_url(api)
    api_chat = f"{api.rstrip('/')}/v1/chat/completions"
    rows: list[dict[str, Any]] = []
    for i in range(trials):
        uid = f"inv_a_{uuid.uuid4().hex[:8]}"
        base = f"chain_{uuid.uuid4().hex[:12]}"
        try:
            prev6 = plant_prefix(model, uid, base, through_turn=6)
            r = plant_turn_headroom(
                api_chat, model, api_root,
                user_id=uid, base_session=base, turn_index=7,
                prev_hash=prev6, user_msg=n7_msg, compress_user=True,
                compress_assistant=True, max_garbled_retries=4,
            )
            n7 = wait_for_chain_turn_block(uid, base, 7, api_root=api_root)
            n7_tok = int(n7.get("numTokens") or 0) if n7 else 0
            n8 = probe_decode(
                model, uid, base, turn_index=8, prev_hash=r.new_prev_hash,
                user_msg=n8_msg, trials=2,
            )
            rows.append({
                "trial": i,
                "n7_tokens": n7_tok,
                "healthy_size": n7_tok >= HEADROOM_N7_MIN and n7_tok > POISON_N7_MAX,
                "poison_size": n7_tok <= POISON_N7_MAX,
                "still_garbled": r.still_garbled,
                "n8_garbled_2": n8.garbled,
                "pass_a1": (
                    n7_tok > POISON_N7_MAX
                    and n7_tok >= HEADROOM_N7_MIN
                    and not r.still_garbled
                ),
                "pass_a2": n8.garbled == 0,
            })
        except Exception as exc:
            rows.append({"trial": i, "error": str(exc)})
        time.sleep(0.5)
    ok_a1 = sum(1 for r in rows if r.get("pass_a1"))
    ok_a2 = sum(1 for r in rows if r.get("pass_a2"))
    tok_rows = [r.get("n7_tokens") for r in rows if "n7_tokens" in r]
    return {
        "trials": trials,
        "pass_a1_full_capture": f"{ok_a1}/{trials}",
        "pass_a2_n8_clean": f"{ok_a2}/{trials}",
        "token_min": min(tok_rows) if tok_rows else None,
        "token_max": max(tok_rows) if tok_rows else None,
        "rows": rows,
        "verdict_a1": ok_a1 == trials,
        "verdict_a2": ok_a2 == trials,
    }


def _mamba_matrix(
    api: str,
    model: str,
    chain_keys: list[str],
    modes: list[int],
) -> list[RecallRow]:
    hotel_q, hotel_exp = RECALL[3]
    rows: list[RecallRow] = []
    for ck in chain_keys:
        user_id, base_session, _ = CHAINS[ck]
        for mode in modes:
            label = f"B{mode}_{ck}"
            rows.append(_recall_probe(
                api, model,
                hypothesis=label,
                chain_key=ck,
                user_id=user_id,
                base_session=base_session,
                question=hotel_q,
                expected=hotel_exp,
                mamba_mode=mode,
            ))
            time.sleep(0.8)
        # env default (no override)
        rows.append(_recall_probe(
            api, model,
            hypothesis=f"B_default_{ck}",
            chain_key=ck,
            user_id=user_id,
            base_session=base_session,
            question=hotel_q,
            expected=hotel_exp,
            mamba_mode=None,
        ))
        time.sleep(0.8)
    return rows


def _interpret(
    disease_a: dict[str, Any] | None,
    mamba_rows: list[RecallRow],
) -> dict[str, Any]:
    by_chain_mode: dict[str, dict[int, RecallRow]] = {}
    for r in mamba_rows:
        by_chain_mode.setdefault(r.chain, {})[r.mamba_mode] = r

    classic = by_chain_mode.get("classic", {})
    v14 = by_chain_mode.get("v14", {})

    def _pass(row: RecallRow | None) -> bool:
        return bool(row and row.pass_recall and not row.garbled and not row.error)

    out: dict[str, Any] = {
        "disease_a": disease_a,
        "hypotheses": {},
    }
    if disease_a:
        out["hypotheses"]["A1_full_capture"] = {
            "claim": "N7 prefilled capture is not truncated (~2398 poison size)",
            "result": disease_a.get("verdict_a1"),
            "evidence": disease_a.get("pass_a1_full_capture"),
        }
        out["hypotheses"]["A2_n8_after_n7"] = {
            "claim": "Healthy N7 does not poison N8 decode",
            "result": disease_a.get("verdict_a2"),
            "evidence": disease_a.get("pass_a2_n8_clean"),
        }

    if classic:
        out["hypotheses"]["B1_mode1_classic"] = {
            "claim": "Mode 1 passes hotel recall on classic chain",
            "result": _pass(classic.get(1)),
            "mode0_pass": _pass(classic.get(0)),
            "mode2_pass": _pass(classic.get(2)),
            "mode3_pass": _pass(classic.get(3)),
        }
    if v14:
        m1 = _pass(v14.get(1))
        m0 = _pass(v14.get(0))
        m2 = _pass(v14.get(2))
        m3 = _pass(v14.get(3))
        out["hypotheses"]["B_disease_b_mamba"] = {
            "claim": "v14 failure is Mamba replay (mode 0/2/3 beats mode 1)",
            "mode1_pass": m1,
            "mode0_pass": m0,
            "mode2_pass": m2,
            "mode3_pass": m3,
            "mamba_likely_if": (not m1) and (m0 or m2 or m3),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--chains", default="classic,v14")
    parser.add_argument("--modes", default="0,1,2,3")
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--capture-trials", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    api = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    model = resolve_model(args.base_url)
    chain_keys = [c.strip() for c in args.chains.split(",") if c.strip()]
    modes = [int(m) for m in args.modes.split(",") if m.strip().isdigit()]

    disease_a = None
    if not args.skip_capture:
        print("=== Disease A: N7 capture geometry ===", flush=True)
        disease_a = _disease_a_capture(api, model, trials=args.capture_trials)
        print(json.dumps(disease_a, indent=2), flush=True)

    print("=== Disease B: Mamba mode matrix (hotel recall) ===", flush=True)
    mamba_rows = _mamba_matrix(api, model, chain_keys, modes)
    for r in mamba_rows:
        status = "PASS" if r.pass_recall and not r.garbled else "FAIL"
        if r.garbled:
            status = "GARBLED"
        if r.error:
            status = f"ERR:{r.error[:40]}"
        print(
            f"  {r.hypothesis} mode={r.mamba_mode} ({r.mamba_name}) "
            f"{status} {r.answer_preview[:80]!r}",
            flush=True,
        )

    report = {
        "model": model,
        "disease_a": disease_a,
        "mamba_rows": [asdict(r) for r in mamba_rows],
        "interpretation": _interpret(disease_a, mamba_rows),
        "mamba_mode_legend": MAMBA_MODES,
        "pass2_note": (
            "Turn capture skips Pass-2; modes 0-3 are inject-time SSM replay only."
        ),
    }
    out = args.out or (_TIER1.parent / "results" / "mamba_invalidation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}", flush=True)
    print(json.dumps(report["interpretation"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
