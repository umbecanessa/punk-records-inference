#!/usr/bin/env python3
"""KV composition invalidation — prefix vs agent blocks vs single-block.

Mamba modes 0–3 are flat on v14; this isolates *which attention KV blocks*
cause hotel recall failure.

Uses ``memory_force_inject`` with ``inject_layout=resume`` (bypasses chain
trim-from-tail semantics of ``memory_resume_max_tokens``).

Usage:
  python bench/tier1/kv_composition_invalidation.py \\
      --base-url http://127.0.0.1:8000
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

from chain_helpers import fetch_user_memories, select_chain_latest, TURN_ROLES  # noqa: E402
from headroom_helpers import mixed_noise_user_message, warmup_headroom
from headroom_sweep_lib import plant_turn_headroom
from nls_kvp_helpers import api_root_from_chat_url, enrich_kv_params
from probe_garble_isolate import fetch_chain_blocks, plant_prefix, probe_decode, wait_for_chain_turn_block
from pri.text_quality import is_garbled_response
from sweep_lib import (
    RECALL,
    SYSTEM_PROMPT,
    apply_resume_inject_caps,
    noise_prompt,
    resolve_model,
    score_recall_clean,
    strip_think,
)

V14_USER = "hr_sweep_8a9d65f547"
V14_BASE = "chain_thread_dae1b8c9f4df"
CLASSIC_USER = "hr_sweep_12c6137747"
CLASSIC_BASE = "chain_thread_eceebeab8bfc"
HOTEL_Q, HOTEL_EXP = RECALL[3]
LARGE_TOK = 1000


@dataclass
class ProbeRow:
    label: str
    chain: str
    block_count: int
    inject_tokens: int
    large_blocks: int
    pass_recall: bool
    garbled: bool
    preview: str
    error: str = ""


def _snapshots(blocks: list[dict]) -> list[dict]:
    return [
        {"path": b["kvPath"], "num_tokens": int(b["numTokens"] or 0)}
        for b in blocks
        if b.get("kvPath") and int(b.get("numTokens") or 0) > 0
    ]


def _all_chain_blocks(
    api_root: str,
    user_id: str,
    base_session: str,
) -> list[dict]:
    mems = fetch_user_memories(api_root, user_id, include_kv=True, limit=500)
    return select_chain_latest(
        mems, base_session, k=0, max_tokens=0, roles=TURN_ROLES,
    )


def _blocks_through_turn(
    api_root: str,
    user_id: str,
    base_session: str,
    through_turn: int,
) -> list[dict]:
    mems = fetch_user_memories(api_root, user_id, include_kv=True, limit=500)
    all_blocks = select_chain_latest(
        mems, base_session, k=0, max_tokens=0, roles=TURN_ROLES,
    )
    return [
        b for b in all_blocks
        if 0 < int(b.get("turnIndex") or -1) <= through_turn
    ]


def _block_at_turn(
    api_root: str,
    user_id: str,
    base_session: str,
    turn: int,
) -> list[dict]:
    mems = fetch_user_memories(api_root, user_id, include_kv=True, limit=500)
    all_blocks = select_chain_latest(
        mems, base_session, k=0, max_tokens=0, roles=TURN_ROLES,
    )
    return [b for b in all_blocks if int(b.get("turnIndex") or -1) == turn]


def _force_probe(
    api: str,
    model: str,
    *,
    label: str,
    chain: str,
    blocks: list[dict],
    api_root: str,
) -> ProbeRow:
    snaps = _snapshots(blocks)
    inject_tok = sum(s["num_tokens"] for s in snaps)
    large = sum(1 for b in blocks if int(b.get("numTokens") or 0) >= LARGE_TOK)
    kv: dict[str, str] = {
        "memory_user": "force_inv",
        "memory_ring": "general",
        "memory_no_capture": "1",
        "memory_force_inject": json.dumps(snaps),
        "memory_inject_layout": "resume",
    }
    kv = enrich_kv_params(kv, SYSTEM_PROMPT, api_root=api_root, model=model)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": HOTEL_Q},
        ],
        "max_tokens": 160,
        "temperature": 0.0,
        "kv_transfer_params": kv,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"kvcomp_{label}_{uuid.uuid4().hex[:8]}",
    }
    try:
        r = requests.post(api, json=body, timeout=300)
        if r.status_code >= 400:
            return ProbeRow(
                label, chain, len(snaps), inject_tok, large,
                False, False, "", error=f"HTTP {r.status_code}: {r.text[:120]}",
            )
        text = strip_think(r.json()["choices"][0]["message"]["content"] or "")
        scored = score_recall_clean(text, HOTEL_EXP)
        return ProbeRow(
            label, chain, len(snaps), inject_tok, large,
            scored["pass"],
            scored.get("garbled") or is_garbled_response(text),
            text[:140],
        )
    except Exception as exc:
        return ProbeRow(
            label, chain, len(snaps), inject_tok, large,
            False, False, "", error=str(exc),
        )


def _resume_probe(
    api: str,
    model: str,
    *,
    label: str,
    chain: str,
    user_id: str,
    base_session: str,
    api_root: str,
) -> ProbeRow:
    kv: dict[str, str] = apply_resume_inject_caps({
        "memory_user": user_id,
        "memory_ring": "general",
        "memory_no_capture": "1",
        "memory_inject_mode": "resume",
        "memory_base_session": base_session,
    })
    # Do NOT enrich_kv_params here: memory_sys_prompt_hash triggers system-block
    # prepend which breaks recall on chains that fit the trim budget (classic proof).
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": HOTEL_Q},
        ],
        "max_tokens": 160,
        "temperature": 0.0,
        "kv_transfer_params": kv,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"kvcomp_{label}_{uuid.uuid4().hex[:8]}",
    }
    try:
        r = requests.post(api, json=body, timeout=300)
        if r.status_code >= 400:
            return ProbeRow(label, chain, -1, -1, -1, False, False, "",
                           error=f"HTTP {r.status_code}")
        text = strip_think(r.json()["choices"][0]["message"]["content"] or "")
        scored = score_recall_clean(text, HOTEL_EXP)
        return ProbeRow(
            label, chain, -1, -1, -1,
            scored["pass"],
            scored.get("garbled") or is_garbled_response(text),
            text[:140],
        )
    except Exception as exc:
        return ProbeRow(label, chain, -1, -1, -1, False, False, "", error=str(exc))


def _disease_a(api: str, model: str, api_root: str) -> dict[str, Any]:
    warmup_headroom()
    n7_msg, _ = mixed_noise_user_message(3, noise_prompt(3), agent_every=4)
    n8_msg = noise_prompt(4)
    uid = f"kv_a_{uuid.uuid4().hex[:8]}"
    base = f"chain_{uuid.uuid4().hex[:12]}"
    prev6 = plant_prefix(model, uid, base, through_turn=6)
    r = plant_turn_headroom(
        api, model, api_root,
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
    return {
        "n7_tokens": n7_tok,
        "still_garbled": r.still_garbled,
        "n8_garbled_2": n8.garbled,
        "healthy": n7_tok > 2450 and n7_tok >= 1200 and not r.still_garbled,
        "n8_clean": n8.garbled == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--skip-disease-a", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    api_root = api_root_from_chat_url(args.base_url)
    api = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    model = resolve_model(args.base_url)
    rows: list[ProbeRow] = []

    v14_all = _all_chain_blocks(api_root, V14_USER, V14_BASE)
    v14_max_turn = max(
        (int(b.get("turnIndex") or 0) for b in v14_all),
        default=0,
    )
    print(
        f"v14 chain: {len(v14_all)} blocks, max_turn={v14_max_turn}, "
        f"tokens={sum(int(b.get('numTokens') or 0) for b in v14_all)}",
        flush=True,
    )

    scenarios: list[tuple[str, str, str, str, int | str]] = [
        ("v14_T3_only", "v14", V14_USER, V14_BASE, 3),
        ("v14_T1-T3", "v14", V14_USER, V14_BASE, 3),
        ("v14_T1-T6", "v14", V14_USER, V14_BASE, 6),
        ("v14_T1-T7", "v14", V14_USER, V14_BASE, 7),
        ("v14_T1-T8", "v14", V14_USER, V14_BASE, 8),
        ("v14_T1-T10", "v14", V14_USER, V14_BASE, 10),
        ("v14_T1-T15", "v14", V14_USER, V14_BASE, 15),
        ("v14_T1-T20", "v14", V14_USER, V14_BASE, 20),
        ("v14_T1-T30", "v14", V14_USER, V14_BASE, 30),
        ("v14_T7_only", "v14", V14_USER, V14_BASE, "turn7"),
        ("classic_T1-T3", "classic", CLASSIC_USER, CLASSIC_BASE, 3),
    ]

    for label, chain, uid, base, through in scenarios:
        if through == "turn7":
            blocks = _block_at_turn(api_root, uid, base, 7)
        elif through == "all":
            blocks = v14_all if uid == V14_USER else _all_chain_blocks(api_root, uid, base)
        else:
            blocks = _blocks_through_turn(api_root, uid, base, int(through))
        row = _force_probe(api, model, label=label, chain=chain, blocks=blocks, api_root=api_root)
        rows.append(row)
        status = "PASS" if row.pass_recall and not row.garbled else (
            "GARBLED" if row.garbled else "FAIL"
        )
        err = f" err={row.error[:80]!r}" if row.error else ""
        print(
            f"  {label}: {status} blocks={row.block_count} tok={row.inject_tokens} "
            f"large={row.large_blocks} {row.preview[:70]!r}{err}",
            flush=True,
        )
        time.sleep(0.8)

    for label, chain, uid, base in [
        ("v14_resume_full", "v14", V14_USER, V14_BASE),
        ("classic_resume_full", "classic", CLASSIC_USER, CLASSIC_BASE),
    ]:
        row = _resume_probe(
            api, model, label=label, chain=chain,
            user_id=uid, base_session=base, api_root=api_root,
        )
        rows.append(row)
        status = "PASS" if row.pass_recall else "FAIL"
        err = f" err={row.error[:100]!r}" if row.error else ""
        print(f"  {label}: {status} {row.preview[:80]!r}{err}", flush=True)
        time.sleep(0.8)

    disease_a = None
    if not args.skip_disease_a:
        print("=== Disease A: fresh N7 capture ===", flush=True)
        disease_a = _disease_a(api, model, api_root)
        print(json.dumps(disease_a, indent=2), flush=True)

    report = {
        "model": model,
        "probes": [asdict(r) for r in rows],
        "disease_a": disease_a,
        "interpretation": _interpret(rows, disease_a),
    }
    out = args.out or (_TIER1.parent / "results" / "kv_composition_invalidation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}", flush=True)
    print(json.dumps(report["interpretation"], indent=2), flush=True)
    return 0


def _interpret(rows: list[ProbeRow], disease_a: dict | None) -> dict[str, Any]:
    by_label = {r.label: r for r in rows}

    def ok(name: str) -> bool:
        r = by_label.get(name)
        return bool(r and r.pass_recall and not r.garbled and not r.error)

    out: dict[str, Any] = {}
    if disease_a:
        out["A_capture_geometry"] = {
            "healthy_n7_size": disease_a.get("healthy"),
            "n8_clean_after_n7": disease_a.get("n8_clean"),
            "n7_tokens": disease_a.get("n7_tokens"),
        }
    out["C_prefix_facts"] = {
        "v14_T3_only": ok("v14_T3_only"),
        "v14_T1-T3": ok("v14_T1-T3"),
        "claim": "Fact blocks alone carry Bellagio KV",
    }
    out["C_agent_poison"] = {
        "T1-T6_ok": ok("v14_T1-T6"),
        "T1-T7_ok": ok("v14_T1-T7"),
        "T1-T8_ok": ok("v14_T1-T8"),
        "T1-T10_ok": ok("v14_T1-T10"),
        "T1-T15_ok": ok("v14_T1-T15"),
        "T1-T20_ok": ok("v14_T1-T20"),
        "T1-T30_ok": ok("v14_T1-T30"),
        "T7_alone_recalls": ok("v14_T7_only"),
    }
    out["controls"] = {
        "classic_T1-T3": ok("classic_T1-T3"),
        "classic_resume_full": ok("classic_resume_full"),
        "v14_resume_full": ok("v14_resume_full"),
    }
    out["resume_vs_force"] = {
        "force_T1-T7_ok": ok("v14_T1-T7"),
        "resume_full_ok": ok("v14_resume_full"),
        "claim": (
            "Ladder locates first prefix length where recall fails; "
            "resume vs force at same turn count isolates resume-path bugs."
        ),
    }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
