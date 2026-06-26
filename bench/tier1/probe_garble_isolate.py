#!/usr/bin/env python3
"""Isolate garble under exact v13 mixed-sweep conditions.

Replicates: T1-T3 hygiene, N4-N6 headroom classic, N7 agent capture, N8 classic probe.
Measures probe-vs-capture divergence at N7 and N8 garble rate after N7 commit.

Usage:
  python bench/tier1/probe_garble_isolate.py
  python bench/tier1/probe_garble_isolate.py --trials 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from headroom_helpers import (
    compress_assistant_for_capture,
    compress_user_payload,
    mixed_noise_user_message,
    warmup_headroom,
)
from headroom_sweep_lib import _capture_prefilled_assistant, plant_turn_headroom
from pri.text_quality import is_garbled_response
from sweep_lib import (
    FACTS,
    SYSTEM_PROMPT,
    assistant_for_history,
    block_hash,
    delete_session_captures,
    noise_prompt,
    plant_turn,
    plant_turn_hygiene,
    resolve_model,
    strip_think,
)

BASE = "http://127.0.0.1:8000"
API = f"{BASE}/v1/chat/completions"
API_ROOT = f"{BASE}/v1"


@dataclass
class CaptureDiag:
    turn_index: int
    user_phase: str
    probe_garbled: bool
    cap_garbled: bool
    harness_would_commit: bool  # line-167 path: capture_asst and not probe_garbled
    probe_len: int
    cap_tail_len: int
    capture_asst_len: int
    block_tokens: int
    block_preview_garbled: bool
    block_session: str
    new_hash: str


@dataclass
class ProbeBatch:
    label: str
    trials: int
    garbled: int
    samples: list[dict[str, Any]]


def fetch_chain_blocks(
    user_id: str,
    base_session: str,
    *,
    api_root: str | None = None,
) -> list[dict]:
    """Latest block per turn_index for a chain (any block role)."""
    root = (api_root or API_ROOT).rstrip("/")
    if not root.endswith("/admin"):
        admin = root.replace("/v1", "") + "/admin" if "/v1" in root else root + "/admin"
    else:
        admin = root
    url = f"{admin}/memory/user-memories?user_id={user_id}&limit=500&include_kv=1"
    with urllib.request.urlopen(url) as resp:
        mems = json.load(resp).get("memories") or []
    rows = [
        m for m in mems
        if (m.get("baseSessionId") or "") == base_session
        and int(m.get("numTokens") or 0) > 0
    ]
    latest: dict[int, dict] = {}
    for block in rows:
        ti = int(block.get("turnIndex") or 0)
        ts = float(block.get("timestamp") or 0)
        prev = latest.get(ti)
        if prev is None or ts >= float(prev.get("timestamp") or 0):
            latest[ti] = block
    return [latest[t] for t in sorted(latest)]


def wait_for_chain_turn_block(
    user_id: str,
    base_session: str,
    turn_index: int,
    *,
    api_root: str | None = None,
    retries: int = 12,
    sleep_s: float = 0.5,
) -> dict | None:
    """Poll admin index until ``turn_index`` block is visible (post-capture lag)."""
    for attempt in range(retries):
        block = block_at_turn(fetch_chain_blocks(user_id, base_session, api_root=api_root), turn_index)
        if block is not None and int(block.get("numTokens") or 0) > 0:
            return block
        if attempt + 1 < retries:
            time.sleep(sleep_s)
    return None


def block_at_turn(blocks: list[dict], turn: int) -> dict | None:
    for b in blocks:
        if int(b.get("turnIndex") or 0) == turn:
            return b
    return None


def plant_prefix(
    model: str,
    user_id: str,
    base_session: str,
    *,
    through_turn: int,
) -> str:
    """Plant exact sweep prefix through ``through_turn`` (inclusive). Returns prev_hash."""
    prev_hash = ""
    turn_index = 0

    for fact in FACTS:
        turn_index += 1
        if turn_index > through_turn:
            return prev_hash
        r = plant_turn_hygiene(
            API, model, API_ROOT,
            user_id=user_id, base_session=base_session, turn_index=turn_index,
            prev_hash=prev_hash, user_msg=fact, max_garbled_retries=4,
        )
        if r.still_garbled:
            raise RuntimeError(f"fact T{turn_index} garbled")
        prev_hash = r.new_prev_hash
        time.sleep(0.3)

    noise_done = 0
    while turn_index < through_turn:
        msg, kind = mixed_noise_user_message(
            noise_done, noise_prompt(noise_done), agent_every=4,
        )
        compress_user = kind == "agent"
        turn_index += 1
        r = plant_turn_headroom(
            API, model, API_ROOT,
            user_id=user_id, base_session=base_session, turn_index=turn_index,
            prev_hash=prev_hash, user_msg=msg, compress_user=compress_user,
            compress_assistant=True, max_garbled_retries=4,
        )
        if r.still_garbled:
            raise RuntimeError(f"noise N{turn_index} garbled/neutral failed")
        prev_hash = r.new_prev_hash
        noise_done += 1
        time.sleep(0.3)

    return prev_hash


def probe_decode(
    model: str,
    user_id: str,
    base_session: str,
    *,
    turn_index: int,
    prev_hash: str,
    user_msg: str,
    trials: int,
) -> ProbeBatch:
    rows: list[dict[str, Any]] = []
    garbled = 0
    for i in range(trials):
        raw, _, _ = plant_turn(
            API, model,
            user_id=user_id, base_session=base_session,
            turn_index=turn_index, prev_hash=prev_hash,
            user_msg=user_msg, capture=False,
        )
        asst, is_g = assistant_for_history(raw)
        text = strip_think(raw or "")
        if is_g:
            garbled += 1
        rows.append({
            "trial": i,
            "garbled": is_g,
            "len": len(asst or text),
            "preview": text[:120],
        })
        time.sleep(0.2)
    return ProbeBatch(
        label=f"N{turn_index}",
        trials=trials,
        garbled=garbled,
        samples=rows[:3],
    )


def instrument_n7_capture(
    model: str,
    user_id: str,
    base_session: str,
    prev_hash: str,
    user_msg: str,
    *,
    compress_user: bool = True,
) -> CaptureDiag:
    """One N7 plant with explicit probe/capture/harness logging (v13 path)."""
    plant_user = user_msg
    user_phase = "raw"
    if compress_user:
        plant_user, _ = compress_user_payload(user_msg, model="gpt-4o")
        user_phase = "compressed"

    probe_content, _, _ = plant_turn(
        API, model,
        user_id=user_id, base_session=base_session, turn_index=7,
        prev_hash=prev_hash, user_msg=plant_user, capture=False,
    )
    probe_asst, probe_garbled = assistant_for_history(probe_content)

    capture_asst = probe_asst
    if probe_asst:
        capture_asst, _ = compress_assistant_for_capture(
            SYSTEM_PROMPT, plant_user, probe_asst, model="gpt-4o",
        )

    cap_content, new_hash, block_session = _capture_prefilled_assistant(
        API, model, api_root=API_ROOT,
        user_id=user_id, base_session=base_session, turn_index=7,
        prev_hash=prev_hash, user_msg=plant_user,
        assistant_text=capture_asst or probe_asst or "",
        max_tokens=8,
    )
    cap_asst, cap_garbled = assistant_for_history(cap_content)
    harness_would_commit = bool(capture_asst) and not probe_garbled and not cap_garbled

    time.sleep(0.5)
    blocks = fetch_chain_blocks(user_id, base_session)
    blk = block_at_turn(blocks, 7)
    preview = (blk.get("preview") or "") if blk else ""
    tokens = int(blk.get("numTokens") or 0) if blk else 0

    return CaptureDiag(
        turn_index=7,
        user_phase=user_phase,
        probe_garbled=probe_garbled,
        cap_garbled=cap_garbled,
        harness_would_commit=harness_would_commit,
        probe_len=len(probe_asst or ""),
        cap_tail_len=len(cap_asst or ""),
        capture_asst_len=len(capture_asst or ""),
        block_tokens=tokens,
        block_preview_garbled=is_garbled_response(preview),
        block_session=block_session,
        new_hash=new_hash,
    )


def run_trial(model: str, trials: int) -> dict[str, Any]:
    user_id = f"iso_{uuid.uuid4().hex[:8]}"
    base_session = f"chain_{uuid.uuid4().hex[:12]}"
    n7_msg, _ = mixed_noise_user_message(3, noise_prompt(3), agent_every=4)
    n8_msg = noise_prompt(4)  # Odyssey — same as v13 N8

    # --- A: through N6, probe N7 agent (no N7 capture) — control ---
    prev6 = plant_prefix(model, user_id, base_session, through_turn=6)
    blocks6 = fetch_chain_blocks(user_id, base_session)
    inject6 = sum(int(b.get("numTokens") or 0) for b in blocks6)
    probe_n7_no_capture = probe_decode(
        model, user_id, base_session,
        turn_index=7, prev_hash=prev6, user_msg=n7_msg, trials=trials,
    )

    # --- B: fresh chain, through N7 capture (exact v13), probe N8 ---
    user_id_b = f"iso_{uuid.uuid4().hex[:8]}"
    base_b = f"chain_{uuid.uuid4().hex[:12]}"
    prev7 = plant_prefix(model, user_id_b, base_b, through_turn=7)
    blocks7 = fetch_chain_blocks(user_id_b, base_b)
    blk7 = block_at_turn(blocks7, 7)
    inject7 = sum(int(b.get("numTokens") or 0) for b in blocks7)
    n7_preview_g = is_garbled_response((blk7.get("preview") or "") if blk7 else "")
    n7_tokens = int(blk7.get("numTokens") or 0) if blk7 else 0
    probe_n8_after_n7 = probe_decode(
        model, user_id_b, base_b,
        turn_index=8, prev_hash=prev7, user_msg=n8_msg, trials=trials,
    )
    # --- B2: exact v13 N8 plant (headroom prefilled capture, not probe-only) ---
    user_id_b2 = f"iso_{uuid.uuid4().hex[:8]}"
    base_b2 = f"chain_{uuid.uuid4().hex[:12]}"
    prev7b = plant_prefix(model, user_id_b2, base_b2, through_turn=7)
    n8_plant = plant_turn_headroom(
        API, model, API_ROOT,
        user_id=user_id_b2, base_session=base_b2, turn_index=8,
        prev_hash=prev7b, user_msg=n8_msg, compress_user=False,
        compress_assistant=True, max_garbled_retries=4,
    )

    # --- C: instrument single N7 capture on fresh prefix-through-N6 ---
    user_id_c = f"iso_{uuid.uuid4().hex[:8]}"
    base_c = f"chain_{uuid.uuid4().hex[:12]}"
    prev6c = plant_prefix(model, user_id_c, base_c, through_turn=6)
    diag = instrument_n7_capture(
        model, user_id_c, base_c, prev6c, n7_msg, compress_user=True,
    )
    probe_n8_after_diag = probe_decode(
        model, user_id_c, base_c,
        turn_index=8, prev_hash=diag.new_hash, user_msg=n8_msg, trials=trials,
    )

    return {
        "user_ids": {"A": user_id, "B": user_id_b, "C": user_id_c},
        "A_n6_probe_n7_no_capture": {
            "inject_tokens": inject6,
            "blocks": len(blocks6),
            **asdict(probe_n7_no_capture),
            "garbled_rate": round(probe_n7_no_capture.garbled / trials, 2),
        },
        "B_n7_captured_probe_n8": {
            "inject_tokens": inject7,
            "n7_block_tokens": n7_tokens,
            "n7_preview_garbled": n7_preview_g,
            **asdict(probe_n8_after_n7),
            "garbled_rate": round(probe_n8_after_n7.garbled / trials, 2),
        },
        "B2_n7_captured_plant_n8": {
            "still_garbled": n8_plant.still_garbled,
            "neutral_fallback": n8_plant.neutral_fallback,
            "garbled_probe_attempts": n8_plant.garbled_probe_attempts,
            "deletes": n8_plant.deletes,
        },
        "C_instrumented_n7_capture": {
            **asdict(diag),
            "probe_n8": asdict(probe_n8_after_diag),
            "n8_garbled_rate": round(probe_n8_after_diag.garbled / trials, 2),
        },
    }


def replay_v13_n8(model: str, trials: int) -> dict[str, Any]:
    """Probe N8 on the actual failed v13 chain (through N7 block on disk)."""
    user_id = "hr_sweep_0a9c117e40"
    base_session = "chain_thread_d858e5064c36"
    blocks = fetch_chain_blocks(user_id, base_session)
    blocks_through_7 = [b for b in blocks if int(b.get("turnIndex") or 0) <= 7]
    blk7 = block_at_turn(blocks, 7)
    if not blk7:
        return {"error": "N7 block missing on v13 chain"}
    prev_hash = block_hash(str(blk7.get("sessionId") or ""))
    inject = sum(int(b.get("numTokens") or 0) for b in blocks_through_7)
    n8_msg = noise_prompt(4)
    batch = probe_decode(
        model, user_id, base_session,
        turn_index=8, prev_hash=prev_hash, user_msg=n8_msg, trials=trials,
    )
    return {
        "inject_tokens_through_n7": inject,
        "n7_tokens": int(blk7.get("numTokens") or 0),
        "n7_preview_garbled": is_garbled_response(blk7.get("preview") or ""),
        **asdict(batch),
        "garbled_rate": round(batch.garbled / trials, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--replay-v13", action="store_true")
    parser.add_argument("--n7-sweep", type=int, default=0, help="Repeat instrumented N7 capture N times")
    parser.add_argument("--out", default="/home/wasnaga/punk-records-inference/bench/results/garble_isolate.json")
    args = parser.parse_args()

    warmup_headroom()
    model = resolve_model(BASE)
    print(f"model={model} trials={args.trials} runs={args.runs}", flush=True)

    if args.n7_sweep:
        n7_msg, _ = mixed_noise_user_message(3, noise_prompt(3), agent_every=4)
        n8_msg = noise_prompt(4)
        rows = []
        for i in range(args.n7_sweep):
            uid = f"sw_{uuid.uuid4().hex[:8]}"
            base = f"chain_{uuid.uuid4().hex[:12]}"
            prev6 = plant_prefix(model, uid, base, through_turn=6)
            d = instrument_n7_capture(model, uid, base, prev6, n7_msg, compress_user=True)
            n8 = probe_decode(model, uid, base, turn_index=8, prev_hash=d.new_hash, user_msg=n8_msg, trials=3)
            poison = d.probe_garbled is False and d.cap_garbled is True
            bad_n8 = n8.garbled > 0
            rows.append({**asdict(d), "n8_garbled_3": n8.garbled, "divergence": poison, "n8_fails": bad_n8})
            print(
                f"  {i+1}/{args.n7_sweep} probe_g={d.probe_garbled} cap_g={d.cap_garbled} "
                f"diverge={poison} block={d.block_tokens}tok n8_g={n8.garbled}/3",
                flush=True,
            )
            time.sleep(0.3)
        div = sum(1 for r in rows if r["divergence"])
        fail = sum(1 for r in rows if r["n8_fails"])
        print(f"summary: divergence={div}/{args.n7_sweep} n8_any_garbled={fail}/{args.n7_sweep}", flush=True)
        out = {"n7_sweep": args.n7_sweep, "rows": rows, "divergence_count": div, "n8_fail_count": fail}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return 0

    if args.replay_v13:
        print("\n=== D: replay v13 chain through N7 → probe N8 ===", flush=True)
        d = replay_v13_n8(model, args.trials)
        print(json.dumps(d, indent=2), flush=True)
        return 0

    all_runs: list[dict] = []
    for i in range(args.runs):
        print(f"\n=== run {i + 1}/{args.runs} ===", flush=True)
        row = run_trial(model, args.trials)
        all_runs.append(row)
        a = row["A_n6_probe_n7_no_capture"]
        b = row["B_n7_captured_probe_n8"]
        c = row["C_instrumented_n7_capture"]
        print(
            f"A  N6→probe N7 (no capture): {a['garbled']}/{args.trials} garbled "
            f"inject={a['inject_tokens']}tok",
            flush=True,
        )
        print(
            f"B  N7 captured → probe N8:    {b['garbled']}/{args.trials} garbled "
            f"n7={b['n7_block_tokens']}tok preview_garbled={b['n7_preview_garbled']}",
            flush=True,
        )
        b2 = row["B2_n7_captured_plant_n8"]
        print(
            f"B2 N7 captured → plant N8:    garbled={b2['still_garbled']} "
            f"neutral={b2['neutral_fallback']} probe_attempts={b2['garbled_probe_attempts']}",
            flush=True,
        )
        print(
            f"C  instrument N7: probe_g={c['probe_garbled']} cap_g={c['cap_garbled']} "
            f"would_commit={c['harness_would_commit']} block={c['block_tokens']}tok "
            f"→ N8 {c['n8_garbled_rate']}",
            flush=True,
        )

    out = {"model": model, "trials": args.trials, "runs": args.runs, "results": all_runs}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
