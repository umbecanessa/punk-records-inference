#!/usr/bin/env python3
"""Turn-chain production sweep: Marco facts + noise at cp20/40/60/80.

Plants a single growing session with turn-level capture, then scores TEXT vs
RESUME (and optional resume_overflow arm D) at each checkpoint.

Garbled assistant responses trigger admin capture delete + retry so poisoned
Mamba state does not remain in the inject chain (see sweep_lib.plant_turn_hygiene).

Requires vLLM with ``NLS_CHAIN_CAPTURE_MODE=turn``.

Usage:
    python bench/tier1/turn_sweep.py --base-url http://127.0.0.1:8000
    python bench/tier1/turn_sweep.py --base-url http://127.0.0.1:8000 --checkpoints 20,40,60,80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

_TIER1 = Path(__file__).resolve().parent
_REPO = _TIER1.parents[1]
for path in (_REPO, _REPO / "bench" / "opencode", _TIER1):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sweep_lib import (  # noqa: E402
    FACTS,
    RECALL,
    chain_turn_stats,
    estimate_chat_tokens,
    estimate_user_only_tokens,
    noise_prompt,
    plant_turn_hygiene,
    resolve_model,
    send_recall_arm,
    wait_for_vllm,
)
from nls_kvp_helpers import api_root_from_chat_url  # noqa: E402


def _save_checkpoint(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("PRI_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument(
        "--checkpoints",
        default="20,40,60,80",
        help="Comma-separated cumulative noise turn counts after facts",
    )
    parser.add_argument("--with-arm-d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--garbled-retries",
        type=int,
        default=2,
        help="Delete capture and retry when assistant output is garbled",
    )
    parser.add_argument(
        "--stop-on-fact-garble",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    api_root = api_root_from_chat_url(args.base_url)
    if not wait_for_vllm(api_root):
        print("FATAL: vLLM not healthy", file=sys.stderr)
        return 1

    api = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    model = os.environ.get("PRI_MODEL") or resolve_model(args.base_url)
    checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    user_id = f"turn_sweep_{uuid.uuid4().hex[:10]}"
    base_session = f"chain_thread_{uuid.uuid4().hex[:12]}"
    capture_mode = os.environ.get("NLS_CHAIN_CAPTURE_MODE", "turn")

    payload: dict = {
        "version": 2,
        "garbled_capture_guard": True,
        "garbled_retries": args.garbled_retries,
        "capture_mode": capture_mode,
        "user_id": user_id,
        "base_session": base_session,
        "model": model,
        "checkpoints": checkpoints,
        "garbled_stripped": [],
        "garbled_captures_deleted": 0,
        "results": [],
    }

    out_path = args.out or Path("bench/results") / "turn_sweep_cp20_80.json"

    print("=" * 72)
    print("TURN CHAIN PRODUCTION SWEEP (garbled capture guard ON)")
    print(f"capture_mode={capture_mode} user={user_id} model={model}")
    print("=" * 72)

    turns: list[tuple[str, str]] = []
    turn_index = 0
    prev_hash = ""
    noise_done = 0
    garbled_stripped: list[dict] = []
    total_deletes = 0

    for fact in FACTS:
        turn_index += 1
        print(f"  [fact T{turn_index}] {fact[:55]}...")
        asst, prev_hash, session, still_garbled, deletes = plant_turn_hygiene(
            api, model, api_root,
            user_id=user_id,
            base_session=base_session,
            turn_index=turn_index,
            prev_hash=prev_hash,
            user_msg=fact,
            max_garbled_retries=args.garbled_retries,
        )
        total_deletes += deletes
        if still_garbled:
            garbled_stripped.append({
                "label": f"fact-T{turn_index}",
                "user": fact[:120],
                "session": session,
                "deletes": deletes,
            })
            if args.stop_on_fact_garble:
                print(f"FATAL: garbled fact plant at T{turn_index} after {deletes} delete(s)")
                payload["garbled_stripped"] = garbled_stripped
                payload["garbled_captures_deleted"] = total_deletes
                _save_checkpoint(out_path, payload)
                return 2
        turns.append((fact, asst))
        time.sleep(0.5)

    payload["garbled_stripped"] = garbled_stripped
    payload["garbled_captures_deleted"] = total_deletes

    for cp in checkpoints:
        while noise_done < cp:
            msg = noise_prompt(noise_done)
            turn_index += 1
            print(f"  [noise N{turn_index}] {msg[:55]}...")
            asst, prev_hash, session, still_garbled, deletes = plant_turn_hygiene(
                api, model, api_root,
                user_id=user_id,
                base_session=base_session,
                turn_index=turn_index,
                prev_hash=prev_hash,
                user_msg=msg,
                max_garbled_retries=args.garbled_retries,
            )
            total_deletes += deletes
            if still_garbled:
                garbled_stripped.append({
                    "label": f"noise-N{turn_index}",
                    "user": msg[:120],
                    "session": session,
                    "deletes": deletes,
                })
                print(f"  [GARBLED] kept empty history, deleted capture ({deletes} attempt(s))")
            turns.append((msg, asst))
            noise_done += 1
            time.sleep(0.5)

        payload["garbled_stripped"] = garbled_stripped
        payload["garbled_captures_deleted"] = total_deletes

        stats = chain_turn_stats(api_root, user_id, base_session)
        text_est = estimate_chat_tokens(api_root, model, turns)
        text_user = estimate_user_only_tokens(api_root, model, turns)

        print(
            f"\n=== CP noise={cp} turns={len(turns)} "
            f"turn_blocks={stats['blocks']}/{stats['tokens']}tok "
            f"text~{text_est} inject_ratio={stats['tokens'] / max(text_est, 1):.2%} "
            f"garbled_deletes={total_deletes} ===",
        )

        text_pass = resume_pass = arm_d_pass = 0
        text_rows: list[dict] = []
        resume_rows: list[dict] = []
        arm_d_rows: list[dict] = []

        for question, expected in RECALL:
            text_row = send_recall_arm(
                api, model, arm="text", turns=turns, user_id=user_id,
                base_session=base_session, checkpoint=cp,
                question=question, expected=expected,
            )
            resume_row = send_recall_arm(
                api, model, arm="resume", turns=turns, user_id=user_id,
                base_session=base_session, checkpoint=cp,
                question=question, expected=expected,
            )
            text_rows.append({"question": question, **text_row})
            resume_rows.append({"question": question, **resume_row})
            if text_row.get("pass_clean"):
                text_pass += 1
            if resume_row.get("pass_clean"):
                resume_pass += 1

            if args.with_arm_d:
                arm_d_row = send_recall_arm(
                    api, model, arm="arm_d", turns=turns, user_id=user_id,
                    base_session=base_session, checkpoint=cp,
                    question=question, expected=expected,
                )
                arm_d_rows.append({"question": question, **arm_d_row})
                if arm_d_row.get("pass_clean"):
                    arm_d_pass += 1

        row = {
            "checkpoint_noise": cp,
            "conversation_turns": len(turns),
            "turn_blocks": stats["blocks"],
            "turn_tokens": stats["tokens"],
            "turn_by_role": stats["by_role"],
            "text_est_tokens": text_est,
            "text_user_only_tokens": text_user,
            "text_pass_clean": text_pass,
            "resume_pass_clean": resume_pass,
            "total": len(RECALL),
            "text_recall": text_rows,
            "resume_recall": resume_rows,
        }
        if args.with_arm_d:
            row["arm_d_pass_clean"] = arm_d_pass
            row["arm_d_recall"] = arm_d_rows

        payload["results"].append(row)
        _save_checkpoint(out_path, payload)

        arm_suffix = f" ARM-D={arm_d_pass}/{len(RECALL)}" if args.with_arm_d else ""
        print(
            f"  TEXT={text_pass}/{len(RECALL)} "
            f"RESUME={resume_pass}/{len(RECALL)}{arm_suffix}",
        )

    print(f"\nSaved {out_path} (garbled_captures_deleted={total_deletes})")

    resume_ok = all(
        r["resume_pass_clean"] >= r["text_pass_clean"]
        for r in payload["results"]
    )
    return 0 if resume_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
