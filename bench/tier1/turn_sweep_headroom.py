#!/usr/bin/env python3
"""Turn sweep with Headroom compression layered on PRI chain capture.

Hypothesis: smaller per-turn KV (compressed tool payloads + compressed assistant
prefill) lowers cumulative inject tokens so RESUME recall survives cp60+.

Usage (GX10):
    /home/wasnaga/headroom-venv/bin/python bench/tier1/turn_sweep_headroom.py \\
        --base-url http://127.0.0.1:8000 \\
        --checkpoints 60,80,100 \\
        --noise-mode mixed
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

_TIER1 = Path(__file__).resolve().parent
_REPO = _TIER1.parents[1]
for path in (_REPO, _REPO / "bench" / "opencode", _TIER1):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from headroom_helpers import (  # noqa: E402
    agent_noise_user_message,
    headroom_available,
    mixed_noise_user_message,
    warmup_headroom,
)
from headroom_sweep_lib import plant_turn_headroom  # noqa: E402
from nls_kvp_helpers import api_root_from_chat_url  # noqa: E402
from sweep_lib import (  # noqa: E402
    FACTS,
    NEUTRAL_TURN_USER,
    RECALL,
    PlantHygieneResult,
    apply_resume_inject_caps,
    chain_turn_stats,
    estimate_chat_tokens,
    noise_prompt,
    plant_neutral_required,
    plant_turn_hygiene,
    resolve_model,
    send_recall_arm,
    sweep_resume_max_tokens,
    wait_for_vllm,
)


def _save_checkpoint(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _noise_turn(mode: str, index: int, *, agent_every: int) -> tuple[str, str, bool]:
    """Return (user_message, kind_label, compress_user)."""
    if mode == "agent":
        return agent_noise_user_message(index), "agent", True
    if mode == "classic":
        return noise_prompt(index), "classic", False
    msg, kind = mixed_noise_user_message(
        index, noise_prompt(index), agent_every=agent_every,
    )
    return msg, kind, kind == "agent"


def _run_geometry_audit(
    *,
    api_root: str,
    user_id: str,
    base_session: str,
    out_path: Path,
) -> dict | None:
    script = _TIER1 / "geometry_audit.py"
    if not script.is_file():
        return None
    cmd = [
        sys.executable,
        str(script),
        "--user-id", user_id,
        "--base-session", base_session,
        "--base-url", api_root,
        "--out", str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        if out_path.is_file():
            return json.loads(out_path.read_text(encoding="utf-8"))
        if proc.stdout.strip():
            return json.loads(proc.stdout)
    except Exception as exc:
        return {"verdict": "error", "findings": [str(exc)]}
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("PRI_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--checkpoints", default="60,80,100")
    parser.add_argument(
        "--noise-mode",
        choices=("classic", "agent", "mixed"),
        default="mixed",
        help="classic = Q&A; agent = tool JSON; mixed = mostly Q&A + periodic tool turns",
    )
    parser.add_argument(
        "--agent-every",
        type=int,
        default=4,
        help="In mixed mode, every Nth noise turn is an agent tool-output turn",
    )
    parser.add_argument(
        "--compress-user",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Headroom-compress user payloads (default: on for agent/mixed agent turns only)",
    )
    parser.add_argument(
        "--compress-assistant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Probe full assistant, compress, re-capture with prefilled assistant (all noise turns)",
    )
    parser.add_argument(
        "--geometry-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run offline geometry audit after final checkpoint",
    )
    parser.add_argument("--with-arm-d", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--garbled-retries", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not headroom_available():
        print("FATAL: pip install headroom-ai (use headroom-venv on GX10)", file=sys.stderr)
        return 1

    print("[headroom] warming up compression models...")
    warmup_headroom()
    print("[headroom] warmup done")

    api_root = api_root_from_chat_url(args.base_url)
    if not wait_for_vllm(api_root):
        print("FATAL: vLLM not healthy", file=sys.stderr)
        return 1

    api = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    model = os.environ.get("PRI_MODEL") or resolve_model(args.base_url)
    checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    user_id = f"hr_sweep_{uuid.uuid4().hex[:10]}"
    base_session = f"chain_thread_{uuid.uuid4().hex[:12]}"
    capture_mode = os.environ.get("NLS_CHAIN_CAPTURE_MODE", "turn")

    payload: dict = {
        "version": 2,
        "experiment": "headroom_plus_pri",
        "noise_mode": args.noise_mode,
        "agent_every": args.agent_every,
        "compress_assistant": args.compress_assistant,
        "capture_mode": capture_mode,
        "resume_max_tokens": sweep_resume_max_tokens(),
        "user_id": user_id,
        "base_session": base_session,
        "model": model,
        "checkpoints": checkpoints,
        "noise_turn_kinds": [],
        "headroom_user_compress": [],
        "garbled_neutral_fallbacks": [],
        "garbled_stripped": [],
        "garbled_captures_deleted": 0,
        "results": [],
    }

    out_path = args.out or Path("bench/results") / f"turn_sweep_headroom_{args.noise_mode}.json"
    geom_out = out_path.with_name(out_path.stem + "_geometry.json")

    print("=" * 72)
    print("HEADROOM + PRI TURN SWEEP")
    print(
        f"noise_mode={args.noise_mode} agent_every={args.agent_every} "
        f"compress_assistant={args.compress_assistant}",
    )
    print(f"user={user_id} model={model} checkpoints={checkpoints}")
    print(f"neutral_user={NEUTRAL_TURN_USER!r}")
    print("=" * 72)

    turns: list[tuple[str, str]] = []
    turn_index = 0
    prev_hash = ""
    noise_done = 0
    garbled_stripped: list[dict] = []
    garbled_neutral_fallbacks: list[dict] = []
    total_deletes = 0

    def _record_headroom_user(result: PlantHygieneResult, *, turn_index: int, kind: str) -> None:
        if result.headroom_user_phase == "n/a":
            return
        payload["headroom_user_compress"].append({
            "turn_index": turn_index,
            "kind": kind,
            "phase": result.headroom_user_phase,
            "tokens_saved": result.headroom_user_tokens_saved,
            "compressed_probe_failures": result.headroom_compressed_probe_failures,
            "neutral_fallback": result.neutral_fallback,
        })

    def _headroom_compress_summary() -> dict:
        rows = payload["headroom_user_compress"]
        agent_rows = [r for r in rows if r.get("kind") == "agent"]
        compressed = [r for r in agent_rows if r.get("phase") == "compressed"]
        raw = [r for r in agent_rows if r.get("phase") == "raw"]
        neutral = [r for r in agent_rows if r.get("phase") in ("neutral", "plain")]
        failed = [r for r in agent_rows if r.get("phase") == "failed"]
        return {
            "agent_turns": len(agent_rows),
            "captured_compressed": len(compressed),
            "captured_raw_fallback": len(raw),
            "captured_neutral_or_plain": len(neutral),
            "failed": len(failed),
            "compress_success_rate": (
                round(len(compressed) / len(agent_rows), 3) if agent_rows else None
            ),
            "total_tokens_saved": sum(int(r.get("tokens_saved") or 0) for r in compressed),
            "total_compressed_probe_failures": sum(
                int(r.get("compressed_probe_failures") or 0) for r in agent_rows
            ),
        }

    def _record_plant(label: str, result: PlantHygieneResult, *, kind: str = "classic") -> bool:
        nonlocal total_deletes, prev_hash
        total_deletes += result.deletes
        if result.neutral_fallback and not result.still_garbled:
            garbled_neutral_fallbacks.append({
                "label": label,
                "turn_index": turn_index,
                "kind": kind,
                "original_user": result.original_user_msg[:120],
                "substitute_user": result.user_text[:120],
                "substitute_assistant": (result.assistant_text or "")[:120],
                "session": result.block_session,
                "deletes": result.deletes,
                "garbled_probe_attempts": result.garbled_probe_attempts,
                "headroom_user_phase": result.headroom_user_phase,
            })
            print(
                f"  [NEUTRAL] {label} captured safe substitute after "
                f"{result.garbled_probe_attempts} garbled probe(s)",
            )
        if result.still_garbled:
            garbled_stripped.append({
                "label": label,
                "turn_index": turn_index,
                "kind": kind,
                "user": result.original_user_msg[:120],
                "session": result.block_session,
                "deletes": result.deletes,
            })
            return False
        _record_headroom_user(result, turn_index=turn_index, kind=kind)
        prev_hash = result.new_prev_hash
        turns.append((result.user_text, result.assistant_text))
        return True

    for fact in FACTS:
        turn_index += 1
        print(f"  [fact T{turn_index}] {fact[:55]}...")
        result = plant_turn_hygiene(
            api, model, api_root,
            user_id=user_id,
            base_session=base_session,
            turn_index=turn_index,
            prev_hash=prev_hash,
            user_msg=fact,
            max_garbled_retries=args.garbled_retries,
        )
        if not _record_plant(f"fact-T{turn_index}", result, kind="fact"):
            result = plant_neutral_required(
                api, model, api_root,
                user_id=user_id,
                base_session=base_session,
                turn_index=turn_index,
                prev_hash=prev_hash,
                original_user_msg=fact,
                deletes=result.deletes,
                garbled_probe_attempts=result.garbled_probe_attempts,
            )
        if not _record_plant(f"fact-T{turn_index}", result, kind="fact"):
            print(f"FATAL: fact T{turn_index} — neutral capture could not advance chain")
            payload["garbled_stripped"] = garbled_stripped
            payload["garbled_neutral_fallbacks"] = garbled_neutral_fallbacks
            payload["garbled_captures_deleted"] = total_deletes
            _save_checkpoint(out_path, payload)
            return 2
        time.sleep(0.5)

    for cp in checkpoints:
        while noise_done < cp:
            msg, kind, compress_user_default = _noise_turn(
                args.noise_mode, noise_done, agent_every=args.agent_every,
            )
            compress_user = (
                compress_user_default
                if args.compress_user is None
                else args.compress_user
            )
            compress_asst = args.compress_assistant
            turn_index += 1
            print(f"  [noise N{turn_index} {kind}] {msg[:70]}...")
            payload["noise_turn_kinds"].append({
                "noise_index": noise_done,
                "turn_index": turn_index,
                "kind": kind,
                "compress_user": compress_user,
                "compress_assistant": compress_asst,
            })
            result = plant_turn_headroom(
                api, model, api_root,
                user_id=user_id,
                base_session=base_session,
                turn_index=turn_index,
                prev_hash=prev_hash,
                user_msg=msg,
                max_garbled_retries=args.garbled_retries,
                compress_user=compress_user,
                compress_assistant=compress_asst,
            )
            if result.still_garbled:
                result = plant_neutral_required(
                    api, model, api_root,
                    user_id=user_id,
                    base_session=base_session,
                    turn_index=turn_index,
                    prev_hash=prev_hash,
                    original_user_msg=msg,
                    deletes=result.deletes,
                    garbled_probe_attempts=result.garbled_probe_attempts,
                )
            if not _record_plant(f"noise-N{turn_index}", result, kind=kind):
                print(f"FATAL: N{turn_index} — neutral capture could not advance chain")
                payload["garbled_stripped"] = garbled_stripped
                payload["garbled_neutral_fallbacks"] = garbled_neutral_fallbacks
                payload["garbled_captures_deleted"] = total_deletes
                _save_checkpoint(out_path, payload)
                return 2
            noise_done += 1
            time.sleep(0.5)

        payload["garbled_stripped"] = garbled_stripped
        payload["garbled_neutral_fallbacks"] = garbled_neutral_fallbacks
        payload["garbled_captures_deleted"] = total_deletes
        stats = chain_turn_stats(api_root, user_id, base_session)
        text_est = estimate_chat_tokens(api_root, model, turns)
        hr = _headroom_compress_summary()
        payload["headroom_user_compress_summary"] = hr

        print(
            f"\n=== CP noise={cp} turns={len(turns)} "
            f"turn_blocks={stats['blocks']}/{stats['tokens']}tok "
            f"text~{text_est} baseline_cp60_inject=17131 ===",
        )
        print(
            f"  headroom user compress: {hr['captured_compressed']}/{hr['agent_turns']} "
            f"agent turns via compressed "
            f"({hr['captured_raw_fallback']} raw fallback, {hr['failed']} failed) "
            f"saved={hr['total_tokens_saved']}tok "
            f"probe_fails={hr['total_compressed_probe_failures']} "
            f"neutral_fallbacks={len(garbled_neutral_fallbacks)} "
            f"stripped={len(garbled_stripped)}",
        )

        text_pass = resume_pass = 0
        text_rows: list[dict] = []
        resume_rows: list[dict] = []

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
            text_pass += int(bool(text_row.get("pass_clean")))
            resume_pass += int(bool(resume_row.get("pass_clean")))

        row = {
            "checkpoint_noise": cp,
            "conversation_turns": len(turns),
            "turn_blocks": stats["blocks"],
            "turn_tokens": stats["tokens"],
            "text_est_tokens": text_est,
            "baseline_inject_cp60": 17131,
            "baseline_inject_cp80": 23543,
            "text_pass_clean": text_pass,
            "resume_pass_clean": resume_pass,
            "total": len(RECALL),
            "text_recall": text_rows,
            "resume_recall": resume_rows,
        }
        payload["results"].append(row)
        _save_checkpoint(out_path, payload)
        print(f"  TEXT={text_pass}/{len(RECALL)} RESUME={resume_pass}/{len(RECALL)}")

    if args.geometry_audit:
        audit = _run_geometry_audit(
            api_root=api_root,
            user_id=user_id,
            base_session=base_session,
            out_path=geom_out,
        )
        if audit:
            payload["geometry_audit"] = {
                "verdict": audit.get("verdict"),
                "findings": audit.get("findings"),
                "path": str(geom_out),
            }
            _save_checkpoint(out_path, payload)
            print(f"\nGeometry audit: {audit.get('verdict')} -> {geom_out}")
            for finding in audit.get("findings") or []:
                print(f"  - {finding}")

    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
