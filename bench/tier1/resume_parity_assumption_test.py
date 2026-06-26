#!/usr/bin/env python3
"""Test 0.92->0.99 parity assumptions on a frozen turn-sweep chain.

Runs four experiment arms on an existing ``turn_sweep*.json`` chain (must still
be on disk):

  A. Phantom microscope — minimal/full vs resume; per-stage query cos
     (``attn_input_hs``, ``ssm_state``, ``deltanet_out_hs``)
  B. Recall probes — 5× Marco questions; correlate pass/garble with parity
  C. Inject scope — hotel probe with max_blocks=3 vs full chain
  D. Decode stability — isolated vs sequential hotel; 3× repeat

Microscope **compare** requires ``torch`` (run inside vLLM container or copy
captures out). HTTP capture requests can run from the host.

Usage:
    python bench/tier1/resume_parity_assumption_test.py \\
        --from-sweep bench/results/.../turn_sweep_cp60_80_garble_inv.json \\
        --checkpoint 80 \\
        --base-url http://127.0.0.1:8000

Inside container (after host sends captures or full run):
    docker exec pri-inference python3 -u /tmp/pri-garble/bench/tier1/resume_parity_assumption_test.py \\
        --from-sweep /tmp/sweep.json --checkpoint 80 --compare-only
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
from garble_root_cause import _interpret, _recall_once, _scan_chain_previews  # noqa: E402
from microscope_lib import (  # noqa: E402
    MICROSCOPE_DIR,
    PARITY_THRESHOLD,
    clear_microscope_dir,
    compare_at_positions,
    compare_pair_query,
    find_latest_capture,
    layer_profile_query,
)
from nls_kvp_helpers import (  # noqa: E402
    api_root_from_chat_url,
    token_count_inline_history,
)
from sweep_lib import (  # noqa: E402
    EMPTY_ASSISTANT_PLACEHOLDER,
    FACTS,
    RECALL,
    SYSTEM_PROMPT,
    chain_turn_stats,
    noise_prompt,
    resolve_model,
    wait_for_vllm,
)
(path: Path) -> list[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [(str(r["user"]), str(r.get("assistant") or "")) for r in raw]


def reconstruct_turns_from_sweep(sweep: dict, checkpoint: int) -> tuple[list[tuple[str, str]], dict]:
    """Best-effort user/assistant pairs from sweep metadata (assistants often empty)."""
    neutral_by_turn = {
        int(x["turn_index"]): str(x.get("substitute_assistant") or "")
        for x in (sweep.get("garbled_neutral_fallbacks") or [])
    }
    stripped_turns = {
        int(x["turn_index"])
        for x in (sweep.get("garbled_stripped") or [])
    }
    turns: list[tuple[str, str]] = []
    turn_idx = 0
    missing_assistants = 0

    for fact in FACTS:
        turn_idx += 1
        asst = neutral_by_turn.get(turn_idx, "")
        if not asst and turn_idx not in neutral_by_turn:
            missing_assistants += 1
        turns.append((fact, asst))

    for noise_i in range(checkpoint):
        turn_idx += 1
        user = noise_prompt(noise_i)
        asst = neutral_by_turn.get(turn_idx, "")
        if turn_idx in stripped_turns and not asst:
            asst = ""
        elif not asst and turn_idx not in neutral_by_turn:
            missing_assistants += 1
        turns.append((user, asst))

    meta = {
        "source": "sweep_reconstruct",
        "turn_count": len(turns),
        "missing_assistants_est": missing_assistants,
        "neutral_substitutes": len(neutral_by_turn),
        "garbled_stripped": len(stripped_turns),
        "assistants_incomplete": missing_assistants > 0,
    }
    return turns, meta


def turns_matched_to_budget(
    api_root: str,
    model: str,
    turns: list[tuple[str, str]],
    question: str,
    budget: int,
) -> list[tuple[str, str]]:
    if budget <= 0 or not turns:
        return turns
    best: list[tuple[str, str]] = turns[-1:]
    for k in range(1, len(turns) + 1):
        subset = turns[-k:]
        tok = token_count_inline_history(
            api_root,
            model,
            system_prompt=SYSTEM_PROMPT,
            turns=subset,
            question=question,
            include_assistant=any(asst for _, asst in subset),
            empty_assistant_placeholder=EMPTY_ASSISTANT_PLACEHOLDER,
        )
        if tok <= budget:
            best = subset
    return best


def inline_messages(
    turns: list[tuple[str, str]],
    question: str,
    *,
    minimal: bool = False,
) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if not minimal:
        for user_text, asst_text in turns:
            msgs.append({"role": "user", "content": user_text})
            msgs.append({
                "role": "assistant",
                "content": asst_text or EMPTY_ASSISTANT_PLACEHOLDER,
            })
    msgs.append({"role": "user", "content": question})
    return msgs


def recall_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def send_microscope(
    api: str,
    model: str,
    messages: list[dict],
    *,
    tag: str,
    user_id: str,
    kv_extras: dict | None = None,
) -> str:
    mem: dict[str, str] = {
        "memory_user": user_id,
        "memory_ring": "general",
        "memory_no_capture": "1",
        "microscope": MICROSCOPE_DIR,
        "microscope_tag": tag,
    }
    if kv_extras:
        mem.update(kv_extras)

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.0,
        "kv_transfer_params": mem,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"pri_parity_{tag}_{uuid.uuid4().hex[:8]}",
    }
    response = requests.post(api, json=body, timeout=300)
    response.raise_for_status()
    return (response.json()["choices"][0]["message"]["content"] or "").strip()


def capture_phantom_arm(
    api: str,
    model: str,
    *,
    turns: list[tuple[str, str]],
    matched: list[tuple[str, str]],
    question: str,
    user_id: str,
    base_session: str,
    tag_suffix: str,
    skip_full: bool,
) -> dict[str, str | None]:
    tags: dict[str, str] = {}

    if not skip_full:
        send_microscope(
            api, model,
            inline_messages(turns, question),
            tag=f"inline_full{tag_suffix}",
            user_id=user_id,
            kv_extras={"memory_off": "1"},
        )
        time.sleep(2)

        send_microscope(
            api, model,
            inline_messages(matched, question),
            tag=f"inline_matched{tag_suffix}",
            user_id=user_id,
            kv_extras={"memory_off": "1"},
        )
        time.sleep(2)

    send_microscope(
        api, model,
        inline_messages(turns, question, minimal=True),
        tag=f"inline_minimal{tag_suffix}",
        user_id=user_id,
        kv_extras={"memory_off": "1"},
    )
    time.sleep(2)

    send_microscope(
        api, model,
        recall_messages(question),
        tag=f"resume_inject{tag_suffix}",
        user_id=user_id,
        kv_extras={
            "memory_inject_mode": "resume",
            "memory_base_session": base_session,
        },
    )
    time.sleep(2)

    for key, tag in (
        ("inline_full", f"inline_full{tag_suffix}"),
        ("inline_matched", f"inline_matched{tag_suffix}"),
        ("inline_minimal", f"inline_minimal{tag_suffix}"),
        ("resume_inject", f"resume_inject{tag_suffix}"),
    ):
        tags[key] = find_latest_capture(tag)
    return tags


def run_recall_battery(
    api: str,
    model: str,
    *,
    user_id: str,
    base_session: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question, expected in RECALL:
        row = _recall_once(
            api, model,
            arm="resume",
            user_id=user_id,
            base_session=base_session,
            question=question,
            expected=expected,
        )
        rows.append(row)
        time.sleep(0.3)
    return rows


def evaluate_assumptions(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Map experiment results to H1–H5 verdicts."""
    verdicts: list[dict[str, Any]] = []
    parity = report.get("parity") or {}
    primary = parity.get("primary_question") or {}
    stages = (primary.get("minimal_vs_resume") or {}).get("stages") or {}

    attn = float((stages.get("attn_input_hs") or {}).get("query_cosine_avg") or 0.0)
    ssm = float((stages.get("ssm_state") or {}).get("query_cosine_avg") or 0.0)
    gap = attn - ssm

    if ssm > 0 and attn > 0:
        if gap >= 0.05:
            v = "supported"
            note = f"SSM query cos ({ssm:.3f}) trails attn ({attn:.3f}) by {gap:.3f}"
        elif gap <= -0.05:
            v = "refuted"
            note = f"SSM ({ssm:.3f}) meets/exceeds attn ({attn:.3f}); gap is not in DeltaNet"
        else:
            v = "inconclusive"
            note = f"Attn/ssm gap {gap:.3f} < 0.05 — both move together at query token"
    else:
        v = "inconclusive"
        note = "Missing attn or ssm capture tensors"
    verdicts.append({"id": "H1_ssm_vs_attn", "verdict": v, "evidence": note})

    worst = (stages.get("attn_input_hs") or {}).get("worst_layer") or {}
    worst_cos = float(worst.get("cosine") or 0.0)
    if worst_cos > 0 and attn > 0:
        if worst_cos < 0.85 and attn >= 0.90:
            v = "supported"
            note = f"Worst attn layer L{worst.get('layer')} cos={worst_cos:.3f} while mean={attn:.3f}"
        elif worst_cos >= 0.90:
            v = "refuted"
            note = f"All attn layers ≥0.90 (worst={worst_cos:.3f}); cliff is not layer-local attn"
        else:
            v = "inconclusive"
            note = f"Worst layer cos={worst_cos:.3f}, mean={attn:.3f}"
    else:
        v = "inconclusive"
        note = "No per-layer attn profile"
    verdicts.append({"id": "H2_worst_layer_cliff", "verdict": v, "evidence": note})

    inj = report.get("inject_scope") or {}
    facts = inj.get("resume_facts_only_3blocks") or {}
    full = inj.get("resume_full_chain") or {}
    if facts.get("pass_clean") is not None and full.get("pass_clean") is not None:
        if facts.get("pass_clean") and not full.get("pass_clean"):
            v = "supported"
            note = "Prefix 3-block inject passes; full chain fails — tail poison"
        elif not facts.get("pass_clean") and not full.get("pass_clean"):
            v = "refuted"
            note = "Facts-only inject also fails — not tail-only pollution"
        elif facts.get("pass_clean") and full.get("pass_clean"):
            v = "refuted"
            note = "Full chain passes on hotel re-probe — garble may be stochastic/thermal"
        else:
            v = "inconclusive"
            note = "Prefix passes but full fails on different metric"
    else:
        v = "inconclusive"
        note = "Inject scope experiment incomplete"
    verdicts.append({"id": "H3_tail_poison", "verdict": v, "evidence": note})

    iso = (report.get("decode_stability") or {}).get("isolated_hotel_resume") or {}
    seq = (report.get("decode_stability") or {}).get("sequential_through_hotel") or {}
    if iso.get("pass_clean") is not None and seq.get("pass_clean") is not None:
        if iso.get("pass_clean") and not seq.get("pass_clean"):
            v = "supported"
            note = "Isolated hotel OK, fails after prior probes — warm GPU / back-to-back pressure"
        elif not iso.get("pass_clean"):
            v = "supported"
            note = "Hotel fails even isolated — stable inject/decode failure at this chain length"
        elif iso.get("pass_clean") and seq.get("pass_clean"):
            v = "refuted"
            note = "Hotel stable isolated and sequential — sweep garble may be operational"
        else:
            v = "inconclusive"
            note = "Mixed sequential/isolated outcomes"
    else:
        v = "inconclusive"
        note = "Decode stability not measured"
    verdicts.append({"id": "H4_warm_decode", "verdict": v, "evidence": note})

    scan = report.get("chain_scan") or {}
    neutrals = len(scan.get("neutral_meta_previews") or [])
    if neutrals > 0 and attn > 0 and attn < 0.95:
        v = "inconclusive"
        note = (
            f"{neutrals} neutral-substitute blocks + attn cos {attn:.3f} — "
            "correlation possible; need A/B on cleaned chain"
        )
    elif neutrals > 0 and attn >= 0.95:
        v = "refuted"
        note = f"{neutrals} neutral blocks but high parity ({attn:.3f}) — pollution not limiting cos"
    elif neutrals == 0:
        v = "inconclusive"
        note = "No neutral-substitute previews in chain index"
    else:
        v = "inconclusive"
        note = "Neutral count vs parity unclear"
    verdicts.append({"id": "H5_neutral_pollution", "verdict": v, "evidence": note})

    high_cos_garble = False
    for row in report.get("recall_probes") or []:
        if row.get("garbled") and attn >= 0.90:
            high_cos_garble = True
            break
    if high_cos_garble:
        verdicts.append({
            "id": "H6_decode_not_repr",
            "verdict": "supported",
            "evidence": (
                f"Garbled decode with attn cos ≥{attn:.2f} — hybrid decode path, not representation"
            ),
        })

    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-sweep", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, default=80)
    parser.add_argument("--base-url", default=os.environ.get("PRI_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--turns-json", type=Path, default=None)
    parser.add_argument(
        "--question-index",
        type=int,
        default=4,
        help="1-based RECALL index for microscope (default 4 = hotel)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip HTTP captures; compare existing microscope tags on disk",
    )
    parser.add_argument("--skip-recall", action="store_true")
    parser.add_argument("--skip-inject-scope", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sweep = json.loads(args.from_sweep.read_text(encoding="utf-8"))
    user_id = sweep["user_id"]
    base_session = sweep["base_session"]
    api_root = api_root_from_chat_url(args.base_url)
    api = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    model = sweep.get("model") or os.environ.get("PRI_MODEL") or resolve_model(args.base_url)

    sweep_row = next(
        (r for r in (sweep.get("results") or []) if r.get("checkpoint_noise") == args.checkpoint),
        None,
    )

    if args.turns_json and args.turns_json.is_file():
        turns = load_turns_json(args.turns_json)
        turns_meta = {"source": str(args.turns_json), "turn_count": len(turns), "assistants_incomplete": False}
    else:
        turns, turns_meta = reconstruct_turns_from_sweep(sweep, args.checkpoint)

    skip_full = bool(turns_meta.get("assistants_incomplete"))
    q_idx = max(1, min(len(RECALL), args.question_index)) - 1
    question, _expected = RECALL[q_idx]
    tag_suffix = f"_cp{args.checkpoint}_q{q_idx + 1}"

    stats = chain_turn_stats(api_root, user_id, base_session)
    inject_tokens = stats["tokens"]
    matched = turns_matched_to_budget(api_root, model, turns, question, inject_tokens)

    report: dict[str, Any] = {
        "version": 1,
        "sweep_json": str(args.from_sweep),
        "checkpoint": args.checkpoint,
        "user_id": user_id,
        "base_session": base_session,
        "model": model,
        "microscope_dir": MICROSCOPE_DIR,
        "parity_threshold": PARITY_THRESHOLD,
        "sweep_row_scores": {
            "text_pass_clean": sweep_row.get("text_pass_clean") if sweep_row else None,
            "resume_pass_clean": sweep_row.get("resume_pass_clean") if sweep_row else None,
            "turn_tokens": sweep_row.get("turn_tokens") if sweep_row else stats["tokens"],
            "turn_blocks": sweep_row.get("turn_blocks") if sweep_row else stats["blocks"],
        },
        "turns_meta": turns_meta,
        "microscope_question": question,
        "microscope_question_index": q_idx + 1,
        "inject_tokens": inject_tokens,
        "matched_turns": len(matched),
    }

    print("=" * 72)
    print("RESUME PARITY ASSUMPTION TEST")
    print(f"chain user={user_id} cp={args.checkpoint} inject={inject_tokens}tok")
    if skip_full:
        print("  [warn] assistants incomplete — skipping inline_full/matched captures")
    print("=" * 72)

    if not args.compare_only:
        if not wait_for_vllm(api_root):
            print("FATAL: vLLM not healthy", file=sys.stderr)
            return 1

        report["chain_scan"] = _scan_chain_previews(api_root, user_id, base_session)

        if not args.skip_recall:
            print("\n--- B: Recall probes (RESUME) ---")
            report["recall_probes"] = run_recall_battery(
                api, model, user_id=user_id, base_session=base_session,
            )
            passed = sum(1 for r in report["recall_probes"] if r.get("pass_clean"))
            print(f"  RESUME recall: {passed}/{len(RECALL)}")

        if not args.skip_inject_scope:
            hotel_q, hotel_exp = RECALL[3]
            print("\n--- C/D: Inject scope + decode stability ---")
            decode: dict[str, Any] = {}
            decode["isolated_hotel_resume"] = _recall_once(
                api, model, arm="resume", user_id=user_id, base_session=base_session,
                question=hotel_q, expected=hotel_exp,
            )
            for q, exp in RECALL[:3]:
                _recall_once(
                    api, model, arm="resume", user_id=user_id, base_session=base_session,
                    question=q, expected=exp,
                )
                time.sleep(0.3)
            decode["sequential_through_hotel"] = _recall_once(
                api, model, arm="resume", user_id=user_id, base_session=base_session,
                question=hotel_q, expected=hotel_exp,
            )
            repeats: list[dict] = []
            for i in range(3):
                repeats.append(_recall_once(
                    api, model, arm="resume", user_id=user_id, base_session=base_session,
                    question=hotel_q, expected=hotel_exp,
                ))
                time.sleep(0.5)
            decode["hotel_resume_repeats"] = repeats
            report["decode_stability"] = decode

            report["inject_scope"] = {
                "resume_facts_only_3blocks": _recall_once(
                    api, model, arm="resume", user_id=user_id, base_session=base_session,
                    question=hotel_q, expected=hotel_exp, max_blocks=3,
                ),
                "resume_full_chain": decode["isolated_hotel_resume"],
            }
            print(
                f"  hotel max_blocks=3: {report['inject_scope']['resume_facts_only_3blocks'].get('pass_clean')} "
                f"full: {report['inject_scope']['resume_full_chain'].get('pass_clean')}",
            )

        print("\n--- A: Microscope captures ---")
        clear_microscope_dir()
        captures = capture_phantom_arm(
            api, model,
            turns=turns,
            matched=matched,
            question=question,
            user_id=user_id,
            base_session=base_session,
            tag_suffix=tag_suffix,
            skip_full=skip_full,
        )
        report["captures"] = captures
    else:
        report["chain_scan"] = _scan_chain_previews(api_root, user_id, base_session)
        captures = {
            "inline_full": find_latest_capture(f"inline_full{tag_suffix}"),
            "inline_matched": find_latest_capture(f"inline_matched{tag_suffix}"),
            "inline_minimal": find_latest_capture(f"inline_minimal{tag_suffix}"),
            "resume_inject": find_latest_capture(f"resume_inject{tag_suffix}"),
        }
        report["captures"] = captures

    print("\n--- A: Parity compare (query token) ---")
    parity: dict[str, Any] = {"primary_question": {}}
    pairs = [
        ("minimal_vs_resume", f"inline_minimal{tag_suffix}", f"resume_inject{tag_suffix}"),
    ]
    if not skip_full:
        pairs.extend([
            ("full_vs_resume", f"inline_full{tag_suffix}", f"resume_inject{tag_suffix}"),
            ("matched_vs_resume", f"inline_matched{tag_suffix}", f"resume_inject{tag_suffix}"),
        ])

    for label, tag_a, tag_b in pairs:
        result = compare_pair_query(tag_a, tag_b)
        parity["primary_question"][label] = result
        if result.get("error"):
            print(f"  {label}: ERROR {result['error']}")
            continue
        attn = result["stages"]["attn_input_hs"]["query_cosine_avg"]
        ssm = result["stages"]["ssm_state"]["query_cosine_avg"]
        print(f"  {label}: attn={attn:.4f} ssm={ssm:.4f} gap={result['attn_vs_ssm_gap']:.4f}")

    min_tag_a = f"inline_minimal{tag_suffix}"
    res_tag_b = f"resume_inject{tag_suffix}"
    parity["boundary"] = {
        "minimal_pos0_vs_resume_pos0_attn": compare_at_positions(
            min_tag_a, res_tag_b,
            text_pos=0, kv_pos=0,
            label="minimal[0] vs resume[0]",
            stage_suffix="attn_input_hs",
        ),
        "minimal_pos0_vs_resume_pos0_ssm": compare_at_positions(
            min_tag_a, res_tag_b,
            text_pos=0, kv_pos=0,
            label="minimal[0] vs resume[0]",
            stage_suffix="ssm_state",
        ),
        "minimal_query_vs_resume_query_ssm": compare_at_positions(
            min_tag_a, res_tag_b,
            text_pos=-1, kv_pos=-1,
            label="minimal[-1] vs resume[-1]",
            stage_suffix="ssm_state",
        ),
    }
    parity["layer_profile_minimal_vs_resume"] = layer_profile_query(min_tag_a, res_tag_b)
    report["parity"] = parity

    report["assumption_verdicts"] = evaluate_assumptions(report)
    if report.get("decode_stability") and report.get("inject_scope"):
        report["garble_interpretation"] = _interpret(
            {
                "chain_scan": report.get("chain_scan"),
                "isolated_hotel_resume": report["decode_stability"].get("isolated_hotel_resume"),
                "sequential_through_hotel": report["decode_stability"].get("sequential_through_hotel"),
                "resume_facts_only_3blocks": report["inject_scope"].get("resume_facts_only_3blocks"),
                "resume_full_chain": report["inject_scope"].get("resume_full_chain"),
                "hotel_resume_repeats": report["decode_stability"].get("hotel_resume_repeats"),
            },
            sweep_row,
        )

    print("\n" + "=" * 72)
    print("ASSUMPTION VERDICTS")
    for row in report["assumption_verdicts"]:
        print(f"  [{row['verdict']:>12}] {row['id']}: {row['evidence']}")
    print("=" * 72)

    out_path = args.out or args.from_sweep.with_name(
        args.from_sweep.stem + f"_parity_assumption_cp{args.checkpoint}.json",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
