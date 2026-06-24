#!/usr/bin/env python3
"""Classify bench failures from overnight/rerun JSON artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pri.text_quality import is_garbled_response

GARBLE_RE = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\u0400-\u04ff]|denun|prezi|contempora",
    re.I,
)
REFUSAL_RE = re.compile(
    r"i don.?t have (access|information|any record)|could you tell me",
    re.I,
)
EMPTY_OR_SHORT = re.compile(r"^\s*$")


def classify_answer(text: str) -> str:
    if EMPTY_OR_SHORT.match(text or ""):
        return "empty"
    if is_garbled_response(text) or GARBLE_RE.search(text or ""):
        return "garbled"
    if REFUSAL_RE.search(text or ""):
        return "refusal"
    return "normal"


def audit_inject_compare(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {
        "file": path.name,
        "user_id": d.get("user_id"),
        "base_session": d.get("base_session"),
        "text_backend": d.get("text_backend"),
        "noise_turns": d.get("noise_turns"),
        "arms": {},
    }
    results = d.get("results") or {}
    summary = d.get("summary") or {}
    for arm in ("text", "resume", "resume_overflow"):
        rows = results.get(arm) or []
        fails = [r for r in rows if not r.get("pass")]
        out["arms"][arm] = {
            "pass": (summary.get(arm) or {}).get("pass", d.get(f"{arm}_pass")),
            "total": 5,
            "failures": [],
        }
        for r in fails:
            ans = r.get("answer") or ""
            kind = classify_answer(ans)
            out["arms"][arm]["failures"].append({
                "question": r.get("question", "")[:60],
                "kind": kind,
                "backend": r.get("backend"),
                "reasoning_tokens": (r.get("usage") or {}).get("completion_tokens_details", {})
                if isinstance((r.get("usage") or {}).get("completion_tokens_details"), dict)
                else None,
                "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                "preview": ans[:120].replace("\n", " "),
            })
    return out


def audit_marco(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    fails = []
    for key in ("text_results", "resume_results"):
        for r in d.get(key) or []:
            if not r.get("pass"):
                ans = r.get("answer") or ""
                fails.append({
                    "arm": r.get("arm"),
                    "backend": r.get("backend"),
                    "question": r.get("question", "")[:60],
                    "kind": classify_answer(ans),
                    "preview": ans[:120].replace("\n", " "),
                    "reasoning_tokens": (r.get("usage") or {}).get("completion_tokens_details"),
                })
    return {
        "file": path.name,
        "user_id": d.get("user_id"),
        "base_session": d.get("base_session"),
        "text_backend": d.get("text_backend"),
        "text_pass": d.get("text_pass"),
        "resume_pass": d.get("resume_pass"),
        "failures": fails,
    }


def audit_turn_sweep(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    cps = []
    for r in d.get("results") or []:
        cp = r.get("checkpoint_noise")
        cps.append({
            "checkpoint": cp,
            "text": r.get("text_pass_clean"),
            "resume": r.get("resume_pass_clean"),
            "arm_d": r.get("arm_d_pass_clean"),
            "blocks": r.get("turn_blocks"),
            "inject_tok": r.get("turn_tokens"),
        })
        for arm, key in (("resume", "resume_recall"), ("text", "text_recall")):
            for probe in r.get(key) or []:
                if not probe.get("pass_clean") and not probe.get("pass"):
                    ans = probe.get("answer_preview") or probe.get("answer") or ""
                    if not ans and arm == "resume":
                        continue
    neutrals = d.get("garbled_neutral_fallbacks") or []
    stripped = d.get("garbled_stripped") or []
    return {
        "file": path.name,
        "user_id": d.get("user_id"),
        "base_session": d.get("base_session"),
        "checkpoints": cps,
        "neutral_fallbacks": len(neutrals),
        "still_garbled_stripped": len(stripped),
        "garbled_deletes": d.get("garbled_captures_deleted"),
    }


def infer_root_cause(arm: str, kind: str, backend: str, text_backend: str) -> str:
    if backend == "openrouter" or (arm == "text" and text_backend == "openrouter"):
        if kind == "empty":
            return "HARNESS: OpenRouter Qwen3.5 reasoning burns max_tokens — no enable_thinking:false"
        return "BASELINE: OpenRouter model behavior / scoring"
    if kind == "garbled":
        if arm in ("resume", "resume_overflow"):
            return "PRODUCT: hybrid KV inject decode collapse (or unguarded garbled capture in chain)"
        return "PRODUCT/UNKNOWN: garbled on TEXT inline"
    if kind == "refusal":
        return "PRODUCT: policy refusal under inject stress (not decode collapse)"
    if kind == "empty" and arm == "text":
        return "HARNESS or BASELINE: empty completion"
    return "UNKNOWN"


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "bench/results/overnight_20260624_003614"
    if not out_dir.is_dir():
        print(f"Missing {out_dir}", file=sys.stderr)
        return 1

    report: dict = {"out_dir": str(out_dir), "runs": [], "isolation": [], "findings": []}

    # Isolation: unique user_ids
    user_ids: list[tuple[str, str, str]] = []
    for p in sorted(out_dir.glob("*.json")):
        if p.name == "manifest.json":
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        uid = d.get("user_id")
        bs = d.get("base_session")
        if uid and bs:
            user_ids.append((p.name, uid, bs))

    report["isolation"] = [{"file": f, "user_id": u, "base_session": b} for f, u, b in user_ids]
    uids = {u for _, u, _ in user_ids}
    report["unique_user_count"] = len(uids)

    for p in sorted(out_dir.glob("inject_mode_compare*.json")):
        if "rerun" in p.name:
            continue
        ac = audit_inject_compare(p)
        for arm, data in ac["arms"].items():
            for fail in data.get("failures") or []:
                fail["root_cause"] = infer_root_cause(
                    arm, fail["kind"], fail.get("backend") or "pri", ac.get("text_backend") or "local"
                )
        report["runs"].append({"type": "inject_mode_compare", **ac})

    for p in sorted(out_dir.glob("tier1_marco_facts*.json")):
        am = audit_marco(p)
        for fail in am["failures"]:
            fail["root_cause"] = infer_root_cause(
                fail["arm"] or "text",
                fail["kind"],
                fail.get("backend") or "pri",
                am.get("text_backend") or "local",
            )
        report["runs"].append({"type": "marco_facts", **am})

    sweep = out_dir / "turn_sweep_cp20_80_v5.json"
    if sweep.is_file():
        report["turn_sweep"] = audit_turn_sweep(sweep)

    geo = out_dir / "geometry_audit_turn_sweep_v5.json"
    if geo.is_file():
        g = json.loads(geo.read_text(encoding="utf-8"))
        report["geometry"] = {"verdict": g.get("verdict"), "blocks": g.get("block_count")}

    manifest = out_dir / "manifest_opencode_t2.json"
    if manifest.is_file():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        report["manifest_proof"] = {
            "rope_start_kvp": m.get("memory_capture_start_t2"),
            "manifest_on_disk": m.get("manifest") is not None,
        }

    # Aggregate findings
    harness = product = baseline = unknown = 0
    for run in report["runs"]:
        for arm_data in run.get("arms", {}).values():
            for fail in arm_data.get("failures") or []:
                rc = fail.get("root_cause", "")
                if rc.startswith("HARNESS"):
                    harness += 1
                elif rc.startswith("BASELINE"):
                    baseline += 1
                elif rc.startswith("PRODUCT"):
                    product += 1
                else:
                    unknown += 1
        for fail in run.get("failures") or []:
            rc = fail.get("root_cause", "")
            if rc.startswith("HARNESS"):
                harness += 1
            elif rc.startswith("BASELINE"):
                baseline += 1
            elif rc.startswith("PRODUCT"):
                product += 1
            else:
                unknown += 1

    report["summary"] = {
        "harness_failures": harness,
        "baseline_failures": baseline,
        "product_failures": product,
        "unknown_failures": unknown,
    }

    out_json = out_dir / "failure_audit.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
