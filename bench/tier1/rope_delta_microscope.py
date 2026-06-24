#!/usr/bin/env python3
"""Microscopic RoPE delta analysis for resume inject packs.

Computes delta-uniformity KPI (target ~0.99+) and flags outlier blocks with
phantom / manifest anomalies. Use after geometry_audit JSON exists:

  python bench/tier1/rope_delta_microscope.py \\
      --geometry bench/results/overnight_.../geometry_audit_turn_sweep_v5.json \\
      --sweep bench/results/overnight_.../turn_sweep_cp20_80_v5.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _mode_delta(rows: list[dict]) -> int | None:
    deltas = [int(r["rope_delta"]) for r in rows if r.get("rope_delta") is not None]
    if not deltas:
        return None
    return Counter(deltas).most_common(1)[0][0]


def analyze_geometry(
    geometry: dict,
    *,
    sweep: dict | None = None,
) -> dict:
    rows = list(geometry.get("rope_pack") or [])
    if not rows:
        return {"error": "empty rope_pack"}

    mode = _mode_delta(rows)
    outliers: list[dict] = []
    for r in rows:
        delta = int(r.get("rope_delta") or 0)
        if mode is not None and delta != mode:
            pack = int(r.get("pack_offset") or 0)
            phantom = int(r.get("capture_num_phantom") or 0)
            rope_start = int(r.get("manifest_rope_start") or 0)
            rope_old = int(r.get("rope_old_effective") or 0)
            expected_phantom = max(0, rope_old - rope_start) if rope_start else pack
            outliers.append({
                "index": r.get("index"),
                "turn_index": r.get("turn_index"),
                "session_id": r.get("session_id"),
                "rope_delta": delta,
                "mode_delta": mode,
                "pack_offset": pack,
                "manifest_rope_start": rope_start,
                "capture_num_phantom": phantom,
                "rope_old_effective": rope_old,
                "expected_phantom_from_rope_old": expected_phantom,
                "phantom_mismatch": phantom - expected_phantom if phantom else None,
                "num_tokens": r.get("num_tokens"),
            })

    n = len(rows)
    uniform_count = sum(1 for r in rows if int(r.get("rope_delta") or 0) == mode)
    delta_uniformity = round(uniform_count / n, 4) if n and mode is not None else 0.0

    deltas = [int(r["rope_delta"]) for r in rows]
    continuity = geometry.get("chain_continuity") or {}

    report: dict = {
        "block_count": n,
        "total_inject_tokens": geometry.get("total_inject_tokens"),
        "verdict": geometry.get("verdict"),
        "mode_rope_delta": mode,
        "delta_uniformity": delta_uniformity,
        "delta_uniformity_pct": round(delta_uniformity * 100, 2),
        "uniform_blocks": uniform_count,
        "outlier_blocks": len(outliers),
        "delta_distribution": dict(Counter(deltas)),
        "delta_stdev": round(statistics.pstdev(deltas), 4) if len(deltas) > 1 else 0.0,
        "manifest_rope_gaps": continuity.get("gap_count"),
        "uniform_manifest_rope_start": continuity.get("uniform_manifest_rope_start"),
        "shared_rope_start": continuity.get("shared_rope_start"),
        "outliers": outliers,
        "kpi_notes": (
            "delta_uniformity = fraction of blocks sharing mode RoPE delta "
            "(research target ~0.92–0.99; 1.0 = perfectly uniform pack rotation)."
        ),
    }

    if sweep:
        neutrals = {
            int(x["turn_index"]): x
            for x in (sweep.get("garbled_neutral_fallbacks") or [])
        }
        for o in outliers:
            ti = o.get("turn_index")
            if ti in neutrals:
                o["neutral_fallback"] = neutrals[ti].get("label")
                o["garbled_probe_attempts"] = neutrals[ti].get("garbled_probe_attempts")

    if delta_uniformity >= 0.99:
        report["kpi_grade"] = "excellent"
    elif delta_uniformity >= 0.92:
        report["kpi_grade"] = "research_baseline"
    else:
        report["kpi_grade"] = "needs_review"

    if outliers:
        report["likely_causes"] = []
        if any(o.get("phantom_mismatch") for o in outliers):
            report["likely_causes"].append(
                "capture_num_phantom mismatch on outlier block(s) — connector "
                "may have written wrong num_phantom at capture (check pri/connector.py)"
            )
        if any(o.get("neutral_fallback") for o in outliers):
            report["likely_causes"].append(
                "outlier aligns with garbled neutral-fallback turn — garbled plant "
                "before substitute capture may corrupt phantom metadata"
            )
        if continuity.get("gap_count"):
            report["likely_causes"].append(
                "manifest rope_start/rope_end gaps between blocks (expected for "
                "per-request rope_start=22 resume-shaped captures; not pack gaps)"
            )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    sweep = None
    if args.sweep and args.sweep.is_file():
        sweep = json.loads(args.sweep.read_text(encoding="utf-8"))

    report = analyze_geometry(geometry, sweep=sweep)
    out_path = args.out or args.geometry.with_name(
        args.geometry.stem.replace("geometry_audit", "rope_delta_microscope") + ".json"
    )
    if out_path == args.geometry:
        out_path = args.geometry.with_name("rope_delta_microscope.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"RoPE delta microscope — {args.geometry.name}")
    print(f"  blocks:           {report.get('block_count')}")
    print(f"  mode delta:         {report.get('mode_rope_delta')}")
    print(f"  delta_uniformity:   {report.get('delta_uniformity_pct')}% ({report.get('uniform_blocks')}/{report.get('block_count')})")
    print(f"  KPI grade:          {report.get('kpi_grade')}")
    print(f"  verdict (input):    {report.get('verdict')}")
    print(f"  outliers:           {report.get('outlier_blocks')}")
    for o in report.get("outliers") or []:
        print(
            f"    turn {o.get('turn_index')}: delta={o.get('rope_delta')} "
            f"phantom={o.get('capture_num_phantom')} "
            f"expected~{o.get('expected_phantom_from_rope_old')} "
            f"{o.get('neutral_fallback') or ''}"
        )
    print(f"  saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
