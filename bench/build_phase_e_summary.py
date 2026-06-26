#!/usr/bin/env python3
"""Build benchmark publication summary from canonical bench artifacts.

Usage:
    python bench/build_phase_e_summary.py \\
        --run-dir bench/results/overnight_20260624_003614 \\
        --out-json bench/results/overnight_20260624_003614/phase_e_summary.json \\
        --out-md bench/results/overnight_20260624_003614/BENCHMARK_SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent

# Canonical artifact filenames relative to run-dir (post-fix / reasoning_none preferred).
CANONICAL = {
    "inject_short_local": "inject_mode_compare_20260624_003614_short.json",
    "inject_long12_local": "inject_mode_compare_20260624_003614_long12_postfix.json",
    "inject_short_openrouter": "inject_mode_compare_short_openrouter_reasoning_none.json",
    "inject_long12_openrouter": "inject_mode_compare_long12_openrouter_reasoning_none.json",
    "inject_long12_resume4096": "inject_mode_compare_long12_resume4096_phase_e.json",
    "marco_local": "tier1_marco_facts_20260624_003614_marco.json",
    "marco_openrouter": "tier1_marco_facts_openrouter_reasoning_none.json",
    "opencode_pri": "opencode_long_session_20260624_003614.json",
    "opencode_baseline": "opencode_long_session_baseline_seed42.json",
    "turn_sweep": "turn_sweep_cp20_80_v5.json",
    "turn_sweep_garble_inv": "turn_sweep_cp60_80_garble_inv.json",
    "geometry_v5": "geometry_audit_turn_sweep_v5_fixed.json",
    "geometry_garble_inv": "geometry_audit_garble_inv.json",
    "rope_microscope_v5": "rope_delta_microscope_fixed.json",
    "rope_microscope_garble_inv": "rope_delta_microscope_garble_inv.json",
    "store_stats": "store_stats.json",
    "garble_cause_cp60": "turn_sweep_cp60_80_garble_inv_garble_cause_cp60.json",
}


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def _arm_row(summary: dict, arm: str) -> dict[str, Any]:
    s = summary.get(arm) or {}
    usage = s.get("usage") or {}
    reqs = max(int(usage.get("requests") or 1), 1)
    prompt = int(usage.get("prompt_tokens") or 0)
    return {
        "recall_pass": s.get("pass"),
        "recall_total": s.get("total"),
        "latency_ms_mean": s.get("latency_ms_mean"),
        "latency_ms_p95": s.get("latency_ms_p95"),
        "prompt_tokens_total": prompt,
        "prompt_tokens_mean": round(prompt / reqs, 1),
        "completion_tokens_total": int(usage.get("completion_tokens") or 0),
    }


def _inject_table(data: dict | None, label: str) -> dict[str, Any] | None:
    if not data:
        return None
    summary = data.get("summary") or {}
    text = _arm_row(summary, "text")
    resume = _arm_row(summary, "resume")
    overflow = _arm_row(summary, "resume_overflow")
    text_prompt = text["prompt_tokens_mean"]
    return {
        "label": label,
        "text_backend": data.get("text_backend", "local"),
        "openrouter_model": data.get("openrouter_model"),
        "noise_turns": data.get("noise_turns"),
        "resume_max_tokens": data.get("resume_max_tokens"),
        "text": text,
        "resume": resume,
        "resume_overflow": overflow,
        "delta_prompt_tokens_mean_vs_text": round(text_prompt - resume["prompt_tokens_mean"], 1),
        "recommended_default": "resume",
        "artifact": label,
    }


def _marco_table(data: dict | None, label: str) -> dict[str, Any] | None:
    if not data:
        return None
    return {
        "label": label,
        "text_backend": data.get("text_backend", "local"),
        "text_pass": data.get("text_pass"),
        "text_total": data.get("text_total"),
        "resume_pass": data.get("resume_pass"),
        "resume_total": data.get("resume_total"),
    }


def _opencode_table(data: dict | None, label: str) -> dict[str, Any] | None:
    if not data:
        return None
    return {
        "label": label,
        "baseline": data.get("baseline", False),
        "recall_passed": data.get("recall_passed"),
        "recall_total": data.get("recall_total"),
        "work_turns": data.get("work_turns"),
        "seed": data.get("seed"),
    }


def _sweep_table(data: dict | None, label: str) -> list[dict[str, Any]]:
    if not data:
        return []
    rows: list[dict[str, Any]] = []
    for r in data.get("results") or []:
        rows.append({
            "checkpoint_noise": r.get("checkpoint_noise"),
            "turn_tokens": r.get("turn_tokens"),
            "turn_blocks": r.get("turn_blocks"),
            "text_pass": r.get("text_pass_clean"),
            "resume_pass": r.get("resume_pass_clean"),
            "arm_d_pass": r.get("arm_d_pass_clean"),
            "total": r.get("total"),
        })
    return rows


def _geometry_kpi(data: dict | None, microscope: dict | None, label: str) -> dict[str, Any] | None:
    if not data and not microscope:
        return None
    return {
        "label": label,
        "verdict": (data or {}).get("verdict"),
        "blocks": (microscope or {}).get("block_count") or (data or {}).get("blocks"),
        "delta_uniformity_pct": (microscope or {}).get("delta_uniformity_pct"),
        "mode_rope_delta": (microscope or {}).get("mode_rope_delta"),
        "kpi_grade": (microscope or {}).get("kpi_grade"),
        "note": "delta_uniformity = RoPE pack re-rotation consistency (agent-room KPI; not retrieval cosine)",
    }


def _store_table(data: dict | None) -> dict[str, Any] | None:
    if not data:
        return None
    admin = data.get("admin_stats") or {}
    cap_bytes = int(data.get("capture_bytes_total") or 0)
    return {
        "capture_count": data.get("capture_count"),
        "index_row_count": data.get("index_row_count"),
        "capture_disk_mb": round(cap_bytes / (1024 * 1024), 2),
        "data_dir_mb": round(int(data.get("data_dir_bytes") or 0) / (1024 * 1024), 2),
        "admin_total_tokens": admin.get("total_tokens"),
    }


def _phase_e_primary(inject_long: dict | None) -> dict[str, Any]:
    """Phase E headline row from long12 local inject compare."""
    if not inject_long:
        return {}
    t = inject_long["text"]
    r = inject_long["resume"]
    o = inject_long["resume_overflow"]
    return {
        "scenario": "inject_mode_compare long12 (local, garbled guard)",
        "recall_at_5": {
            "TEXT": f"{t['recall_pass']}/{t['recall_total']}",
            "RESUME": f"{r['recall_pass']}/{r['recall_total']}",
            "OVERFLOW": f"{o['recall_pass']}/{o['recall_total']}",
        },
        "mean_prompt_tokens": {
            "TEXT": t["prompt_tokens_mean"],
            "RESUME": r["prompt_tokens_mean"],
            "OVERFLOW": o["prompt_tokens_mean"],
            "delta_text_minus_resume": inject_long["delta_prompt_tokens_mean_vs_text"],
        },
        "mean_latency_ms": {
            "TEXT": t["latency_ms_mean"],
            "RESUME": r["latency_ms_mean"],
            "OVERFLOW": o["latency_ms_mean"],
        },
    }


def _render_md(report: dict) -> str:
    lines = [
        "# Benchmark proof summary",
        "",
        f"Generated: {report.get('generated_at')}",
        f"Run dir: `{report.get('run_dir')}`",
        f"Git SHA: `{report.get('git_sha') or 'unknown'}`",
        f"Default inject mode: **{report.get('recommended_inject_mode', 'resume')}**",
        "",
        "## Headline (long chain, local PRI)",
        "",
    ]
    pe = report.get("phase_e_primary") or {}
    if pe:
        r = pe.get("recall_at_5") or {}
        p = pe.get("mean_prompt_tokens") or {}
        lat = pe.get("mean_latency_ms") or {}
        lines.extend([
            "| Metric | TEXT | RESUME | OVERFLOW | Δ vs TEXT (prompt tok) |",
            "|--------|------|--------|----------|-------------------------|",
            f"| Recall @5 | {r.get('TEXT')} | {r.get('RESUME')} | {r.get('OVERFLOW')} | — |",
            f"| Mean prompt tok | {p.get('TEXT')} | {p.get('RESUME')} | {p.get('OVERFLOW')} | {p.get('delta_text_minus_resume')} |",
            f"| Mean latency ms | {lat.get('TEXT')} | {lat.get('RESUME')} | {lat.get('OVERFLOW')} | — |",
            "",
        ])

    lines.extend([
        "## Inject mode compare",
        "",
        "| Scenario | TEXT | RESUME | OVERFLOW | Δ prompt (mean) |",
        "|----------|------|--------|----------|-----------------|",
    ])
    for row in report.get("inject_mode_compare") or []:
        t, r, o = row["text"], row["resume"], row["resume_overflow"]
        lines.append(
            f"| {row['label']} ({row.get('text_backend')}) "
            f"| {t['recall_pass']}/{t['recall_total']} "
            f"| {r['recall_pass']}/{r['recall_total']} "
            f"| {o['recall_pass']}/{o['recall_total']} "
            f"| {row['delta_prompt_tokens_mean_vs_text']} |"
        )
    lines.append("")

    lines.extend([
        "## Tier-1 Marco",
        "",
        "| Run | TEXT | RESUME |",
        "|-----|------|--------|",
    ])
    for m in report.get("marco") or []:
        lines.append(
            f"| {m['label']} ({m.get('text_backend')}) "
            f"| {m.get('text_pass')}/{m.get('text_total')} "
            f"| {m.get('resume_pass')}/{m.get('resume_total')} |"
        )
    lines.append("")

    lines.extend([
        "## OpenCode long session (seed 42)",
        "",
        "| Arm | RECALL |",
        "|-----|--------|",
    ])
    for oc in report.get("opencode") or []:
        tag = "baseline (memory_off)" if oc.get("baseline") else "PRI"
        lines.append(f"| {tag} | {oc.get('recall_passed')}/{oc.get('recall_total')} |")
    lines.append("")

    lines.extend([
        "## Turn sweep (TEXT / RESUME / ARM-D)",
        "",
        "| cp | inject tok | TEXT | RESUME | ARM-D |",
        "|----|------------|------|--------|-------|",
    ])
    for sweep_name, rows in (report.get("turn_sweeps") or {}).items():
        for row in rows:
            ad = row.get("arm_d_pass")
            ad_s = f"{ad}/{row.get('total')}" if ad is not None else "—"
            lines.append(
                f"| {sweep_name} cp{row.get('checkpoint_noise')} "
                f"| {row.get('turn_tokens')} "
                f"| {row.get('text_pass')}/{row.get('total')} "
                f"| {row.get('resume_pass')}/{row.get('total')} "
                f"| {ad_s} |"
            )
    lines.append("")

    geo = report.get("geometry") or []
    if geo:
        lines.extend([
            "## RoPE pack geometry (delta_uniformity KPI)",
            "",
            "| Chain | Verdict | Blocks | delta_uniformity | mode Δ |",
            "|-------|---------|--------|------------------|--------|",
        ])
        for g in geo:
            lines.append(
                f"| {g.get('label')} | {g.get('verdict')} | {g.get('blocks')} "
                f"| {g.get('delta_uniformity_pct')}% | {g.get('mode_rope_delta')} |"
            )
        lines.append("")

    store = report.get("store") or {}
    if store:
        lines.extend([
            "## Storage (turn-sweep session)",
            "",
            f"- Captures: {store.get('capture_count')} files, **{store.get('capture_disk_mb')} MB**",
            f"- Index rows: {store.get('index_row_count')}",
            f"- Data dir: {store.get('data_dir_mb')} MB",
            "",
        ])

    lim = report.get("known_limitations") or []
    if lim:
        lines.extend(["## Known limitations (documented, not bench failures)", ""])
        for item in lim:
            lines.append(f"- {item}")
        lines.append("")

    lines.extend([
        "## Canonical artifacts",
        "",
        "See `canonical_artifacts.json` in this folder for full paths.",
        "",
        "## Research analysis (extended)",
        "",
        "Detailed pages with charts: [`research/README.md`](research/README.md)",
        "",
        "| Topic | File |",
        "|-------|------|",
        "| Token efficiency | [research/02_token_efficiency.md](research/02_token_efficiency.md) |",
        "| Latency | [research/03_latency_analysis.md](research/03_latency_analysis.md) |",
        "| Computational cost | [research/04_computational_cost.md](research/04_computational_cost.md) |",
        "| Storage | [research/05_storage_footprint.md](research/05_storage_footprint.md) |",
        "| Turn-sweep scaling | [research/06_turn_sweep_scaling.md](research/06_turn_sweep_scaling.md) |",
        "| RoPE geometry | [research/07_rope_geometry.md](research/07_rope_geometry.md) |",
        "| Failure modes | [research/08_failure_modes.md](research/08_failure_modes.md) |",
        "| Energy & cost | [research/09_energy_and_cost.md](research/09_energy_and_cost.md) |",
        "",
    ])
    return "\n".join(lines)


def build_report(run_dir: Path) -> dict[str, Any]:
    # Prefer phase_e resume4096; fall back to ropefix rerun.
    resume4096_name = CANONICAL["inject_long12_resume4096"]
    if not (run_dir / resume4096_name).is_file():
        fallback = run_dir / "inject_mode_compare_long12_resume4096_ropefix.json"
        if fallback.is_file():
            CANONICAL_LOCAL = {**CANONICAL, "inject_long12_resume4096": fallback.name}
        else:
            CANONICAL_LOCAL = CANONICAL
    else:
        CANONICAL_LOCAL = CANONICAL

    paths = {k: run_dir / v for k, v in CANONICAL_LOCAL.items()}
    loaded = {k: _load(p) for k, p in paths.items()}

    inject_rows = [
        _inject_table(loaded["inject_short_local"], "short chain (0 noise)"),
        _inject_table(loaded["inject_long12_local"], "long12 chain"),
        _inject_table(loaded["inject_short_openrouter"], "short OpenRouter TEXT"),
        _inject_table(loaded["inject_long12_openrouter"], "long12 OpenRouter TEXT"),
        _inject_table(loaded["inject_long12_resume4096"], "long12 resume_max_tokens=4096"),
    ]
    inject_rows = [r for r in inject_rows if r]

    marco_rows = [
        _marco_table(loaded["marco_local"], "Marco local"),
        _marco_table(loaded["marco_openrouter"], "Marco OpenRouter"),
    ]
    marco_rows = [r for r in marco_rows if r]

    opencode_rows = [
        _opencode_table(loaded["opencode_pri"], "OpenCode PRI"),
        _opencode_table(loaded["opencode_baseline"], "OpenCode baseline"),
    ]
    opencode_rows = [r for r in opencode_rows if r]

    long12 = next((r for r in inject_rows if "long12 chain" in r.get("label", "")), None)

    missing = [k for k, p in paths.items() if k in CANONICAL_LOCAL and not p.is_file()]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.relative_to(_ROOT)).replace("\\", "/"),
        "git_sha": _git_sha(),
        "recommended_inject_mode": "resume",
        "recommended_inject_mode_rationale": (
            "Short + long12 inject compare: RESUME matches or beats OVERFLOW on recall; "
            "resume_overflow does not recover cp60+ turn-sweep cliff."
        ),
        "phase_e_primary": _phase_e_primary(long12),
        "inject_mode_compare": inject_rows,
        "marco": marco_rows,
        "opencode": opencode_rows,
        "turn_sweeps": {
            "v5_overnight": _sweep_table(loaded["turn_sweep"], "v5"),
            "garble_investigation": _sweep_table(loaded["turn_sweep_garble_inv"], "garble_inv"),
        },
        "geometry": [
            _geometry_kpi(loaded["geometry_v5"], loaded["rope_microscope_v5"], "turn_sweep v5 (RoPE fix audit)"),
            _geometry_kpi(loaded["geometry_garble_inv"], loaded["rope_microscope_garble_inv"], "garble_inv chain"),
        ],
        "store": _store_table(loaded["store_stats"]),
        "garble_root_cause_summary": (loaded["garble_cause_cp60"] or {}).get("interpretation"),
        "known_limitations": [
            "Turn sweep cp60+: RESUME garbled decode while TEXT 5/5 (~17–23k inject tokens) — inject-mediated, not RoPE geometry.",
            "Facts-only inject (max_blocks=3) still garbles at cp60+ — not tail-noise text pollution alone.",
            "OpenRouter TEXT requires reasoning.effort=none (see openrouter_client.py).",
            "Long-chain RESUME cliff is a product limitation until inject/decode fix lands.",
        ],
        "missing_artifacts": missing,
        "canonical_paths": {k: str(v.relative_to(_ROOT)).replace("\\", "/") for k, v in paths.items() if v.is_file()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=_ROOT / "bench" / "results" / "overnight_20260624_003614",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_json = args.out_json or run_dir / "phase_e_summary.json"
    out_md = args.out_md or run_dir / "BENCHMARK_SUMMARY.md"

    report = build_report(run_dir)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(report), encoding="utf-8")

    canonical = run_dir / "canonical_artifacts.json"
    canonical.write_text(
        json.dumps({"canonical": report["canonical_paths"], "missing": report["missing_artifacts"]}, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if report["missing_artifacts"]:
        print(f"Missing ({len(report['missing_artifacts'])}): {', '.join(report['missing_artifacts'])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
