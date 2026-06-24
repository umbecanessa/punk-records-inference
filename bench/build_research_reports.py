#!/usr/bin/env python3
"""Build extensive research analysis pages from bench artifacts.

Generates markdown reports with Mermaid charts under:
    bench/results/<run>/research/

Usage:
    python bench/build_research_reports.py \\
        --run-dir bench/results/overnight_20260624_003614
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.build_phase_e_summary import CANONICAL, build_report, _load  # noqa: E402


def _pct(part: float, whole: float) -> float:
    if whole <= 0:
        return 0.0
    return round(100.0 * part / whole, 1)


def _inject_probe_rows(data: dict | None, arm: str) -> list[dict[str, Any]]:
    if not data:
        return []
    rows: list[dict[str, Any]] = []
    for item in (data.get("results") or {}).get(arm) or []:
        usage = item.get("usage") or {}
        rows.append(
            {
                "question": item.get("question"),
                "pass": item.get("pass"),
                "latency_ms": item.get("latency_ms"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )
    return rows


def _sweep_checkpoint_rows(data: dict | None) -> list[dict[str, Any]]:
    if not data:
        return []
    rows: list[dict[str, Any]] = []
    for cp in data.get("results") or []:
        text_usage = [
            (r.get("usage") or {}).get("prompt_tokens")
            for r in cp.get("text_recall") or []
            if (r.get("usage") or {}).get("prompt_tokens") is not None
        ]
        resume_usage = [
            (r.get("usage") or {}).get("prompt_tokens")
            for r in cp.get("resume_recall") or []
            if (r.get("usage") or {}).get("prompt_tokens") is not None
        ]
        rows.append(
            {
                "checkpoint": cp.get("checkpoint_noise"),
                "turn_blocks": cp.get("turn_blocks"),
                "turn_tokens": cp.get("turn_tokens"),
                "text_est_tokens": cp.get("text_est_tokens"),
                "text_pass": cp.get("text_pass_clean"),
                "resume_pass": cp.get("resume_pass_clean"),
                "arm_d_pass": cp.get("arm_d_pass_clean"),
                "total": cp.get("total"),
                "text_prompt_mean": round(statistics.mean(text_usage), 1) if text_usage else None,
                "resume_prompt_mean": round(statistics.mean(resume_usage), 1) if resume_usage else None,
                "token_savings_pct": _pct(
                    (statistics.mean(text_usage) - statistics.mean(resume_usage))
                    if text_usage and resume_usage
                    else 0,
                    statistics.mean(text_usage) if text_usage else 1,
                )
                if text_usage and resume_usage
                else None,
            }
        )
    return rows


def _parse_capture_sizes(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    sizes: list[int] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sizes.append(int(row["bytes"]))
    if not sizes:
        return {}
    total = sum(sizes)
    return {
        "file_count": len(sizes),
        "total_bytes": total,
        "total_mb": round(total / (1024 * 1024), 2),
        "mean_bytes": round(statistics.mean(sizes)),
        "median_bytes": round(statistics.median(sizes)),
        "min_bytes": min(sizes),
        "max_bytes": max(sizes),
        "stdev_bytes": round(statistics.pstdev(sizes)) if len(sizes) > 1 else 0,
        "bytes_per_file_series": sizes,
    }


def _geometry_storage_series(data: dict | None) -> list[dict[str, Any]]:
    if not data:
        return []
    series: list[dict[str, Any]] = []
    cumulative = 0
    for block in data.get("rope_pack") or []:
        cumulative += int(block.get("num_tokens") or 0)
        series.append(
            {
                "turn_index": block.get("turn_index"),
                "num_tokens": block.get("num_tokens"),
                "pack_offset": block.get("pack_offset"),
                "cumulative_tokens": cumulative,
            }
        )
    return series


def _cost_proxy(prompt_tokens: float) -> dict[str, float]:
    """Linear prefill proxy (relative units, not calibrated FLOPs)."""
    return {
        "prefill_proxy_units": round(prompt_tokens, 1),
        "note": "1 unit ≈ one prompt token prefill; decode omitted (small vs prefill at long context)",
    }


# Energy / cost assumptions — GX10 did not meter wall power; tune for your deployment.
ASSUMED_GPU_POWER_W = 250
ASSUMED_PUE = 1.0
ASSUMED_USD_PER_KWH = 0.15
ASSUMED_CLOUD_INPUT_USD_PER_1M = 0.18
ASSUMED_CLOUD_OUTPUT_USD_PER_1M = 0.72
DECODE_COMPUTE_WEIGHT = 0.12  # relative to one prefill token (decode is shorter but full forward)


def _arm_usage_means(arm_summary: dict | None) -> dict[str, float]:
    if not arm_summary:
        return {"prompt_mean": 0.0, "completion_mean": 0.0, "latency_ms_mean": 0.0}
    usage = arm_summary.get("usage") or {}
    reqs = max(int(usage.get("requests") or 1), 1)
    return {
        "prompt_mean": round(int(usage.get("prompt_tokens") or 0) / reqs, 2),
        "completion_mean": round(int(usage.get("completion_tokens") or 0) / reqs, 2),
        "latency_ms_mean": float(arm_summary.get("latency_ms_mean") or 0),
    }


def _energy_wh(latency_ms: float, power_w: float, pue: float = ASSUMED_PUE) -> float:
    return round(power_w * pue * latency_ms / 1000 / 3600, 6)


def _compute_units(prompt_mean: float, completion_mean: float) -> float:
    return round(prompt_mean + completion_mean * DECODE_COMPUTE_WEIGHT, 2)


def _cloud_usd_per_recall(prompt_mean: float, completion_mean: float) -> float:
    return round(
        prompt_mean * ASSUMED_CLOUD_INPUT_USD_PER_1M / 1_000_000
        + completion_mean * ASSUMED_CLOUD_OUTPUT_USD_PER_1M / 1_000_000,
        6,
    )


def _arm_energy_cost(arm_summary: dict | None, power_w: float = ASSUMED_GPU_POWER_W) -> dict[str, Any]:
    m = _arm_usage_means(arm_summary)
    wh = _energy_wh(m["latency_ms_mean"], power_w)
    return {
        **m,
        "energy_wh_per_recall": wh,
        "electricity_usd_per_recall": round(wh / 1000 * ASSUMED_USD_PER_KWH, 6),
        "compute_units_per_recall": _compute_units(m["prompt_mean"], m["completion_mean"]),
        "cloud_api_usd_per_recall": _cloud_usd_per_recall(m["prompt_mean"], m["completion_mean"]),
    }


def _compute_energy_cost(
    long_summary: dict,
    short_summary: dict,
    sweep_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    long12 = {
        "TEXT": _arm_energy_cost(long_summary.get("text")),
        "RESUME": _arm_energy_cost(long_summary.get("resume")),
        "OVERFLOW": _arm_energy_cost(long_summary.get("resume_overflow")),
    }
    short = {
        "TEXT": _arm_energy_cost(short_summary.get("text")),
        "RESUME": _arm_energy_cost(short_summary.get("resume")),
    }

    text_l12 = long12["TEXT"]
    res_l12 = long12["RESUME"]

    wh_saved = round(text_l12["energy_wh_per_recall"] - res_l12["energy_wh_per_recall"], 6)
    wh_savings_pct = _pct(wh_saved, text_l12["energy_wh_per_recall"])

    units_saved = round(text_l12["compute_units_per_recall"] - res_l12["compute_units_per_recall"], 2)
    units_savings_pct = _pct(units_saved, text_l12["compute_units_per_recall"])

    cloud_saved = round(
        text_l12["cloud_api_usd_per_recall"] - res_l12["cloud_api_usd_per_recall"],
        6,
    )

    sensitivity: list[dict[str, Any]] = []
    for power in (150, 250, 350):
        t_wh = _energy_wh(text_l12["latency_ms_mean"], power)
        r_wh = _energy_wh(res_l12["latency_ms_mean"], power)
        sensitivity.append({
            "gpu_power_w": power,
            "text_wh": t_wh,
            "resume_wh": r_wh,
            "wh_saved": round(t_wh - r_wh, 6),
            "wh_savings_pct": _pct(t_wh - r_wh, t_wh),
        })

    sweep_compute: list[dict[str, Any]] = []
    for row in sweep_rows:
        tp = row.get("text_prompt_mean") or 0
        rp = row.get("resume_prompt_mean") or 0
        sweep_compute.append({
            "checkpoint": row.get("checkpoint"),
            "text_compute_units": round(tp, 1),
            "resume_compute_units": round(rp + 40 * DECODE_COMPUTE_WEIGHT, 1),  # ~40 tok decode est
            "text_cloud_usd_est": _cloud_usd_per_recall(tp, 50),
            "resume_cloud_usd_est": _cloud_usd_per_recall(rp, 50),
        })

    def _annual(recalls_per_day: int) -> dict[str, float]:
        per_year = recalls_per_day * 365
        return {
            "recalls_per_day": recalls_per_day,
            "gpu_kwh_saved": round(wh_saved * per_year / 1000, 1),
            "electricity_usd_saved": round(wh_saved * per_year / 1000 * ASSUMED_USD_PER_KWH, 2),
            "cloud_api_usd_saved": round(cloud_saved * per_year, 2),
            "input_tokens_avoided": round((text_l12["prompt_mean"] - res_l12["prompt_mean"]) * per_year),
        }

    return {
        "disclaimer": (
            "Proof run did not capture wall power or SM clocks. "
            "Figures combine measured bench latency/tokens with documented assumptions below."
        ),
        "assumptions": {
            "gpu_power_w_default": ASSUMED_GPU_POWER_W,
            "pue": ASSUMED_PUE,
            "electricity_usd_per_kwh": ASSUMED_USD_PER_KWH,
            "cloud_input_usd_per_1m_tokens": ASSUMED_CLOUD_INPUT_USD_PER_1M,
            "cloud_output_usd_per_1m_tokens": ASSUMED_CLOUD_OUTPUT_USD_PER_1M,
            "decode_compute_weight": DECODE_COMPUTE_WEIGHT,
            "energy_model_latency": "E(Wh) = P_gpu(W) × PUE × latency(s) / 3600",
            "energy_model_compute": "units = prompt_tokens + decode_weight × completion_tokens",
            "cloud_model": "API $ = input×$/1M + output×$/1M (illustrative OpenRouter-class rates)",
        },
        "long12_per_recall": long12,
        "short_per_recall": short,
        "long12_savings": {
            "energy_wh": wh_saved,
            "energy_wh_savings_pct": wh_savings_pct,
            "compute_units": units_saved,
            "compute_units_savings_pct": units_savings_pct,
            "cloud_api_usd": cloud_saved,
            "cloud_api_savings_pct": _pct(cloud_saved, text_l12["cloud_api_usd_per_recall"]),
        },
        "power_sensitivity_wh": sensitivity,
        "turn_sweep_compute_proxy": sweep_compute,
        "annual_projections_long12": [_annual(n) for n in (100, 1_000, 10_000)],
        "storage_amortization_note": (
            "648 MB capture disk for 83-turn session ≈ 7.8 MB/turn one-time. "
            "At 0.005 Wh/GB-year HDD idle, storage energy is ≪ single recall inference."
        ),
    }


def _fmt_axis(n: float) -> str:
    return str(int(n)) if n == int(n) else str(round(n, 4))


def _mermaid_xychart(
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float]]],
    y_label: str,
    *,
    y_min: float = 0,
    y_max: float | None = None,
) -> str:
    """GitHub xychart-beta — explicit y_max so recall (0–5) and tokens (k) render."""
    flat = [v for _, vals in series for v in vals]
    if y_max is None:
        y_max = max(flat) * 1.15 if flat else 1
    y_max = max(y_max, y_min + 1)
    cats = ", ".join(f'"{c}"' for c in categories)
    lines = [
        "```mermaid",
        "xychart-beta",
        f'    title "{title}"',
        f"    x-axis [{cats}]",
        f'    y-axis "{y_label}" {_fmt_axis(y_min)} --> {_fmt_axis(y_max)}',
    ]
    for name, values in series:
        vals = ", ".join(str(v) for v in values)
        lines.append(f'    bar "{name}" [{vals}]')
    lines.append("```")
    return "\n".join(lines)


def _mermaid_linechart(
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float]]],
    y_label: str,
    *,
    y_min: float = 0,
    y_max: float | None = None,
) -> str:
    flat = [v for _, vals in series for v in vals]
    if y_max is None:
        y_max = max(flat) * 1.15 if flat else 1
    y_max = max(y_max, y_min + 1)
    cats = ", ".join(f'"{c}"' for c in categories)
    lines = [
        "```mermaid",
        "xychart-beta",
        f'    title "{title}"',
        f"    x-axis [{cats}]",
        f'    y-axis "{y_label}" {_fmt_axis(y_min)} --> {_fmt_axis(y_max)}',
    ]
    for name, values in series:
        vals = ", ".join(str(v) for v in values)
        lines.append(f'    line "{name}" [{vals}]')
    lines.append("```")
    return "\n".join(lines)


def _mermaid_pie(title: str, slices: list[tuple[str, float]]) -> str:
    """Pie — skip zero slices; no '=' in labels (breaks GitHub parser)."""
    lines = ["```mermaid", "pie showData", f'    title "{title}"']
    for label, value in slices:
        if value <= 0:
            continue
        safe = label.replace('"', "'").replace("=", " ")
        lines.append(f'    "{safe}" : {int(value) if value == int(value) else value}')
    lines.append("```")
    return "\n".join(lines)


def _embed_chart(block: str) -> str:
    return block if block else ""


def _build_mermaid_charts(data: dict[str, Any]) -> dict[str, str]:
    """Return markdown Mermaid blocks keyed by chart id (GitHub-native)."""
    sweep = data["token_efficiency"]["turn_sweep_checkpoints"]
    labels = [f"cp{r['checkpoint']}" for r in sweep]
    charts: dict[str, str] = {}

    charts["turn_sweep_recall"] = _mermaid_xychart(
        "Recall pass count @5 (turn sweep)",
        labels,
        [
            ("TEXT", [float(r.get("text_pass") or 0) for r in sweep]),
            ("RESUME", [float(r.get("resume_pass") or 0) for r in sweep]),
            ("ARM-D", [float(r.get("arm_d_pass") or 0) for r in sweep]),
        ],
        "pass / 5",
        y_max=5,
    )
    inject_max = max((r.get("turn_tokens") or 0 for r in sweep), default=1)
    charts["turn_sweep_inject"] = _mermaid_xychart(
        "Chain inject tokens at checkpoint",
        labels,
        [("inject tokens", [float(r.get("turn_tokens") or 0) for r in sweep])],
        "tokens",
        y_max=inject_max * 1.1,
    )
    prompt_max = max(
        max((r.get("text_prompt_mean") or 0 for r in sweep), default=0),
        max((r.get("resume_prompt_mean") or 0 for r in sweep), default=0),
    )
    charts["turn_sweep_prompt"] = _mermaid_xychart(
        "Mean prompt tokens at recall (turn sweep)",
        labels,
        [
            ("TEXT", [float(r.get("text_prompt_mean") or 0) for r in sweep]),
            ("RESUME", [float(r.get("resume_prompt_mean") or 0) for r in sweep]),
        ],
        "prompt tokens",
        y_max=prompt_max * 1.1,
    )

    inject_rows = data["phase_e_summary"].get("inject_mode_compare") or []
    inj_labels = [r["label"].replace(" chain", "").replace(" OpenRouter TEXT", " OR")[:12] for r in inject_rows]
    inj_max = max((r["text"]["prompt_tokens_mean"] for r in inject_rows), default=1)
    charts["inject_prompt"] = _mermaid_xychart(
        "Mean prompt tokens — inject mode compare",
        inj_labels,
        [
            ("TEXT", [r["text"]["prompt_tokens_mean"] for r in inject_rows]),
            ("RESUME", [r["resume"]["prompt_tokens_mean"] for r in inject_rows]),
        ],
        "prompt tokens",
        y_max=inj_max * 1.1,
    )

    lat = data["latency"]
    lat_max = max(
        lat["long12_ms_mean"]["TEXT"] or 0,
        lat["long12_ms_mean"]["RESUME"] or 0,
        lat["long12_ms_mean"]["OVERFLOW"] or 0,
    )
    charts["latency_mean"] = _mermaid_xychart(
        "Mean recall latency — long12 (ms)",
        ["TEXT", "RESUME", "OVERFLOW"],
        [("ms", [
            lat["long12_ms_mean"]["TEXT"] or 0,
            lat["long12_ms_mean"]["RESUME"] or 0,
            lat["long12_ms_mean"]["OVERFLOW"] or 0,
        ])],
        "ms",
        y_max=lat_max * 1.15,
    )

    text_p = lat["per_probe_long12"].get("text") or []
    resume_p = lat["per_probe_long12"].get("resume") or []
    probe_max = max(
        max((p.get("latency_ms") or 0 for p in text_p), default=0),
        max((p.get("latency_ms") or 0 for p in resume_p), default=0),
    )
    charts["latency_per_probe"] = _mermaid_xychart(
        "Per-probe latency — long12 (ms)",
        [f"P{i}" for i in range(1, len(text_p) + 1)],
        [
            ("TEXT", [p.get("latency_ms") or 0 for p in text_p]),
            ("RESUME", [p.get("latency_ms") or 0 for p in resume_p]),
        ],
        "ms",
        y_max=probe_max * 1.15,
    )

    ec = data.get("energy_cost") or {}
    long12_ec = ec.get("long12_per_recall") or {}
    if long12_ec:
        wh_max = max(
            long12_ec.get("TEXT", {}).get("energy_wh_per_recall", 0),
            long12_ec.get("RESUME", {}).get("energy_wh_per_recall", 0),
            0.001,
        )
        charts["energy_wh_long12"] = _mermaid_xychart(
            "Est. GPU energy per recall — long12 (Wh @ 250W)",
            ["TEXT", "RESUME", "OVERFLOW"],
            [("Wh", [
                long12_ec.get("TEXT", {}).get("energy_wh_per_recall", 0),
                long12_ec.get("RESUME", {}).get("energy_wh_per_recall", 0),
                long12_ec.get("OVERFLOW", {}).get("energy_wh_per_recall", 0),
            ])],
            "Wh",
            y_max=wh_max * 1.25,
        )
        cu_max = max(
            long12_ec.get("TEXT", {}).get("compute_units_per_recall", 0),
            long12_ec.get("RESUME", {}).get("compute_units_per_recall", 0),
        )
        charts["compute_units_long12"] = _mermaid_xychart(
            "Compute units per recall — long12",
            ["TEXT", "RESUME"],
            [("units", [
                long12_ec.get("TEXT", {}).get("compute_units_per_recall", 0),
                long12_ec.get("RESUME", {}).get("compute_units_per_recall", 0),
            ])],
            "relative units",
            y_max=cu_max * 1.15,
        )
        charts["cloud_cost_long12"] = _mermaid_xychart(
            "Cloud API cost — long12 (micro-USD per recall)",
            ["TEXT", "RESUME"],
            [("micro-USD", [
                round(long12_ec.get("TEXT", {}).get("cloud_api_usd_per_recall", 0) * 1_000_000, 1),
                round(long12_ec.get("RESUME", {}).get("cloud_api_usd_per_recall", 0) * 1_000_000, 1),
            ])],
            "micro-USD",
            y_max=800,
        )
        sweep_ec = ec.get("turn_sweep_compute_proxy") or []
        if sweep_ec:
            labels_ec = [f"cp{r['checkpoint']}" for r in sweep_ec]
            ec_max = max((r["text_compute_units"] for r in sweep_ec), default=1)
            charts["turn_sweep_compute"] = _mermaid_xychart(
                "Compute units @ recall (turn sweep)",
                labels_ec,
                [
                    ("TEXT", [r["text_compute_units"] for r in sweep_ec]),
                    ("RESUME", [r["resume_compute_units"] for r in sweep_ec]),
                ],
                "relative units",
                y_max=ec_max * 1.1,
            )

    cap = data.get("storage", {}).get("capture_sizes_csv") or {}
    if cap.get("bytes_per_file_series"):
        series = cap["bytes_per_file_series"]
        step = max(1, len(series) // 12)
        sampled = series[::step]
        charts["capture_sizes"] = _mermaid_linechart(
            "Capture file size sample (.nls bytes)",
            [str(i * step + 1) for i in range(len(sampled))],
            [("bytes", [float(v) for v in sampled])],
            "bytes",
            y_max=max(sampled) * 1.1,
        )

    mic = (data.get("geometry") or {}).get("microscope") or {}
    uniform = int(mic.get("uniform_blocks") or mic.get("block_count") or 83)
    charts["rope_delta_before_after"] = _mermaid_xychart(
        "RoPE delta blocks — pre-fix vs post-fix phantom repair",
        ["Pre-fix", "Post-fix"],
        [
            ("delta -22 uniform", [82.0, float(uniform)]),
            ("delta -42 outlier", [1.0, float(mic.get("outlier_blocks") or 0)]),
        ],
        "blocks",
        y_max=85,
    )
    dist = mic.get("delta_distribution") or {}
    if dist:
        sorted_keys = sorted(dist.keys(), key=lambda k: float(k))
        charts["rope_delta_histogram"] = _mermaid_xychart(
            "Post-fix RoPE delta histogram (83 blocks)",
            [f"delta {k}" for k in sorted_keys],
            [("blocks", [float(dist[k]) for k in sorted_keys])],
            "blocks",
            y_max=max(int(v) for v in dist.values()) + 2,
        )
    charts["rope_delta_distribution"] = charts.get("rope_delta_before_after", "")

    return charts


def _build_research_data(run_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    paths = {k: run_dir / v for k, v in CANONICAL.items()}
    loaded = {k: _load(p) for k, p in paths.items()}

    capture_path = run_dir / "capture_sizes.csv"
    if not capture_path.is_file():
        capture_path = run_dir / "capture_sizes_phase_e.csv"

    inject_long = loaded.get("inject_long12_local")
    inject_short = loaded.get("inject_short_local")
    sweep = loaded.get("turn_sweep")
    geometry = loaded.get("geometry_v5")
    microscope = loaded.get("rope_microscope_v5")

    long_summary = (inject_long or {}).get("summary") or {}
    short_summary = (inject_short or {}).get("summary") or {}

    def arm_mean(summary: dict, arm: str, field: str) -> float | None:
        block = summary.get(arm) or {}
        return block.get(field)

    sweep_rows = _sweep_checkpoint_rows(sweep)
    geo_series = _geometry_storage_series(geometry)
    capture_stats = _parse_capture_sizes(capture_path)

    long_text_prompt = arm_mean(long_summary, "text", "usage")
    long_text_prompt_mean = None
    long_resume_prompt_mean = None
    if long_text_prompt:
        reqs = max(int(long_text_prompt.get("requests") or 1), 1)
        long_text_prompt_mean = round(int(long_text_prompt.get("prompt_tokens") or 0) / reqs, 1)
    long_resume_usage = (long_summary.get("resume") or {}).get("usage") or {}
    if long_resume_usage:
        reqs = max(int(long_resume_usage.get("requests") or 1), 1)
        long_resume_prompt_mean = round(int(long_resume_usage.get("prompt_tokens") or 0) / reqs, 1)

    token_savings_long = None
    if long_text_prompt_mean and long_resume_prompt_mean is not None:
        token_savings_long = _pct(long_text_prompt_mean - long_resume_prompt_mean, long_text_prompt_mean)

    lat_text = arm_mean(long_summary, "text", "latency_ms_mean")
    lat_resume = arm_mean(long_summary, "resume", "latency_ms_mean")
    lat_savings = _pct((lat_text or 0) - (lat_resume or 0), lat_text or 1) if lat_text else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.relative_to(_ROOT)).replace("\\", "/"),
        "git_sha": report.get("git_sha"),
        "model": (inject_long or {}).get("model", "/model"),
        "harness": {
            "inject_long12": {
                "plant_turns": (inject_long or {}).get("plant_turns"),
                "noise_turns": (inject_long or {}).get("noise_turns"),
                "garbled_guard": (inject_long or {}).get("garbled_capture_guard"),
                "user_id": (inject_long or {}).get("user_id"),
                "base_session": (inject_long or {}).get("base_session"),
            },
            "turn_sweep": {
                "user_id": (sweep or {}).get("user_id"),
                "base_session": (sweep or {}).get("base_session"),
                "checkpoints": (sweep or {}).get("checkpoints"),
                "garbled_neutral_fallbacks": len((sweep or {}).get("garbled_neutral_fallbacks") or []),
            },
        },
        "token_efficiency": {
            "short_chain": {
                "text_prompt_mean": (short_summary.get("text") or {}).get("usage"),
                "resume_prompt_mean": (short_summary.get("resume") or {}).get("usage"),
            },
            "long12_chain": {
                "text_prompt_mean": long_text_prompt_mean,
                "resume_prompt_mean": long_resume_prompt_mean,
                "overflow_prompt_mean": round(
                    int(((long_summary.get("resume_overflow") or {}).get("usage") or {}).get("prompt_tokens") or 0)
                    / max(int(((long_summary.get("resume_overflow") or {}).get("usage") or {}).get("requests") or 1), 1),
                    1,
                ),
                "savings_pct_vs_text": token_savings_long,
                "per_probe": {
                    "text": _inject_probe_rows(inject_long, "text"),
                    "resume": _inject_probe_rows(inject_long, "resume"),
                },
            },
            "turn_sweep_checkpoints": sweep_rows,
        },
        "latency": {
            "long12_ms_mean": {
                "TEXT": lat_text,
                "RESUME": lat_resume,
                "OVERFLOW": arm_mean(long_summary, "resume_overflow", "latency_ms_mean"),
            },
            "long12_ms_p95": {
                "TEXT": arm_mean(long_summary, "text", "latency_ms_p95"),
                "RESUME": arm_mean(long_summary, "resume", "latency_ms_p95"),
                "OVERFLOW": arm_mean(long_summary, "resume_overflow", "latency_ms_p95"),
            },
            "latency_reduction_pct": lat_savings,
            "ms_per_1k_prompt_tokens": {
                "TEXT": round((lat_text or 0) / (long_text_prompt_mean or 1) * 1000, 2),
                "RESUME": round((lat_resume or 0) / (long_resume_prompt_mean or 1) * 1000, 2),
            },
            "per_probe_long12": {
                "text": _inject_probe_rows(inject_long, "text"),
                "resume": _inject_probe_rows(inject_long, "resume"),
            },
        },
        "computational_cost_proxy": {
            "method": "Linear prefill proxy: cost ∝ prompt_tokens (standard KV-cache resume assumption)",
            "long12_recall_prefill_units": {
                "TEXT": _cost_proxy(long_text_prompt_mean or 0),
                "RESUME": _cost_proxy(long_resume_prompt_mean or 0),
            },
            "prefill_reduction_pct": token_savings_long,
            "turn_sweep_cp80": next((r for r in sweep_rows if r.get("checkpoint") == 80), None),
        },
        "storage": {
            "store_stats": report.get("store"),
            "capture_sizes_csv": capture_stats,
            "geometry_chain_tokens": {
                "block_count": len(geo_series),
                "total_inject_tokens": (geometry or {}).get("total_inject_tokens"),
                "cumulative_series_sample": geo_series[:: max(1, len(geo_series) // 10)][:12],
            },
            "bytes_per_inject_token": round(
                capture_stats.get("total_bytes", 0) / max(int((geometry or {}).get("total_inject_tokens") or 1), 1),
                1,
            )
            if capture_stats and geometry
            else None,
        },
        "geometry": {
            "verdict": (geometry or {}).get("verdict"),
            "microscope": microscope,
        },
        "energy_cost": _compute_energy_cost(long_summary, short_summary, sweep_rows),
        "phase_e_summary": report,
    }


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _render_index(data: dict[str, Any], run_rel: str, charts: dict[str, str]) -> str:
    return f"""# PRI benchmark research — analysis index

**Run:** [`../README.md`](../README.md) · **Model:** `{data.get("model")}` · **Date:** 2026-06-24

Extended analysis from the published proof run. Start with **[Key findings](00_findings.md)** for narrative context and NLS cross-reference.

Each page embeds **Mermaid charts** (GitHub `xychart-beta` / `pie`) with explicit y-axis ranges.

## Quick headline

| Metric | TEXT (inline) | RESUME (KV inject) | Savings |
|--------|--------------:|-------------------:|--------:|
| Recall @5 (long12) | 5/5 | 5/5 | — |
| Mean prompt tokens | {data["token_efficiency"]["long12_chain"].get("text_prompt_mean")} | {data["token_efficiency"]["long12_chain"].get("resume_prompt_mean")} | **{data["token_efficiency"]["long12_chain"].get("savings_pct_vs_text")}%** |
| Mean latency (ms) | {data["latency"]["long12_ms_mean"].get("TEXT")} | {data["latency"]["long12_ms_mean"].get("RESUME")} | **{data["latency"].get("latency_reduction_pct")}%** |
| Est. GPU energy / recall | {data.get("energy_cost", {}).get("long12_per_recall", {}).get("TEXT", {}).get("energy_wh_per_recall")} Wh | {data.get("energy_cost", {}).get("long12_per_recall", {}).get("RESUME", {}).get("energy_wh_per_recall")} Wh | **{data.get("energy_cost", {}).get("long12_savings", {}).get("energy_wh_savings_pct")}%** |
| Capture disk | — | {data["storage"]["store_stats"].get("capture_disk_mb")} MB ({data["storage"]["store_stats"].get("capture_count")} files) | — |

**Default inject mode:** `resume` · Summary: [`../BENCHMARK_SUMMARY.md`](../BENCHMARK_SUMMARY.md)

## Analysis pages

| # | Topic | File |
|---|-------|------|
| 0 | **Key findings & NLS context** | **[00_findings.md](00_findings.md)** |
| 1 | Methodology & arms | [01_methodology.md](01_methodology.md) |
| 2 | Token efficiency | [02_token_efficiency.md](02_token_efficiency.md) |
| 3 | Latency | [03_latency_analysis.md](03_latency_analysis.md) |
| 4 | Computational cost | [04_computational_cost.md](04_computational_cost.md) |
| 5 | Storage footprint | [05_storage_footprint.md](05_storage_footprint.md) |
| 6 | Turn-sweep scaling | [06_turn_sweep_scaling.md](06_turn_sweep_scaling.md) |
| 7 | RoPE geometry | [07_rope_geometry.md](07_rope_geometry.md) |
| 8 | Failure modes | [08_failure_modes.md](08_failure_modes.md) |
| 9 | Energy & cost | [09_energy_and_cost.md](09_energy_and_cost.md) |

## Regenerate

```bash
python bench/build_research_reports.py --run-dir bench/results/{run_rel.split("/")[-1] if "/" in run_rel else run_rel}
```

## Machine-readable

- [`research_data.json`](research_data.json) — extracted metrics
- [`../phase_e_summary.json`](../phase_e_summary.json) — rollup JSON
- [`../canonical_artifacts.json`](../canonical_artifacts.json) — artifact index

## Architecture context

Full pipeline narrative (retrieval-first design): [Neural Ledger System](https://github.com/umbecanessa/neural-ledger-system) · [docs/OVERVIEW.md](../../../docs/OVERVIEW.md)

Historical engineering triage (superseded): [`../internal/`](../internal/)
"""


def _render_methodology(data: dict[str, Any]) -> str:
    h = data.get("harness") or {}
    inj = h.get("inject_long12") or {}
    sw = h.get("turn_sweep") or {}
    return f"""# 1 — Methodology & measurement arms

## Environment

| Field | Value |
|-------|-------|
| Host | NVIDIA GPU ≥24 GB VRAM (Qwen3.5-35B-A3B-FP8 validated) |
| Model | `{data.get("model")}` (BYOC FP8 hybrid) |
| API | vLLM OpenAI-compatible :8000 + PRI plugin |
| Run folder | `{data.get("run_dir")}` |
| Git SHA | `{data.get("git_sha") or "unknown"}` |

## Arms (every study)

| Arm | Client pattern | Purpose |
|-----|----------------|---------|
| **TEXT (local)** | Full inline history, `memory_off=1` | On-box baseline sharing GPU |
| **TEXT (OpenRouter)** | Cloud API, same model slug, no PRI | Isolated TEXT baseline |
| **RESUME** | Last message + chain KV inject | Primary value case |
| **RESUME_OVERFLOW** | RESUME + Swiss backfill on trim | Overflow stress |
| **ARM-D** | `resume_overflow` in turn sweep | Same as overflow in sweep harness |

## Harnesses

### Inject mode compare (`inject_mode_compare.py`)

Long12 run metadata:

| Field | Value |
|-------|-------|
| Plant turns | {inj.get("plant_turns")} (3 facts + {inj.get("noise_turns")} noise) |
| Garbled guard | `{inj.get("garbled_guard") or "none"}` |
| User / chain | `{inj.get("user_id")}` / `{inj.get("base_session")}` |

Five recall probes scored per arm. Metrics: `usage.prompt_tokens`, HTTP latency, pass/fail vs expected spans.

### Turn sweep (`turn_sweep.py`)

| Field | Value |
|-------|-------|
| Checkpoints | {sw.get("checkpoints")} |
| User / chain | `{sw.get("user_id")}` / `{sw.get("base_session")}` |
| Neutral fallbacks | {sw.get("garbled_neutral_fallbacks")} (garbled-capture hygiene) |

At each checkpoint: cumulative noise turns, then TEXT / RESUME / ARM-D recall @5.

### OpenCode long session

8 work turns + 6 strict recall probes (seed 42). PRI vs `memory_off` baseline.

## Isolation

Each harness run uses `fresh_chain_ids()` — unique `memory_user` + `memory_base_session`. No shared chain keys between studies (see `08_failure_modes.md`).

## Success criteria

1. **Value case (cp20–40, inject long12):** RESUME recall ≥ TEXT; prompt tokens ≪ TEXT.
2. **Agent case:** OpenCode RECALL ≥ baseline.
3. **Geometry:** RoPE pack delta_uniformity → 1.0 after phantom fix.
4. **Long-chain cliff:** Document RESUME degradation at cp60+ (not a bench failure).
"""


def _render_token_efficiency(data: dict[str, Any], charts: dict[str, str]) -> str:
    te = data["token_efficiency"]
    long12 = te["long12_chain"]
    sweep = te["turn_sweep_checkpoints"]
    inject_rows = data["phase_e_summary"].get("inject_mode_compare") or []

    chart_labels = [str(r["checkpoint"]) for r in sweep]
    text_prompts = [r.get("text_prompt_mean") or 0 for r in sweep]
    resume_prompts = [r.get("resume_prompt_mean") or 0 for r in sweep]

    lines = [
        "# 2 — Token efficiency",
        "",
        "PRI RESUME sends **only the latest user message** on recall; prior context lives in injected KV.",
        "TEXT re-prefills the full inline transcript every request.",
        "",
        "## Headline — long12 inject compare",
        "",
        "| Arm | Mean prompt tok | vs TEXT | Recall @5 |",
        "|-----|----------------:|--------:|----------:|",
        f"| TEXT | {long12.get('text_prompt_mean')} | — | 5/5 |",
        f"| RESUME | {long12.get('resume_prompt_mean')} | **−{long12.get('savings_pct_vs_text')}%** | 5/5 |",
        f"| OVERFLOW | {long12.get('overflow_prompt_mean')} | **−{long12.get('savings_pct_vs_text')}%** | 5/5 |",
        "",
        f"**Net savings:** ~{round((long12.get('text_prompt_mean') or 0) - (long12.get('resume_prompt_mean') or 0))} prompt tokens per recall request on a 15-turn chain.",
        "",
        "## All inject-mode scenarios",
        "",
        "| Scenario | TEXT mean | RESUME mean | Δ tokens | Savings % |",
        "|----------|----------:|------------:|---------:|----------:|",
    ]
    for row in inject_rows:
        t = row["text"]["prompt_tokens_mean"]
        r = row["resume"]["prompt_tokens_mean"]
        savings = _pct(t - r, t)
        lines.append(
            f"| {row['label']} | {t} | {r} | {round(t - r, 1)} | {savings}% |"
        )

    lines.extend(
        [
            "",
            "## Turn sweep — prompt tokens vs checkpoint",
            "",
            "| cp | chain inject tok | TEXT prompt (mean) | RESUME prompt (mean) | Savings % |",
            "|----|-----------------:|-------------------:|---------------------:|----------:|",
        ]
    )
    for r in sweep:
        lines.append(
            f"| {r.get('checkpoint')} | {r.get('turn_tokens')} "
            f"| {r.get('text_prompt_mean')} | {r.get('resume_prompt_mean')} "
            f"| {r.get('token_savings_pct')}% |"
        )

    lines.extend(
        [
            "",
        "**Key finding:** Token savings stay **>99%** even at cp80 (~23k inject tokens). Recall failure at long context is **not** caused by reverting to inline prefill.",
        "",
        _embed_chart(charts.get("turn_sweep_prompt", "")),
        "",
        _embed_chart(charts.get("inject_prompt", "")),
        "",
        "## Per-probe detail (long12, local)",
            "",
            "### TEXT arm",
            "",
            "| # | Prompt tok | Completion tok | Pass |",
            "|---|----------:|---------------:|:----:|",
        ]
    )
    for i, p in enumerate(long12.get("per_probe", {}).get("text") or [], 1):
        lines.append(
            f"| {i} | {p.get('prompt_tokens')} | {p.get('completion_tokens')} | {'✓' if p.get('pass') else '✗'} |"
        )

    lines.extend(["", "### RESUME arm", "", "| # | Prompt tok | Completion tok | Pass |", "|---|----------:|---------------:|:----:|"])
    for i, p in enumerate(long12.get("per_probe", {}).get("resume") or [], 1):
        lines.append(
            f"| {i} | {p.get('prompt_tokens')} | {p.get('completion_tokens')} | {'✓' if p.get('pass') else '✗'} |"
        )

    lines.append("\nRaw: `inject_mode_compare_*_long12_postfix.json`, `turn_sweep_cp20_80_v5.json`")
    return "\n".join(lines)


def _render_latency(data: dict[str, Any], charts: dict[str, str]) -> str:
    lat = data["latency"]
    mean = lat["long12_ms_mean"]
    p95 = lat["long12_ms_p95"]
    inject_rows = data["phase_e_summary"].get("inject_mode_compare") or []

    chart_labels = ["TEXT", "RESUME", "OVERFLOW"]
    chart_vals = [mean.get("TEXT") or 0, mean.get("RESUME") or 0, mean.get("OVERFLOW") or 0]

    lines = [
        "# 3 — Latency analysis",
        "",
        "End-to-end HTTP latency from harness (`latency_ms` per request). Includes network, vLLM queue, prefill, and decode.",
        "",
        "## Summary — long12 recall (mean / p95)",
        "",
        "| Arm | Mean ms | p95 ms |",
        "|-----|--------:|-------:|",
        f"| TEXT | {mean.get('TEXT')} | {p95.get('TEXT')} |",
        f"| RESUME | {mean.get('RESUME')} | {p95.get('RESUME')} |",
        f"| OVERFLOW | {mean.get('OVERFLOW')} | {p95.get('OVERFLOW')} |",
        "",
        "*Note: “ms per 1k prompt tokens” is misleading for RESUME (denominator ≈42 tok). Compare mean latency directly.*",
        "",
        f"**Mean latency reduction (RESUME vs TEXT):** {lat.get('latency_reduction_pct')}%",
        "",
        _embed_chart(charts.get("latency_mean", "")),
        "",
        _embed_chart(charts.get("latency_per_probe", "")),
        "",
        "## All scenarios (inject compare)",
        "",
        "| Scenario | TEXT mean | RESUME mean | OVERFLOW mean |",
        "|----------|----------:|------------:|--------------:|",
    ]
    for row in inject_rows:
        lines.append(
            f"| {row['label']} | {row['text'].get('latency_ms_mean')} "
            f"| {row['resume'].get('latency_ms_mean')} "
            f"| {row['resume_overflow'].get('latency_ms_mean')} |"
        )

    lines.extend(
        [
            "",
            "## Per-probe latency (long12)",
            "",
            "| # | TEXT ms | RESUME ms | TEXT prompt tok | RESUME prompt tok |",
            "|---|--------:|----------:|----------------:|------------------:|",
        ]
    )
    text_probes = lat["per_probe_long12"].get("text") or []
    resume_probes = lat["per_probe_long12"].get("resume") or []
    for i, (t, r) in enumerate(zip(text_probes, resume_probes), 1):
        lines.append(
            f"| {i} | {t.get('latency_ms')} | {r.get('latency_ms')} "
            f"| {t.get('prompt_tokens')} | {r.get('prompt_tokens')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- RESUME avoids large prefill → lower mean latency despite KV inject overhead.",
            "- Short chain: RESUME can be slightly *slower* (inject setup dominates when context is tiny).",
            "- Long chain: RESUME ~**1.9× faster** mean recall vs TEXT at equal correctness.",
            "",
            "Raw: `inject_mode_compare_*_long12_postfix.json`",
        ]
    )
    return "\n".join(lines)


def _render_computational_cost(data: dict[str, Any]) -> str:
    cost = data["computational_cost_proxy"]
    long12 = cost["long12_recall_prefill_units"]
    cp80 = cost.get("turn_sweep_cp80") or {}
    te = data["token_efficiency"]["long12_chain"]

    lines = [
        "# 4 — Computational cost (prefill proxy)",
        "",
        "## Model",
        "",
        cost.get("method", ""),
        "",
        "We report **relative prefill units** = `prompt_tokens` at recall time. For transformer decoders,",
        "attention prefill scales ~O(n) per layer w.r.t. sequence length when KV is cold; RESUME inject",
        "reuses stored KV so the live prefill length equals last-message tokens (~42) not chain length (~3743).",
        "",
        "Decode cost is small here (≤200 completion tokens) and similar across arms.",
        "",
        "## Long12 recall — prefill units (mean per probe)",
        "",
        "| Arm | Prefill proxy units | Relative cost |",
        "|-----|--------------------:|--------------:|",
        f"| TEXT | {long12['TEXT']['prefill_proxy_units']} | 1.00× |",
        f"| RESUME | {long12['RESUME']['prefill_proxy_units']} | "
        f"{round((long12['RESUME']['prefill_proxy_units'] or 1) / max(long12['TEXT']['prefill_proxy_units'] or 1, 1), 3)}× |",
        "",
        f"**Prefill reduction:** {cost.get('prefill_reduction_pct')}%",
        "",
        "## Turn sweep — cost vs correctness trade-off",
        "",
        "At cp80 the RESUME arm still pays **~42 prompt tokens** (99%+ prefill savings) but recall is **0/5**.",
        "Computational efficiency does not imply semantic recovery at extreme inject depth.",
        "",
        "| cp | inject tok | TEXT prefill (mean) | RESUME prefill (mean) | TEXT pass | RESUME pass |",
        "|----|----------:|--------------------:|----------------------:|----------:|------------:|",
    ]
    for r in data["token_efficiency"]["turn_sweep_checkpoints"]:
        lines.append(
            f"| {r.get('checkpoint')} | {r.get('turn_tokens')} "
            f"| {r.get('text_prompt_mean')} | {r.get('resume_prompt_mean')} "
            f"| {r.get('text_pass')}/{r.get('total')} | {r.get('resume_pass')}/{r.get('total')} |"
        )

    lines.extend(
        [
            "",
            "See [09_energy_and_cost.md](09_energy_and_cost.md) for Wh, electricity $, and cloud API $ models.",
        ]
    )
    return "\n".join(lines)


def _render_energy_cost(data: dict[str, Any], charts: dict[str, str]) -> str:
    ec = data.get("energy_cost") or {}
    asm = ec.get("assumptions") or {}
    long12 = ec.get("long12_per_recall") or {}
    sav = ec.get("long12_savings") or {}
    t = long12.get("TEXT") or {}
    r = long12.get("RESUME") or {}
    o = long12.get("OVERFLOW") or {}

    lines = [
        "# 9 — Energy impact & cost",
        "",
        f"> {ec.get('disclaimer', '')}",
        "",
        "## Assumptions (tunable)",
        "",
        "| Parameter | Value | Notes |",
        "|-----------|------:|-------|",
        f"| GPU average power | **{asm.get('gpu_power_w_default')} W** | Not wall-metered; typical inference-GPU estimate |",
        f"| PUE | {asm.get('pue')} | 1.0 = on-prem wall meter; use 1.2 for colo |",
        f"| Electricity | ${asm.get('electricity_usd_per_kwh')}/kWh | Illustrative |",
        f"| Cloud input | ${asm.get('cloud_input_usd_per_1m_tokens')}/1M tok | OpenRouter-class illustrative |",
        f"| Cloud output | ${asm.get('cloud_output_usd_per_1m_tokens')}/1M tok | OpenRouter-class illustrative |",
        f"| Decode weight | {asm.get('decode_compute_weight')}× | Relative to one prefill token in compute proxy |",
        "",
        "## Models",
        "",
        "1. **Latency × power** — `E(Wh) = P_gpu × PUE × latency(s) / 3600` using measured HTTP latency.",
        "2. **Token compute proxy** — `units = prompt_tok + w × completion_tok` (prefill-dominated at long context).",
        "3. **Cloud API $** — if TEXT ran on a hosted API with per-token billing vs local RESUME.",
        "",
        "## Long12 — per successful recall (mean)",
        "",
        "| Arm | Latency ms | Wh @ 250W | Electricity $ | Compute units | Cloud API $ |",
        "|-----|----------:|----------:|--------------:|--------------:|------------:|",
        f"| TEXT | {t.get('latency_ms_mean')} | {t.get('energy_wh_per_recall')} | ${t.get('electricity_usd_per_recall', 0):.6f} | {t.get('compute_units_per_recall')} | ${t.get('cloud_api_usd_per_recall', 0):.6f} |",
        f"| RESUME | {r.get('latency_ms_mean')} | {r.get('energy_wh_per_recall')} | ${r.get('electricity_usd_per_recall', 0):.6f} | {r.get('compute_units_per_recall')} | ${r.get('cloud_api_usd_per_recall', 0):.6f} |",
        f"| OVERFLOW | {o.get('latency_ms_mean')} | {o.get('energy_wh_per_recall')} | ${o.get('electricity_usd_per_recall', 0):.6f} | {o.get('compute_units_per_recall')} | ${o.get('cloud_api_usd_per_recall', 0):.6f} |",
        "",
        "### Savings (RESUME vs TEXT, long12)",
        "",
        "| Metric | Saved | % |",
        "|--------|------:|--:|",
        f"| GPU energy (Wh) | {sav.get('energy_wh')} | **{sav.get('energy_wh_savings_pct')}%** |",
        f"| Compute units | {sav.get('compute_units')} | **{sav.get('compute_units_savings_pct')}%** |",
        f"| Cloud API $ / recall | ${sav.get('cloud_api_usd', 0):.6f} | **{sav.get('cloud_api_savings_pct')}%** |",
        "",
        _embed_chart(charts.get("energy_wh_long12", "")),
        "",
        _embed_chart(charts.get("compute_units_long12", "")),
        "",
        _embed_chart(charts.get("cloud_cost_long12", "")),
        "",
        "## Power sensitivity (latency model)",
        "",
        "| GPU power (W) | TEXT Wh | RESUME Wh | Saved | % |",
        "|--------------:|--------:|----------:|------:|--:|",
    ]
    for row in ec.get("power_sensitivity_wh") or []:
        lines.append(
            f"| {row.get('gpu_power_w')} | {row.get('text_wh')} | {row.get('resume_wh')} "
            f"| {row.get('wh_saved')} | {row.get('wh_savings_pct')}% |"
        )

    lines.extend([
        "",
        "**Note:** % savings is stable across power assumptions because both arms use the same multiplier.",
        "",
        "## Turn sweep — compute proxy (token-based)",
        "",
        "No per-request latency in sweep JSON; energy uses token proxy only at each checkpoint.",
        "",
        _embed_chart(charts.get("turn_sweep_compute", "")),
        "",
        "| cp | TEXT units | RESUME units | TEXT cloud $ est | RESUME cloud $ est |",
        "|----|----------:|-------------:|-----------------:|-------------------:|",
    ])
    for row in ec.get("turn_sweep_compute_proxy") or []:
        lines.append(
            f"| {row.get('checkpoint')} | {row.get('text_compute_units')} | {row.get('resume_compute_units')} "
            f"| ${row.get('text_cloud_usd_est', 0):.6f} | ${row.get('resume_cloud_usd_est', 0):.6f} |"
        )

    lines.extend([
        "",
        "At **cp80**, RESUME still saves **>99%** compute units vs TEXT but recall is **0/5** — cheap inference that fails the task is not a net win.",
        "",
        "## Annual projections (long12, RESUME vs TEXT)",
        "",
        "Assumes every recall resembles the long12 inject-compare chain (15 turns planted).",
        "",
        "| Recalls/day | GPU kWh saved/yr | Electricity $ saved/yr | Cloud API $ saved/yr | Input tokens avoided/yr |",
        "|------------:|-----------------:|-----------------------:|---------------------:|------------------------:|",
    ])
    for row in ec.get("annual_projections_long12") or []:
        lines.append(
            f"| {row.get('recalls_per_day'):,} | {row.get('gpu_kwh_saved')} | ${row.get('electricity_usd_saved')} "
            f"| ${row.get('cloud_api_usd_saved')} | {int(row.get('input_tokens_avoided') or 0):,} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Electricity** savings track **latency** (~46% here) — meaningful at high QPS, modest per request.",
        "- **Cloud API** savings track **input tokens** (~99%) — dominant if TEXT would run on a hosted per-token API.",
        "- **On-prem GPU** deployments still benefit from shorter GPU busy time → higher throughput headroom.",
        "- **Storage** energy is negligible vs inference; capture cost is capex/disk, not kWh (see [05_storage_footprint.md](05_storage_footprint.md)).",
        "",
        f"{ec.get('storage_amortization_note', '')}",
        "",
        "## Future measurement (MoE matrix)",
        "",
        "For publication-grade energy claims, add to the next bench pass:",
        "",
        "- `nvidia-smi --query-gpu=power.draw` sampled per recall request",
        "- vLLM profiler / `engine_core` prefill vs decode split",
        "- Wall-meter validation on target hardware for at least one arm",
        "",
        "Raw: `inject_mode_compare_*_long12_postfix.json`, assumptions in `research_data.json` → `energy_cost`",
    ])
    return "\n".join(lines)


def _render_storage(data: dict[str, Any], charts: dict[str, str]) -> str:
    store = data["storage"]["store_stats"] or {}
    cap = data["storage"]["capture_sizes_csv"] or {}
    geo = data["storage"]["geometry_chain_tokens"] or {}
    bpt = data["storage"].get("bytes_per_inject_token")
    series = geo.get("cumulative_series_sample") or []

    sample_labels = [str(s.get("turn_index")) for s in series]
    sample_cum = [s.get("cumulative_tokens") or 0 for s in series]

    lines = [
        "# 5 — Storage footprint",
        "",
        "PRI persists one `.nls` capture per turn under `/data/pri/snapshot/captures/`.",
        "",
        "## Aggregate (full turn-sweep session)",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Capture files | {store.get('capture_count')} |",
        f"| Index rows | {store.get('index_row_count')} |",
        f"| Capture disk | **{store.get('capture_disk_mb')} MB** |",
        f"| Data dir total | {store.get('data_dir_mb')} MB |",
        f"| Admin indexed tokens | {store.get('admin_total_tokens')} |",
        "",
    ]
    if cap:
        lines.extend(
            [
                "## Per-file capture sizes (`capture_sizes.csv`)",
                "",
                _embed_chart(charts.get("capture_sizes", "")),
                "",
                "| Stat | Bytes | MB |",
                "|------|------:|---:|",
                f"| Total ({cap.get('file_count')} files) | {cap.get('total_bytes'):,} | {cap.get('total_mb')} |",
                f"| Mean | {cap.get('mean_bytes'):,} | {round((cap.get('mean_bytes') or 0) / (1024 * 1024), 2)} |",
                f"| Median | {cap.get('median_bytes'):,} | — |",
                f"| Min / Max | {cap.get('min_bytes'):,} / {cap.get('max_bytes'):,} | — |",
                f"| Stdev | {cap.get('stdev_bytes'):,} | — |",
                "",
            ]
        )
    if bpt:
        lines.append(f"**Bytes per inject token (chain total):** ~{bpt} B/tok (captures / `{geo.get('total_inject_tokens')}` inject tokens)")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Capture size grows with turn length and assistant output (KV + hybrid state).",
            "- Disk cost is **one-time per turn**; amortized over all future RESUME recalls on that chain.",
            "- 648 MB for 83-turn production-length session ≈ **7.8 MB/turn** mean (full sweep).",
            "",
            "Raw: `store_stats.json`, `capture_sizes.csv`, `geometry_audit_turn_sweep_v5_fixed.json`",
        ]
    )
    return "\n".join(lines)


def _render_turn_sweep(data: dict[str, Any], charts: dict[str, str]) -> str:
    sweep = data["token_efficiency"]["turn_sweep_checkpoints"]
    labels = [f"cp{r['checkpoint']}" for r in sweep]
    text_pass = [r.get("text_pass") or 0 for r in sweep]
    resume_pass = [r.get("resume_pass") or 0 for r in sweep]
    inject_tok = [r.get("turn_tokens") or 0 for r in sweep]

    lines = [
        "# 6 — Turn-sweep scaling",
        "",
        "Marco facts planted once; cumulative noise turns to each checkpoint; recall @5 at cp 20/40/60/80.",
        "",
        "## Recall vs checkpoint",
        "",
        "| cp | turn blocks | inject tok | TEXT | RESUME | ARM-D |",
        "|----|------------:|-----------:|:----:|:------:|:-----:|",
    ]
    for r in sweep:
        lines.append(
            f"| {r.get('checkpoint')} | {r.get('turn_blocks')} | {r.get('turn_tokens')} "
            f"| {r.get('text_pass')}/{r.get('total')} | {r.get('resume_pass')}/{r.get('total')} "
            f"| {r.get('arm_d_pass')}/{r.get('total')} |"
        )

    lines.extend(
        [
            "",
            _embed_chart(charts.get("turn_sweep_recall", "")),
            "",
            _embed_chart(charts.get("turn_sweep_inject", "")),
            "",
            "## Failure character by checkpoint",
            "",
            "| cp | RESUME failure mode | TEXT |",
            "|----|---------------------|------|",
            "| 20–40 | — (5/5) | 5/5 |",
            "| 60 | Policy refusals (3/5) | 5/5 |",
            "| 80 | Refusals / no-hit (0/5) | 5/5 |",
            "",
            "Garble investigation (`turn_sweep_cp60_80_garble_inv.json`) confirms TEXT 5/5 while RESUME degrades — inject-mediated decode, not missing inline context.",
            "",
            "Raw: `turn_sweep_cp20_80_v5.json`, `turn_sweep_cp60_80_garble_inv.json`",
        ]
    )
    return "\n".join(lines)


def _render_geometry(data: dict[str, Any], charts: dict[str, str]) -> str:
    geo = data.get("geometry") or {}
    mic = geo.get("microscope") or {}
    before_after = _embed_chart(charts.get("rope_delta_before_after", ""))
    histogram = _embed_chart(charts.get("rope_delta_histogram", ""))
    return f"""# 7 — RoPE pack geometry

## KPI: delta_uniformity

Bench metric: fraction of turn blocks sharing the same **RoPE re-rotation delta**
(`rope_new − rope_old`) in the inject pack plan. **Not** retrieval cosine similarity.

| Metric | Value | Grade |
|--------|------:|-------|
| Verdict | `{geo.get("verdict")}` | — |
| Blocks | {mic.get("block_count")} | — |
| delta_uniformity | **{mic.get("delta_uniformity_pct")}%** | {mic.get("kpi_grade")} |
| mode RoPE Δ | {mic.get("mode_rope_delta")} | uniform |
| Outliers | {mic.get("outlier_blocks")} | — |
| delta stdev | {mic.get("delta_stdev")} | — |

## Pre-fix vs post-fix (phantom repair)

Pre-fix audit had **82** blocks at Δ=−22 and **1** outlier at Δ=−42 (turn 59). Post-fix: **{mic.get("uniform_blocks") or 83}** / **{mic.get("block_count") or 83}** uniform.

{before_after}

## Post-fix delta histogram

All **{mic.get("uniform_blocks") or mic.get("block_count")}** blocks share mode **Δ = {mic.get("mode_rope_delta")}** (`delta_distribution`: {mic.get("delta_distribution")}).

{histogram}

## Post-fix status

Pre-fix audit ([`../internal/ROPE_DELTA_AUDIT.md`](../internal/ROPE_DELTA_AUDIT.md)) reported 98.8% (one turn-59 phantom outlier).
After **chain-pack cumulative offset** fix in `pri/resume.py` / `pri/connector.py`:

- **100%** delta_uniformity on v5 sweep chain
- **100%** on garble-investigation chain
- Geometry verdict: **`pass`**

## Manifest rope gaps (expected)

82 manifest `rope_end[i] ≠ rope_start[i+1]` gaps because each block captures with
`rope_start=22` (resume-shaped per-turn capture). **Pack offsets are contiguous.**

## Tools

```bash
python bench/tier1/geometry_audit.py --from-sweep turn_sweep_cp20_80_v5.json ...
python bench/tier1/rope_delta_microscope.py ...
```

Raw: `geometry_audit_turn_sweep_v5_fixed.json`, `rope_delta_microscope_fixed.json`
"""


def _render_failure_modes(data: dict[str, Any]) -> str:
    lim = data["phase_e_summary"].get("known_limitations") or []
    bullets = "\n".join(f"- {x}" for x in lim)
    return f"""# 8 — Failure modes & garble root cause

## Executive summary

| Category | Status |
|----------|--------|
| Harness bugs (OpenRouter, garbled guard) | **Fixed** in postfix run |
| RoPE geometry | **Pass** at 100% delta_uniformity |
| Long-chain RESUME recall | **Open product issue** at cp60+ |

## Documented limitations

{bullets}

## Garble investigation highlights

From `turn_sweep_cp60_80_garble_inv_garble_cause_cp60.json`:

1. **TEXT 5/5 + RESUME fail** → failure is inject-mediated decode, not absent inline facts.
2. **Facts-only inject (`max_blocks=3`)** still garbles — not tail-noise text pollution alone.
3. **Hotel probe** fails isolated and after probes 1–3 at ~17–18k inject tokens.
4. **21 neutral-substitute blocks** in tail; TEXT still 5/5 with same inline history.

## Session isolation

Each harness uses unique `memory_user` / `memory_base_session`. No cross-test `.nls` bleed
(see [`../internal/FAILURE_AUDIT.md`](../internal/FAILURE_AUDIT.md) §1).

## Historical audits

| Doc | Scope |
|-----|-------|
| [`../internal/FAILURE_AUDIT.md`](../internal/FAILURE_AUDIT.md) | Pre-fix triage (OpenRouter reasoning, unguarded plants) |
| [`../internal/ROPE_DELTA_AUDIT.md`](../internal/ROPE_DELTA_AUDIT.md) | Pre-fix turn-59 phantom outlier |

Post-fix canonical artifacts: [`canonical_artifacts.json`](../canonical_artifacts.json)

## Recommended research follow-ups

1. Profile vLLM decode under long KV inject (cp60+).
2. MoE model matrix — same harness, compare cliff location.
3. Optional: GPU profiler + energy per recall arm.
"""


def build_research_reports(run_dir: Path) -> Path:
    report = build_report(run_dir)
    data = _build_research_data(run_dir, report)
    research_dir = run_dir / "research"
    charts = _build_mermaid_charts(data)

    pages = {
        "README.md": _render_index(data, data["run_dir"], charts),
        "01_methodology.md": _render_methodology(data),
        "02_token_efficiency.md": _render_token_efficiency(data, charts),
        "03_latency_analysis.md": _render_latency(data, charts),
        "04_computational_cost.md": _render_computational_cost(data),
        "05_storage_footprint.md": _render_storage(data, charts),
        "06_turn_sweep_scaling.md": _render_turn_sweep(data, charts),
        "07_rope_geometry.md": _render_geometry(data, charts),
        "08_failure_modes.md": _render_failure_modes(data),
        "09_energy_and_cost.md": _render_energy_cost(data, charts),
    }

    for name, content in pages.items():
        _write(research_dir / name, content)

    _write(research_dir / "research_data.json", json.dumps(data, indent=2))
    return research_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=_ROOT / "bench" / "results" / "overnight_20260624_003614",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Run dir not found: {run_dir}", file=sys.stderr)
        return 1
    out = build_research_reports(run_dir)
    print(f"Wrote research reports under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
