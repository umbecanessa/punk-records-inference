"""Plug-and-play RoPE pack self-check and inject snapshot balancing.

Turn-capture chains store ``rope_start`` per block (usually sys-prompt strip),
not cumulative manifest positions — manifest ``rope_end`` vs next ``rope_start``
gaps are **expected** on Tier B. Inject uses pack cumulative offset +
``rope_start + phantom_at_capture`` for re-rotation (see ``inject_geometry_audit``).

This module runs at startup and optionally normalizes resume inject snapshots
before geometry preflight so new models fail only on real pack errors
(``fail_rope_pack`` = inconsistent deltas), not cosmetic manifest gaps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("nls_rope_pack_balance")


def infer_capture_tier(profile: dict[str, Any]) -> str:
    """Return ``A`` (hybrid+mamba), ``B`` (kv-only), or ``C`` (experimental)."""
    family = str(
        profile.get("topology", {}).get("architecture_family")
        or profile.get("env_exports", {}).get("PRI_ARCHITECTURE_FAMILY")
        or "",
    )
    mamba = str(profile.get("env_exports", {}).get("NLS_RESUME_MAMBA_DELTA_SUM", "0"))
    if family in ("qwen_next_hybrid", "hybrid_unknown") and mamba not in ("0", ""):
        return "A"
    if family in ("dense_or_unknown", "moe_dense"):
        return "B"
    return "C"


def expected_findings_for_tier(tier: str, profile: dict[str, Any]) -> list[str]:
    """Human-readable expectations so matrix logs are not misread."""
    topo = profile.get("topology") or {}
    full = len(topo.get("full_attention_layers") or [])
    linear = len(topo.get("linear_attention_layers") or [])
    n_layers = int(topo.get("num_hidden_layers") or 0)
    lines = [
        f"Tier {tier}: inject is K/V + RoPE"
        + (" + Mamba delta-sum" if tier == "A" else " only (no SSM compounding)"),
    ]
    if tier == "B" and linear and full:
        lines.append(
            f"Partial layer capture expected: {full} full-attn / {n_layers} layers "
            f"({linear} linear/sliding layers skipped in .nls readback)",
        )
    if tier in ("A", "B"):
        lines.append(
            "Turn-capture manifest rope_end→rope_start gaps are OK; "
            "pack continuity + uniform resume RoPE delta matter",
        )
        lines.append(
            "capture_start is resolved at runtime via vLLM /tokenize messages "
            "(pri.chat_template) — no per-model hard-coded strip lengths",
        )
    return lines


def balance_resume_snapshots(snapshots: list[dict]) -> list[dict]:
    """Sort by turn_index and ensure pack fields exist for geometry audit."""
    if not snapshots:
        return snapshots
    ordered = sorted(
        snapshots,
        key=lambda s: (
            int(s.get("turn_index") if s.get("turn_index") is not None else -1),
            str(s.get("path") or ""),
        ),
    )
    out: list[dict] = []
    for snap in ordered:
        row = dict(snap)
        row.setdefault("strip_prefix", 0)
        if row.get("turn_index") is None:
            row["turn_index"] = -1
        out.append(row)
    return out


def audit_and_balance_resume_config(cfg: dict | None) -> tuple[dict | None, dict[str, Any]]:
    """Normalize snapshots then run geometry audit. Returns (cfg, summary)."""
    from pri.inject_geometry_audit import summarize_geometry_audit

    empty: dict[str, Any] = {"verdict": "pass", "findings": [], "block_count": 0}
    if not cfg or cfg.get("inject_layout") != "resume":
        return cfg, empty

    snaps = balance_resume_snapshots(list(cfg.get("snapshots") or []))
    cfg = dict(cfg)
    cfg["snapshots"] = snaps
    summary = summarize_geometry_audit(
        snaps,
        rope_offset=0,
        resume_mode=True,
        mamba_delta_sum=int(cfg.get("mamba_delta_sum", 0) or 0),
    )
    return cfg, summary


def run_profile_self_check(profile_path: str | Path) -> dict[str, Any]:
    """Startup hook: log tier expectations from ``model_profile.json``."""
    path = Path(profile_path)
    if not path.is_file():
        return {"ok": False, "error": f"missing {path}"}

    profile = json.loads(path.read_text(encoding="utf-8"))
    tier = infer_capture_tier(profile)
    expectations = expected_findings_for_tier(tier, profile)
    report = {
        "ok": True,
        "tier": tier,
        "model_type": profile.get("topology", {}).get("model_type"),
        "architecture_family": profile.get("topology", {}).get("architecture_family"),
        "expectations": expectations,
    }
    logger.info(
        "NLS rope pack self-check: tier=%s family=%s — %s",
        tier,
        report.get("architecture_family"),
        "; ".join(expectations[:2]),
    )
    for line in expectations[2:]:
        logger.info("NLS rope pack self-check: %s", line)
    return report
