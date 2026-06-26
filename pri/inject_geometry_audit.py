"""Inject geometry audit — RoPE pack plan, Mamba provenance, capture consistency.

Validates that a resume inject pack is geometrically consistent before loading KV:

  - Per-block ``rope_start`` → cumulative pack offset mapping
  - Mamba delta-sum mode vs chain block order
  - Strip prefix and token counts vs manifest

Turn-capture resume replay uses **pack cumulative offset** as the phantom
for RoPE re-rotation. Manifest ``capture_num_phantom`` is chain provenance
(tokens before this block in ``base_session`` order), not live request
``num_phantom`` from the HTTP inject path.

Called at resume inject time inside ``pri.connector`` and offline by
``bench/tier1/geometry_audit.py``. Abort controlled by
``NLS_RESUME_ABORT_ON_ROPE_FAIL`` (default on).
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("nls_inject_geometry_audit")


@dataclass
class BlockRopeAudit:
    index: int
    path: str
    session_id: str
    turn_index: int
    role: str
    num_tokens: int
    strip_prefix: int
    manifest_rope_start: int
    manifest_rope_end: int
    pack_offset: int
    rope_old_effective: int
    rope_new: int
    rope_delta: int
    resume_zero_delta_expected: bool
    capture_num_phantom: int
    sys_prompt_hash: str
    issues: list[str] = field(default_factory=list)


def _resolve_kv_path(kv_path: str) -> str:
    """Map container paths to host paths when auditing outside Docker."""
    path = Path(kv_path)
    if path.exists():
        return str(path)

    mem_dir = os.environ.get("NLS_MEMORY_DIR", "/data/pri").rstrip("/")
    snapshot_dir = os.environ.get("NLS_SNAPSHOT_DIR", f"{mem_dir}/snapshot").rstrip("/")
    replacements = (
        ("/data/pri/snapshot", snapshot_dir),
        ("/data/pri", mem_dir),
    )
    for old, new in replacements:
        if kv_path.startswith(old):
            alt = kv_path.replace(old, new, 1)
            if Path(alt).exists():
                return alt

    name = path.name
    for base in (
        f"{snapshot_dir}/captures",
        f"{mem_dir}/snapshot/captures",
    ):
        alt = str(Path(base) / name)
        if Path(alt).exists():
            return alt
    return kv_path


def _read_manifest(kv_path: str) -> dict | None:
    try:
        from pri.format import read_manifest
        return read_manifest(_resolve_kv_path(kv_path))
    except Exception:
        return None


def audit_rope_pack_plan(
    snapshots: list[dict],
    *,
    rope_offset: int = 0,
    resume_mode: bool = False,
) -> list[BlockRopeAudit]:
    """Per-block RoPE re-rotation plan (mirrors ``_load_multi_snapshots``)."""
    rows: list[BlockRopeAudit] = []
    total_tokens = 0

    for idx, snap in enumerate(snapshots):
        path = str(snap.get("path") or "")
        n_tok = int(snap.get("num_tokens") or 0)
        strip = int(snap.get("strip_prefix") or 0)
        pack_offset = rope_offset + total_tokens

        manifest = _read_manifest(path) if path.endswith(".nls") else None
        rope_start = int(snap.get("rope_start") or 0)
        rope_end = 0
        capture_phantom = 0
        sys_hash = ""
        session_id = ""
        turn_index = -1
        role = ""
        if manifest:
            rope_start = int(manifest.get("rope_start", rope_start) or 0)
            rope_end = int(manifest.get("rope_end", 0) or 0)
            capture_phantom = int(manifest.get("capture_num_phantom", 0) or 0)
            sys_hash = str(manifest.get("sys_prompt_hash", "") or "")
            session_id = str(manifest.get("session_id", "") or "")
            turn_index = int(manifest.get("turn_index", -1))
            role = str(manifest.get("role", "") or "")

        phantom_at_capture = 0
        if resume_mode:
            if role == "turn":
                phantom_at_capture = total_tokens
            elif capture_phantom > 0:
                phantom_at_capture = capture_phantom
            rope_old = max(strip, rope_start + phantom_at_capture)
        else:
            rope_old = max(strip, rope_start)
        rope_new = pack_offset
        rope_delta = rope_new - rope_old

        issues: list[str] = []
        if n_tok <= 0:
            issues.append("num_tokens<=0")
        if not Path(_resolve_kv_path(path)).exists():
            issues.append("kv file missing")

        rows.append(BlockRopeAudit(
            index=idx,
            path=path,
            session_id=session_id,
            turn_index=turn_index,
            role=role,
            num_tokens=n_tok,
            strip_prefix=strip,
            manifest_rope_start=rope_start,
            manifest_rope_end=rope_end,
            pack_offset=pack_offset,
            rope_old_effective=rope_old,
            rope_new=rope_new,
            rope_delta=rope_delta,
            resume_zero_delta_expected=resume_mode,
            capture_num_phantom=capture_phantom,
            sys_prompt_hash=sys_hash,
            issues=issues,
        ))
        total_tokens += n_tok

    return rows


def audit_chain_continuity(rope_rows: list[BlockRopeAudit]) -> dict[str, Any]:
    """Check manifest rope_end[i] vs rope_start[i+1] and pack contiguity."""
    gaps: list[dict] = []
    for i in range(len(rope_rows) - 1):
        left = rope_rows[i]
        right = rope_rows[i + 1]
        if left.manifest_rope_end <= 0:
            continue
        expected = left.manifest_rope_end
        actual = right.manifest_rope_start
        if expected != actual:
            gaps.append({
                "after_block": i,
                "before_block": i + 1,
                "left_rope_end": expected,
                "right_rope_start": actual,
                "gap": actual - expected,
            })

    pack_gaps: list[dict] = []
    for i in range(len(rope_rows) - 1):
        left = rope_rows[i]
        right = rope_rows[i + 1]
        expected = left.pack_offset + left.num_tokens
        actual = right.pack_offset
        if expected != actual:
            pack_gaps.append({
                "after_block": i,
                "expected_pack": expected,
                "actual_pack": actual,
            })

    starts = [r.manifest_rope_start for r in rope_rows]
    uniform_start = len(set(starts)) == 1 and len(starts) > 1

    return {
        "gap_count": len(gaps),
        "gaps": gaps,
        "continuous": len(gaps) == 0,
        "pack_continuous": len(pack_gaps) == 0,
        "pack_gaps": pack_gaps,
        "uniform_manifest_rope_start": uniform_start,
        "shared_rope_start": starts[0] if uniform_start else None,
    }


def audit_sys_prompt_consistency(
    rope_rows: list[BlockRopeAudit],
    live_hash: str | None,
) -> dict[str, Any]:
    hashes = sorted({r.sys_prompt_hash for r in rope_rows if r.sys_prompt_hash})
    issues: list[str] = []
    mismatched: list[dict] = []
    if len(hashes) > 1:
        issues.append(f"chain has {len(hashes)} distinct sys_prompt_hash values")
    if live_hash:
        mismatched = [
            {
                "block": r.index,
                "session_id": r.session_id,
                "hash": r.sys_prompt_hash,
            }
            for r in rope_rows
            if r.sys_prompt_hash and r.sys_prompt_hash != live_hash
        ]
        if mismatched:
            issues.append(
                f"{len(mismatched)} blocks disagree with live sys hash {live_hash}"
            )
    return {
        "distinct_hashes": hashes,
        "live_hash": live_hash or "",
        "mismatch_blocks": mismatched,
        "issues": issues,
    }


def audit_mamba_resume_plan(
    snapshots: list[dict],
    mamba_delta_sum: int,
) -> dict[str, Any]:
    """Summarize Mamba inject strategy and last-block key presence."""
    mode_names = {
        0: "genesis_only",
        1: "genesis_plus_sum_deltas",
        2: "genesis_plus_last_delta",
        3: "resume_last_block_verbatim",
    }
    last_path = _resolve_kv_path(str(snapshots[-1]["path"])) if snapshots else ""
    mamba_keys: list[str] = []
    issues: list[str] = []

    if last_path and Path(last_path).exists():
        try:
            from pri.format import load_nls
            data = load_nls(last_path)
            mamba_keys = sorted(
                k for k in data if "mamba_ssm" in k or "mamba_conv" in k
            )
        except Exception as exc:
            issues.append(f"failed to load last block mamba: {exc}")
    else:
        issues.append("last chain block path missing")

    if mamba_delta_sum == 3 and len(snapshots) > 1:
        issues.append(
            "mode=3 uses ONLY last block Mamba; prior blocks' SSM not compounded"
        )

    return {
        "mamba_delta_sum": mamba_delta_sum,
        "mode_name": mode_names.get(mamba_delta_sum, f"unknown_{mamba_delta_sum}"),
        "last_block_path": last_path,
        "last_block_mamba_keys": len(mamba_keys),
        "block_count": len(snapshots),
        "issues": issues,
    }


def audit_capture_provenance(rope_rows: list[BlockRopeAudit]) -> dict[str, Any]:
    """Phantom-at-capture signals from manifest (incl. inferred from rope_start)."""
    rows: list[dict] = []
    for r in rope_rows:
        inferred_phantom = r.manifest_rope_start > 0 or r.capture_num_phantom > 0
        rows.append({
            "block": r.index,
            "turn_index": r.turn_index,
            "role": r.role,
            "capture_num_phantom": r.capture_num_phantom,
            "manifest_rope_start": r.manifest_rope_start,
            "inferred_resume_shaped_capture": inferred_phantom,
        })
    phantom_captures = sum(1 for x in rows if x["inferred_resume_shaped_capture"])
    return {
        "blocks": rows,
        "resume_shaped_capture_count": phantom_captures,
        "all_cold_inline_captures": phantom_captures == 0,
    }


def summarize_geometry_audit(
    snapshots: list[dict],
    *,
    rope_offset: int = 0,
    resume_mode: bool = False,
    mamba_delta_sum: int = 0,
    live_sys_hash: str | None = None,
) -> dict[str, Any]:
    rope_rows = audit_rope_pack_plan(
        snapshots, rope_offset=rope_offset, resume_mode=resume_mode,
    )
    continuity = audit_chain_continuity(rope_rows)
    sys_audit = audit_sys_prompt_consistency(rope_rows, live_sys_hash)
    mamba_audit = audit_mamba_resume_plan(snapshots, mamba_delta_sum)
    provenance = audit_capture_provenance(rope_rows)

    findings: list[str] = []
    system_rows = [r for r in rope_rows if r.role == "system"]
    pack_turn_rows = [r for r in rope_rows if r.role != "system"]
    non_zero = [r for r in pack_turn_rows if r.rope_delta != 0]
    delta_values = sorted({r.rope_delta for r in pack_turn_rows})
    if resume_mode:
        if system_rows:
            findings.append(
                f"resume pack includes {len(system_rows)} system block(s) at pack offset "
                f"(RoPE delta={system_rows[0].rope_delta} expected for cold system KV)"
            )
        if len(delta_values) > 1:
            findings.append(
                f"inconsistent resume RoPE deltas across blocks: {delta_values}"
            )
        elif non_zero and delta_values:
            findings.append(
                f"uniform resume RoPE delta={delta_values[0]} on "
                f"{len(pack_turn_rows)} turn block(s) (expected after phantom-aware rotate)"
            )
    elif non_zero:
        findings.append(
            f"{len(non_zero)}/{len(rope_rows)} blocks need RoPE rerotation"
        )
    if not continuity["continuous"]:
        findings.append(
            f"manifest rope discontinuity: {continuity['gap_count']} gap(s)"
        )
    if continuity.get("uniform_manifest_rope_start"):
        shared = continuity.get("shared_rope_start")
        findings.append(
            f"turn blocks use per-request rope_start={shared}; resume pack uses "
            f"rope_old=rope_start+phantom_at_capture"
        )
    findings.extend(sys_audit.get("issues") or [])
    findings.extend(mamba_audit.get("issues") or [])
    if not provenance["all_cold_inline_captures"]:
        findings.append(
            f"{provenance['resume_shaped_capture_count']} blocks captured with "
            f"non-zero rope_start/phantom (resume-shaped, not cold inline)"
        )

    block_issues = [r for r in rope_rows if r.issues]
    if block_issues:
        findings.append(f"{len(block_issues)} blocks have per-block issues")

    if not findings:
        verdict = "pass"
    elif resume_mode and len(delta_values) <= 1:
        verdict = "pass"
    elif resume_mode and len(delta_values) > 1:
        verdict = "fail_rope_pack"
    elif resume_mode and non_zero:
        verdict = "warn"
    else:
        verdict = "warn"

    return {
        "verdict": verdict,
        "findings": findings,
        "total_inject_tokens": sum(int(s.get("num_tokens") or 0) for s in snapshots),
        "block_count": len(snapshots),
        "resume_mode": resume_mode,
        "rope_offset": rope_offset,
        "mamba_delta_sum": mamba_delta_sum,
        "rope_pack": [asdict(r) for r in rope_rows],
        "chain_continuity": continuity,
        "sys_prompt": sys_audit,
        "mamba": mamba_audit,
        "capture_provenance": provenance,
    }


def resume_inject_aborted(summary: dict[str, Any]) -> bool:
    """True when resume pack geometry is unsafe and inject should be skipped."""
    if summary.get("verdict") != "fail_rope_pack":
        return False
    return os.environ.get("NLS_RESUME_ABORT_ON_ROPE_FAIL", "1") != "0"


def log_geometry_audit(summary: dict[str, Any], *, req_id: str = "") -> None:
    """Structured log for connector hot path."""
    prefix = f"req={req_id[:16]} " if req_id else ""
    logger.info(
        "NLS inject geometry audit: %sverdict=%s blocks=%d tokens=%d findings=%s",
        prefix,
        summary.get("verdict"),
        summary.get("block_count"),
        summary.get("total_inject_tokens"),
        summary.get("findings"),
    )
    for row in summary.get("rope_pack") or []:
        if row.get("rope_delta") != 0 or row.get("issues"):
            logger.warning(
                "NLS inject geometry block[%d]: turn=%s role=%s "
                "rope_delta=%d pack=%d manifest_start=%d phantom=%d issues=%s",
                row.get("index"),
                row.get("turn_index"),
                row.get("role"),
                row.get("rope_delta"),
                row.get("pack_offset"),
                row.get("manifest_rope_start"),
                row.get("capture_num_phantom"),
                row.get("issues"),
            )
