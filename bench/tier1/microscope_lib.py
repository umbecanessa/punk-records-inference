"""Microscope capture comparison utilities (attn vs SSM parity).

Runs inside the vLLM container where ``/tmp/nls_microscope`` captures land.
Host-side scripts send HTTP requests; compare step needs ``torch``.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import torch

MICROSCOPE_DIR = os.environ.get("NLS_MICROSCOPE_DIR", "/tmp/nls_microscope")
PARITY_THRESHOLD = float(os.environ.get("NLS_RESUME_PARITY_COS", "0.999"))

FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
DELTANET_LAYERS = [i for i in range(40) if i not in FULL_ATTN_LAYERS]

STAGE_SUFFIXES = (
    "attn_input_hs",
    "ssm_state",
    "deltanet_out_hs",
)


def clear_microscope_dir() -> None:
    os.makedirs(MICROSCOPE_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(MICROSCOPE_DIR, "microscope_*.pt")):
        os.remove(path)


def find_latest_capture(tag: str) -> str | None:
    pattern = os.path.join(MICROSCOPE_DIR, f"microscope_{tag}_*.pt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def capture_seq_len(path: str, *, stage_suffix: str = "attn_input_hs") -> int:
    data = torch.load(path, map_location="cpu", weights_only=True)
    for layer_idx in range(40):
        key = f"L{layer_idx}_{stage_suffix}"
        if key not in data:
            continue
        tensor = data[key]
        if tensor.dim() >= 2:
            return int(tensor.shape[0])
        return 1
    return 0


def cosine_at_position(
    text_path: str,
    kv_path: str,
    *,
    text_pos: int = -1,
    kv_pos: int = -1,
    stage_suffix: str = "attn_input_hs",
) -> dict[str, Any]:
    text_data = torch.load(text_path, map_location="cpu", weights_only=True)
    kv_data = torch.load(kv_path, map_location="cpu", weights_only=True)

    results: dict[str, Any] = {}
    layer_rows: list[dict] = []
    for layer_idx in range(40):
        key = f"L{layer_idx}_{stage_suffix}"
        if key not in text_data or key not in kv_data:
            continue
        text_tensor = text_data[key].float()
        kv_tensor = kv_data[key].float()
        if text_tensor.dim() < 1 or kv_tensor.dim() < 1:
            continue
        if text_tensor.dim() >= 2:
            ti = text_pos if text_pos >= 0 else text_tensor.shape[0] + text_pos
            ki = kv_pos if kv_pos >= 0 else kv_tensor.shape[0] + kv_pos
            if not (0 <= ti < text_tensor.shape[0] and 0 <= ki < kv_tensor.shape[0]):
                continue
            text_flat = text_tensor[ti].flatten()
            kv_flat = kv_tensor[ki].flatten()
        else:
            text_flat = text_tensor.flatten()
            kv_flat = kv_tensor.flatten()

        text_norm = text_flat.norm().item()
        kv_norm = kv_flat.norm().item()
        if text_norm < 1e-8 or kv_norm < 1e-8:
            cosine = 0.0
        else:
            cosine = float(torch.dot(text_flat, kv_flat) / (text_norm * kv_norm))
        l2 = float((text_flat - kv_flat).norm().item())
        results[key] = {"cosine": cosine, "l2": l2}
        layer_type = "ATT" if layer_idx in FULL_ATTN_LAYERS else "DN"
        layer_rows.append({
            "layer": layer_idx,
            "layer_type": layer_type,
            "cosine": round(cosine, 6),
            "l2": round(l2, 4),
        })

    avg = (
        sum(results[k]["cosine"] for k in results if k.startswith("L"))
        / len(results)
        if results else 0.0
    )
    attn_keys = [k for k in results if "attn_input_hs" in k]
    dn_keys = [k for k in results if "ssm_state" in k or "deltanet_out_hs" in k]
    attn_avg = (
        sum(results[k]["cosine"] for k in attn_keys) / len(attn_keys)
        if attn_keys else None
    )
    dn_avg = (
        sum(results[k]["cosine"] for k in dn_keys) / len(dn_keys)
        if dn_keys else None
    )

    return {
        **results,
        "_stage": stage_suffix,
        "_query_cosine_avg": round(avg, 6),
        "_attn_layer_avg": round(attn_avg, 6) if attn_avg is not None else None,
        "_deltanet_layer_avg": round(dn_avg, 6) if dn_avg is not None else None,
        "_text_seq_len": capture_seq_len(text_path, stage_suffix=stage_suffix),
        "_kv_seq_len": capture_seq_len(kv_path, stage_suffix=stage_suffix),
        "_text_pos": text_pos,
        "_kv_pos": kv_pos,
        "layers": layer_rows,
        "worst_layer": min(layer_rows, key=lambda r: r["cosine"]) if layer_rows else None,
        "worst_attn_layer": (
            min(
                (r for r in layer_rows if r["layer_type"] == "ATT"),
                key=lambda r: r["cosine"],
            )
            if any(r["layer_type"] == "ATT" for r in layer_rows) else None
        ),
        "worst_deltanet_layer": (
            min(
                (r for r in layer_rows if r["layer_type"] == "DN"),
                key=lambda r: r["cosine"],
            )
            if any(r["layer_type"] == "DN" for r in layer_rows) else None
        ),
    }


def compare_pair_query(
    tag_a: str,
    tag_b: str,
    *,
    path_a: str | None = None,
    path_b: str | None = None,
) -> dict[str, Any]:
    """Query-token parity for attn, SSM, and DeltaNet output stages."""
    file_a = path_a or find_latest_capture(tag_a)
    file_b = path_b or find_latest_capture(tag_b)
    if not file_a or not file_b:
        return {
            "error": "missing capture",
            "tag_a": tag_a,
            "tag_b": tag_b,
            "path_a": file_a,
            "path_b": file_b,
        }

    by_stage: dict[str, Any] = {}
    for stage in STAGE_SUFFIXES:
        res = cosine_at_position(
            file_a, file_b,
            text_pos=-1, kv_pos=-1,
            stage_suffix=stage,
        )
        avg = float(res.get("_query_cosine_avg", 0.0))
        by_stage[stage] = {
            "query_cosine_avg": round(avg, 6),
            "parity_pass": avg >= PARITY_THRESHOLD,
            "layer_count": len(res.get("layers") or []),
            "worst_layer": res.get("worst_layer"),
            "worst_attn_layer": res.get("worst_attn_layer"),
            "worst_deltanet_layer": res.get("worst_deltanet_layer"),
            "layers": res.get("layers") or [],
        }

    attn_avg = by_stage.get("attn_input_hs", {}).get("query_cosine_avg", 0.0)
    ssm_avg = by_stage.get("ssm_state", {}).get("query_cosine_avg", 0.0)
    return {
        "tag_a": tag_a,
        "tag_b": tag_b,
        "path_a": file_a,
        "path_b": file_b,
        "text_seq_len": capture_seq_len(file_a),
        "kv_seq_len": capture_seq_len(file_b),
        "threshold": PARITY_THRESHOLD,
        "stages": by_stage,
        "attn_vs_ssm_gap": round(float(attn_avg) - float(ssm_avg), 6),
    }


def compare_at_positions(
    tag_a: str,
    tag_b: str,
    *,
    path_a: str | None = None,
    path_b: str | None = None,
    text_pos: int,
    kv_pos: int,
    label: str,
    stage_suffix: str = "attn_input_hs",
) -> dict[str, Any]:
    file_a = path_a or find_latest_capture(tag_a)
    file_b = path_b or find_latest_capture(tag_b)
    if not file_a or not file_b:
        return {"error": "missing capture", "label": label}

    res = cosine_at_position(
        file_a, file_b,
        text_pos=text_pos, kv_pos=kv_pos,
        stage_suffix=stage_suffix,
    )
    avg = float(res.get("_query_cosine_avg", 0.0))
    return {
        "label": label,
        "stage": stage_suffix,
        "tag_a": tag_a,
        "tag_b": tag_b,
        "text_pos": text_pos,
        "kv_pos": kv_pos,
        "cosine_avg": round(avg, 6),
        "parity_pass": avg >= PARITY_THRESHOLD,
        "worst_layer": res.get("worst_layer"),
        "layers": res.get("layers") or [],
    }


def layer_profile_query(tag_a: str, tag_b: str) -> dict[str, Any]:
    """Per-layer attn vs ssm at query token for one pair."""
    pair = compare_pair_query(tag_a, tag_b)
    if pair.get("error"):
        return pair

    attn_layers = {
        row["layer"]: row["cosine"]
        for row in pair["stages"]["attn_input_hs"]["layers"]
    }
    ssm_layers = {
        row["layer"]: row["cosine"]
        for row in pair["stages"]["ssm_state"]["layers"]
    }
    profile: list[dict] = []
    for layer_idx in sorted(set(attn_layers) | set(ssm_layers)):
        attn_cos = attn_layers.get(layer_idx)
        ssm_cos = ssm_layers.get(layer_idx)
        gap = None
        if attn_cos is not None and ssm_cos is not None:
            gap = round(attn_cos - ssm_cos, 6)
        profile.append({
            "layer": layer_idx,
            "layer_type": "ATT" if layer_idx in FULL_ATTN_LAYERS else "DN",
            "attn_cosine": attn_cos,
            "ssm_cosine": ssm_cos,
            "attn_minus_ssm": gap,
        })
    return {
        "tag_a": tag_a,
        "tag_b": tag_b,
        "attn_vs_ssm_gap": pair["attn_vs_ssm_gap"],
        "profile": profile,
    }
