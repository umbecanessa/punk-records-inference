"""Model probe + inject-profile env for Punk Records Inference startup.

Reads ``config.json`` from the mounted checkpoint, derives hybrid layer
topology (full-attention vs linear/Mamba), and writes a cached profile plus
shell env exports under ``NLS_MEMORY_DIR``.

Inject profiles gate Swiss / neural-scoring env vars:

  resume           — chain inject only (default v0.1 bench profile)
  resume_overflow  — chain + Swiss backfill when trim evicts tokens
  swiss            — legacy pool retrieval primary (not default)

Usage (container ``start.sh``):

    python3 -m pri.startup_profile \\
      --model-path "${MODEL_PATH}" \\
      --memory-dir "${NLS_MEMORY_DIR}" \\
      --inject-mode "${NLS_API_INJECT_MODE:-resume}" \\
      --write-env "${NLS_MEMORY_DIR}/profile.env"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("pri.startup_profile")

PROFILE_VERSION = 1
DEFAULT_QWEN_FULL_ATTN = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
DEFAULT_QWEN_LINEAR = [i for i in range(40) if i not in DEFAULT_QWEN_FULL_ATTN]
DEFAULT_QWEN_DELTA_PROBES = [2, 14, 26, 38]

INJECT_PROFILES = frozenset({"resume", "resume_overflow", "swiss"})


def _evenly_spaced(items: list[int], count: int) -> list[int]:
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    picks: list[int] = []
    seen: set[int] = set()
    for i in range(count):
        idx = int(round(i * step))
        layer = items[idx]
        if layer not in seen:
            picks.append(layer)
            seen.add(layer)
    return picks


def _config_fingerprint(config_path: Path) -> str:
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return digest[:16]


@dataclass
class ModelTopology:
    architecture_family: str
    num_hidden_layers: int
    full_attention_layers: list[int]
    linear_attention_layers: list[int]
    num_experts: int
    head_dim: int
    num_kv_heads: int
    rope_theta: float
    model_type: str = ""


@dataclass
class ModelProfile:
    version: int
    config_fingerprint: str
    model_path: str
    inject_mode: str
    topology: ModelTopology
    delta_fact_probe_layers: list[int] = field(default_factory=list)
    neural_score_layers: list[int] = field(default_factory=list)
    v_suppression_at_layer: int = -1
    env_exports: dict[str, str] = field(default_factory=dict)


def probe_model_config(model_path: str | Path) -> ModelTopology:
    """Derive hybrid layer topology from HuggingFace ``config.json``."""
    root = Path(model_path)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found under {root}")

    raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    tc = raw.get("text_config") or raw

    n_layers = int(tc.get("num_hidden_layers") or raw.get("num_hidden_layers") or 40)
    layer_types: list[str] = list(tc.get("layer_types") or raw.get("layer_types") or [])

    full_attn = [
        i for i, lt in enumerate(layer_types) if lt == "full_attention"
    ]
    model_type = str(raw.get("model_type") or tc.get("model_type") or "")
    model_type_l = model_type.lower()

    if not full_attn and not layer_types:
        # Dense checkpoints (Llama, Gemma, …) have no layer_types — all layers
        # are full attention. Do not use full_attention_interval fallback.
        if any(tag in model_type_l for tag in ("llama", "gemma", "mistral", "phi")):
            full_attn = list(range(n_layers))
        else:
            interval = int(tc.get("full_attention_interval") or 4)
            full_attn = list(range(interval - 1, n_layers, interval))

    linear_attn = [i for i in range(n_layers) if i not in full_attn]

    num_experts = int(
        tc.get("num_experts")
        or tc.get("n_routed_experts")
        or tc.get("num_local_experts")
        or raw.get("num_experts")
        or raw.get("n_routed_experts")
        or 0
    )

    hidden = int(tc.get("hidden_size") or raw.get("hidden_size") or 2048)
    n_heads = int(tc.get("num_attention_heads") or raw.get("num_attention_heads") or 16)
    n_kv = int(
        tc.get("num_key_value_heads")
        or raw.get("num_key_value_heads")
        or n_heads
    )
    head_dim = int(tc.get("head_dim") or hidden // max(n_heads, 1))
    rope_theta = float(tc.get("rope_theta") or raw.get("rope_theta") or 10_000_000)

    if layer_types and linear_attn:
        family = "qwen_next_hybrid"
    elif "mamba" in model_type.lower() or (
        "hybrid" in model_type.lower() and linear_attn
    ):
        family = "hybrid_unknown"
    elif "gemma" in model_type.lower() or "llama" in model_type.lower():
        family = "dense_or_unknown"
    elif "enggpt" in model_type.lower() or model_type.lower().endswith("_moe"):
        family = "moe_dense"
    else:
        family = "dense_or_unknown"

    return ModelTopology(
        architecture_family=family,
        num_hidden_layers=n_layers,
        full_attention_layers=full_attn,
        linear_attention_layers=linear_attn,
        num_experts=num_experts,
        head_dim=head_dim,
        num_kv_heads=n_kv,
        rope_theta=rope_theta,
        model_type=model_type,
    )


def derive_probe_layers(topology: ModelTopology) -> tuple[list[int], list[int], int]:
    """Return (delta_fact_probes, neural_score_layers, v_suppression_layer)."""
    n_layers = max(1, topology.num_hidden_layers)
    all_layers = list(range(n_layers))
    linear = list(topology.linear_attention_layers)
    full = list(topology.full_attention_layers)

    if not full and not linear:
        interval = max(1, n_layers // 10)
        full = list(range(interval - 1, n_layers, interval))
        linear = [i for i in all_layers if i not in full]

    delta_probes = _evenly_spaced(linear, 4) if linear else _evenly_spaced(all_layers, 4)
    score_layers = full if full else _evenly_spaced(all_layers, min(10, n_layers))
    v_layer = full[len(full) // 2] if full else all_layers[len(all_layers) // 2]
    return delta_probes, score_layers, v_layer


def derive_vllm_runtime_env(topology: ModelTopology) -> dict[str, str]:
    """vLLM CLI hints from probed topology (written to profile.env for start.sh)."""
    family = topology.architecture_family
    is_hybrid = family in ("qwen_next_hybrid", "hybrid_unknown") and bool(
        topology.linear_attention_layers,
    )
    is_moe_dense = family == "moe_dense" or topology.num_experts > 1
    env: dict[str, str] = {
        "PRI_ARCHITECTURE_FAMILY": family,
        "PRI_MODEL_TYPE": topology.model_type or "unknown",
    }
    if is_hybrid:
        env.update({
            "PRI_VLLM_MAMBA_CACHE": "1",
            "PRI_VLLM_HYBRID_KV": "1",
            "NLS_RESUME_MAMBA_DELTA_SUM": os.environ.get("NLS_RESUME_MAMBA_DELTA_SUM", "1"),
            "PRI_VLLM_TOOL_PARSER": os.environ.get("PRI_VLLM_TOOL_PARSER", "qwen3_coder"),
            "PRI_VLLM_REASONING_PARSER": os.environ.get("PRI_VLLM_REASONING_PARSER", "qwen3"),
        })
    elif is_moe_dense:
        # MoE without Mamba/DeltaNet (e.g. EngGPT2-16B-A3B) — K/V resume, no SSM compounding
        env.update({
            "PRI_VLLM_MAMBA_CACHE": "0",
            "PRI_VLLM_HYBRID_KV": "0",
            "NLS_RESUME_MAMBA_DELTA_SUM": "0",
            "NLS_STRIP_INJECT_SYS_BLOCK_LEN": "0",
            "PRI_VLLM_TOOL_PARSER": "",
            "PRI_VLLM_REASONING_PARSER": "",
            "PRI_VLLM_MODEL_IMPL": "transformers",
        })
    else:
        # Dense / attention-only (Gemma, Llama, etc.) — K/V resume without SSM compounding
        env.update({
            "PRI_VLLM_MAMBA_CACHE": "0",
            "PRI_VLLM_HYBRID_KV": "0",
            "NLS_RESUME_MAMBA_DELTA_SUM": "0",
            "NLS_STRIP_INJECT_SYS_BLOCK_LEN": "0",
            "PRI_VLLM_TOOL_PARSER": "",
            "PRI_VLLM_REASONING_PARSER": "",
        })
    return env


def inject_mode_env(
    inject_mode: str,
    *,
    delta_probes: list[int],
    score_layers: list[int],
    v_suppression_layer: int,
) -> dict[str, str]:
    """Build env exports for the selected inject profile."""
    mode = (inject_mode or "resume").strip().lower()
    if mode not in INJECT_PROFILES:
        mode = "resume"

    delta_csv = ",".join(str(x) for x in delta_probes)
    score_csv = ",".join(str(x) for x in score_layers)

    common: dict[str, str] = {
        "NLS_DELTA_FACT_PROBE_LAYERS": delta_csv,
        "NLS_NEURAL_SCORE_LAYERS": score_csv,
        "NLS_V_SUPPRESSION_AT_LAYER": str(v_suppression_layer),
    }

    if mode == "resume":
        common.update({
            "NLS_NEURAL_SCORING": "0",
            "NLS_V_SUPPRESSION": "0",
            "NLS_DELTA_FACT": "1",
        })
    elif mode == "resume_overflow":
        common.update({
            "NLS_NEURAL_SCORING": "1",
            "NLS_V_SUPPRESSION": "1",
            "NLS_NEURAL_COARSE_K": "10",
            "NLS_NEURAL_FINAL_K": "5",
            "NLS_V_SUPPRESSION_KEEP_K": "5",
            "NLS_RESUME_SWISS_MAX_TOKENS": "256",
            "NLS_DELTA_FACT": "1",
        })
    else:  # swiss
        common.update({
            "NLS_NEURAL_SCORING": "1",
            "NLS_V_SUPPRESSION": "1",
            "NLS_NEURAL_COARSE_K": "20",
            "NLS_NEURAL_FINAL_K": "5",
            "NLS_V_SUPPRESSION_KEEP_K": "5",
            "NLS_DELTA_FACT": "1",
            "NLS_INJECT_MODE": "swiss",
        })

    common["PRI_INJECT_PROFILE"] = mode
    return common


def build_profile(
    model_path: str | Path,
    *,
    inject_mode: str = "resume",
) -> ModelProfile:
    root = Path(model_path)
    config_path = root / "config.json"
    topology = probe_model_config(root)
    delta_probes, score_layers, v_layer = derive_probe_layers(topology)
    fingerprint = _config_fingerprint(config_path)
    env_exports = inject_mode_env(
        inject_mode,
        delta_probes=delta_probes,
        score_layers=score_layers,
        v_suppression_layer=v_layer,
    )
    env_exports.update(derive_vllm_runtime_env(topology))
    return ModelProfile(
        version=PROFILE_VERSION,
        config_fingerprint=fingerprint,
        model_path=str(root),
        inject_mode=(inject_mode or "resume").strip().lower(),
        topology=topology,
        delta_fact_probe_layers=delta_probes,
        neural_score_layers=score_layers,
        v_suppression_at_layer=v_layer,
        env_exports=env_exports,
    )


def profile_json_path(memory_dir: str | Path) -> Path:
    return Path(memory_dir) / "model_profile.json"


def profile_env_path(memory_dir: str | Path) -> Path:
    return Path(memory_dir) / "profile.env"


def profile_is_current(profile: ModelProfile, memory_dir: str | Path) -> bool:
    path = profile_json_path(memory_dir)
    if not path.is_file():
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        saved.get("version") == PROFILE_VERSION
        and saved.get("config_fingerprint") == profile.config_fingerprint
        and saved.get("inject_mode") == profile.inject_mode
    )


def profile_to_json_dict(profile: ModelProfile) -> dict[str, Any]:
    data = asdict(profile)
    data["topology"] = asdict(profile.topology)
    return data


def write_profile(profile: ModelProfile, memory_dir: str | Path) -> Path:
    mem = Path(memory_dir)
    mem.mkdir(parents=True, exist_ok=True)
    out = profile_json_path(mem)
    out.write_text(
        json.dumps(profile_to_json_dict(profile), indent=2),
        encoding="utf-8",
    )
    return out


def write_env_file(profile: ModelProfile, env_path: str | Path) -> Path:
    path = Path(env_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by pri.startup_profile — do not edit by hand",
        f"# inject_mode={profile.inject_mode} fingerprint={profile.config_fingerprint}",
    ]
    for key, val in sorted(profile.env_exports.items()):
        escaped = val.replace("'", "'\\''")
        lines.append(f"export {key}='{escaped}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def apply_env_exports(
    exports: dict[str, str],
    *,
    respect_existing: bool = True,
) -> list[str]:
    """Apply exports to ``os.environ``. Returns keys that were set."""
    applied: list[str] = []
    for key, val in exports.items():
        if respect_existing and key in os.environ and os.environ[key].strip():
            continue
        os.environ[key] = val
        applied.append(key)
    return applied


def run_startup_profile(
    model_path: str | Path,
    memory_dir: str | Path,
    *,
    inject_mode: str = "resume",
    write_env: str | Path | None = None,
    force: bool = False,
) -> ModelProfile:
    profile = build_profile(model_path, inject_mode=inject_mode)
    if force or not profile_is_current(profile, memory_dir):
        write_profile(profile, memory_dir)
        env_target = Path(write_env) if write_env else profile_env_path(memory_dir)
        write_env_file(profile, env_target)
        logger.info(
            "Wrote model profile inject=%s family=%s layers=%d full_attn=%d linear=%d",
            profile.inject_mode,
            profile.topology.architecture_family,
            profile.topology.num_hidden_layers,
            len(profile.topology.full_attention_layers),
            len(profile.topology.linear_attention_layers),
        )
        try:
            from pri.rope_pack_balance import run_profile_self_check

            run_profile_self_check(profile_json_path(memory_dir))
        except Exception as exc:
            logger.warning("RoPE pack self-check skipped: %s", exc)
    else:
        logger.info(
            "Model profile unchanged (fingerprint=%s inject=%s)",
            profile.config_fingerprint,
            profile.inject_mode,
        )
    apply_env_exports(profile.env_exports)
    return profile


def _shell_quote(val: str) -> str:
    if not val:
        return "''"
    if all(c.isalnum() or c in "._-/+:" for c in val):
        return val
    return "'" + val.replace("'", "'\\''") + "'"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument(
        "--memory-dir",
        default=os.environ.get("NLS_MEMORY_DIR", "/data/pri"),
    )
    parser.add_argument(
        "--inject-mode",
        default=os.environ.get("NLS_API_INJECT_MODE", "resume_overflow"),
    )
    parser.add_argument(
        "--write-env",
        default="",
        help="Write shell exports to this file (default: MEMORY_DIR/profile.env)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--apply-shell",
        action="store_true",
        help="Print export lines to stdout for eval/source",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="[startup_profile] %(message)s",
    )

    if not args.model_path:
        print("ERROR: --model-path or MODEL_PATH required", file=sys.stderr)
        return 2

    env_path = args.write_env or str(profile_env_path(args.memory_dir))
    profile = run_startup_profile(
        args.model_path,
        args.memory_dir,
        inject_mode=args.inject_mode,
        write_env=env_path,
        force=args.force,
    )

    if args.apply_shell:
        for key, val in sorted(profile.env_exports.items()):
            print(f"export {key}={_shell_quote(val)}")
    else:
        topo = profile.topology
        print(
            f"profile ok inject={profile.inject_mode} "
            f"family={topo.architecture_family} "
            f"layers={topo.num_hidden_layers} "
            f"full_attn={len(topo.full_attention_layers)} "
            f"delta_probes={profile.delta_fact_probe_layers} "
            f"env={env_path}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
