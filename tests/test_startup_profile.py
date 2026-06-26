"""Tests for pri.startup_profile model probe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pri.startup_profile import (
    build_profile,
    inject_mode_env,
    probe_model_config,
    profile_is_current,
    run_startup_profile,
    write_profile,
)


FIXTURE = Path(__file__).parent / "fixtures" / "qwen35_hybrid_config.json"


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(FIXTURE.read_text(encoding="utf-8"))
    return root


def test_probe_qwen_hybrid_layers(model_dir: Path) -> None:
    topo = probe_model_config(model_dir)
    assert topo.num_hidden_layers == 40
    assert topo.full_attention_layers == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    assert len(topo.linear_attention_layers) == 30
    assert topo.num_experts == 512
    assert topo.architecture_family == "qwen_next_hybrid"


def test_inject_mode_resume_disables_neural() -> None:
    env = inject_mode_env(
        "resume",
        delta_probes=[2, 14, 26, 38],
        score_layers=[3, 7, 11],
        v_suppression_layer=11,
    )
    assert env["NLS_NEURAL_SCORING"] == "0"
    assert env["NLS_V_SUPPRESSION"] == "0"
    assert env["PRI_INJECT_PROFILE"] == "resume"


def test_inject_mode_overflow_enables_neural() -> None:
    env = inject_mode_env(
        "resume_overflow",
        delta_probes=[2, 14],
        score_layers=[3, 7],
        v_suppression_layer=7,
    )
    assert env["NLS_NEURAL_SCORING"] == "1"
    assert env["NLS_V_SUPPRESSION"] == "1"
    assert env["NLS_RESUME_SWISS_MAX_TOKENS"] == "256"


def test_profile_write_and_cache_hit(model_dir: Path, tmp_path: Path) -> None:
    mem = tmp_path / "pri"
    profile = build_profile(model_dir, inject_mode="resume")
    write_profile(profile, mem)
    assert (mem / "model_profile.json").is_file()
    assert profile_is_current(profile, mem)


def test_run_startup_profile_sets_env(
    model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mem = tmp_path / "pri"
    monkeypatch.delenv("NLS_NEURAL_SCORING", raising=False)
    run_startup_profile(model_dir, mem, inject_mode="resume", force=True)
    import os
    assert os.environ.get("NLS_NEURAL_SCORING") == "0"
    assert os.environ.get("NLS_DELTA_FACT_PROBE_LAYERS")


def test_dense_vllm_runtime_env() -> None:
    from pri.startup_profile import ModelTopology, derive_vllm_runtime_env

    topo = ModelTopology(
        architecture_family="dense_or_unknown",
        num_hidden_layers=32,
        full_attention_layers=list(range(32)),
        linear_attention_layers=[],
        num_experts=0,
        head_dim=128,
        num_kv_heads=8,
        rope_theta=10000.0,
        model_type="gemma3",
    )
    env = derive_vllm_runtime_env(topo)
    assert env["PRI_VLLM_MAMBA_CACHE"] == "0"
    assert env["NLS_RESUME_MAMBA_DELTA_SUM"] == "0"
    assert env["PRI_VLLM_TOOL_PARSER"] == ""


def test_hybrid_vllm_runtime_env(model_dir: Path) -> None:
    from pri.startup_profile import derive_vllm_runtime_env, probe_model_config

    env = derive_vllm_runtime_env(probe_model_config(model_dir))
    assert env["PRI_VLLM_MAMBA_CACHE"] == "1"
    assert env["NLS_RESUME_MAMBA_DELTA_SUM"] == "1"


def test_profile_json_roundtrip(model_dir: Path) -> None:
    from pri.startup_profile import profile_to_json_dict

    profile = build_profile(model_dir, inject_mode="resume_overflow")
    data = profile_to_json_dict(profile)
    assert data["inject_mode"] == "resume_overflow"
