"""Tests for resume Mamba mode override on the wire."""

from __future__ import annotations

from pri.resume import apply_mamba_mode_override, build_resume_inject_config


def test_apply_mamba_mode_override_sets_delta_sum():
    cfg = {"mamba_delta_sum": 1, "snapshots": []}
    apply_mamba_mode_override(cfg, {"memory_mamba_mode": "2"})
    assert cfg["mamba_delta_sum"] == 2


def test_apply_mamba_mode_override_noop_when_missing():
    cfg = {"mamba_delta_sum": 1}
    apply_mamba_mode_override(cfg, {})
    assert cfg["mamba_delta_sum"] == 1


def test_build_resume_inject_config_default_mode_from_env(monkeypatch):
    monkeypatch.setenv("NLS_RESUME_MAMBA_DELTA_SUM", "1")

    class _Block:
        kv_path = "/x.nls"
        num_tokens = 10
        rope_start = 22
        turn_index = 1
        role = "turn"
        session_id = "s"
        ring_type = "general"
        meta_score = 0.0

    cfg = build_resume_inject_config([_Block()])  # type: ignore[list-item]
    assert cfg is not None
    assert cfg["mamba_delta_sum"] == 1
