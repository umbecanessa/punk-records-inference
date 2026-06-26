"""Round-trip tests for pri.format (.nls save/load)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from pri.format import load_nls, read_manifest, save_nls
from spec.validate import validate_nls_file


def test_nls_roundtrip_preserves_tensors():
    data = {
        "layer_0_k": torch.randn(32, 8, dtype=torch.bfloat16),
        "layer_0_v": torch.randn(32, 8, dtype=torch.bfloat16),
        "layer_3_mamba_ssm": torch.randn(16, 4, dtype=torch.bfloat16),
        "_meta_seq_len": torch.tensor([32]),
        "_meta_has_mamba": torch.tensor([1]),
    }
    extra = {
        "user_id": "test_user",
        "session_id": "sess_roundtrip",
        "role": "user",
        "rope_start": 10,
        "rope_end": 42,
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.nls"
        save_nls(data, path, extra_manifest=extra)
        assert path.stat().st_size > 64

        manifest = read_manifest(path)
        assert manifest is not None
        assert manifest["session_id"] == "sess_roundtrip"
        assert manifest["rope_start"] == 10
        assert "block_hash" in manifest

        assert validate_nls_file(path) == []

        loaded = load_nls(path)
        assert loaded["layer_0_k"].shape == data["layer_0_k"].shape
        assert torch.allclose(loaded["layer_0_k"].float(), data["layer_0_k"].float(), atol=0.02)


def test_read_manifest_without_full_load():
    data = {
        "layer_0_k": torch.zeros(4, 2),
        "_meta_seq_len": torch.tensor([4]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mini.nls"
        save_nls(data, path, extra_manifest={"session_id": "x"})
        manifest = read_manifest(path)
        assert manifest is not None
        assert manifest["seq_len"] == 4
