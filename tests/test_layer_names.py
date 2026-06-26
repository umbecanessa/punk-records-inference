"""Tests for pri.layer_names — vLLM module path → layer index."""

from __future__ import annotations

from pri.layer_names import classify_layer_names, extract_layer_index


def test_classic_vllm_self_attn() -> None:
    assert extract_layer_index("model.layers.12.self_attn") == 12
    assert extract_layer_index("layers.0.self_attn.k_proj") == 0


def test_transformers_backend_compact_attn() -> None:
    assert extract_layer_index("0.attn") == 0
    assert extract_layer_index("79.attn") == 79


def test_decoder_wrapper() -> None:
    assert extract_layer_index("decoder.layers.3.self_attn") == 3


def test_unmapped_returns_none() -> None:
    assert extract_layer_index("embed_tokens") is None
    assert extract_layer_index("") is None


def test_classify_layer_names() -> None:
    names = ["0.attn", "1.attn", "lm_head"]
    out = classify_layer_names(names)
    assert out["mapped"] == ["0.attn", "1.attn"]
    assert out["unmapped"] == ["lm_head"]
