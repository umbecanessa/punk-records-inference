"""Tests for chain pack phantom (resume RoPE replay semantics)."""

from __future__ import annotations

from pri.inject_geometry_audit import audit_rope_pack_plan, summarize_geometry_audit
from pri.resume import chain_pack_phantom_before_turn, collect_chain_blocks


def test_chain_pack_phantom_before_turn_sums_prior_blocks(tmp_path):
    kv_path = tmp_path / "block.nls"
    kv_path.write_bytes(b"\x00")

    class _Mem:
        def __init__(self, *, turn: int, tokens: int, sid: str) -> None:
            self.user_id = "user1"
            self.base_session_id = "chain_x"
            self.role = "turn"
            self.kv_path = str(kv_path)
            self.num_tokens = tokens
            self.rope_start = 22
            self.turn_index = turn
            self.session_id = sid
            self.id = sid
            self.ring_type = "general"
            self.meta_score = 0.0
            self.timestamp = float(turn)

    class _Store:
        _memories = [
            _Mem(turn=1, tokens=100, sid="chain_x_t1"),
            _Mem(turn=2, tokens=50, sid="chain_x_t2"),
            _Mem(turn=3, tokens=75, sid="chain_x_t3"),
        ]

    assert chain_pack_phantom_before_turn(_Store(), "user1", "chain_x", 1) == 0
    assert chain_pack_phantom_before_turn(_Store(), "user1", "chain_x", 2) == 100
    assert chain_pack_phantom_before_turn(_Store(), "user1", "chain_x", 4) == 225


def test_audit_turn_blocks_use_pack_offset_not_stale_manifest_phantom():
    """Turn role: pack cumulative offset wins over wrong capture_num_phantom."""
    snapshots = [
        {
            "path": "/data/pri/snapshot/captures/a.nls",
            "num_tokens": 100,
            "strip_prefix": 0,
            "rope_start": 22,
        },
        {
            "path": "/data/pri/snapshot/captures/b.nls",
            "num_tokens": 84,
            "strip_prefix": 0,
            "rope_start": 22,
        },
    ]

    def _fake_manifest(path: str) -> dict | None:
        if path.endswith("a.nls"):
            return {
                "role": "turn",
                "turn_index": 1,
                "rope_start": 22,
                "rope_end": 122,
                "capture_num_phantom": 0,
            }
        if path.endswith("b.nls"):
            return {
                "role": "turn",
                "turn_index": 2,
                "rope_start": 22,
                "rope_end": 106,
                "capture_num_phantom": 16049,
            }
        return None

    import pri.inject_geometry_audit as audit_mod

    original = audit_mod._read_manifest
    audit_mod._read_manifest = lambda p: _fake_manifest(p)  # type: ignore[assignment]
    try:
        rows = audit_rope_pack_plan(snapshots, resume_mode=True)
        assert rows[0].rope_delta == -22
        assert rows[1].rope_delta == -22
        summary = summarize_geometry_audit(
            snapshots, resume_mode=True, mamba_delta_sum=1,
        )
        assert summary["verdict"] == "pass"
        assert len({r.rope_delta for r in rows}) == 1
    finally:
        audit_mod._read_manifest = original


def test_resume_system_block_excluded_from_turn_delta_uniformity():
    """Prepended system KV (delta 0) must not fail turn-chain uniformity (-22)."""
    snapshots = [
        {
            "path": "/data/pri/snapshot/captures/sys.nls",
            "num_tokens": 22,
            "strip_prefix": 0,
            "rope_start": 0,
            "turn_index": -1,
            "role": "system",
        },
        {
            "path": "/data/pri/snapshot/captures/a.nls",
            "num_tokens": 100,
            "strip_prefix": 0,
            "rope_start": 22,
            "turn_index": 1,
            "role": "turn",
        },
    ]

    def _fake_manifest(path: str) -> dict | None:
        if path.endswith("sys.nls"):
            return {
                "role": "system",
                "turn_index": -1,
                "rope_start": 0,
                "rope_end": 22,
                "capture_num_phantom": 0,
            }
        if path.endswith("a.nls"):
            return {
                "role": "turn",
                "turn_index": 1,
                "rope_start": 22,
                "rope_end": 122,
                "capture_num_phantom": 0,
            }
        return None

    import pri.inject_geometry_audit as audit_mod

    original = audit_mod._read_manifest
    audit_mod._read_manifest = lambda p: _fake_manifest(p)  # type: ignore[assignment]
    try:
        rows = audit_rope_pack_plan(snapshots, resume_mode=True)
        assert rows[0].role == "system"
        assert rows[0].rope_delta == 0
        assert rows[1].rope_delta == -22
        summary = summarize_geometry_audit(
            snapshots, resume_mode=True, mamba_delta_sum=1,
        )
        assert summary["verdict"] == "pass"
        assert "inconsistent resume RoPE deltas" not in " ".join(summary["findings"])
    finally:
        audit_mod._read_manifest = original


def test_collect_chain_blocks_dedupes_by_turn(tmp_path):
    kv_path = tmp_path / "block.nls"
    kv_path.write_bytes(b"\x00")

    class _Mem:
        def __init__(self, *, turn: int, ts: float, sid: str, tokens: int) -> None:
            self.user_id = "u"
            self.base_session_id = "base"
            self.role = "turn"
            self.kv_path = str(kv_path)
            self.num_tokens = tokens
            self.rope_start = 22
            self.turn_index = turn
            self.session_id = sid
            self.id = sid
            self.ring_type = "general"
            self.meta_score = 0.0
            self.timestamp = ts

    class _Store:
        _memories = [
            _Mem(turn=1, ts=1.0, sid="t1_old", tokens=10),
            _Mem(turn=1, ts=2.0, sid="t1_new", tokens=20),
        ]

    blocks = collect_chain_blocks(_Store(), "u", "base")
    assert len(blocks) == 1
    assert blocks[0].num_tokens == 20
    assert chain_pack_phantom_before_turn(_Store(), "u", "base", 2) == 20
