"""Smoke tests for capture/resume chain helpers (no GPU)."""

from __future__ import annotations

from pri.capture import (
    default_resume_roles,
    is_turn_capture_mode,
    resume_turn_requires_inject,
    turn_capture_prefill_slice_start,
)
from pri.resume import collect_chain_blocks


def test_default_resume_roles_includes_turn_in_turn_mode():
    roles = default_resume_roles()
    if is_turn_capture_mode():
        assert "turn" in roles
    else:
        assert "user" in roles


def test_collect_chain_blocks_empty_store():
    class _EmptyStore:
        _memories = []

    blocks = collect_chain_blocks(_EmptyStore(), "user1", "chain_x")
    assert blocks == []


def test_collect_chain_blocks_keeps_latest_per_turn(tmp_path):
    kv_path = tmp_path / "block.nls"
    kv_path.write_bytes(b"\x00")

    class _Mem:
        def __init__(self, *, turn: int, ts: float, sid: str) -> None:
            self.user_id = "user1"
            self.base_session_id = "chain_x"
            self.role = "turn"
            self.kv_path = str(kv_path)
            self.num_tokens = 10
            self.rope_start = 0
            self.turn_index = turn
            self.session_id = sid
            self.id = sid
            self.ring_type = "general"
            self.meta_score = 0.0
            self.timestamp = ts

    class _Store:
        _memories = [
            _Mem(turn=1, ts=1.0, sid="chain_x_t1_user_old"),
            _Mem(turn=1, ts=2.0, sid="chain_x_t1_user_new"),
            _Mem(turn=2, ts=1.0, sid="chain_x_t2_user_old"),
            _Mem(turn=2, ts=3.0, sid="chain_x_t2_user_new"),
        ]

    blocks = collect_chain_blocks(_Store(), "user1", "chain_x")
    assert [b.session_id for b in blocks] == [
        "chain_x_t1_user_new",
        "chain_x_t2_user_new",
    ]


def test_turn_capture_prefill_slice_start_after_resume_strip():
    """Resume strip already removed system — do not double-slice at cap_start."""
    start, rope = turn_capture_prefill_slice_start(
        capture_start=22,
        prefill_end=2420,
        resume_stripped_sys=22,
    )
    assert start == 0
    assert rope == 22


def test_turn_capture_prefill_slice_start_cold_inline():
    start, rope = turn_capture_prefill_slice_start(
        capture_start=22,
        prefill_end=100,
        resume_stripped_sys=0,
    )
    assert start == 22
    assert rope == 22


def test_resume_turn_requires_inject_after_t1():
    assert resume_turn_requires_inject("resume", 2) is True
    assert resume_turn_requires_inject("resume_overflow", 7) is True
    assert resume_turn_requires_inject("resume", 1) is False
    assert resume_turn_requires_inject("swiss", 3) is False
