"""Smoke tests for capture/resume chain helpers (no GPU)."""

from __future__ import annotations

from pri.capture import default_resume_roles, is_turn_capture_mode
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
