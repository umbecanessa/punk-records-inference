"""Tests for resume inject system-block prepend gating."""

from __future__ import annotations

from pathlib import Path

import pytest

from pri.resume import (
    resume_prepend_sys_block_enabled,
    try_resume_config,
)


class _Mem:
    def __init__(
        self,
        *,
        role: str,
        kv_path: Path,
        turn: int = 1,
        sys_hash: str = "",
    ) -> None:
        self.user_id = "u1"
        self.base_session_id = "chain_a"
        self.role = role
        self.kv_path = str(kv_path)
        self.num_tokens = 10
        self.rope_start = 22
        self.turn_index = turn
        self.session_id = f"s_{role}_{turn}"
        self.id = self.session_id
        self.ring_type = "general"
        self.meta_score = 0.0
        self.timestamp = float(turn)
        self.sys_prompt_hash = sys_hash


class _Store:
    def __init__(self, memories: list[_Mem]) -> None:
        self._memories = memories


def test_resume_prepend_sys_block_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NLS_RESUME_PREPEND_SYS_BLOCK", raising=False)
    assert resume_prepend_sys_block_enabled() is False


def test_try_resume_config_skips_sys_prepend_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NLS_RESUME_PREPEND_SYS_BLOCK", "0")
    turn_kv = tmp_path / "turn.nls"
    sys_kv = tmp_path / "sys.nls"
    turn_kv.write_bytes(b"\x00")
    sys_kv.write_bytes(b"\x00")
    store = _Store(
        [
            _Mem(role="turn", kv_path=turn_kv, turn=1),
            _Mem(role="system", kv_path=sys_kv, turn=-1, sys_hash="abc123"),
        ],
    )
    cfg = try_resume_config(
        store,
        "u1",
        "chain_a",
        sys_prompt_hash="abc123",
    )
    assert cfg is not None
    assert len(cfg["snapshots"]) == 1


def test_try_resume_config_prepends_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NLS_RESUME_PREPEND_SYS_BLOCK", "1")
    turn_kv = tmp_path / "turn.nls"
    sys_kv = tmp_path / "sys.nls"
    turn_kv.write_bytes(b"\x00")
    sys_kv.write_bytes(b"\x00")
    store = _Store(
        [
            _Mem(role="turn", kv_path=turn_kv, turn=1),
            _Mem(role="system", kv_path=sys_kv, turn=-1, sys_hash="abc123"),
        ],
    )
    cfg = try_resume_config(
        store,
        "u1",
        "chain_a",
        sys_prompt_hash="abc123",
    )
    assert cfg is not None
    assert len(cfg["snapshots"]) == 2
