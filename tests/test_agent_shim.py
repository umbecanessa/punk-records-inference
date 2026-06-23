"""Smoke tests for agent middleware helpers (no vLLM required)."""

from __future__ import annotations

from pri.middleware import agent_shim as shim


def test_strip_agent_messages_keeps_current_turn():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
    ]
    stripped = shim._strip_agent_messages_for_resume(messages)
    roles = [m["role"] for m in stripped]
    assert roles == ["system", "user"]
    assert stripped[-1]["content"] == "turn 2"


def test_is_agent_mode_detects_tools():
    assert shim._is_agent_mode({"messages": [], "tools": [{"type": "function"}]})
    assert not shim._is_agent_mode({"messages": [{"role": "user", "content": "hi"}]})


def test_sys_prompt_hash_stable():
    h1 = shim._sys_prompt_hash("hello")
    h2 = shim._sys_prompt_hash("hello")
    assert h1 == h2
    assert len(h1) == 16


def test_capture_mode_env_default():
    from pri.capture import is_turn_capture_mode

    assert is_turn_capture_mode() in (True, False)


def test_ensure_kv_transfer_params_mutates_body():
    body: dict = {"messages": []}
    kvp = shim._ensure_kv_transfer_params(body)
    kvp["memory_session"] = "chain_test_t1_user"
    assert body["kv_transfer_params"]["memory_session"] == "chain_test_t1_user"
