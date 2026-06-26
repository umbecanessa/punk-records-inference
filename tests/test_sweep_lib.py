"""Tests for sweep_lib plant_turn capture flag."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from bench.tier1 import sweep_lib


def test_plant_turn_no_capture_sets_memory_no_capture():
    captured: dict = {}

    def fake_post(_url, json=None, timeout=180):
        captured["body"] = json
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        return response

    with patch.object(sweep_lib.requests, "post", side_effect=fake_post):
        content, new_hash, session = sweep_lib.plant_turn(
            "http://127.0.0.1:8000/v1/chat/completions",
            "/model",
            user_id="u1",
            base_session="chain_x",
            turn_index=2,
            prev_hash="abc",
            user_msg="hello",
            capture=False,
        )

    assert content == "ok"
    assert new_hash == "abc"
    assert session == "chain_x_t2_user"
    kvp = captured["body"]["kv_transfer_params"]
    assert kvp["memory_no_capture"] == "1"
    assert kvp["memory_inject_mode"] == "resume"


def test_plant_turn_capture_advances_hash():
    with patch.object(sweep_lib.requests, "post") as fake_post:
        fake_post.return_value = MagicMock(
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
        )
        _, new_hash, session = sweep_lib.plant_turn(
            "http://127.0.0.1:8000/v1/chat/completions",
            "/model",
            user_id="u1",
            base_session="chain_x",
            turn_index=2,
            prev_hash="abc",
            user_msg="hello",
            capture=True,
        )

    assert new_hash == sweep_lib.block_hash(session)
    assert new_hash != "abc"


def test_plant_turn_hygiene_neutral_fallback_after_garbled_probes():
    calls: list[dict] = []

    def fake_post(_url, json=None, timeout=180):
        calls.append(json)
        user_msg = json["messages"][-1]["content"]
        capture = json["kv_transfer_params"].get("memory_no_capture") != "1"
        if user_msg == sweep_lib.NEUTRAL_TURN_USER and capture:
            content = "No problem — let me know if you want to try again."
        else:
            content = "evere vue 勻nton 中文垃圾 token soup " * 4
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": content}}],
        }
        return response

    with patch.object(sweep_lib.requests, "post", side_effect=fake_post):
        with patch.object(sweep_lib, "delete_session_captures", return_value={"deleted": 0}):
            result = sweep_lib.plant_turn_hygiene(
                "http://127.0.0.1:8000/v1/chat/completions",
                "/model",
                "http://127.0.0.1:8000",
                user_id="u1",
                base_session="chain_x",
                turn_index=5,
                prev_hash="abc",
                user_msg="Teach me something substantive about mycology.",
                max_garbled_retries=2,
            )

    assert result.neutral_fallback is True
    assert result.still_garbled is False
    assert result.user_text == sweep_lib.NEUTRAL_TURN_USER
    assert result.original_user_msg.startswith("Teach me")
    assert result.assistant_text
    assert result.new_prev_hash != "abc"
    assert any(
        c.get("messages")
        and c["messages"][-1]["content"] == sweep_lib.NEUTRAL_TURN_USER
        and c["kv_transfer_params"].get("memory_no_capture") != "1"
        for c in calls
    )
