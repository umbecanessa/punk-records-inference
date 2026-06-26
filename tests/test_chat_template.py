"""Tests for model-native chat template boundaries."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pri.chat_template import compute_capture_start, token_count_messages


def test_compute_capture_start_messages_delta():
    def fake_post(url, json=None, timeout=None):
        del timeout
        assert url.endswith("/tokenize")
        messages = json.get("messages") or []
        has_system = any(m.get("role") == "system" for m in messages)
        response = MagicMock()
        response.raise_for_status = MagicMock()
        if has_system:
            response.json.return_value = {"count": 28}
        else:
            response.json.return_value = {"count": 10}
        return response

    with patch("pri.chat_template.requests.post", side_effect=fake_post):
        assert compute_capture_start(
            "You are helpful.",
            api_root="http://127.0.0.1:8000",
            model="/model",
        ) == 18


def test_token_count_messages_empty():
    assert token_count_messages("http://x", "m", []) == 0
