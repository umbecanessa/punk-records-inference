"""Shared planting, recall, and scoring for production-length tier-1 sweeps."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
import uuid
from pathlib import Path

import requests

_TIER1 = Path(__file__).resolve().parent
_BENCH = _TIER1.parent
_REPO = _BENCH.parent
for path in (_REPO, _BENCH / "opencode", _TIER1):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nls_kvp_helpers import (  # noqa: E402
    _token_count_cached,
    api_root_from_chat_url,
    enrich_kv_params,
    render_system_turn_like_qwen,
)

from chain_helpers import fetch_user_memories, select_chain_latest, TURN_ROLES  # noqa: E402
from recall_helpers import score_recall_any, normalize_recall_text  # noqa: E402

from pri.text_quality import is_garbled_response  # noqa: E402

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"

SYSTEM_PROMPT = (
    "You are a personal assistant with persistent memory. "
    "Answer from prior conversation context when available."
)

FACTS = [
    "My name is Marco and I live in Milan, Italy. I work as an architect.",
    "I have a golden retriever named Luna. She's 3 years old and loves swimming.",
    "Last weekend I went to Lake Como with my wife Sofia. We stayed at Hotel Bellagio.",
]

NOISE_BANK = [
    "What's the capital of Norway and what is it known for?",
    "Explain the difference between TCP and UDP in networking.",
    "What are the main ingredients in a classic Margherita pizza?",
    "How does photosynthesis work in broadleaf trees?",
    "Summarize the plot of The Odyssey in three sentences.",
    "What is the speed of light in a vacuum?",
    "Describe how CRISPR gene editing works at a high level.",
    "What caused the fall of the Roman Western Empire?",
    "How do you tune a piano by ear?",
    "What is the Higgs boson and why does it matter?",
    "Explain blockchain consensus without jargon.",
    "What are best practices for sourdough starter maintenance?",
    "How does a heat pump work in winter?",
    "What is the difference between Baroque and Rococo art?",
    "Describe the water cycle for a ten-year-old.",
]

RECALL = [
    ("What's my name and where do I live?", ["Marco", "Milan"]),
    ("What's my dog's name?", ["Luna"]),
    ("Where did I go last weekend and who with?", ["Lake Como", "Sofia"]),
    ("What hotel did I stay at?", ["Bellagio"]),
    ("What do I do for work?", ["architect"]),
]


def strip_think(text: str) -> str:
    if "<think>" in text.lower():
        return re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
    return text


def block_hash(session_id: str) -> str:
    return hashlib.sha256(session_id.encode()).hexdigest()[:16]


def wait_for_vllm(api_root: str, *, timeout_s: int = 180) -> bool:
    deadline = time.time() + timeout_s
    url = f"{api_root.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def resolve_model(base_url: str) -> str:
    response = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=15)
    response.raise_for_status()
    models = response.json().get("data") or []
    if not models:
        raise RuntimeError("no models from /v1/models")
    return models[0]["id"]


def assistant_for_history(raw: str) -> tuple[str, bool]:
    cleaned = strip_think(raw or "").strip()
    if is_garbled_response(cleaned):
        return "", True
    return cleaned, False


def score_recall_clean(content: str, expected_keywords: list[str]) -> dict:
    clean = strip_think(content or "")
    garbled = is_garbled_response(clean)
    base = score_recall_any(clean, expected_keywords)
    return {
        **base,
        "garbled": garbled,
        "pass_clean": bool(base.get("pass")) and not garbled,
    }


def noise_prompt(index: int) -> str:
    if index < len(NOISE_BANK):
        return NOISE_BANK[index]
    topics = (
        "volcanology", "number theory", "Renaissance art", "ocean currents",
        "compiler design", "mycology", "orbital mechanics", "linguistics",
    )
    topic = topics[index % len(topics)]
    return (
        f"Teach me something substantive about {topic}. "
        f"Use two short paragraphs with concrete details (session turn {index + 1})."
    )


def inline_messages(turns: list[tuple[str, str]], question: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_text, asst_text in turns:
        messages.append({"role": "user", "content": user_text})
        if asst_text:
            messages.append({"role": "assistant", "content": asst_text})
    messages.append({"role": "user", "content": question})
    return messages


def estimate_chat_tokens(
    api_root: str,
    model: str,
    turns: list[tuple[str, str]],
    question: str = RECALL[0][0],
) -> int:
    parts = [render_system_turn_like_qwen(SYSTEM_PROMPT)]
    for user_text, asst_text in turns:
        parts.append(f"{_IM_START}user\n{user_text}{_IM_END}\n")
        if asst_text:
            parts.append(f"{_IM_START}assistant\n{asst_text}{_IM_END}\n")
    parts.append(f"{_IM_START}user\n{question}{_IM_END}\n{_IM_START}assistant\n")
    return _token_count_cached(api_root, model, "".join(parts))


def estimate_user_only_tokens(
    api_root: str,
    model: str,
    turns: list[tuple[str, str]],
    question: str = RECALL[0][0],
) -> int:
    parts = [render_system_turn_like_qwen(SYSTEM_PROMPT)]
    for user_text, _asst in turns:
        parts.append(f"{_IM_START}user\n{user_text}{_IM_END}\n")
    parts.append(f"{_IM_START}user\n{question}{_IM_END}\n{_IM_START}assistant\n")
    return _token_count_cached(api_root, model, "".join(parts))


def chain_turn_stats(api_root: str, user_id: str, base_session: str) -> dict:
    memories = fetch_user_memories(api_root, user_id, include_kv=True)
    blocks = select_chain_latest(
        memories, base_session, k=10**9, max_tokens=0, roles=TURN_ROLES,
    )
    by_role: dict[str, int] = {}
    tokens = 0
    for block in blocks:
        role = block.get("role") or "user"
        by_role[role] = by_role.get(role, 0) + 1
        tokens += int(block.get("numTokens") or 0)
    return {"blocks": len(blocks), "tokens": tokens, "by_role": by_role}


def delete_session_captures(api_root: str, session_ids: list[str]) -> dict:
    """Remove poisoned .nls blocks from index (admin API)."""
    if not session_ids:
        return {"deleted": 0}
    response = requests.post(
        f"{api_root.rstrip('/')}/admin/memory/delete",
        json={"session_ids": session_ids},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def plant_turn(
    api: str,
    model: str,
    *,
    user_id: str,
    base_session: str,
    turn_index: int,
    prev_hash: str,
    user_msg: str,
    repetition_penalty: float = 1.15,
) -> tuple[str, str]:
    block_session = f"{base_session}_t{turn_index}_user"
    kv: dict[str, str] = {
        "memory_user": user_id,
        "memory_ring": "general",
        "memory_block_role": "user",
        "memory_base_session": base_session,
        "memory_session": block_session,
        "memory_turn_index": str(turn_index),
        "memory_text": user_msg,
    }
    if prev_hash:
        kv["memory_prev_hash"] = prev_hash
    if turn_index == 1:
        kv["memory_silo"] = "1"
    else:
        kv["memory_inject_mode"] = "resume"

    kv = enrich_kv_params(kv, SYSTEM_PROMPT, api_root=api_root_from_chat_url(api), model=model)

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
        "repetition_penalty": repetition_penalty,
        "kv_transfer_params": kv,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"pri_sweep_{user_id}_{turn_index}_{uuid.uuid4().hex[:8]}",
    }
    response = requests.post(api, json=body, timeout=180)
    response.raise_for_status()
    data = response.json()
    content = (data["choices"][0]["message"]["content"] or "").strip()
    return content, block_hash(block_session), block_session


def plant_turn_hygiene(
    api: str,
    model: str,
    api_root: str,
    *,
    user_id: str,
    base_session: str,
    turn_index: int,
    prev_hash: str,
    user_msg: str,
    repetition_penalty: float = 1.15,
    max_garbled_retries: int = 2,
) -> tuple[str, str, str, bool, int]:
    """Plant one turn; delete capture and retry when assistant output is garbled.

    Returns (assistant_text, new_prev_hash, block_session, was_garbled, deletes).
    On persistent garbled, returns empty assistant and does not advance prev_hash.
    """
    deletes = 0
    for attempt in range(max(1, max_garbled_retries)):
        content, new_hash, block_session = plant_turn(
            api,
            model,
            user_id=user_id,
            base_session=base_session,
            turn_index=turn_index,
            prev_hash=prev_hash,
            user_msg=user_msg,
            repetition_penalty=repetition_penalty,
        )
        asst, garbled = assistant_for_history(content)
        if not garbled:
            return asst, new_hash, block_session, False, deletes

        delete_session_captures(api_root, [block_session])
        deletes += 1
        if attempt + 1 < max_garbled_retries:
            time.sleep(0.3)

    return "", prev_hash, block_session, True, deletes


def send_recall_arm(
    api: str,
    model: str,
    *,
    arm: str,
    turns: list[tuple[str, str]],
    user_id: str,
    base_session: str,
    checkpoint: int,
    question: str,
    expected: list[str],
) -> dict:
    if arm == "text":
        messages = inline_messages(turns, question)
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    kv: dict[str, str] = {
        "memory_user": user_id,
        "memory_ring": "general",
        "memory_no_capture": "1",
    }
    if arm == "text":
        kv["memory_off"] = "1"
    elif arm == "arm_d":
        kv["memory_inject_mode"] = "resume_overflow"
        kv["memory_base_session"] = base_session
    else:
        kv["memory_inject_mode"] = "resume"
        kv["memory_base_session"] = base_session

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.0,
        "kv_transfer_params": kv,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"pri_turn_sweep_{arm}_cp{checkpoint}_{uuid.uuid4().hex[:8]}",
    }
    try:
        response = requests.post(api, json=body, timeout=300)
        if response.status_code >= 400:
            return {"pass_clean": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
        data = response.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()
        usage = data.get("usage") or {}
    except Exception as exc:
        return {"pass_clean": False, "error": str(exc)}

    return {
        **score_recall_clean(content, expected),
        "answer_preview": normalize_recall_text(content)[:160],
        "usage": usage,
    }
