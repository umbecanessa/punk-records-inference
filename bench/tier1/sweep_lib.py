"""Shared planting, recall, and scoring for production-length tier-1 sweeps."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import requests

_TIER1 = Path(__file__).resolve().parent
_BENCH = _TIER1.parent
_REPO = _BENCH.parent
for path in (_REPO, _BENCH / "opencode", _TIER1):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nls_kvp_helpers import (  # noqa: E402
    api_root_from_chat_url,
    enrich_kv_params,
    token_count_inline_history,
)

from chain_helpers import fetch_user_memories, select_chain_latest, TURN_ROLES  # noqa: E402
from recall_helpers import score_recall_any, normalize_recall_text  # noqa: E402

from pri.text_quality import is_garbled_response  # noqa: E402

SYSTEM_PROMPT = (
    "You are a personal assistant with persistent memory. "
    "Answer from prior conversation context when available."
)

# vLLM rejects consecutive user roles; garbled plants may leave assistant empty.
EMPTY_ASSISTANT_PLACEHOLDER = "Acknowledged."

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

# Substitute turn when the real prompt/decode is garbled after retries — keeps turn_index
# contiguous without poisoning the chain (no skip, no empty assistant).
NEUTRAL_TURN_USER = "I'm having trouble generating a response."

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
        messages.append({
            "role": "assistant",
            "content": asst_text or EMPTY_ASSISTANT_PLACEHOLDER,
        })
    messages.append({"role": "user", "content": question})
    return messages


def estimate_chat_tokens(
    api_root: str,
    model: str,
    turns: list[tuple[str, str]],
    question: str = RECALL[0][0],
) -> int:
    return token_count_inline_history(
        api_root,
        model,
        system_prompt=SYSTEM_PROMPT,
        turns=turns,
        question=question,
        include_assistant=True,
        empty_assistant_placeholder=EMPTY_ASSISTANT_PLACEHOLDER,
    )


def estimate_user_only_tokens(
    api_root: str,
    model: str,
    turns: list[tuple[str, str]],
    question: str = RECALL[0][0],
) -> int:
    return token_count_inline_history(
        api_root,
        model,
        system_prompt=SYSTEM_PROMPT,
        turns=turns,
        question=question,
        include_assistant=False,
    )


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


@dataclass(frozen=True)
class PlantHygieneResult:
    user_text: str
    assistant_text: str
    new_prev_hash: str
    block_session: str
    still_garbled: bool
    deletes: int
    neutral_fallback: bool
    original_user_msg: str
    garbled_probe_attempts: int


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
    max_tokens: int = 300,
    capture: bool = True,
) -> tuple[str, str, str]:
    """Send one chain plant turn. When ``capture=False``, sets ``memory_no_capture=1``."""
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
    if not capture:
        kv["memory_no_capture"] = "1"

    kv = enrich_kv_params(kv, SYSTEM_PROMPT, api_root=api_root_from_chat_url(api), model=model)

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
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
    new_hash = block_hash(block_session) if capture else prev_hash
    return content, new_hash, block_session


def _plant_neutral_turn(
    api: str,
    model: str,
    api_root: str,
    *,
    user_id: str,
    base_session: str,
    turn_index: int,
    prev_hash: str,
    deletes: int,
) -> PlantHygieneResult:
    """Capture a short neutral turn so turn_index stays contiguous."""
    block_session = f"{base_session}_t{turn_index}_user"
    for attempt in range(4):
        cap_content, new_hash, block_session = plant_turn(
            api,
            model,
            user_id=user_id,
            base_session=base_session,
            turn_index=turn_index,
            prev_hash=prev_hash,
            user_msg=NEUTRAL_TURN_USER,
            repetition_penalty=1.0,
            max_tokens=64,
            capture=True,
        )
        cap_asst, cap_garbled = assistant_for_history(cap_content)
        if not cap_garbled and cap_asst:
            return PlantHygieneResult(
                user_text=NEUTRAL_TURN_USER,
                assistant_text=cap_asst,
                new_prev_hash=new_hash,
                block_session=block_session,
                still_garbled=False,
                deletes=deletes,
                neutral_fallback=True,
                original_user_msg="",
                garbled_probe_attempts=0,
            )
        delete_session_captures(api_root, [block_session])
        deletes += 1
        time.sleep(0.3)

    return PlantHygieneResult(
        user_text=NEUTRAL_TURN_USER,
        assistant_text="",
        new_prev_hash=prev_hash,
        block_session=block_session,
        still_garbled=True,
        deletes=deletes,
        neutral_fallback=True,
        original_user_msg="",
        garbled_probe_attempts=0,
    )


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
) -> PlantHygieneResult:
    """Probe with no capture, then capture only when assistant output is clean.

    After ``max_garbled_retries`` failed probes/captures on ``user_msg``, substitutes
    a neutral user+assistant turn (``NEUTRAL_TURN_USER``) at the same turn_index
    so the inject chain stays contiguous without poisoned decode.
    """
    deletes = 0
    block_session = f"{base_session}_t{turn_index}_user"
    attempts = max(1, max_garbled_retries)
    garbled_probe_attempts = 0

    for attempt in range(attempts):
        probe_content, _, _ = plant_turn(
            api,
            model,
            user_id=user_id,
            base_session=base_session,
            turn_index=turn_index,
            prev_hash=prev_hash,
            user_msg=user_msg,
            repetition_penalty=repetition_penalty,
            capture=False,
        )
        probe_asst, probe_garbled = assistant_for_history(probe_content)
        if probe_garbled:
            garbled_probe_attempts += 1
            if attempt + 1 < attempts:
                time.sleep(0.3)
            continue

        cap_content, new_hash, block_session = plant_turn(
            api,
            model,
            user_id=user_id,
            base_session=base_session,
            turn_index=turn_index,
            prev_hash=prev_hash,
            user_msg=user_msg,
            repetition_penalty=repetition_penalty,
            capture=True,
        )
        cap_asst, cap_garbled = assistant_for_history(cap_content)
        if not cap_garbled:
            return PlantHygieneResult(
                user_text=user_msg,
                assistant_text=cap_asst,
                new_prev_hash=new_hash,
                block_session=block_session,
                still_garbled=False,
                deletes=deletes,
                neutral_fallback=False,
                original_user_msg=user_msg,
                garbled_probe_attempts=garbled_probe_attempts,
            )

        delete_session_captures(api_root, [block_session])
        deletes += 1
        garbled_probe_attempts += 1
        if attempt + 1 < attempts:
            time.sleep(0.3)

    neutral = _plant_neutral_turn(
        api,
        model,
        api_root,
        user_id=user_id,
        base_session=base_session,
        turn_index=turn_index,
        prev_hash=prev_hash,
        deletes=deletes,
    )
    return PlantHygieneResult(
        user_text=neutral.user_text,
        assistant_text=neutral.assistant_text,
        new_prev_hash=neutral.new_prev_hash,
        block_session=neutral.block_session,
        still_garbled=neutral.still_garbled,
        deletes=neutral.deletes,
        neutral_fallback=True,
        original_user_msg=user_msg,
        garbled_probe_attempts=garbled_probe_attempts,
    )


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
