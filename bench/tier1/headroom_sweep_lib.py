"""Plant turns with Headroom compression layered on PRI chain capture."""

from __future__ import annotations

import os
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

from headroom_helpers import (  # noqa: E402
    agent_noise_user_message,
    compress_assistant_for_capture,
    compress_user_payload,
    headroom_available,
)
from nls_kvp_helpers import (
    api_root_from_chat_url,
    enrich_prefilled_capture_kv_params,
)
from sweep_lib import (  # noqa: E402
    PlantHygieneResult,
    SYSTEM_PROMPT,
    apply_resume_inject_caps,
    assistant_for_history,
    block_hash,
    default_chain_inject_mode,
    delete_session_captures,
    plant_turn,
    plant_turn_hygiene,
    plant_neutral_required,
)


def plant_turn_headroom(
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
    compress_user: bool = False,
    compress_assistant: bool = True,
    capture_max_tokens: int = 8,
) -> PlantHygieneResult:
    """Probe → Headroom compress → capture with prefilled assistant when possible."""
    debug = os.environ.get("HEADROOM_DEBUG") == "1"

    def _dbg(msg: str) -> None:
        if debug:
            print(f"[headroom plant t{turn_index}] {msg}", flush=True)
    if not headroom_available():
        raise RuntimeError("headroom not installed — use /home/wasnaga/headroom-venv/bin/python")

    user_phases: list[tuple[str, str, dict]] = []
    if compress_user:
        compressed, stats = compress_user_payload(user_msg, model="gpt-4o")
        user_phases.append(("compressed", compressed, stats))
        if compressed.strip() != user_msg.strip():
            user_phases.append(("raw", user_msg, {}))
    else:
        user_phases.append(("raw", user_msg, {}))

    deletes = 0
    attempts = max(1, max_garbled_retries)
    garbled_probe_attempts = 0
    compressed_probe_failures = 0
    headroom_stats: list[dict] = []
    tokens_saved = 0
    if compress_user and user_phases and user_phases[0][0] == "compressed":
        tokens_saved = int(user_phases[0][2].get("tokens_saved") or 0)

    def _success(
        *,
        plant_user: str,
        history_asst: str,
        new_hash: str,
        block_session: str,
        phase_label: str,
    ) -> PlantHygieneResult:
        if compress_user:
            print(
                f"    [headroom user] N{turn_index} captured={phase_label} "
                f"saved={tokens_saved}tok compressed_probe_fails={compressed_probe_failures}",
                flush=True,
            )
        return PlantHygieneResult(
            user_text=plant_user,
            assistant_text=history_asst,
            new_prev_hash=new_hash,
            block_session=block_session,
            still_garbled=False,
            deletes=deletes,
            neutral_fallback=False,
            original_user_msg=user_msg,
            garbled_probe_attempts=garbled_probe_attempts,
            headroom_user_phase=phase_label if compress_user else "n/a",
            headroom_user_tokens_saved=tokens_saved if compress_user else 0,
            headroom_compressed_probe_failures=compressed_probe_failures if compress_user else 0,
        )

    for phase_label, plant_user, user_stats in user_phases:
        _dbg(f"user phase={phase_label} len={len(plant_user)}")
        if user_stats:
            headroom_stats.append({"stage": f"user_{phase_label}", **user_stats})
        block_session = f"{base_session}_t{turn_index}_user"
        phase_attempts = 2 if phase_label == "compressed" and len(user_phases) > 1 else attempts

        for attempt in range(phase_attempts):
            probe_content, _, _ = plant_turn(
                api,
                model,
                user_id=user_id,
                base_session=base_session,
                turn_index=turn_index,
                prev_hash=prev_hash,
                user_msg=plant_user,
                repetition_penalty=repetition_penalty,
                capture=False,
            )
            probe_asst, probe_garbled = assistant_for_history(probe_content)
            _dbg(
                f"phase={phase_label} attempt {attempt + 1}/{phase_attempts} "
                f"probe_garbled={probe_garbled} len={len(probe_asst)} "
                f"raw_preview={(probe_content or '')[:80]!r}",
            )
            if probe_garbled:
                garbled_probe_attempts += 1
                if phase_label == "compressed":
                    compressed_probe_failures += 1
                if attempt + 1 < phase_attempts:
                    time.sleep(1.0 if phase_label == "compressed" else 0.3)
                continue

            capture_asst = probe_asst
            if compress_assistant and probe_asst:
                capture_asst, asst_stats = compress_assistant_for_capture(
                    SYSTEM_PROMPT, plant_user, probe_asst, model="gpt-4o",
                )
                headroom_stats.append({"stage": "assistant", **asst_stats})

            cap_content, new_hash, block_session = _capture_prefilled_assistant(
                api,
                model,
                api_root=api_root,
                user_id=user_id,
                base_session=base_session,
                turn_index=turn_index,
                prev_hash=prev_hash,
                user_msg=plant_user,
                assistant_text=capture_asst,
                max_tokens=capture_max_tokens,
            )
            cap_asst, cap_garbled = assistant_for_history(cap_content)
            _dbg(
                f"capture capture_asst_len={len(capture_asst or '')} "
                f"cap_garbled={cap_garbled} cap_tail_len={len(cap_asst)}",
            )
            history_asst = capture_asst or cap_asst or probe_asst
            if capture_asst and not probe_garbled and not cap_garbled:
                return _success(
                    plant_user=plant_user,
                    history_asst=history_asst,
                    new_hash=new_hash,
                    block_session=block_session,
                    phase_label=phase_label,
                )
            if cap_garbled and not probe_garbled:
                _dbg(
                    f"probe clean but capture garbled — reject commit "
                    f"(cap_tail_len={len(cap_asst or '')})",
                )

            delete_session_captures(api_root, [block_session])
            deletes += 1
            garbled_probe_attempts += 1
            if attempt + 1 < phase_attempts:
                time.sleep(0.3)

    block_session = f"{base_session}_t{turn_index}_user"

    if compress_assistant:
        plain = plant_turn_hygiene(
            api,
            model,
            api_root,
            user_id=user_id,
            base_session=base_session,
            turn_index=turn_index,
            prev_hash=prev_hash,
            user_msg=user_msg,
            max_garbled_retries=min(2, attempts),
        )
        _dbg(f"plain hygiene still_garbled={plain.still_garbled} neutral={plain.neutral_fallback}")
        if not plain.still_garbled:
            phase = "plain" if not plain.neutral_fallback else "neutral"
            if compress_user:
                print(
                    f"    [headroom user] N{turn_index} captured={phase} "
                    f"(assistant compress skipped) "
                    f"compressed_probe_fails={compressed_probe_failures}",
                    flush=True,
                )
            return PlantHygieneResult(
                user_text=plain.user_text,
                assistant_text=plain.assistant_text,
                new_prev_hash=plain.new_prev_hash,
                block_session=plain.block_session,
                still_garbled=False,
                deletes=deletes + plain.deletes,
                neutral_fallback=plain.neutral_fallback,
                original_user_msg=user_msg,
                garbled_probe_attempts=garbled_probe_attempts + plain.garbled_probe_attempts,
                headroom_user_phase=(
                    phase if compress_user else "n/a"
                ),
                headroom_user_tokens_saved=0,
                headroom_compressed_probe_failures=compressed_probe_failures,
            )

    neutral = plant_neutral_required(
        api,
        model,
        api_root,
        user_id=user_id,
        base_session=base_session,
        turn_index=turn_index,
        prev_hash=prev_hash,
        original_user_msg=user_msg,
        deletes=deletes,
        garbled_probe_attempts=garbled_probe_attempts,
    )
    _dbg(f"neutral required still_garbled={neutral.still_garbled}")
    neutral_phase = "failed" if neutral.still_garbled else "neutral"
    if compress_user:
        print(
            f"    [headroom user] N{turn_index} captured={neutral_phase} "
            f"saved=0tok compressed_probe_fails={compressed_probe_failures}",
            flush=True,
        )
    return PlantHygieneResult(
        user_text=neutral.user_text,
        assistant_text=neutral.assistant_text,
        new_prev_hash=neutral.new_prev_hash,
        block_session=neutral.block_session,
        still_garbled=neutral.still_garbled,
        deletes=neutral.deletes,
        neutral_fallback=not neutral.still_garbled,
        original_user_msg=user_msg,
        garbled_probe_attempts=neutral.garbled_probe_attempts,
        headroom_user_phase=neutral_phase if compress_user else "n/a",
        headroom_user_tokens_saved=0,
        headroom_compressed_probe_failures=compressed_probe_failures,
    )


def _capture_prefilled_assistant(
    api: str,
    model: str,
    *,
    api_root: str,
    user_id: str,
    base_session: str,
    turn_index: int,
    prev_hash: str,
    user_msg: str,
    assistant_text: str,
    max_tokens: int,
) -> tuple[str, str, str]:
    """Capture turn KV with assistant prefilled (minimal decode tail)."""
    block_session = f"{base_session}_t{turn_index}_user"
    kv: dict[str, str] = {
        "memory_user": user_id,
        "memory_ring": "general",
        "memory_block_role": "user",
        "memory_base_session": base_session,
        "memory_session": block_session,
        "memory_turn_index": str(turn_index),
    }
    if prev_hash:
        kv["memory_prev_hash"] = prev_hash
    if turn_index == 1:
        kv["memory_silo"] = "1"
    else:
        kv["memory_inject_mode"] = default_chain_inject_mode()

    kv = enrich_prefilled_capture_kv_params(
        kv, SYSTEM_PROMPT, api_root=api_root, model=model,
    )
    kv = apply_resume_inject_caps(kv)
    kv["memory_text"] = user_msg

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "kv_transfer_params": kv,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"pri_hr_{user_id}_{turn_index}_{uuid.uuid4().hex[:8]}",
    }
    response = requests.post(api, json=body, timeout=180)
    response.raise_for_status()
    data = response.json()
    content = (data["choices"][0]["message"]["content"] or "").strip()
    new_hash = block_hash(block_session)
    return content, new_hash, block_session


__all__ = ["agent_noise_user_message", "plant_turn_headroom"]
