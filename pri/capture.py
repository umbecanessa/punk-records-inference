"""Turn capture mode — ``dual`` user/assistant blocks vs unified ``turn`` snapshots.

Controlled by ``NLS_CHAIN_CAPTURE_MODE`` (default ``turn`` for v0.1).

  - ``turn`` — one ``.nls`` per HTTP turn covering user + assistant decode (resume path).
  - ``dual`` — separate user and assistant captures per turn (legacy).

``default_resume_roles()`` returns which block roles ``pri.resume`` includes when
walking the chain. See ``docs/reference/env-vars.md``.
"""

from __future__ import annotations

import os

# ``dual`` (default) — separate user + assistant .nls blocks per HTTP turn.
# ``turn`` — one contiguous user+assistant snapshot per turn for resume inject.
CHAIN_CAPTURE_MODE = os.environ.get("NLS_CHAIN_CAPTURE_MODE", "turn").strip().lower()


def is_turn_capture_mode() -> bool:
    return CHAIN_CAPTURE_MODE == "turn"


def default_resume_roles() -> frozenset[str]:
    if is_turn_capture_mode():
        raw = os.environ.get("NLS_RESUME_ROLES", "turn,tool")
    else:
        raw = os.environ.get("NLS_RESUME_ROLES", "user,tool")
    return frozenset(r.strip() for r in raw.split(",") if r.strip())


def turn_capture_prefill_slice_start(
    *,
    capture_start: int,
    prefill_end: int,
    resume_stripped_sys: int,
) -> tuple[int, int]:
    """Return ``(kv_slice_start, manifest_rope_start)`` for turn-mode readback.

    Resume inject strips the live system prefix when the phantom pack already
    includes a system block (``memory_capture_start`` / ``memory_sys_prompt_hash``).
    KV slicing must then begin at 0 in post-phantom coordinates — applying
    ``capture_start`` again would amputate user content (v13 N7 poison geometry).
    """
    manifest_rope_start = max(0, capture_start)
    if resume_stripped_sys > 0 and manifest_rope_start > 0:
        return 0, manifest_rope_start
    return max(0, min(manifest_rope_start, prefill_end)), manifest_rope_start


def is_prefilled_capture_kvp(kvp: dict) -> bool:
    raw = kvp.get("memory_prefilled_capture", "")
    return str(raw).strip().lower() in ("1", "true", "yes")


def resume_turn_requires_inject(inject_mode: str, turn_index: int) -> bool:
    """Resume chain turns after T1 must inject phantom KV before capture."""
    mode = str(inject_mode or "").strip().lower()
    return mode in ("resume", "resume_overflow") and int(turn_index) > 1
