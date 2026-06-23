"""Chain capture mode: dual user/assistant blocks vs unified turn snapshots."""

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
