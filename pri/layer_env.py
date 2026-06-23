"""Parse comma-separated layer index lists from environment variables."""

from __future__ import annotations

import os


def parse_layer_list_env(env_key: str, *, fallback: list[int]) -> list[int]:
    raw = (os.environ.get(env_key) or "").strip()
    if not raw:
        return list(fallback)
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or list(fallback)
