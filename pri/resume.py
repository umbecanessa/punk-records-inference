"""Chain resume — inject ordered ``base_session`` blocks without Swiss retrieval.

Collects blockchain-linked ``.nls`` blocks for a ``memory_base_session``, packs them
in turn order, and builds inject config for ``pri.connector``.

v0.1 default (``NLS_API_INJECT_MODE=resume_overflow``):

  - Swiss retrieval is skipped when the chain walk succeeds.
  - Attention K/V: packed in turn order with per-block RoPE re-rotation from each
    manifest ``rope_start`` to the cumulative pack offset.
  - Mamba: ``NLS_RESUME_MAMBA_DELTA_SUM`` (default 1) = genesis + Σ(block−genesis)
    across the chain; 3 = last block verbatim; 2 = genesis + last delta only.
  - System-prefix strip is disabled for every block in the pack.
  - Pass-2 compound capture is skipped for resume requests (connector-side).

Capture mode (``NLS_CHAIN_CAPTURE_MODE``):

  - ``turn`` — one contiguous user+assistant snapshot per HTTP turn (v0.1 default).
  - ``dual`` — separate user + assistant blocks per turn (legacy).

When no chain blocks exist, callers fall back to Swiss auto-retrieval.
See ``docs/getting-started/concepts.md``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pri.store import MemoryStore

from pri.capture import default_resume_roles

logger = logging.getLogger("nls_chain_resume")

INJECT_MODE = os.environ.get("NLS_INJECT_MODE", "swiss").strip().lower()
RESUME_ROLES = default_resume_roles()


def resume_prepend_sys_block_enabled() -> bool:
    """Whether resume inject prepends a stored system ``.nls`` block.

    Default off: the live HTTP request already carries ``system``; resume inject
    strips it via ``memory_capture_start``. Prepending a stored system block
    duplicates phantom geometry and breaks recall (classic chain proof).
    """
    raw = str(os.environ.get("NLS_RESUME_PREPEND_SYS_BLOCK", "0") or "").strip().lower()
    return raw in ("1", "true", "yes")
# user → tool → turn ordering within the same turn_index
_ROLE_ORDER = {"user": 0, "tool": 1, "turn": 1, "assistant": 2}


@dataclass(frozen=True)
class ChainBlock:
    kv_path: str
    num_tokens: int
    rope_start: int
    turn_index: int
    role: str
    session_id: str
    ring_type: str = "general"
    meta_score: float = 0.0


def inject_mode_from_kvp(kvp: dict) -> str:
    """Resolve inject mode: per-request kvp overrides env default."""
    raw = str(kvp.get("memory_inject_mode", "") or "").strip().lower()
    if raw:
        return raw
    return INJECT_MODE


def is_resume_mode(kvp: dict) -> bool:
    mode = inject_mode_from_kvp(kvp)
    return mode in ("resume", "resume_overflow")


def is_resume_overflow_mode(kvp: dict) -> bool:
    """Arm D: resume chain + Swiss retrieval augmentation for evicted/extra facts."""
    if inject_mode_from_kvp(kvp) == "resume_overflow":
        return True
    return str(kvp.get("memory_resume_swiss_overflow", "") or "").strip() in (
        "1", "true", "yes",
    )


def find_system_block(
    store: "MemoryStore",
    sys_prompt_hash: str,
) -> ChainBlock | None:
    """Return the content-addressed system KV block for resume prepend."""
    if not sys_prompt_hash:
        return None
    for mem in store._memories:
        if mem.role != "system":
            continue
        if str(getattr(mem, "sys_prompt_hash", "") or "") != sys_prompt_hash:
            continue
        if not mem.kv_path or mem.num_tokens <= 0:
            continue
        if not Path(mem.kv_path).exists():
            logger.warning(
                "Resume: system block file missing for hash=%s",
                sys_prompt_hash[:12],
            )
            continue
        rope_start = int(getattr(mem, "rope_start", 0) or 0)
        return ChainBlock(
            kv_path=mem.kv_path,
            num_tokens=int(mem.num_tokens),
            rope_start=rope_start,
            turn_index=-1,
            role="system",
            session_id=mem.session_id or mem.id,
            ring_type=mem.ring_type or "general",
            meta_score=float(getattr(mem, "meta_score", 0.0) or 0.0),
        )
    return None


def collect_chain_blocks(
    store: "MemoryStore",
    user_id: str,
    base_session_id: str,
    *,
    roles: frozenset[str] = RESUME_ROLES,
) -> list[ChainBlock]:
    if not base_session_id or not user_id:
        return []

    latest_by_slot: dict[tuple[int, str], tuple[float, ChainBlock]] = {}
    for mem in store._memories:
        if mem.user_id != user_id:
            continue
        if mem.base_session_id != base_session_id:
            continue
        role = getattr(mem, "role", "") or "user"
        if role not in roles:
            continue
        if not mem.kv_path or mem.num_tokens <= 0:
            continue
        if not Path(mem.kv_path).exists():
            logger.warning(
                "Resume: missing file for %s (%s)", mem.session_id, mem.kv_path,
            )
            continue
        rope_start = int(getattr(mem, "rope_start", 0) or 0)
        if rope_start <= 0:
            try:
                from pri.format import read_manifest
                manifest = read_manifest(mem.kv_path)
                if manifest:
                    rope_start = int(manifest.get("rope_start", 0) or 0)
            except Exception:
                rope_start = 0
        turn_index = int(getattr(mem, "turn_index", -1))
        block = ChainBlock(
            kv_path=mem.kv_path,
            num_tokens=int(mem.num_tokens),
            rope_start=rope_start,
            turn_index=turn_index,
            role=role,
            session_id=mem.session_id or mem.id,
            ring_type=mem.ring_type or "general",
            meta_score=float(getattr(mem, "meta_score", 0.0) or 0.0),
        )
        slot = (turn_index, role)
        ts = float(getattr(mem, "timestamp", 0) or 0)
        prev = latest_by_slot.get(slot)
        if prev is None or ts >= prev[0]:
            latest_by_slot[slot] = (ts, block)

    blocks = [block for _, block in latest_by_slot.values()]
    blocks.sort(
        key=lambda b: (
            b.turn_index if b.turn_index >= 0 else 10**9,
            _ROLE_ORDER.get(b.role, 99),
            b.session_id,
        ),
    )
    turn_indices = {b.turn_index for b in blocks if b.role == "turn"}
    if turn_indices:
        blocks = [
            b for b in blocks
            if b.role == "turn"
            or b.role == "tool"
            or b.turn_index not in turn_indices
        ]
    return blocks


def trim_chain_blocks(
    blocks: list[ChainBlock],
    *,
    max_blocks: int = 0,
    max_tokens: int = 0,
) -> list[ChainBlock]:
    """Keep the newest blocks within optional block/token budgets (overflow)."""
    if not blocks:
        return blocks
    if max_blocks <= 0 and max_tokens <= 0:
        return blocks

    selected: list[ChainBlock] = []
    token_budget = max_tokens if max_tokens > 0 else 10**9
    block_budget = max_blocks if max_blocks > 0 else 10**9

    for block in reversed(blocks):
        if len(selected) >= block_budget:
            break
        if block.num_tokens > token_budget and selected:
            break
        if block.num_tokens > token_budget and not selected:
            continue
        selected.append(block)
        token_budget -= block.num_tokens

    selected.reverse()
    if len(selected) < len(blocks):
        logger.info(
            "Resume trim: %d → %d blocks, %d → %d tokens",
            len(blocks),
            len(selected),
            sum(b.num_tokens for b in blocks),
            sum(b.num_tokens for b in selected),
        )
    return selected


def chain_pack_phantom_before_turn(
    store: "MemoryStore",
    user_id: str,
    base_session_id: str,
    turn_index: int,
    *,
    roles: frozenset[str] | None = None,
) -> int:
    """Return cumulative chain tokens packed before ``turn_index``.

    Resume RoPE re-rotation uses this pack-replay phantom (prior blocks in
    ``base_session_id`` order). It is **not** the live request ``num_phantom``
    from inject on the wire — that value reflects the current HTTP prefill and
    can diverge from chain pack math after trim, neutral fallback, or register
    layout.
    """
    if not base_session_id or not user_id or turn_index <= 1:
        return 0
    role_set = roles if roles is not None else RESUME_ROLES
    blocks = collect_chain_blocks(
        store, user_id, base_session_id, roles=role_set,
    )
    return sum(
        b.num_tokens
        for b in blocks
        if 0 <= b.turn_index < turn_index
    )


def apply_mamba_mode_override(cfg: dict | None, kvp: dict) -> None:
    """Per-request Mamba inject mode for resume pack (bench invalidation).

    ``memory_mamba_mode`` on the wire overrides ``NLS_RESUME_MAMBA_DELTA_SUM``:
      0 = genesis only
      1 = genesis + Σ(block − genesis)
      2 = genesis + last block delta only
      3 = last block SSM verbatim (resume chain tail)
    """
    if cfg is None:
        return
    raw = str(kvp.get("memory_mamba_mode", "") or "").strip()
    if not raw:
        return
    try:
        cfg["mamba_delta_sum"] = int(raw)
    except ValueError:
        logger.warning("Resume: invalid memory_mamba_mode=%r", raw)


def build_resume_inject_config(blocks: list[ChainBlock]) -> Optional[dict]:
    """Build snapshot_connector auto_config dict for resume inject."""
    if not blocks:
        return None

    snaps = []
    total_tokens = 0
    for block in blocks:
        snaps.append({
            "path": block.kv_path,
            "num_tokens": block.num_tokens,
            "strip_prefix": 0,
            "ring": block.ring_type,
            "sim": 1.0,
            "meta_score": block.meta_score,
            "rope_start": block.rope_start,
            "turn_index": block.turn_index,
            "role": block.role,
        })
        total_tokens += block.num_tokens

    return {
        "multi": True,
        "neural_scoring": False,
        "snapshots": snaps,
        "num_tokens": total_tokens,
        "inject_layout": "resume",
        # Mamba: genesis + Σ(block_ssm − genesis) across the turn chain.
        # Mode 3 (last block verbatim) is available via NLS_RESUME_MAMBA_DELTA_SUM=3.
        "mamba_delta_sum": int(os.environ.get("NLS_RESUME_MAMBA_DELTA_SUM", "1")),
    }


def try_resume_config(
    store: Optional["MemoryStore"],
    user_id: str,
    base_session_id: str,
    *,
    sys_prompt_hash: str = "",
    max_blocks: int = 0,
    max_tokens: int = 0,
) -> Optional[dict]:
    if store is None or not base_session_id:
        return None
    blocks = collect_chain_blocks(store, user_id, base_session_id)
    full_blocks = len(blocks)
    full_tokens = sum(b.num_tokens for b in blocks)
    blocks = trim_chain_blocks(blocks, max_blocks=max_blocks, max_tokens=max_tokens)
    if blocks and sys_prompt_hash and resume_prepend_sys_block_enabled():
        sys_block = find_system_block(store, sys_prompt_hash)
        if sys_block is not None:
            blocks = [sys_block] + blocks
            logger.info(
                "Resume: prepended system block (%d tok) hash=%s",
                sys_block.num_tokens,
                sys_prompt_hash[:12],
            )
        else:
            logger.warning(
                "Resume: no system block for hash=%s — inject order may be wrong",
                sys_prompt_hash[:12],
            )
    elif blocks and sys_prompt_hash and not resume_prepend_sys_block_enabled():
        logger.debug(
            "Resume: sys hash=%s present; system prepend disabled "
            "(live request carries system prefix)",
            sys_prompt_hash[:12],
        )
    if not blocks:
        logger.info(
            "Resume: no chain blocks for user=%s base_session=%s",
            user_id, base_session_id,
        )
        return None
    cfg = build_resume_inject_config(blocks)
    if cfg:
        cfg["_trim_evicted_blocks"] = max(0, full_blocks - len(blocks))
        cfg["_trim_evicted_tokens"] = max(0, full_tokens - cfg["num_tokens"])
        logger.info(
            "Resume: chain ready user=%s base_session=%s blocks=%d tokens=%d "
            "turns=%s trim_evicted=%d tok",
            user_id,
            base_session_id,
            len(blocks),
            cfg["num_tokens"],
            sorted({b.turn_index for b in blocks if b.turn_index >= 0}),
            cfg["_trim_evicted_tokens"],
        )
    return cfg
