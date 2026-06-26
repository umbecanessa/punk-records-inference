"""vLLM KV connector — capture and inject ``.nls`` snapshots (``NLSSnapshotConnector``).

Central hook registered via ``--kv-transfer-config`` in ``docker/start.sh``. Implements
the read/write path for Punk Records Inference:

  **Capture (write)** — after decode, serialize attention KV and hybrid recurrent state
  into compressed ``.nls`` manifests under ``NLS_SNAPSHOT_DIR``.

  **Inject (read)** — before prefill, load prior chain blocks (resume) or Swiss-ranked
  memories (overflow / legacy swiss profile) directly into vLLM's paged KV cache without
  dummy padding tokens.

Scheduler/worker split:

  - Scheduler reads inject config and reports externally-computed token count so real
    prompt positions start after injected KV.
  - Worker ``start_load_kv()`` writes K/V (and Mamba state) into allocated blocks
    before the model forward pass.

Configuration: ``NLS_SNAPSHOT_DIR``, ``NLS_API_INJECT_MODE`` (via ``startup_profile``),
per-request ``kv_transfer_params``. See ``docs/ARCHITECTURE.md`` and
``docs/CLIENT_CONTRACT.md``.
"""

import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from pri.capture import (
    is_turn_capture_mode,
    resume_turn_requires_inject,
    turn_capture_prefill_slice_start,
)

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# --- Auto-memory (Non-Stateless LLM) ---
try:
    from pri import retrieve as _auto_mem
    logger.info("NLS auto-memory loaded (enabled=%s)", _auto_mem.is_enabled())
except ImportError:
    _auto_mem = None
    logger.info("NLS auto-memory not available")

# --- Neural Scorer (model-native reranking) ---
try:
    from pri import scorer as _neural
    logger.info("NLS neural scorer loaded (enabled=%s)", _neural.is_enabled())
except ImportError:
    _neural = None
    logger.info("NLS neural scorer not available")

_last_prompt_token_ids: list[int] = []

# KL #607: strip a leading prefix of every injected memory's K/V tensor, to
# remove the templated system-block that was captured with each session.
# The ingest-time system block is `<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n`
# which tokenizes to a constant number of positions. For bench_longmemeval.py
# this is 25 tokens (Qwen 3.5 tokenizer, default SYSTEM_PROMPT). Set to 0 to
# disable. Only affects the multi-snapshot injection path.
STRIP_INJECT_SYS_BLOCK_LEN = int(os.environ.get("NLS_STRIP_INJECT_SYS_BLOCK_LEN", "0"))
if STRIP_INJECT_SYS_BLOCK_LEN > 0:
    logger.info(
        "NLS inject-time system-block strip: ACTIVE, stripping first %d positions "
        "of every injected memory (NLS_STRIP_INJECT_SYS_BLOCK_LEN)",
        STRIP_INJECT_SYS_BLOCK_LEN,
    )


def _get_strip_for_memory(kv_path: str) -> int:
    """KL #648: Per-memory strip based on manifest rope_start.

    Memories captured with capture_start>0 already had the system prompt
    stripped at capture time (rope_start stored in manifest). Applying the
    global STRIP on top would double-strip and destroy user content.
    Only apply STRIP when the memory includes the system prompt (rope_start==0).
    """
    if STRIP_INJECT_SYS_BLOCK_LEN == 0:
        return 0
    try:
        from pri.format import read_manifest
        manifest = read_manifest(kv_path)
        if manifest and manifest.get("rope_start", 0) > 0:
            return 0
    except Exception:
        pass
    return STRIP_INJECT_SYS_BLOCK_LEN


def _snapshots_from_retrieve_results(
    results: list,
    *,
    exclude_paths: set[str] | None = None,
    max_tokens: int = 0,
) -> tuple[list[dict], int]:
    """Build inject snapshot dicts from auto-memory retrieve() hits."""
    exclude_paths = exclude_paths or set()
    snaps: list[dict] = []
    token_budget = max_tokens if max_tokens > 0 else 10**9

    for r in results:
        kv_path, num_tokens, sim, ring, meta_s = r
        if kv_path in exclude_paths:
            continue
        strip = _get_strip_for_memory(kv_path)
        eff = max(num_tokens - strip, 0)
        if STRIP_ASSISTANT_KEEP_RATIO > 0 and eff > 0:
            eff = max(int(eff * STRIP_ASSISTANT_KEEP_RATIO), 1)
        if eff <= 0:
            continue
        if eff > token_budget and snaps:
            break
        if eff > token_budget and not snaps:
            continue
        snaps.append({
            "path": kv_path,
            "num_tokens": eff,
            "strip_prefix": strip,
            "ring": ring,
            "sim": sim,
            "meta_score": meta_s,
        })
        token_budget -= eff

    return snaps, sum(s["num_tokens"] for s in snaps)


def _kvp_truthy(kvp: dict, key: str) -> bool:
    return str(kvp.get(key, "") or "").strip().lower() in ("1", "true", "yes")


def _should_skip_auto_retrieval(kvp: dict) -> bool:
    """Agent chain turn-1 silo and resume-mode paths skip Swiss retrieval."""
    if _kvp_truthy(kvp, "memory_silo") or _kvp_truthy(kvp, "memory_no_retrieval"):
        return True
    if _kvp_truthy(kvp, "memory_off"):
        return True
    try:
        turn_idx = int(kvp.get("memory_turn_index", "-1") or "-1")
    except (TypeError, ValueError):
        turn_idx = -1
    base_sess = str(kvp.get("memory_base_session", "") or "").strip()
    if turn_idx == 1 and base_sess:
        return True
    resume_mode = str(kvp.get("memory_inject_mode", "") or "").strip().lower()
    if resume_mode in ("resume", "resume_overflow") and turn_idx > 1:
        return True
    return False


def _audit_resume_config_or_abort(cfg: dict) -> dict | None:
    """Drop resume inject config when RoPE pack geometry is inconsistent."""
    if not cfg or cfg.get("inject_layout") != "resume":
        return cfg
    try:
        from pri.inject_geometry_audit import log_geometry_audit, resume_inject_aborted
        from pri.rope_pack_balance import audit_and_balance_resume_config

        cfg, audit = audit_and_balance_resume_config(cfg)
        log_geometry_audit(audit)
        if resume_inject_aborted(audit):
            logger.error(
                "NLS resume inject ABORTED: %s — cold prefill this request",
                audit.get("findings"),
            )
            return None
    except Exception as exc:
        logger.warning("NLS resume geometry preflight skipped: %s", exc)
    return cfg


def _resolve_prefilled_capture_boundary(
    kvp: dict,
    *,
    user_id: str,
) -> tuple[int, str]:
    """Recover KL #648 boundary fields when prefilled capture omits them on the wire."""
    try:
        cs = int(kvp.get("memory_capture_start", 0) or 0)
    except (TypeError, ValueError):
        cs = 0
    sh = str(kvp.get("memory_sys_prompt_hash", "") or "")
    if cs > 0 and sh:
        return cs, sh
    if _auto_mem is None or not getattr(_auto_mem, "_store", None):
        return cs, sh
    store = _auto_mem._store
    base_sess = str(kvp.get("memory_base_session", "") or "")
    if not sh:
        for mem in reversed(store._memories):
            if mem.user_id != user_id:
                continue
            if str(getattr(mem, "role", "")) == "system":
                cand = str(getattr(mem, "sys_prompt_hash", "") or "")
                if cand:
                    sh = cand
                    break
    if not sh and base_sess:
        for mem in reversed(store._memories):
            if mem.user_id != user_id or mem.base_session_id != base_sess:
                continue
            cand = str(getattr(mem, "sys_prompt_hash", "") or "")
            if cand:
                sh = cand
                break
    if cs <= 0 and base_sess:
        for mem in store._memories:
            if mem.user_id != user_id or mem.base_session_id != base_sess:
                continue
            if int(getattr(mem, "turn_index", -1)) == 1:
                rs = int(getattr(mem, "rope_start", 0) or 0)
                if rs > 0:
                    cs = rs
                    break
    if sh:
        from pri.resume import find_system_block

        sys_block = find_system_block(store, sh)
        if sys_block is not None and sys_block.num_tokens > 0 and cs <= 0:
            cs = int(sys_block.num_tokens)
    return cs, sh


def _compaction_overflow_snaps(
    store: Any,
    user_id: str,
    base_session_id: str,
    exclude_paths: set[str],
    max_tokens: int,
) -> tuple[list[dict], int]:
    """Pin latest compaction-context block(s) for Arm D overflow."""
    if max_tokens <= 0 or store is None:
        return [], 0
    snaps: list[dict] = []
    token_budget = max_tokens
    for mem in store.find_compaction_context_memories(
        user_id,
        base_session_id,
        exclude_paths=exclude_paths,
        limit=2,
    ):
        strip = _get_strip_for_memory(mem.kv_path)
        eff = max(mem.num_tokens - strip, 0)
        if eff <= 0 or eff > token_budget:
            continue
        snaps.append({
            "path": mem.kv_path,
            "num_tokens": eff,
            "strip_prefix": strip,
            "ring": mem.ring_type,
            "sim": 1.0,
            "meta_score": mem.meta_score,
            "rope_start": mem.rope_start,
            "turn_index": mem.turn_index,
            "role": mem.role,
        })
        token_budget -= eff
    return snaps, sum(s["num_tokens"] for s in snaps)

# Each captured memory follows the template:
#   [system block] [user turn ~50%] [assistant turn ~50%]
# Assistant responses echo user facts and add noise, degrading extraction.
# When enabled, only the first STRIP_ASSISTANT_KEEP_RATIO of each memory's
# tokens (after system-block strip) are injected.
STRIP_ASSISTANT_KEEP_RATIO = float(os.environ.get("NLS_STRIP_ASSISTANT_KEEP_RATIO", "0"))
if STRIP_ASSISTANT_KEEP_RATIO > 0:
    logger.info(
        "NLS inject-time assistant strip: ACTIVE, keeping first %.0f%% of each "
        "memory's tokens (NLS_STRIP_ASSISTANT_KEEP_RATIO)",
        STRIP_ASSISTANT_KEEP_RATIO * 100,
    )
elif STRIP_ASSISTANT_KEEP_RATIO < 0:
    logger.info(
        "NLS inject-time assistant strip: SEGMENT-BASED V-SUPPRESSION (ratio=%s). "
        "Will zero V-vectors at assistant positions per .nls manifest segments.",
        STRIP_ASSISTANT_KEEP_RATIO,
    )

# KL #630: Memory salience amplification ("new car effect").
# Scale injected K/V tensors to boost attention toward memory positions.
# K-scaling increases Q@K dot products → more attention weight on memory.
# V-scaling amplifies the output signal when attention lands on memory.
# Also supports per-request override via kv_transfer_params:
#   memory_k_scale, memory_v_scale
KV_K_SCALE = float(os.environ.get("NLS_KV_K_SCALE", "1.0"))
KV_V_SCALE = float(os.environ.get("NLS_KV_V_SCALE", "1.0"))
if KV_K_SCALE != 1.0 or KV_V_SCALE != 1.0:
    logger.info(
        "NLS memory salience amplification: K_SCALE=%.2f, V_SCALE=%.2f",
        KV_K_SCALE, KV_V_SCALE,
    )

# Per-request capture registry (KL #476): maps request_id -> capture info
# Populated by get_num_new_matched_tokens, consumed by capture/readback paths
_capture_registry: dict[str, dict] = {}

# DeltaNet-only injection registry (KL #625 compounding).
# Maps request_id -> .nls path or "session:<id>" for deferred resolution.
_deltanet_init_registry: dict[str, str] = {}

# KL #625: Block indices where DeltaNet was seeded — must NOT be zeroed
# by clear_linear_attention_cache_for_new_sequences during prefill.
_deltanet_seeded_blocks: set[int] = set()

# Module-level shared state for readback captures.
# Populated by scheduler-side request_finished_all_groups,
# consumed by worker-side _process_pending_captures.
_pending_readback_captures: dict[str, dict] = {}
_finished_readback_ids: set[str] = set()

from pri.layer_names import classify_layer_names, extract_layer_index


def _extract_layer_idx(layer_name: str) -> int | None:
    return extract_layer_index(layer_name)


def _resolve_lm_config(hf):
    """Causal-LM config for nested wrappers (e.g. Gemma3 ``text_config``).

    vLLM's top-level ``hf_config`` for multimodal / wrapper models often omits
    ``head_dim`` and ``num_key_value_heads``; capture readback width checks then
    fall back to wrong defaults and skip every layer.
    """
    tc = getattr(hf, "text_config", None)
    if tc is not None and getattr(tc, "num_hidden_layers", None) is not None:
        return tc
    return hf


# ── Bounded byte-aware snapshot LRU ──────────────────────────────────
#
# KL #736 piece 1: replace the unbounded `_snapshot_cache: dict` that
# leaked every loaded `.nls`/`.kvz` for the process lifetime. On
# OpenCode sessions where one tool block (e.g. a 20k-token PRD read) is
# captured per turn, each retrieval round re-dequantises ~1-2 GiB of
# bf16 tensors into the python heap (or, on GB10 unified memory, into a
# region that the CUDA allocator cannot reclaim). Without an explicit
# eviction path those tensors are pinned by the dict for the lifetime
# of the engine and the working set climbs monotonically until OOM
# kicks the engine ~30 turns in.
#
# This LRU bounds the cache by BOTH entries AND summed tensor bytes
# (whichever cap hits first), drops refs on evict so torch's refcount
# falls to zero, and calls `torch.cuda.empty_cache()` periodically so
# the caching allocator actually releases blocks back to the pool. The
# downstream `_load_multi_snapshots` path already shallow-copies before
# mutating, so evicting a hot path while it is in use is safe — the
# caller already holds the dict it needs.
class _SnapshotLRU:
    """Byte-aware LRU for dequantised snapshot dicts."""

    # Default caps sized for a single-user demo on GB10. Tuneable via
    # env so the harness can sweep them without code edits.
    _DEFAULT_MAX_ENTRIES = 32
    _DEFAULT_MAX_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB
    _EMPTY_CACHE_EVERY = 4  # empty_cache() once per N evictions

    def __init__(
        self,
        max_entries: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        try:
            env_entries = int(os.environ.get("NLS_SNAPSHOT_CACHE_MAX_ENTRIES", "0"))
        except Exception:
            env_entries = 0
        try:
            env_bytes = int(os.environ.get("NLS_SNAPSHOT_CACHE_MAX_BYTES", "0"))
        except Exception:
            env_bytes = 0

        self._max_entries: int = (
            max_entries if max_entries is not None else (env_entries or self._DEFAULT_MAX_ENTRIES)
        )
        self._max_bytes: int = (
            max_bytes if max_bytes is not None else (env_bytes or self._DEFAULT_MAX_BYTES)
        )
        # Path -> data dict, in LRU order (oldest = front).
        self._data: "OrderedDict[str, dict]" = OrderedDict()
        # Mirror of estimated bytes per entry — recomputed on insert so
        # we never iterate the full LRU just to know its total weight.
        self._bytes_per_path: dict[str, int] = {}
        self._total_bytes: int = 0
        self._evict_counter: int = 0
        # Stats for the harness scorecard and admin endpoint.
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    @staticmethod
    def _estimate_bytes(data: dict) -> int:
        """Sum the size of any torch.Tensor inside `data` (recursive)."""
        total = 0
        # Snapshot dicts are usually flat {key: tensor_or_meta}, but a
        # few legacy paths nest lists/dicts (e.g. multi_snapshots meta).
        # Handle both shapes without importing typing helpers.
        stack: list = [data]
        while stack:
            obj = stack.pop()
            if isinstance(obj, torch.Tensor):
                try:
                    total += int(obj.numel()) * int(obj.element_size())
                except Exception:
                    pass
            elif isinstance(obj, dict):
                stack.extend(obj.values())
            elif isinstance(obj, (list, tuple)):
                stack.extend(obj)
        return total

    def get(self, path: str) -> dict | None:
        entry = self._data.get(path)
        if entry is None:
            self.misses += 1
            return None
        # LRU touch — move to end (most-recently-used).
        self._data.move_to_end(path)
        self.hits += 1
        return entry

    def put(self, path: str, data: dict) -> None:
        if path in self._data:
            # Drop the old measurement before reinsert; tensor identity
            # may have changed (e.g. Pass-2 rewrote the target file).
            self._total_bytes -= self._bytes_per_path.pop(path, 0)
            self._data.pop(path, None)
        size = self._estimate_bytes(data)
        self._data[path] = data
        self._bytes_per_path[path] = size
        self._total_bytes += size
        self._enforce_caps()

    def pop(self, path: str, default=None):
        """Explicit removal — used by the Pass-2 compound-merge path
        which rewrites a target snapshot and needs the next load to
        pick up fresh tensors."""
        entry = self._data.pop(path, default)
        if entry is default:
            return default
        self._total_bytes -= self._bytes_per_path.pop(path, 0)
        # Drop tensor refs so the allocator can reclaim them.
        self._release(entry)
        return entry

    def _enforce_caps(self) -> None:
        # Evict oldest until both caps are satisfied. We always preserve
        # at least one entry — the just-inserted hot one — so an
        # oversized single snapshot doesn't get evicted before it's
        # used. (The next put() will evict it then.)
        while len(self._data) > 1 and (
            len(self._data) > self._max_entries
            or self._total_bytes > self._max_bytes
        ):
            oldest_path, oldest_data = self._data.popitem(last=False)
            self._total_bytes -= self._bytes_per_path.pop(oldest_path, 0)
            self._release(oldest_data)
            self.evictions += 1
            self._evict_counter += 1
            if self._evict_counter >= self._EMPTY_CACHE_EVERY:
                self._evict_counter = 0
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    @staticmethod
    def _release(data: dict) -> None:
        """Drop refs to any tensors in `data` so refcount → 0."""
        try:
            stack: list = [data]
            while stack:
                obj = stack.pop()
                if isinstance(obj, dict):
                    for k in list(obj.keys()):
                        v = obj[k]
                        if isinstance(v, (dict, list, tuple)):
                            stack.append(v)
                        # Drop in-place so the alias inside the dict
                        # itself doesn't pin the tensor.
                        obj[k] = None
                elif isinstance(obj, list):
                    for i in range(len(obj)):
                        v = obj[i]
                        if isinstance(v, (dict, list, tuple)):
                            stack.append(v)
                        obj[i] = None
        except Exception:
            pass

    def __contains__(self, path: str) -> bool:
        return path in self._data

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        return {
            "entries": len(self._data),
            "max_entries": self._max_entries,
            "total_bytes": self._total_bytes,
            "max_bytes": self._max_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


# ── Metadata passed from Scheduler → Worker ──────────────────────────


@dataclass
class SnapshotReqMeta:
    slot_mapping: torch.Tensor   # maps real snapshot positions -> KV cache slots
    snapshot_path: str
    num_snapshot_tokens: int      # actual tokens with real K,V data
    multi_snapshots: list | None = None  # list of {"path", "num_tokens", "offset"}
    mamba_block_map: dict | None = None  # group_idx -> block_id for Mamba state
    register_slot_mapping: torch.Tensor | None = None  # V2.2 register slots physical mapping
    mamba_delta_sum: int = 0  # 0=genesis, 1=sum-deltas, 2=last-delta
    deltanet_init_path: str = ""  # KL #625: .nls path for Mamba-only seeding
    neural_scoring: bool = True  # per-request override (force-inject sets False)
    # Fable C′: "concat" (packed multi-snap) vs "resume" (chain thread inject).
    inject_layout: str = "concat"
    # KL #708: physical phantom layout starts with `num_register` register slots
    # BEFORE the memory tokens. Memory K vectors live at logical positions
    # [num_register .. num_register + num_snapshot_tokens - 1] in the new
    # request, so RoPE re-rotation must use that offset (not 0). Without this
    # shift, every memory carries a -num_register positional bias which the
    # model perceives as a phase mismatch — empirically this manifests as
    # degeneration / token loops once enough memories accumulate.
    num_register: int = 0


@dataclass
class NLSSnapshotMetadata(KVConnectorMetadata):
    requests: list[SnapshotReqMeta] = field(default_factory=list)


# ── Connector ────────────────────────────────────────────────────────


def _patch_prometheus_counter():
    """Prevent negative counter increments from crashing the server."""
    try:
        from prometheus_client.metrics import Counter
        _orig_inc = Counter.inc

        def _safe_inc(self, amount=1, exemplar=None):
            if amount < 0:
                amount = 0
            return _orig_inc(self, amount, exemplar)

        Counter.inc = _safe_inc
        logger.info("NLS: patched Prometheus Counter.inc (clamp negatives)")
    except Exception:
        logger.warning("NLS: failed to patch Prometheus Counter", exc_info=True)


def _patch_mamba_block_aligned_split():
    """Remove the 'not verified yet' assert for external KV tokens."""
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
        orig = Scheduler._mamba_block_aligned_split

        def patched(self, request, num_new_tokens,
                    num_new_local_computed_tokens=0,
                    num_external_computed_tokens=0):
            num_computed_tokens = (
                request.num_computed_tokens
                + num_new_local_computed_tokens
                + num_external_computed_tokens
            )
            if num_computed_tokens < max(
                request.num_prompt_tokens, request.num_tokens - 1
            ):
                block_size = self.cache_config.block_size
                last_cache_position = (
                    request.num_tokens - request.num_tokens % block_size
                )
                if getattr(self, 'use_eagle', False):
                    last_cache_position = max(
                        last_cache_position - block_size, 0
                    )
                after = num_computed_tokens + num_new_tokens
                if after < last_cache_position:
                    num_new_tokens = num_new_tokens // block_size * block_size
                elif num_computed_tokens < last_cache_position < after:
                    num_new_tokens = last_cache_position - num_computed_tokens
            return num_new_tokens

        Scheduler._mamba_block_aligned_split = patched
        logger.info("NLS: patched _mamba_block_aligned_split (removed assert)")
    except Exception:
        logger.warning("NLS: failed to patch _mamba_block_aligned_split",
                       exc_info=True)


_patch_prometheus_counter()
_patch_mamba_block_aligned_split()


class NLSSnapshotConnector(KVConnectorBase_V1, SupportsHMA):

    _instance: "NLSSnapshotConnector | None" = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._composite_block_size = vllm_config.cache_config.block_size
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        # Resolution order: NLS_SNAPSHOT_DIR env var (authoritative, keeps
        # store + connector agreeing on one location) > connector config >
        # legacy default. /tmp is volatile in containers so any deployment
        # MUST set NLS_SNAPSHOT_DIR to persistent storage.
        env_snap = os.environ.get("NLS_SNAPSHOT_DIR", "")
        self._snapshot_dir = Path(
            env_snap or extra.get("snapshot_dir", "/tmp/nls_kv_snapshot")
        )
        self._config_path = self._snapshot_dir / "inject_config.json"
        self._requests_need_load: dict[str, "Request"] = {}
        # NLS FIX prefix-cache-with-memories: track requests where we
        # mutated the prompt to inject phantoms after a prefix-cache hit
        # and deferred them via return-None. Re-entry returns stored data.
        self._deferred_for_cache_mutation: dict[str, dict] = {}
        # KL #736 piece 1: bounded byte-aware LRU replaces the legacy
        # unbounded dict. Sized via NLS_SNAPSHOT_CACHE_MAX_{ENTRIES,BYTES}
        # envs (defaults 32 entries / 4 GiB). See `_SnapshotLRU`.
        self._snapshot_cache: _SnapshotLRU = _SnapshotLRU()
        # KL #736 piece 2: throttle for `torch.cuda.empty_cache()` calls
        # at transient-tensor lifecycle boundaries (post-readback,
        # post-compound-merge, post-request-finished). The CUDA caching
        # allocator does not return blocks to the OS until empty_cache()
        # is called, but each call costs ~1-2ms of GPU sync — so we
        # gate by elapsed wall time (default 5s) rather than calling on
        # every event.
        try:
            self._cuda_release_interval_s: float = float(
                os.environ.get("NLS_CUDA_RELEASE_INTERVAL_S", "5.0")
            )
        except Exception:
            self._cuda_release_interval_s = 5.0
        self._last_cuda_release_ns: int = 0
        self._session_path_cache: dict[str, str] = {}  # session_id -> .nls path
        self._config_cache: dict | None = None
        self._config_mtime: float = 0.0
        self._auto_config: dict | None = None  # cached auto-memory result

        # Post-completion KV capture uses module-level dicts shared between
        # scheduler and worker instances (both live in the same EngineCore process).
        self._kv_snapshot_capture_dir = self._snapshot_dir / "captures"
        self._kv_snapshot_capture_dir.mkdir(parents=True, exist_ok=True)

        NLSSnapshotConnector._instance = self

        hf = _resolve_lm_config(vllm_config.model_config.hf_config)
        hidden = int(getattr(hf, "hidden_size", 0) or 0)
        n_heads = int(getattr(hf, "num_attention_heads", 0) or 0)
        self._rope_head_dim = int(
            getattr(hf, "head_dim", None) or (hidden // max(n_heads, 1) or 256)
        )
        self._rope_theta = getattr(hf, "rope_theta", 10_000_000)
        self._num_kv_heads = int(
            getattr(hf, "num_key_value_heads", None) or max(n_heads, 2)
        )

        # With HMA, each group has its own block size. We need the
        # full-attention group's block size for KV snapshot alignment.
        self._group_block_sizes: list[int] = []
        self._full_attn_group_idx: int = 0
        self._attn_block_size: int = self._composite_block_size
        self._mamba_group_indices: list[int] = []
        self._layer_to_group: dict[int, int] = {}

        if kv_cache_config is not None:
            unmapped_kv_names: list[str] = []
            for gidx, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                bs = getattr(spec, 'block_size', self._composite_block_size)
                self._group_block_sizes.append(bs)
                spec_name = type(spec).__name__
                for layer_name in group.layer_names:
                    lidx = _extract_layer_idx(layer_name)
                    if lidx is not None:
                        self._layer_to_group[lidx] = gidx
                    else:
                        unmapped_kv_names.append(layer_name)
                logger.info(
                    "  group[%d]: %s, block_size=%d, layers=%d (%s...)",
                    gidx, spec_name, bs, len(group.layer_names),
                    group.layer_names[0] if group.layer_names else "?",
                )
                if 'FullAttention' in spec_name:
                    self._full_attn_group_idx = gidx
                    self._attn_block_size = bs
                if 'Mamba' in spec_name:
                    self._mamba_group_indices.append(gidx)
            if unmapped_kv_names:
                sample = ", ".join(unmapped_kv_names[:3])
                logger.warning(
                    "NLS layer index: %d KV layer name(s) unmatched "
                    "(capture/inject will skip them), e.g. %s",
                    len(unmapped_kv_names),
                    sample,
                )

        logger.info(
            "NLSSnapshotConnector init: role=%s, snapshot_dir=%s, "
            "composite_bs=%d, attn_bs=%d, attn_group=%d, groups=%s, "
            "mamba_groups=%s, layer_map_size=%d",
            role, self._snapshot_dir, self._composite_block_size,
            self._attn_block_size, self._full_attn_group_idx,
            self._group_block_sizes, self._mamba_group_indices,
            len(self._layer_to_group),
        )

        # KL #625: Monkey-patch linear_attn cache clearing to respect seeded blocks
        self._patch_linear_attn_cache_clear()

    # ── KL #625: Patch linear_attn cache clearing ──────────────

    @staticmethod
    def _patch_linear_attn_cache_clear():
        """Wrap clear_linear_attention_cache_for_new_sequences to skip
        blocks that were seeded with DeltaNet state for compounding."""
        try:
            import vllm.model_executor.layers.mamba.linear_attn as _la_mod
            _orig_fn = _la_mod.clear_linear_attention_cache_for_new_sequences

            if getattr(_orig_fn, "_nls_patched", False):
                return  # already patched

            def _patched_clear(kv_cache, state_indices_tensor, attn_metadata):
                if not _deltanet_seeded_blocks:
                    return _orig_fn(kv_cache, state_indices_tensor, attn_metadata)

                num_prefills = getattr(attn_metadata, "num_prefills", 0)
                if num_prefills <= 0:
                    return

                num_decode_tokens = getattr(attn_metadata, "num_decode_tokens", 0)
                for prefill_idx in range(num_prefills):
                    q_start = attn_metadata.query_start_loc[
                        num_decode_tokens + prefill_idx
                    ]
                    q_end = attn_metadata.query_start_loc[
                        num_decode_tokens + prefill_idx + 1
                    ]
                    query_len = q_end - q_start
                    context_len = (
                        attn_metadata.seq_lens[num_decode_tokens + prefill_idx]
                        - query_len
                    )
                    if context_len == 0:
                        block_idx = state_indices_tensor[
                            num_decode_tokens + prefill_idx
                        ]
                        bi = int(block_idx.item())
                        if bi in _deltanet_seeded_blocks:
                            _deltanet_seeded_blocks.discard(bi)
                            logger.info(
                                "DeltaNet-init: SKIPPED cache clear for "
                                "block %d (seeded state preserved)",
                                bi,
                            )
                        else:
                            kv_cache[block_idx, ...] = 0

            _patched_clear._nls_patched = True
            _la_mod.clear_linear_attention_cache_for_new_sequences = _patched_clear
            logger.info(
                "KL #625: Patched linear_attn cache clearing for "
                "DeltaNet compounding"
            )
        except ImportError:
            logger.debug("linear_attn module not found, skipping patch")
        except Exception:
            logger.warning(
                "Failed to patch linear_attn cache clearing",
                exc_info=True,
            )

    # ── Config file reader (scheduler-side, cached) ──────────────

    def _read_config(self) -> dict | None:
        try:
            exists = self._config_path.exists()
            if not exists:
                if self._config_cache is not None:
                    logger.info("NLS snapshot config removed: %s", self._config_path)
                self._config_cache = None
                return None
            mtime = os.path.getmtime(self._config_path)
            if mtime != self._config_mtime:
                with open(self._config_path) as f:
                    self._config_cache = json.load(f)
                self._config_mtime = mtime
                logger.info(
                    "NLS snapshot config loaded: %s", self._config_cache
                )
            return self._config_cache
        except Exception:
            logger.warning(
                "Failed to read snapshot config: %s", self._config_path,
                exc_info=True,
            )
            return None

    # ══════════════════════════════════════════════════════════════
    # SCHEDULER-SIDE
    # ══════════════════════════════════════════════════════════════

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # ════════════════════════════════════════════════════════════════
        # Phase 1 — UNCONDITIONAL bookkeeping (must run on every request).
        #
        # Historical bug: this function used to early-return on
        # `num_computed_tokens > 0` BEFORE parsing kv_transfer_params or
        # populating the per-request capture registry. That made every
        # tool-result follow-up turn in agent mode invisible to NLS:
        #   - vLLM prefix-caches the static preamble (system + tools)
        #   - Follow-up turns arrive with num_computed_tokens > 0
        #   - Connector early-returned → no labels parsed → no registry
        #   - Capture step at request_finished found registry_has=False
        #   - Tool blocks NEVER captured, chain semantics never built
        # The fix is to ALWAYS parse labels and register the request, and
        # then skip retrieval/injection on prefix-cache hits (Phase 2).
        # ════════════════════════════════════════════════════════════════
        global _last_prompt_token_ids
        _last_prompt_token_ids = list(request.prompt_token_ids)

        # Parse per-request memory labels from kv_transfer_params.
        # vLLM v3 doesn't populate request.kv_transfer_params from API;
        # the data arrives inside sampling_params.extra_args instead.
        _kvp = getattr(request, "kv_transfer_params", None) or {}
        if not _kvp:
            _sp = getattr(request, "sampling_params", None)
            _ea = getattr(_sp, "extra_args", None) if _sp else None
            if isinstance(_ea, dict):
                _kvp = _ea.get("kv_transfer_params") or {}
        if _kvp:
            logger.info(
                "NLS memory labels: %s",
                {k: v for k, v in _kvp.items() if k.startswith("memory_")},
            )
        # Only update auto-mem context when we actually have new labels
        # — prevents the second-call clobber issue (see KL #714 below).
        if _kvp and _auto_mem is not None and _auto_mem.is_enabled():
            _auto_mem.set_request_context(
                user_id=str(_kvp.get("memory_user", "default")),
                project_id=str(_kvp.get("memory_project", "")),
                ring_type=str(_kvp.get("memory_ring", "general")),
                description=str(_kvp.get("memory_desc", "")),
                session_id=str(_kvp.get("memory_session", "")),
                memory_off=str(_kvp.get("memory_off", "0")) == "1",
                ingest_only=str(_kvp.get("memory_ingest", "0")) == "1",
                no_capture=str(_kvp.get("memory_no_capture", "0")) == "1",
            )

        orig_tokens = request.num_tokens
        # KL #714 fix: vLLM may call get_num_new_matched_tokens multiple
        # times for the same request (e.g. when the scheduler defers and
        # re-evaluates). The original `kv_transfer_params` only arrives on
        # the FIRST call — subsequent calls see an empty `_kvp` because
        # the request object's params have already been consumed. If we
        # unconditionally overwrite `_capture_registry[request_id]`, the
        # second call clobbers the labels (compound_into, sys_prompt_hash,
        # turn_index, etc.) with empty strings, breaking Pass-2 merge,
        # system self-warm, and chain-walk. Fix: only write/update label
        # fields when `_kvp` is non-empty; always update dynamic fields
        # (prompt_token_ids, expected_tokens) since they may shift.
        existing = _capture_registry.get(request.request_id)
        if existing is None:
            existing = {
                "prompt_token_ids": list(request.prompt_token_ids),
                "expected_tokens": orig_tokens,
                "num_phantom": 0,  # updated in Phase 3 if we inject
            }
            _capture_registry[request.request_id] = existing
        else:
            existing["prompt_token_ids"] = list(request.prompt_token_ids)
            existing["expected_tokens"] = orig_tokens

        if _kvp:
            existing.update({
                "user_id": str(_kvp.get("memory_user", "default")),
                "session_id": str(_kvp.get("memory_session", "")),
                "ring_type": str(_kvp.get("memory_ring", "general")),
                "memory_off": str(_kvp.get("memory_off", "0")) == "1",
                "no_capture": str(_kvp.get("memory_no_capture", "0")) == "1",
                "memory_text": str(_kvp.get("memory_text", "")),
                # NLS v2 blockchain fields
                "capture_start": int(_kvp.get("memory_capture_start", 0)),
                "capture_end": int(_kvp.get("memory_capture_end", 0)),
                "block_role": str(_kvp.get("memory_block_role", "")),
                "parent_hash": str(_kvp.get("memory_parent_hash", "")),
                "prev_hash": str(_kvp.get("memory_prev_hash", "")),
                "turn_index": int(_kvp.get("memory_turn_index", -1)),
                "base_session_id": str(_kvp.get("memory_base_session", "")),
                # System-prompt block dedup (KL #708): backend hashes the
                # rendered system prompt and sends a 16-char SHA256 prefix
                # so the capture path can attribute blocks to a known sys-
                # prompt template (enables future cross-request reuse).
                "sys_prompt_hash": str(
                    _kvp.get("memory_sys_prompt_hash", "")
                ),
                # KL #626 two-pass: when set, this request is a Pass-2
                # compound follow-up — the readback only extracts Mamba
                # state and merges it into the target session's existing
                # .nls (no new memory entry created).
                "compound_into": str(_kvp.get("memory_compound_into", "")),
                # When set, the assistant block from this turn's dual-emit
                # readback gets this session_id (so backends can pre-
                # allocate the chain-walk identity instead of inferring it
                # from the primary session_id).
                "asst_session_id": str(_kvp.get("memory_asst_session", "")),
                "inject_mode": str(
                    _kvp.get("memory_inject_mode", "")
                    or os.environ.get("NLS_INJECT_MODE", "swiss")
                ).strip().lower(),
                "compaction_detected": _kvp_truthy(_kvp, "memory_compaction_detected"),
                "prefilled_capture": _kvp_truthy(_kvp, "memory_prefilled_capture"),
            })
            if _kvp_truthy(_kvp, "memory_prefilled_capture"):
                _uid = str(_kvp.get("memory_user", "default"))
                _resolved_cs, _resolved_sh = _resolve_prefilled_capture_boundary(
                    _kvp, user_id=_uid,
                )
                if _resolved_cs > 0:
                    existing["capture_start"] = _resolved_cs
                    _kvp["memory_capture_start"] = str(_resolved_cs)
                if _resolved_sh:
                    existing["sys_prompt_hash"] = _resolved_sh
                    _kvp["memory_sys_prompt_hash"] = _resolved_sh
                if _resolved_cs > 0 or _resolved_sh:
                    logger.info(
                        "NLS prefilled capture: resolved boundary "
                        "capture_start=%s sys_hash=%s",
                        _resolved_cs,
                        (_resolved_sh[:12] + "...") if _resolved_sh else "",
                    )

        # KL #625: DeltaNet compounding registration runs unconditionally
        # too — the capture path needs it on follow-up turns.
        _dn_init_path = str(_kvp.get("memory_deltanet_init", ""))
        _dn_init_session = str(_kvp.get("memory_deltanet_init_session", ""))
        if _dn_init_path or _dn_init_session:
            _deltanet_init_registry[request.request_id] = (
                _dn_init_path or f"session:{_dn_init_session}"
            )
            logger.info(
                "NLS DeltaNet-init registered: req=%s, target=%s",
                request.request_id,
                _dn_init_path or f"session:{_dn_init_session}",
            )

        # ════════════════════════════════════════════════════════════════
        # Phase 2 — Skip retrieval/injection on prefix-cache hits.
        #
        # If vLLM has already computed any tokens of this prompt from the
        # prefix cache, we cannot inject phantom KV at position 0 — the
        # cached prefix already occupies that range. Capture still runs at
        # request_finished and writes the new tokens (e.g. the tool result)
        # onto the existing chain via memory_capture_start/_end labels.
        # ════════════════════════════════════════════════════════════════
        # NLS FIX prefix-cache-with-memories: post-deferral re-entry.
        # If this request was deferred on a previous scheduler tick to
        # mutate its prompt past a stale cache hit, the prompt is already
        # mutated and get_computed_blocks just returned 0 on the fresh
        # hashes. Return the previously-computed ext_tokens.
        if request.request_id in self._deferred_for_cache_mutation:
            info = self._deferred_for_cache_mutation.pop(request.request_id)
            self._auto_config = info["cfg"]
            logger.info(
                "NLS post-defer re-entry: req=%s, ext_tokens=%d, "
                "new_computed=%d",
                request.request_id, info["ext_tokens"], num_computed_tokens,
            )
            return info["ext_tokens"], False

        # Remember whether the scheduler had a stale cache hit; we use
        # this AFTER running retrieve() — if memories exist we mutate the
        # prompt and defer, otherwise we keep the original skip behavior.
        _had_cache_hit = num_computed_tokens > 0

        # ════════════════════════════════════════════════════════════════
        # Phase 3 — Fresh prefill: retrieval + phantom-token injection.
        # ════════════════════════════════════════════════════════════════
        # Ensure forward-pass Q capture is active (needed for attention reranking).
        # Legacy forward-pass Q capture is no longer needed — all captures
        # go through _readback_and_save() which reads KV back from the paged
        # cache. Neural scoring Q data uses its own path.
        self._fwd_capture_enabled = False

        cfg = self._read_config()

        # Auto-memory retrieval: multi-snapshot injection
        # When neural scoring is enabled, retrieve a wider candidate pool
        # (COARSE_K) for model-native reranking during prefill.
        _neural_enabled = _neural is not None and _neural.is_enabled()
        _retrieve_k = _neural.COARSE_K if _neural_enabled else 3

        # Force-inject override: bypass retrieval and inject specific snapshots
        _force_inject_raw = _kvp.get("memory_force_inject", "")
        if _force_inject_raw and cfg is None:
            try:
                force_list = json.loads(_force_inject_raw)
                snaps = []
                for entry in force_list:
                    p = entry["path"]
                    nt = entry["num_tokens"]
                    strip = _get_strip_for_memory(p)
                    eff = max(nt - strip, 0) if strip else nt
                    if eff <= 0:
                        continue
                    snaps.append({
                        "path": p,
                        "num_tokens": eff,
                        "strip_prefix": strip,
                        "ring": "general",
                        "sim": 1.0,
                    })
                if snaps:
                    total_tokens = sum(s["num_tokens"] for s in snaps)
                    _inject_layout = str(
                        _kvp.get("memory_inject_layout", "concat") or "concat",
                    ).strip().lower()
                    cfg = {
                        "multi": True,
                        "neural_scoring": False,
                        "snapshots": snaps,
                        "num_tokens": total_tokens,
                        "mamba_delta_sum": int(_kvp.get("memory_mamba_mode", "0")),
                        "inject_layout": _inject_layout,
                    }
                    self._auto_config = cfg
                    logger.info(
                        "NLS FORCE-INJECT: %d memories, total_tokens=%d",
                        len(snaps), total_tokens,
                    )
            except Exception as e:
                logger.warning("memory_force_inject parse error: %s", e)

        # Fable C′ session-resume (+ Arm D overflow augmentation).
        if cfg is None:
            try:
                from pri.resume import (
                    apply_mamba_mode_override,
                    is_resume_mode,
                    is_resume_overflow_mode,
                    try_resume_config,
                )
                if is_resume_mode(_kvp) and _auto_mem is not None and _auto_mem._store:
                    _base_sess = str(_kvp.get("memory_base_session", ""))
                    _uid = str(_kvp.get("memory_user", "default"))
                    _max_blocks = int(_kvp.get("memory_resume_max_blocks", "0") or "0")
                    _max_tokens = int(_kvp.get("memory_resume_max_tokens", "0") or "0")
                    _sys_hash = str(_kvp.get("memory_sys_prompt_hash", "") or "")
                    _resume_cfg_raw = try_resume_config(
                        _auto_mem._store,
                        _uid,
                        _base_sess,
                        sys_prompt_hash=_sys_hash,
                        max_blocks=_max_blocks,
                        max_tokens=_max_tokens,
                    )
                    _resume_cfg = None
                    if _resume_cfg_raw is not None:
                        _resume_cfg = _audit_resume_config_or_abort(_resume_cfg_raw)
                        if _resume_cfg is None:
                            _cap_abort = _capture_registry.get(request.request_id)
                            if _cap_abort is not None:
                                _cap_abort["resume_inject_aborted"] = True
                    if _resume_cfg is not None:
                        cfg = _resume_cfg
                        apply_mamba_mode_override(cfg, _kvp)
                        self._auto_config = cfg
                        _mode_label = (
                            "resume_overflow"
                            if is_resume_overflow_mode(_kvp)
                            else "resume"
                        )
                        existing["inject_mode"] = _mode_label
                        _capture_registry[request.request_id]["inject_mode"] = _mode_label

                        if is_resume_overflow_mode(_kvp):
                            _evicted = int(_resume_cfg.get("_trim_evicted_tokens", 0) or 0)
                            _swiss_always = str(
                                _kvp.get("memory_resume_swiss_always", "") or "",
                            ).strip().lower() in ("1", "true", "yes")
                            _swiss_max = int(
                                _kvp.get("memory_resume_swiss_max_tokens", "0") or "0",
                            )
                            if _swiss_max < 0:
                                _swiss_max = 0
                            elif _swiss_max <= 0:
                                _swiss_max = int(
                                    os.environ.get("NLS_RESUME_SWISS_MAX_TOKENS", "256"),
                                )
                            if _swiss_max > 0 and (_evicted > 0 or _swiss_always):
                                _exclude = {s["path"] for s in cfg.get("snapshots", [])}
                                _comp_snaps, _comp_tok = _compaction_overflow_snaps(
                                    _auto_mem._store,
                                    _uid,
                                    _base_sess,
                                    _exclude,
                                    _swiss_max,
                                )
                                _exclude.update(s["path"] for s in _comp_snaps)
                                _swiss_budget = max(0, _swiss_max - _comp_tok)
                                _swiss_results = None
                                if _swiss_budget > 0:
                                    _swiss_results = _auto_mem.retrieve(
                                        list(request.prompt_token_ids),
                                        top_k=_retrieve_k,
                                        base_session_id=_base_sess,
                                        boost_compaction_context=True,
                                    )
                                _swiss_snaps: list[dict] = []
                                _swiss_tok = 0
                                if _swiss_results:
                                    _swiss_snaps, _swiss_tok = _snapshots_from_retrieve_results(
                                        _swiss_results,
                                        exclude_paths=_exclude,
                                        max_tokens=_swiss_budget,
                                    )
                                _swiss_snaps = _comp_snaps + _swiss_snaps
                                _swiss_tok = _comp_tok + _swiss_tok
                                if _swiss_snaps:
                                    # Swiss augmentation first; resume chain last
                                    # so mamba_delta_sum=3 uses the last resume block.
                                    cfg["snapshots"] = _swiss_snaps + cfg["snapshots"]
                                    cfg["num_tokens"] = (
                                        cfg.get("num_tokens", 0) + _swiss_tok
                                    )
                                    self._auto_config = cfg
                                    logger.info(
                                        "NLS ARM-D: +%d swiss blocks (%d tok) "
                                        "before resume chain (evicted=%d)",
                                        len(_swiss_snaps),
                                        _swiss_tok,
                                        _evicted,
                                    )
                            elif is_resume_overflow_mode(_kvp) and _swiss_max > 0:
                                logger.info(
                                    "NLS ARM-D: skip swiss (no trim eviction, "
                                    "evicted=%d tok)",
                                    _evicted,
                                )

                        logger.info(
                            "NLS RESUME-INJECT: %d blocks, total_tokens=%d",
                            len(cfg.get("snapshots", [])),
                            cfg.get("num_tokens", 0),
                        )
            except Exception as e:
                logger.warning("Session resume setup failed: %s", e)

        if cfg is None and _auto_mem is not None and _auto_mem.is_enabled():
            if _should_skip_auto_retrieval(_kvp):
                try:
                    _turn_idx = int(_kvp.get("memory_turn_index", "-1") or "-1")
                except (TypeError, ValueError):
                    _turn_idx = -1
                logger.info(
                    "NLS auto-retrieval SKIPPED (silo/resume): turn=%d silo=%s",
                    _turn_idx,
                    _kvp_truthy(_kvp, "memory_silo"),
                )
                results = None
            else:
                results = _auto_mem.retrieve(
                    list(request.prompt_token_ids), top_k=_retrieve_k,
                )
            if results is not None and len(results) > 0:
                if len(results) == 1 and STRIP_INJECT_SYS_BLOCK_LEN == 0:
                    kv_path, num_tokens, sim, ring, meta_s = results[0]
                    cfg = {"path": kv_path, "num_tokens": num_tokens}
                    logger.info(
                        "NLS auto-memory INJECT (single): ring=%s, sim=%.3f, "
                        "tokens=%d, path=%s",
                        ring, sim, num_tokens, kv_path,
                    )
                else:
                    snaps = []
                    for r in results:
                        kv_path, num_tokens, sim, ring, meta_s = r
                        strip = _get_strip_for_memory(kv_path)
                        eff = max(num_tokens - strip, 0)
                        if STRIP_ASSISTANT_KEEP_RATIO > 0 and eff > 0:
                            eff = max(int(eff * STRIP_ASSISTANT_KEEP_RATIO), 1)
                        if eff <= 0:
                            continue
                        snaps.append({
                            "path": kv_path,
                            "num_tokens": eff,
                            "strip_prefix": strip,
                            "ring": ring,
                            "sim": sim,
                            "meta_score": meta_s,
                        })
                    total_tokens = sum(s["num_tokens"] for s in snaps)
                    if snaps:
                        cfg = {
                            "multi": True,
                            "neural_scoring": _neural_enabled,
                            "snapshots": snaps,
                            "num_tokens": total_tokens,
                            "mamba_delta_sum": int(_kvp.get("memory_mamba_mode", "0")),
                        }
                        strip_vals = set(s["strip_prefix"] for s in snaps)
                        logger.info(
                            "NLS auto-memory INJECT (%s): %d memories, "
                            "total_tokens=%d (strip=%s), rings=%s",
                            "neural-score" if _neural_enabled else "multi",
                            len(snaps), total_tokens,
                            strip_vals if len(strip_vals) > 1 else strip_vals.pop(),
                            [s["ring"] for s in snaps],
                        )
                if cfg is not None:
                    self._auto_config = cfg

        num_snap = 0
        if cfg is not None:
            num_snap = cfg.get("num_tokens", 0)

        # Refresh per-user inject stats (resume + swiss + force-inject).
        try:
            from pri.admin import record_retrieval_event

            event: dict = {
                "user_id": str(_kvp.get("memory_user", "default")),
                "prompt_tokens": request.num_tokens,
                "injected_tokens": num_snap,
            }
            if cfg and cfg.get("inject_layout") == "resume":
                event["type"] = "chain_resume"
                event["memories"] = [
                    {
                        "path": s.get("path", ""),
                        "tokens": s.get("num_tokens", 0),
                        "role": "turn",
                    }
                    for s in (cfg.get("snapshots") or [])
                ]
            elif _auto_mem is not None and _auto_mem._last_retrieval:
                event.update(_auto_mem._last_retrieval)
                event["injected_tokens"] = num_snap
            record_retrieval_event(request.request_id, event)
        except Exception:
            pass

        # Update Phase-1 capture registry entry with the now-known phantom
        # token count (registry was populated above; labels + dn_init are
        # already set unconditionally so we don't repeat that work).
        logger.info(
            "NLS capture registry: req=%s, tokens=%d, phantom=%d",
            request.request_id, orig_tokens, max(num_snap, 0),
        )
        _capture_registry[request.request_id]["num_phantom"] = max(num_snap, 0)

        if num_snap <= 0:
            # No KV injection, but capture registry is populated for KV capture.
            # DeltaNet-init (if set) was registered in Phase 1.
            return 0, False

        logger.info(
            "NLS snapshot get_matched: req=%s, snap=%d, prompt=%d",
            request.request_id, num_snap, orig_tokens,
        )

        # Inject phantom prefix tokens so the scheduler sees enough tokens
        # for the external KV. The phantom token_ids are never embedded
        # (they're marked "externally computed"), only their KV slots matter.
        #
        # Layout: [register_slots | memory_injection | real_prompt]
        # Register slots (V2.2): reserved empty KV positions that the
        # streaming scorer can hot-swap memories into during decode.
        is_force_inject = bool(_kvp.get("memory_force_inject", ""))
        _is_resume = bool(cfg and cfg.get("inject_layout") == "resume")
        num_register = 0

        # Resume layout: phantom pack is [system | prior turns…]; the live
        # HTTP prompt still includes system+user. Strip the system prefix so
        # the embedded sequence matches inline chat order.
        if _is_resume and num_snap > 0:
            _resume_snaps = cfg.get("snapshots") or []
            if _resume_snaps and str(_resume_snaps[0].get("role", "")) == "system":
                try:
                    _strip_sys = int(_kvp.get("memory_capture_start", "0") or "0")
                except (TypeError, ValueError):
                    _strip_sys = 0
                if _strip_sys > 0 and len(request.prompt_token_ids) >= _strip_sys:
                    request.prompt_token_ids[:] = request.prompt_token_ids[_strip_sys:]
                    if getattr(request, "_all_token_ids", None):
                        request._all_token_ids[:] = request._all_token_ids[_strip_sys:]
                    request.num_prompt_tokens = max(
                        0, request.num_prompt_tokens - _strip_sys,
                    )
                    logger.info(
                        "NLS resume: stripped %d system tokens from live prompt "
                        "(phantom pack includes system block)",
                        _strip_sys,
                    )
                    _cap_reg = _capture_registry.get(request.request_id)
                    if _cap_reg is not None:
                        _cap_reg["resume_stripped_sys"] = _strip_sys

        if num_snap > 0 and request.request_id not in self._requests_need_load:
            total_phantom = num_register + num_snap
            phantom = [0] * total_phantom
            request.prompt_token_ids[:0] = phantom
            request._all_token_ids[:0] = phantom
            request.num_prompt_tokens += total_phantom
            _capture_registry[request.request_id]["num_register"] = num_register

            # NLS FIX phantom-layout-salt: phantom tokens are all zeros,
            # so without a salt every phantom-prefixed prompt would share
            # the same first-block hash → cross-request collisions (wrong
            # memory KV served from cache, or scheduler asserts firing on
            # over-counted num_computed_tokens).
            #
            # We derive a STABLE salt from the full phantom layout
            # (user, retrieved memory set, register/snap counts). When
            # two requests have the IDENTICAL layout — e.g. consecutive
            # turns of the same conversation where retrieval picks up the
            # same memory set — they share a cache_salt, the cached
            # phantom-block KV (memories + register slots) is reused, and
            # the model gets up to ~5000 tokens of memory "for free" on
            # the follow-up turn (this is the symbiosis between vLLM's
            # prefix cache and NLS injection).
            #
            # Layouts that differ in ANY of (user_id, memory paths,
            # num_register, num_snap) produce different salts → no false
            # matches → correctness preserved.
            try:
                if cfg.get("multi", False):
                    snapshot_paths = sorted(
                        str(s.get("path", "")) for s in cfg.get("snapshots", [])
                    )
                else:
                    snapshot_paths = [str(cfg.get("path", ""))]
                layout_key = json.dumps({
                    "u": str(_kvp.get("memory_user", "default")),
                    "paths": snapshot_paths,
                    "reg": num_register,
                    "snap": num_snap,
                }, sort_keys=True)
                layout_hash = hashlib.sha256(
                    layout_key.encode("utf-8")
                ).hexdigest()[:16]
                request.cache_salt = f"nls_layout_{layout_hash}"
            except Exception:
                # Fallback to a per-request unique salt. We prefer
                # correctness (no false cache match) over perf (cache
                # reuse) when layout hashing fails for any reason.
                try:
                    request.cache_salt = request.request_id
                except Exception:
                    pass

            # Recompute block hashes after token modification (required for
            # prefix caching — old hashes were computed on the original sequence)
            if hasattr(request, 'block_hashes') and hasattr(request, '_block_hasher'):
                request.block_hashes.clear()
                if request._block_hasher is not None:
                    request.block_hashes.extend(request._block_hasher(request))

            # Update registry with post-injection token ids
            _capture_registry[request.request_id]["prompt_token_ids"] = list(
                request.prompt_token_ids
            )

            logger.info(
                "NLS snapshot phantom inject: req=%s, register=%d, "
                "mem_snap=%d, total_phantom=%d, "
                "new_num_tokens=%d (was %d), block_hashes=%d",
                request.request_id, num_register, num_snap,
                num_register + num_snap,
                request.num_tokens, orig_tokens,
                len(getattr(request, 'block_hashes', [])),
            )

        # NLS FIX prefix-cache-with-memories: if the scheduler had a stale
        # cache hit on the bare prompt and we just mutated the prompt with
        # phantoms, the scheduler's local num_new_local_computed_tokens is
        # stale and points at KV slots for the WRONG content. Defer the
        # request so the scheduler re-runs get_computed_blocks against the
        # phantom-prefixed prompt (different first-block hash → no match).
        if _had_cache_hit and num_snap > 0:
            self._deferred_for_cache_mutation[request.request_id] = {
                "ext_tokens": num_snap + num_register,
                "cfg": self._auto_config,
            }
            logger.info(
                "NLS post-cache-hit inject: req=%s, mutated prompt and "
                "deferred (will re-enter with fresh hashes)",
                request.request_id,
            )
            return None, False

        return num_snap + num_register, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request
            logger.info(
                "NLS snapshot alloc: req=%s, external_tokens=%d",
                request.request_id, num_external_tokens,
            )
        elif request.request_id in _deltanet_init_registry:
            self._requests_need_load[request.request_id] = request
            logger.info(
                "NLS snapshot alloc (deltanet-init only): req=%s",
                request.request_id,
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NLSSnapshotMetadata()
        cfg = self._read_config()
        if cfg is None and self._auto_config is not None:
            cfg = self._auto_config

        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id not in self._requests_need_load:
                continue

            # KL #625: DeltaNet-init only (no KV injection)
            _dn_path = _deltanet_init_registry.get(new_req.req_id, "")
            if cfg is None and not _dn_path:
                continue
            if cfg is None and _dn_path:
                # Build minimal meta with only mamba_block_map
                all_block_ids = new_req.block_ids
                mamba_block_map = {}
                for gidx in self._mamba_group_indices:
                    if gidx < len(all_block_ids) and all_block_ids[gidx]:
                        mamba_block_map[gidx] = list(all_block_ids[gidx])
                if mamba_block_map:
                    meta.requests.append(SnapshotReqMeta(
                        slot_mapping=torch.empty(0, dtype=torch.long),
                        snapshot_path="",
                        num_snapshot_tokens=0,
                        mamba_block_map=mamba_block_map,
                        deltanet_init_path=_dn_path,
                    ))
                    logger.info(
                        "NLS DeltaNet-init meta: req=%s, mamba_blocks=%s",
                        new_req.req_id,
                        {g: len(b) for g, b in mamba_block_map.items()},
                    )
                del self._requests_need_load[new_req.req_id]
                continue

            # Multi-snapshot: build a combined path list for the worker
            is_multi = cfg.get("multi", False)
            snapshot_list = None
            if is_multi:
                snapshot_list = cfg.get("snapshots", [])

            num_snap = cfg.get("num_tokens", 0)
            bs = self._composite_block_size

            # block_ids is tuple[list[int], ...] with HMA — one list per group.
            # Use the full-attention group for KV injection.
            all_block_ids = new_req.block_ids
            attn_group = self._full_attn_group_idx

            logger.info(
                "NLS snapshot meta debug: req=%s, num_groups=%d, "
                "block_ids_lens=%s, attn_group=%d, "
                "attn_block_ids=%s, bs=%d",
                new_req.req_id, len(all_block_ids),
                [len(g) for g in all_block_ids], attn_group,
                all_block_ids[attn_group][:5] if attn_group < len(all_block_ids) else "?",
                bs,
            )

            if attn_group >= len(all_block_ids):
                logger.error(
                    "NLS snapshot: attn group %d >= num groups %d",
                    attn_group, len(all_block_ids),
                )
                continue

            group_block_ids = all_block_ids[attn_group]

            # Retrieve register token count from the capture registry
            num_register = _capture_registry.get(
                new_req.req_id, {}
            ).get("num_register", 0)

            total_phantom = num_register + num_snap
            num_blocks_needed = (total_phantom + bs - 1) // bs
            snap_block_ids = group_block_ids[:num_blocks_needed]

            if len(snap_block_ids) < num_blocks_needed:
                logger.error(
                    "NLS snapshot: need %d blocks but only %d in group %d",
                    num_blocks_needed, len(snap_block_ids), attn_group,
                )
                continue

            # Full physical slot mapping for all phantom positions:
            #   [0..num_register-1] = register slots (for streaming scorer)
            #   [num_register..total_phantom-1] = injected memory tokens
            block_offsets = torch.arange(0, bs)
            block_ids_t = torch.tensor(snap_block_ids)
            full_mapping = (
                block_offsets.reshape(1, bs)
                + block_ids_t.reshape(num_blocks_needed, 1) * bs
            ).flatten()
            register_slot_mapping = full_mapping[:num_register]
            slot_mapping = full_mapping[num_register:num_register + num_snap]

            # Build multi-snapshot offset map
            _multi_info = None
            if is_multi and snapshot_list:
                _multi_info = []
                _offset = 0
                for _snap in snapshot_list:
                    _multi_info.append({
                        "path": _snap["path"],
                        "num_tokens": _snap["num_tokens"],
                        "offset": _offset,
                        "strip_prefix": _snap.get("strip_prefix", 0),
                        "ring": _snap.get("ring", "general"),
                        "meta_score": _snap.get("meta_score", 0.0),
                    })
                    _offset += _snap["num_tokens"]

            # Mamba block IDs: all blocks per group (mode=all gives multiple)
            mamba_block_map = {}
            for gidx in self._mamba_group_indices:
                if gidx < len(all_block_ids) and all_block_ids[gidx]:
                    mamba_block_map[gidx] = list(all_block_ids[gidx])
            if mamba_block_map:
                logger.info(
                    "NLS snapshot mamba blocks: req=%s, map=%s",
                    new_req.req_id,
                    {g: len(b) for g, b in mamba_block_map.items()},
                )

            meta.requests.append(SnapshotReqMeta(
                slot_mapping=slot_mapping,
                snapshot_path=cfg.get("path", snapshot_list[0]["path"] if snapshot_list else ""),
                num_snapshot_tokens=num_snap,
                multi_snapshots=_multi_info,
                mamba_block_map=mamba_block_map if mamba_block_map else None,
                register_slot_mapping=(
                    register_slot_mapping if num_register > 0 else None
                ),
                mamba_delta_sum=cfg.get("mamba_delta_sum", 0),
                deltanet_init_path=_dn_path,
                neural_scoring=cfg.get("neural_scoring", True),
                num_register=num_register,
                inject_layout=str(cfg.get("inject_layout", "concat")),
            ))

            logger.info(
                "NLS snapshot meta: req=%s, group=%d, blocks=%d/%d, "
                "register=%d, mem_tokens=%d, total_phantom=%d, "
                "aligned=%d, bs=%d, path=%s",
                new_req.req_id, attn_group,
                num_blocks_needed, len(group_block_ids),
                num_register, num_snap, total_phantom,
                num_blocks_needed * bs, bs, cfg.get("path", "multi-inject"),
            )

        self._requests_need_load.clear()
        self._auto_config = None  # clear stale auto config
        return meta

    # ══════════════════════════════════════════════════════════════
    # WORKER-SIDE
    # ══════════════════════════════════════════════════════════════

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if not _finished_readback_ids:
            return None, None
        done = _finished_readback_ids.copy()
        _finished_readback_ids.clear()
        return done, None

    def _process_pending_captures(self, forward_context: "ForwardContext"):
        """Read KV from paged cache for requests that finished. Prefix-cache safe."""
        if not _pending_readback_captures:
            return

        pending = dict(_pending_readback_captures)
        _pending_readback_captures.clear()

        for req_id, info in pending.items():
            try:
                self._readback_and_save(req_id, info, forward_context)
                _finished_readback_ids.add(req_id)
            except Exception:
                logger.warning(
                    "NLS capture readback failed for %s", req_id[:16],
                    exc_info=True,
                )
                _finished_readback_ids.add(req_id)

    def _readback_and_save(self, req_id: str, info: dict,
                           forward_context: "ForwardContext"):
        """Read KV (attention) + recurrent state (Mamba) from paged cache.

        KL #626 + KL #611: dual-emit per turn. The single readback covers
        BOTH prefill (system + user/tool) AND decode (assistant) positions.
        We slice into:
          - Prefill block: positions ``[cap_start..num_prompt)`` →
            role from ``block_role`` (typically 'user' or 'tool'), with
            cap_start strip removing the system prompt prefix.
          - Decode block:  positions ``[num_prompt..num_prompt+num_decoded)``
            → role 'assistant', no cap_start (decode is past the system
            prompt), chained via ``prev_hash`` to the prefill block hash.

        KL #626 Pass-2 compound mode: when ``info["compound_into"]`` is set,
        the readback only extracts Mamba state and merges it into the
        target session's existing .nls. No new memory entry is created.
        This is what enables the runtime two-pass storage contract for
        the dual-centroid Q-vs-F signal (KL #651).
        """
        # ── Setup: full slot mapping including decode positions ─────────
        block_ids = info["block_ids"]
        mamba_block_ids = info.get("mamba_block_ids", {})
        num_prompt = int(info.get("num_prompt", info.get("num_tokens", 0)))
        num_decoded = max(0, int(info.get("num_decoded", 0)))
        num_phantom = int(info["num_phantom"])
        num_register = int(info.get("num_register", 0))
        bs = self._attn_block_size

        block_offsets = torch.arange(bs)
        block_ids_t = torch.tensor(block_ids)
        num_blocks = len(block_ids)
        full_mapping = (
            block_offsets.reshape(1, bs)
            + block_ids_t.reshape(num_blocks, 1) * bs
        ).flatten()

        # KL #631: total_phantom = register slots + injected memory tokens
        # (both leading the real prompt in the cache layout). Stripping
        # this off post-readback yields the real-token K/V we want to save.
        total_phantom = num_phantom + num_register
        num_total_real = num_prompt + num_decoded
        total_slots_needed = num_total_real + total_phantom

        available_slots = full_mapping.shape[0]
        if total_slots_needed > available_slots:
            # Decode may have allocated additional blocks not yet in our
            # block_ids list, or vice-versa. Clamp and continue with what
            # we can read; the assistant slice may end up shorter than
            # `num_decoded` indicates.
            logger.warning(
                "NLS readback: total_slots_needed=%d > available=%d, clamping",
                total_slots_needed, available_slots,
            )
            total_slots_needed = available_slots
            num_total_real = max(0, total_slots_needed - total_phantom)
            num_decoded = max(0, num_total_real - num_prompt)

        slot_mapping = full_mapping[:total_slots_needed]

        # ── Read all layers into raw_attn_kv + raw_mamba ────────────────
        raw_attn_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        raw_mamba: dict[str, torch.Tensor] = {}
        total_bytes = 0
        attn_layers_read = 0
        mamba_layers_read = 0

        for layer_name in forward_context.no_compile_layers:
            layer = forward_context.no_compile_layers[layer_name]
            kv_cache_attr = getattr(layer, "kv_cache", None)
            if kv_cache_attr is None:
                continue
            layer_idx = _extract_layer_idx(layer_name)
            if layer_idx is None:
                continue
            kv_or_states = (
                kv_cache_attr[0]
                if isinstance(kv_cache_attr, (list, tuple))
                else kv_cache_attr
            )
            # Mamba layer (list/tuple of state tensors)
            if isinstance(kv_or_states, (list, tuple)):
                gidx = self._layer_to_group.get(layer_idx)
                if gidx is None or gidx not in mamba_block_ids:
                    continue
                block_list = mamba_block_ids[gidx]
                state_names = ["conv", "ssm"]
                for si, state_tensor in enumerate(kv_or_states):
                    sname = (
                        state_names[si] if si < len(state_names) else f"s{si}"
                    )
                    try:
                        blocks_data = (
                            state_tensor[block_list].detach().cpu().clone()
                        )
                        raw_mamba[
                            f"layer_{layer_idx}_mamba_{sname}"
                        ] = blocks_data
                        total_bytes += blocks_data.nbytes
                    except Exception as e:
                        logger.warning(
                            "NLS capture mamba L%d %s failed: %s",
                            layer_idx, sname, e,
                        )
                mamba_layers_read += 1
                if mamba_layers_read == 1:
                    logger.info(
                        "NLS capture mamba L%d: group=%d, blocks=%s, "
                        "num_states=%d, shapes=%s",
                        layer_idx, gidx, block_list,
                        len(kv_or_states),
                        [list(s.shape) for s in kv_or_states],
                    )
                continue
            # Attention layer: standard paged KV cache
            kv = kv_or_states
            shape = kv.shape
            device = kv.device
            slots = slot_mapping.to(device=device, dtype=torch.long)
            if len(shape) == 5 and shape[0] == 2:
                page_size = shape[2]
                page_idx = slots // page_size
                offset_idx = slots % page_size
                k_raw = kv[0, page_idx, offset_idx]
                v_raw = kv[1, page_idx, offset_idx]
            elif len(shape) == 4:
                flat = kv.reshape(shape[0] * shape[1], -1)
                kv_both = flat[slots]
                feat = kv_both.shape[-1] // 2
                k_raw = kv_both[:, :feat]
                v_raw = kv_both[:, feat:]
            else:
                k_raw = kv[slots // self._composite_block_size, 0,
                           slots % self._composite_block_size]
                v_raw = kv[slots // self._composite_block_size, 1,
                           slots % self._composite_block_size]
            # Strip leading phantom (register + injected memories), keep
            # the [num_total_real] real positions (prefill + decode).
            if total_phantom > 0:
                k_real = k_raw[total_phantom:].detach().cpu().clone()
                v_real = v_raw[total_phantom:].detach().cpu().clone()
            else:
                k_real = k_raw[:num_total_real].detach().cpu().clone()
                v_real = v_raw[:num_total_real].detach().cpu().clone()
            # Reshape to [N, kv_dim] (matches existing on-disk format)
            k_real = k_real.reshape(k_real.shape[0], -1)
            v_real = v_real.reshape(v_real.shape[0], -1)
            # KL #649: skip DeltaNet pseudo-layers that fall through the
            # attention branch with degenerate (N, 1) shape. Real attention
            # layers have width num_kv_heads * head_dim.
            expected_kv_width = self._num_kv_heads * self._rope_head_dim
            if k_real.shape[-1] != expected_kv_width:
                continue
            raw_attn_kv[layer_idx] = (k_real, v_real)
            total_bytes += k_real.nbytes + v_real.nbytes
            attn_layers_read += 1

        total_layers = attn_layers_read + mamba_layers_read
        if total_layers < 2:
            logger.info(
                "NLS capture readback: only %d layers (attn=%d, mamba=%d), "
                "skipping",
                total_layers, attn_layers_read, mamba_layers_read,
            )
            return

        # ── KL #626 Pass-2: compound merge into existing target ─────────
        compound_into = info.get("compound_into", "")
        if compound_into:
            self._compound_merge_mamba(compound_into, raw_mamba, req_id)
            return

        # ── Normal mode: dual-emit ──────────────────────────────────────
        # Prefill (user/tool) slice in raw_attn_kv coordinates: cap_start
        # strips the system prompt prefix; cap_end (if set) bounds the user
        # content. Default: take all real prompt positions [cap_start..num_prompt).
        cap_start = int(info.get("capture_start", 0))
        cap_end = int(info.get("capture_end", 0))
        prefill_end = cap_end if cap_end > 0 else num_prompt
        prefill_end = min(prefill_end, num_prompt, num_total_real)
        # Resume inject can inflate num_prompt; real prefill length is bounded
        # by prompt_token_ids minus phantom prefix (not phantom KV readback).
        full_prompt_ids = info.get("prompt_token_ids", []) or []
        if full_prompt_ids:
            real_prompt_len = max(0, len(full_prompt_ids) - int(total_phantom or 0))
            if real_prompt_len < prefill_end:
                logger.info(
                    "NLS capture: prefill_end clamped %d -> %d "
                    "(prompt_ids=%d phantom=%d num_prompt=%d)",
                    prefill_end,
                    real_prompt_len,
                    len(full_prompt_ids),
                    total_phantom,
                    num_prompt,
                )
                prefill_end = real_prompt_len
        prefill_start, manifest_rope_start = turn_capture_prefill_slice_start(
            capture_start=cap_start,
            prefill_end=prefill_end,
            resume_stripped_sys=int(info.get("resume_stripped_sys", 0) or 0),
        )

        prefill_role = info.get("block_role", "") or "user"
        prefill_session_id = info.get("session_id", "")

        if is_turn_capture_mode() and prefill_role in ("user", "tool"):
            self._capture_turn_block(
                info=info,
                raw_attn_kv=raw_attn_kv,
                raw_mamba=raw_mamba,
                prefill_start=prefill_start,
                prefill_end=prefill_end,
                cap_start=manifest_rope_start,
                prefill_session_id=prefill_session_id,
                num_prompt=num_prompt,
                num_decoded=num_decoded,
                num_total_real=num_total_real,
                total_phantom=total_phantom,
                mamba_layers_read=mamba_layers_read,
                total_bytes=total_bytes,
                req_id=req_id,
            )
            return

        prefill_path, prefill_block_hash, prefill_real_tokens = self._save_block(
            info=info,
            raw_attn_kv=raw_attn_kv,
            raw_mamba=raw_mamba,
            slice_start=prefill_start,
            slice_end=prefill_end,
            rope_start=cap_start,
            rope_end=prefill_end,
            role=prefill_role,
            session_id=prefill_session_id,
            prev_hash=info.get("prev_hash", ""),
            parent_hash=info.get("parent_hash", ""),
            mamba_layers_read=mamba_layers_read,
            total_bytes=total_bytes,
            req_id=req_id,
        )
        if prefill_path is None:
            return

        # Build the prefill token slice for BM25/semantic indexing. The
        # registry's `prompt_token_ids` is the post-injection sequence
        # `[register_phantom | mem_phantom | system | user]`, so stripping
        # `total_phantom + cap_start` aligns with our K/V slice.
        full_prompt_ids = info.get("prompt_token_ids", [])
        prefill_token_ids: list[int] = []
        if full_prompt_ids:
            tid_start = total_phantom + prefill_start
            tid_end = min(total_phantom + prefill_end, len(full_prompt_ids))
            if tid_end > tid_start:
                prefill_token_ids = list(full_prompt_ids[tid_start:tid_end])

        self._register_block(
            path=prefill_path,
            block_hash=prefill_block_hash,
            real_tokens=prefill_real_tokens,
            info=info,
            role=prefill_role,
            session_id=prefill_session_id,
            token_ids=prefill_token_ids,
            mem_text_override=info.get("memory_text", ""),
            rope_start=cap_start,
        )

        # ── KL #626 plugin-triggered Pass-2 compound (Option A) ─────────
        # Fire-and-forget HTTP loopback for Pass-2 compound merge. Fires
        # right after the prefill block is on disk, REGARDLESS of whether
        # the assistant decode block gets emitted below — short-decode
        # ingest paths (max_tokens=1) produce no assistant block but
        # still benefit from contract-correct compounded Mamba on the
        # prefill block. Recursion + chain-metadata guards live inside
        # `_maybe_fire_loopback_pass2`.
        self._maybe_fire_loopback_pass2(info, prefill_session_id)

        # ── KL #708 + KL #611: system block self-warm ───────────────────
        # If the request carries a `sys_prompt_hash` and we have a system
        # prefix in the readback (cap_start > 0), capture it as its own
        # `role='system'` memory the first time we see this hash. The
        # MemoryStore guard `has_system_block_for_hash` makes this an
        # idempotent no-op for subsequent requests with the same prompt.
        sys_hash = info.get("sys_prompt_hash", "")
        if sys_hash and cap_start > 0:
            self._maybe_self_warm_system_block(
                info=info,
                raw_attn_kv=raw_attn_kv,
                raw_mamba=raw_mamba,
                cap_start=cap_start,
                sys_prompt_hash=sys_hash,
                full_prompt_ids=full_prompt_ids,
                total_phantom=total_phantom,
                mamba_layers_read=mamba_layers_read,
                total_bytes=total_bytes,
                req_id=req_id,
            )

        # ── Decode (assistant) slice ────────────────────────────────────
        # Skip if no real decoded content (max_tokens=1 ingest paths,
        # offline ingestion via prod_toolcall_test.py / ingest_blockchain.py
        # do max_tokens=1 → num_decoded ~ 1 → naturally below threshold).
        _min_decode = int(os.environ.get("NLS_CAPTURE_MIN_DECODE_TOKENS", "4"))
        if num_decoded < _min_decode:
            return

        decode_start = num_prompt
        decode_end = num_prompt + num_decoded
        decode_session_id = (
            info.get("asst_session_id", "")
            or (prefill_session_id + "_asst" if prefill_session_id else "")
        )

        decode_path, decode_block_hash, decode_real_tokens = self._save_block(
            info=info,
            raw_attn_kv=raw_attn_kv,
            raw_mamba=raw_mamba,
            slice_start=decode_start,
            slice_end=decode_end,
            # KL #648 + KL #645: assistant decode is past the system prompt,
            # so rope_start = num_prompt (sentinel >0 to skip global STRIP
            # at injection time). The decode K/V was computed at absolute
            # positions [num_prompt..num_prompt+num_decoded] in this turn's
            # sequence; we record that offset for future RoPE re-rotation.
            rope_start=num_prompt,
            rope_end=decode_end,
            role="assistant",
            session_id=decode_session_id,
            prev_hash=prefill_block_hash,
            parent_hash=info.get("parent_hash", ""),
            mamba_layers_read=mamba_layers_read,
            total_bytes=total_bytes,
            req_id=req_id,
            filename_suffix="_asst",
        )
        if decode_path is None:
            return

        # Assistant tokens come from the request's `output_token_ids`,
        # captured at request_finished_all_groups. This isolates the
        # assistant's text for BM25 without contaminating from the
        # user prompt (which already lives in the prefill block).
        decode_token_ids: list[int] = []
        out_ids = info.get("output_token_ids", []) or []
        if out_ids:
            decode_token_ids = list(out_ids[: num_decoded])

        self._register_block(
            path=decode_path,
            block_hash=decode_block_hash,
            real_tokens=decode_real_tokens,
            info=info,
            role="assistant",
            session_id=decode_session_id,
            token_ids=decode_token_ids,
            mem_text_override="",
            rope_start=num_prompt,
            prev_hash_override=prefill_block_hash,
        )

    def _maybe_fire_loopback_pass2(self, info: dict, prefill_session_id: str) -> None:
        """KL #626 + Option A: fire a localhost HTTP loopback to ourselves
        for the Pass-2 compound merge. No-ops when:
          - The plugin Pass-2 feature is disabled (``NLS_PLUGIN_PASS2=0``).
          - This IS already a Pass-2 (recursion guard via compound_into).
          - The prefill block has no chain metadata (single-shot capture,
            nothing to compound from).
          - We can't find the prior user-block session in the pool.
        """
        if os.environ.get("NLS_PLUGIN_PASS2", "1") == "0":
            return
        if info.get("inject_mode") in ("resume", "resume_overflow"):
            return
        # Recursion guard: a request that was ITSELF a Pass-2 must not
        # spawn another Pass-2 (it would compound a no-op forever).
        if info.get("compound_into"):
            return
        prefill_role = info.get("block_role", "") or "user"
        if prefill_role not in ("user", "tool"):
            return
        base_sid = info.get("base_session_id", "")
        turn_index = int(info.get("turn_index", -1))
        if not base_sid or turn_index < 1:
            # Need explicit chain metadata. Backends that don't pass
            # turn_index (legacy / bare-session captures) get single-pass
            # behavior, which is still a strict improvement over the
            # pre-V4 status quo.
            return

        prev_user_session = self._find_prev_user_session(base_sid, turn_index)
        if not prev_user_session:
            # No prior user block — this is turn 1 of the chain or the
            # prior turn was filtered out. Nothing to seed from.
            return

        memory_text = info.get("memory_text", "")
        if not memory_text:
            # Pass-2 needs the same user content as Pass-1 to evolve
            # Mamba over the same tokens. Fall back to decoding from
            # the captured prompt_token_ids slice; if that's also
            # missing we can't fire safely.
            full = info.get("prompt_token_ids", [])
            tp = int(info.get("num_phantom", 0)) + int(info.get("num_register", 0))
            cs = int(info.get("capture_start", 0))
            np_ = int(info.get("num_prompt", 0))
            if full and tp + np_ <= len(full):
                slice_ids = list(full[tp + cs : tp + np_])
                try:
                    from pri import retrieve as _am
                    tok = getattr(_am, "_tokenizer", None)
                    if tok is not None and slice_ids:
                        memory_text = tok.decode(
                            slice_ids, skip_special_tokens=True
                        )
                except Exception:
                    memory_text = ""
        if not memory_text:
            return

        payload = self._build_pass2_payload(
            info=info,
            target_session=prefill_session_id,
            prev_user_session=prev_user_session,
            memory_text=memory_text,
        )
        # Fire on a daemon thread so the worker readback returns
        # immediately. The loopback request goes through the same vLLM
        # API server we're a part of; uvicorn handles concurrency.
        threading.Thread(
            target=self._do_loopback_pass2,
            args=(payload, prefill_session_id, prev_user_session),
            daemon=True,
            name=f"nls-pass2-{info.get('user_id', '')[:16]}",
        ).start()

    def _find_prev_user_session(
        self, base_session_id: str, turn_index: int,
    ) -> str:
        """Locate the prior turn's user block session_id in the store.

        O(N) scan over `_memories` — fine for typical 1-50 hits per turn.
        Optimization (later): index by `(base_session_id, turn_index, role)`
        in MemoryStore if this becomes a hot path.

        KL #727: defensive search. Originally this matched exactly
        ``turn_index - 1``, but the autoderive layer is currently
        emitting +2 increments per user turn (1, 3, 5, …) instead of
        +1 — most likely because `get_num_new_matched_tokens` runs more
        than once per request and each call advances the chain. Until
        that duplication is rooted out, the exact-match lookup leaves
        Pass-2 silently bailing on EVERY turn because no memory exists
        at the "phantom" odd turn index between two real user blocks.
        Fall back to "the most recent prior user block in this chain"
        so Pass-2 can still find a valid seed. This is also a strict
        improvement once the indexing is fixed: a missing turn (e.g.
        a chitchat-gated message) no longer breaks Mamba compounding
        for the next real turn.
        """
        if turn_index < 1 or not base_session_id:
            return ""
        try:
            from pri import retrieve as auto_memory
        except Exception:
            return ""
        if not auto_memory.is_enabled() or auto_memory._store is None:
            return ""
        best_ti = -1
        best_session = ""
        for m in auto_memory._store._memories:
            if (
                m.role in ("user", "turn")
                and m.base_session_id == base_session_id
                and m.turn_index < turn_index
                and m.turn_index > best_ti
            ):
                best_ti = m.turn_index
                best_session = m.session_id
        return best_session

    def _build_pass2_payload(
        self,
        info: dict,
        target_session: str,
        prev_user_session: str,
        memory_text: str,
    ) -> dict:
        """Construct a Pass-2 loopback payload with byte-identical prefill.

        Original approach (KL #714 v1) used /v1/chat/completions with a
        placeholder system prompt "." — the plugin only had the
        sys_prompt_hash, not the rendered content. That created an
        off-distribution Mamba state because the loopback's prefill
        differed from the primary request's prefill in the system region
        (which the model is most sensitive to).

        Fixed approach (KL #714 v2): replay the exact prompt token IDs
        captured by the registry on the original primary request. We
        strip the phantom prefix (register slots + memory injection
        tokens) so the loopback prefills only the real ``[system | user]``
        tokens — byte-identical to what Pass-1 prefilled. vLLM's
        /v1/completions endpoint accepts ``prompt`` as a list of token
        IDs, bypassing tokenizer/chat-template roundtrips entirely.

        The Mamba state at the end of this prefill, with ``deltanet_init_
        session`` seeding from the prior user turn, is the contract-
        correct compounded Mamba.
        """
        full_prompt_ids = info.get("prompt_token_ids", []) or []
        num_phantom = int(info.get("num_phantom", 0) or 0)
        num_register = int(info.get("num_register", 0) or 0)
        total_phantom = num_phantom + num_register
        if total_phantom > 0 and len(full_prompt_ids) > total_phantom:
            real_prompt_ids = list(full_prompt_ids[total_phantom:])
        else:
            real_prompt_ids = list(full_prompt_ids)

        kv_params: dict = {
            "memory_user": info.get("user_id", "default"),
            "memory_session": target_session,
            "memory_ring": info.get("ring_type", "general"),
            "memory_block_role": info.get("block_role", "") or "user",
            "memory_base_session": info.get("base_session_id", ""),
            "memory_text": memory_text,
            "memory_compound_into": target_session,
            "memory_deltanet_init_session": prev_user_session,
            # Don't register a new memory — the readback's compound
            # branch returns before _save_block runs; this is belt-and-
            # suspenders if a future code path tries to save.
            "memory_no_capture": "1",
            # KL #714 v2: disable retrieval/injection for the Pass-2
            # loopback. The point of Pass-2 is to capture Mamba state
            # at the end of a CLEAN prefill of the original tokens (with
            # Mamba seeded from the prior turn), not to re-run the full
            # retrieval+inject pipeline. Without this, Pass-2 would
            # pull in memories that weren't present at Pass-1, drifting
            # the resulting Mamba off-distribution.
            "memory_off": "1",
        }
        cs = int(info.get("capture_start", 0))
        if cs > 0:
            kv_params["memory_capture_start"] = str(cs)
        sys_hash = info.get("sys_prompt_hash", "")
        if sys_hash:
            kv_params["memory_sys_prompt_hash"] = sys_hash
        ti = int(info.get("turn_index", -1))
        if ti >= 0:
            kv_params["memory_turn_index"] = str(ti)
        ph = info.get("prev_hash", "")
        if ph:
            kv_params["memory_prev_hash"] = ph

        return {
            "model": os.environ.get("NLS_MODEL_PATH") or os.environ.get(
                "MODEL_PATH", "/model"
            ),
            # Raw token IDs guarantee identical prefill. Falls back to
            # the user message text only if real_prompt_ids is empty
            # (registry didn't capture token ids — shouldn't happen).
            "prompt": real_prompt_ids if real_prompt_ids else memory_text,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
            "kv_transfer_params": kv_params,
        }

    def _do_loopback_pass2(
        self,
        payload: dict,
        target_session: str,
        prev_user_session: str,
    ) -> None:
        """Worker thread body for the Pass-2 loopback POST.

        KL #714 v2: posts to /v1/completions (token-id mode) instead of
        /v1/chat/completions. This bypasses chat templating and
        guarantees the prefill is byte-identical to the primary request.
        """
        url = os.environ.get(
            "NLS_LOOPBACK_URL", "http://127.0.0.1:8000",
        ).rstrip("/") + "/v1/completions"
        try:
            import requests as _rq
            r = _rq.post(
                url, json=payload,
                timeout=float(os.environ.get("NLS_PASS2_TIMEOUT", "60")),
            )
            if r.status_code != 200:
                logger.warning(
                    "Plugin Pass-2 loopback (%s ← seed %s): HTTP %d %s",
                    target_session[:32], prev_user_session[:32],
                    r.status_code, r.text[:200],
                )
                return
            # Drain response so the connection releases promptly. The
            # actual merge happens in `_compound_merge_mamba` inside the
            # readback for this same request.
            _ = r.text
            logger.info(
                "Plugin Pass-2 loopback OK: target=%s seed=%s",
                target_session[:32], prev_user_session[:32],
            )
        except Exception as e:
            logger.warning(
                "Plugin Pass-2 loopback EXCEPTION (%s ← %s): %s",
                target_session[:32], prev_user_session[:32], e,
            )

    # ──────────────────────────────────────────────────────────────────
    # KL #626/#611 helpers: per-block save + register, compound merge.
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_garbled_decode(text: str) -> bool:
        from pri.text_quality import is_garbled_response

        return is_garbled_response(text)

    def _decode_token_ids(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        try:
            from pri import retrieve as auto_memory
        except Exception:
            return ""
        tok = getattr(auto_memory, "_tokenizer", None)
        if tok is None and auto_memory._model_path:
            try:
                from transformers import AutoTokenizer

                auto_memory._tokenizer = AutoTokenizer.from_pretrained(
                    auto_memory._model_path, trust_remote_code=True,
                )
                tok = auto_memory._tokenizer
            except Exception:
                return ""
        if tok is None:
            return ""
        try:
            return tok.decode(token_ids, skip_special_tokens=True).strip()
        except Exception:
            return ""

    def _capture_turn_block(
        self,
        *,
        info: dict,
        raw_attn_kv: dict,
        raw_mamba: dict,
        prefill_start: int,
        prefill_end: int,
        cap_start: int,
        prefill_session_id: str,
        num_prompt: int,
        num_decoded: int,
        num_total_real: int,
        total_phantom: int,
        mamba_layers_read: int,
        total_bytes: int,
        req_id: str,
    ) -> None:
        """Save one contiguous user+assistant snapshot per chain turn.

        Replaces dual-emit (separate user / _asst blocks) when
        ``NLS_CHAIN_CAPTURE_MODE=turn``. Mamba at end-of-decode is
        end-of-turn state — Pass-2 compound is skipped.
        """
        _min_decode = int(os.environ.get("NLS_CAPTURE_MIN_DECODE_TOKENS", "4"))
        strip_garbled = os.environ.get("NLS_TURN_STRIP_GARBLED_DECODE", "1") != "0"
        out_ids = info.get("output_token_ids", []) or []
        decode_included = num_decoded >= _min_decode
        if decode_included:
            turn_end = min(num_prompt + num_decoded, num_total_real)
        else:
            turn_end = prefill_end

        decode_garbled_stripped = False
        if decode_included and strip_garbled and out_ids:
            decode_text = self._decode_token_ids(list(out_ids[:num_decoded]))
            if decode_text and self._is_garbled_decode(decode_text):
                turn_end = prefill_end
                decode_included = False
                decode_garbled_stripped = True
                logger.info(
                    "NLS turn capture: garbled decode stripped turn=%s",
                    info.get("turn_index", -1),
                )

        if info.get("prefilled_capture") and decode_garbled_stripped:
            logger.warning(
                "NLS turn capture: refusing prefilled commit turn=%s "
                "(garbled decode stripped; decode=0)",
                info.get("turn_index", -1),
            )
            return

        inject_mode = str(info.get("inject_mode", "") or "").strip().lower()
        turn_idx = int(info.get("turn_index", -1))
        if info.get("resume_inject_aborted"):
            logger.warning(
                "NLS turn capture: refusing resume turn after inject abort "
                "turn=%s session=%s",
                turn_idx,
                (prefill_session_id or "")[:32],
            )
            return
        if resume_turn_requires_inject(inject_mode, turn_idx) and total_phantom <= 0:
            logger.warning(
                "NLS turn capture: refusing resume turn without inject "
                "(phantom=0) turn=%s session=%s",
                turn_idx,
                (prefill_session_id or "")[:32],
            )
            return

        sys_hash = info.get("sys_prompt_hash", "")
        if sys_hash and cap_start > 0:
            full_prompt_ids = info.get("prompt_token_ids", [])
            self._maybe_self_warm_system_block(
                info=info,
                raw_attn_kv=raw_attn_kv,
                raw_mamba=raw_mamba,
                cap_start=cap_start,
                sys_prompt_hash=sys_hash,
                full_prompt_ids=full_prompt_ids,
                total_phantom=total_phantom,
                mamba_layers_read=mamba_layers_read,
                total_bytes=total_bytes,
                req_id=req_id,
            )

        turn_path, turn_hash, turn_tokens = self._save_block(
            info=info,
            raw_attn_kv=raw_attn_kv,
            raw_mamba=raw_mamba,
            slice_start=prefill_start,
            slice_end=turn_end,
            rope_start=cap_start,
            rope_end=turn_end,
            role="turn",
            session_id=prefill_session_id,
            prev_hash=info.get("prev_hash", ""),
            parent_hash=info.get("parent_hash", ""),
            mamba_layers_read=mamba_layers_read,
            total_bytes=total_bytes,
            req_id=req_id,
        )
        if turn_path is None:
            return

        full_prompt_ids = info.get("prompt_token_ids", [])
        turn_token_ids: list[int] = []
        if full_prompt_ids:
            tid_start = total_phantom + prefill_start
            tid_end = min(total_phantom + turn_end, len(full_prompt_ids))
            if tid_end > tid_start:
                turn_token_ids = list(full_prompt_ids[tid_start:tid_end])
        if decode_included and out_ids:
            turn_token_ids.extend(list(out_ids[:num_decoded]))

        self._register_block(
            path=turn_path,
            block_hash=turn_hash,
            real_tokens=turn_tokens,
            info=info,
            role="turn",
            session_id=prefill_session_id,
            token_ids=turn_token_ids,
            mem_text_override=info.get("memory_text", ""),
            rope_start=cap_start,
        )
        logger.info(
            "NLS turn capture: session=%s turn=%s tokens=%d (prefill=%d decode=%d)",
            prefill_session_id,
            info.get("turn_index", -1),
            turn_tokens,
            prefill_end - prefill_start,
            max(0, turn_end - num_prompt),
        )

    def _save_block(
        self,
        info: dict,
        raw_attn_kv: dict,
        raw_mamba: dict,
        slice_start: int,
        slice_end: int,
        rope_start: int,
        rope_end: int,
        role: str,
        session_id: str,
        prev_hash: str,
        parent_hash: str,
        mamba_layers_read: int,
        total_bytes: int,
        req_id: str,
        filename_suffix: str = "",
        conversation_text_override: str = "",
    ) -> tuple[str | None, str, int]:
        """Build save_data for a slice and write the .nls file.

        Returns ``(path, block_hash, real_tokens)`` or ``(None, "", 0)`` on
        failure / empty slice. Mamba state is shared across slices: per
        KL #626, the prefill block's Mamba is provisional (will be replaced
        by the Pass-2 compound merge), and the assistant block's Mamba is
        single-pass-correct because it represents end-of-decode of
        ``[system + user + assistant]``.
        """
        if slice_end <= slice_start:
            return None, "", 0
        real_tokens = slice_end - slice_start

        save_data: dict[str, torch.Tensor] = {}
        for layer_idx, (k_full, v_full) in raw_attn_kv.items():
            save_data[f"layer_{layer_idx}_k"] = k_full[slice_start:slice_end]
            save_data[f"layer_{layer_idx}_v"] = v_full[slice_start:slice_end]
        for k, v in raw_mamba.items():
            save_data[k] = v
        save_data["_meta_seq_len"] = torch.tensor([real_tokens])
        save_data["_meta_has_mamba"] = torch.tensor([mamba_layers_read])

        ts_ms = int(time.time() * 1000)
        filepath = (
            self._kv_snapshot_capture_dir
            / f"kv_snapshot_{ts_ms}{filename_suffix}.nls"
        )

        extra: dict = {
            "user_id": info.get("user_id", "default"),
            "session_id": session_id,
            "ring_type": info.get("ring_type", "general"),
            "req_id": req_id,
            "num_tokens": int(real_tokens),
        }
        if rope_start > 0 or rope_end > 0:
            extra["rope_start"] = int(rope_start)
            extra["rope_end"] = int(
                rope_end if rope_end > 0 else rope_start + real_tokens
            )
        if role:
            extra["role"] = role
        if parent_hash:
            extra["parent_hash"] = parent_hash
        if prev_hash:
            extra["prev_hash"] = prev_hash
        turn_index = info.get("turn_index", -1)
        base_sid = info.get("base_session_id", "")
        if turn_index >= 0:
            extra["turn_index"] = turn_index
        if base_sid:
            extra["base_session_id"] = base_sid
        sys_prompt_hash = info.get("sys_prompt_hash", "")
        if sys_prompt_hash:
            extra["sys_prompt_hash"] = sys_prompt_hash
        capture_phantom = 0
        ti = int(turn_index if turn_index is not None else -1)
        uid = info.get("user_id", "default")
        if role == "turn" and base_sid and ti >= 0:
            try:
                from pri import retrieve as auto_memory
                from pri.resume import chain_pack_phantom_before_turn

                if auto_memory.is_enabled() and auto_memory._store is not None:
                    capture_phantom = chain_pack_phantom_before_turn(
                        auto_memory._store,
                        uid,
                        base_sid,
                        ti,
                    )
            except Exception:
                capture_phantom = 0
        if capture_phantom > 0:
            extra["capture_num_phantom"] = capture_phantom
        # The user/tool block carries the user-provided memory_text.
        # The assistant block's text is decoded from output_token_ids
        # by `_register_block`; do NOT inherit the user prompt as
        # conversation_text on the assistant manifest. The system block
        # gets the rendered system prompt text via the explicit
        # `conversation_text_override` (it's not in info["memory_text"],
        # which carries the USER message).
        if conversation_text_override:
            extra["conversation_text"] = conversation_text_override
        elif role not in ("assistant", "system"):
            mem_text = info.get("memory_text", "")
            if mem_text:
                extra["conversation_text"] = mem_text

        try:
            from pri.format import save_nls
            file_size = save_nls(save_data, filepath, extra_manifest=extra)
        except Exception:
            logger.error(
                "NLS capture save FAILED: %s", filepath, exc_info=True,
            )
            return None, "", 0

        block_hash = extra.get("block_hash", "")

        # KL #625: session_path_cache drives runtime DeltaNet seeding.
        # Update only when the block has a session_id; the assistant
        # block uses its own derived id so prefill/decode don't collide.
        if session_id:
            self._session_path_cache[session_id] = str(filepath)

        logger.info(
            "NLS capture saved [%s slice=%d:%d]: req=%s, %d tokens, "
            "%.1f KB -> %s",
            role, slice_start, slice_end, req_id[:16],
            real_tokens, file_size / 1024, filepath,
        )
        return str(filepath), block_hash, real_tokens

    def _register_block(
        self,
        path: str,
        block_hash: str,
        real_tokens: int,
        info: dict,
        role: str,
        session_id: str,
        token_ids: list[int],
        mem_text_override: str,
        rope_start: int,
        prev_hash_override: str | None = None,
    ) -> None:
        """Register a saved block with MemoryStore (BM25, semantic, delta).

        Applies the KL #651 chitchat capture gate (drops noise like 'Hey',
        'ping', 'ok'). For the assistant block, ``token_ids`` should be
        the decoded tokens (from request.output_token_ids), so BM25 only
        indexes the assistant's text and not the user prompt.
        """
        try:
            from pri import retrieve as auto_memory
        except Exception:
            return

        if not auto_memory.is_enabled() or auto_memory._store is None:
            return
        if not token_ids:
            return

        # Decode token_ids → text for manifest and semantic embedding.
        # If the caller provided memory_text (chat-mode user block), use
        # that verbatim. Otherwise tokenizer-decode for assistant blocks
        # and for chat-mode prefill where memory_text wasn't sent.
        mem_text = mem_text_override
        if not mem_text:
            tok = getattr(auto_memory, "_tokenizer", None)
            if tok is None and auto_memory._model_path:
                try:
                    from transformers import AutoTokenizer
                    auto_memory._tokenizer = AutoTokenizer.from_pretrained(
                        auto_memory._model_path, trust_remote_code=True,
                    )
                    tok = auto_memory._tokenizer
                except Exception:
                    pass
            if tok is not None:
                try:
                    mem_text = tok.decode(token_ids, skip_special_tokens=True)
                except Exception:
                    mem_text = ""

        store = auto_memory._store

        # KL #639/#651: language-agnostic Q-vs-F regex meta-score, applied
        # at retrieval as a debuff and here as a hard chitchat gate.
        from pri.store import compute_meta_score
        m_score = compute_meta_score(
            mem_text or "",
            role="user" if role == "turn" else role,
        )

        _capture_meta_max = float(
            os.environ.get("NLS_CAPTURE_META_MAX", "0.95")
        )
        _capture_min_words = int(
            os.environ.get("NLS_CAPTURE_MIN_WORDS", "4")
        )
        word_count = len((mem_text or "").split())
        if (
            _capture_meta_max > 0
            and m_score >= _capture_meta_max
            and word_count < _capture_min_words
        ):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
            logger.info(
                "NLS capture SKIPPED [%s] (chitchat: meta=%.2f, words=%d): %s",
                role, m_score, word_count,
                (mem_text[:60] + "...") if len(mem_text) > 60 else mem_text,
            )
            return

        try:
            mem_id = store.add(
                token_ids=token_ids,
                kv_path=path,
                num_tokens=real_tokens,
                ring_type=info.get("ring_type", "general"),
                user_id=info.get("user_id", "default"),
                session_id=session_id,
                source="readback_capture",
                description=mem_text or "",
                role=role,
                block_hash=block_hash,
                parent_hash=info.get("parent_hash", ""),
                prev_hash=(
                    prev_hash_override
                    if prev_hash_override is not None
                    else info.get("prev_hash", "")
                ),
                turn_index=info.get("turn_index", -1),
                base_session_id=info.get("base_session_id", ""),
                rope_start=int(rope_start),
                meta_score=m_score,
                sys_prompt_hash=info.get("sys_prompt_hash", ""),
                is_compaction_context=bool(info.get("compaction_detected", False)),
            )
            # KL #726 — feed CLEAN user text to BM25, not the full wire.
            # The wire `token_ids` for a user/tool block contains the
            # whole prompt prefix (system + tool catalog + history),
            # which makes every user memory's tf-vector ~95% identical
            # and renders BM25 useless for differentiating turns. The
            # `mem_text` here is the punkrecords-supplied verbatim user
            # content (or, for fallbacks, a tokenizer-decoded copy of
            # token_ids — still better than nothing). Re-tokenize it
            # and feed THOSE IDs to BM25 so the keyword index actually
            # discriminates between "I have two kids" and "what's my name".
            # KV-cache capture (the .nls file) is untouched — it still
            # owns the full wire span needed for replay/inject.
            bm25_token_ids = token_ids
            bm25_turn_lists = auto_memory._split_turns(token_ids)
            if mem_text:
                tok = getattr(auto_memory, "_tokenizer", None)
                if tok is None and getattr(auto_memory, "_model_path", None):
                    try:
                        from transformers import AutoTokenizer
                        auto_memory._tokenizer = AutoTokenizer.from_pretrained(
                            auto_memory._model_path, trust_remote_code=True,
                        )
                        tok = auto_memory._tokenizer
                    except Exception:
                        tok = None
                if tok is not None:
                    try:
                        clean_ids = tok.encode(
                            mem_text, add_special_tokens=False,
                        )
                        if len(clean_ids) >= 3:
                            bm25_token_ids = clean_ids
                            bm25_turn_lists = [clean_ids]
                    except Exception:
                        pass
            store.add_bm25_data(
                key=session_id or mem_id,
                token_ids=bm25_token_ids,
                turn_texts_token_ids=bm25_turn_lists,
            )
            if mem_text:
                store.update_semantic_embedding(mem_id, mem_text)
            try:
                store.update_delta_energy(mem_id)
            except Exception as _e:
                logger.debug("update_delta_energy failed: %s", _e)
            logger.info(
                "NLS capture registered [%s]: id=%s, tokens=%d, store=%d, "
                "meta=%.2f, text=%s",
                role, mem_id, real_tokens, store.size, m_score,
                (mem_text[:60] + "...") if mem_text and len(mem_text) > 60
                else mem_text or "",
            )
        except Exception as e:
            logger.debug("Auto-memory registration (readback %s): %s", role, e)

    def _maybe_self_warm_system_block(
        self,
        info: dict,
        raw_attn_kv: dict,
        raw_mamba: dict,
        cap_start: int,
        sys_prompt_hash: str,
        full_prompt_ids: list,
        total_phantom: int,
        mamba_layers_read: int,
        total_bytes: int,
        req_id: str,
    ) -> None:
        """KL #708 + KL #611: capture the system-prompt prefix as its own
        ``role='system'`` block on the first request after a deploy that
        carries a ``memory_sys_prompt_hash`` we haven't stored yet.

        Subsequent requests with the same hash see the existing block via
        ``has_system_block_for_hash`` and short-circuit. The system block
        is stored alongside the user/tool/assistant blocks but excluded
        from search by ``NLS_ROLE_FILTER='user,tool'`` — its purpose is
        future cross-request reuse, content-addressed integrity, and a
        canonical anchor for the dual-centroid Q-vs-F signal."""
        try:
            from pri import retrieve as auto_memory
        except Exception:
            return
        if not auto_memory.is_enabled() or auto_memory._store is None:
            return
        store = auto_memory._store

        try:
            if store.has_system_block_for_hash(sys_prompt_hash):
                return
        except Exception:
            return

        # Sanity: the system slice covers positions [0..cap_start) in
        # raw_attn_kv coordinates. Skip if any layer's K is shorter
        # than expected (defensive — shouldn't happen with real
        # readbacks but guards against partial captures).
        any_layer = next(iter(raw_attn_kv.values()), None)
        if any_layer is None:
            return
        if any_layer[0].shape[0] < cap_start:
            return

        sys_session_id = f"__system_{sys_prompt_hash}__"
        # Synthetic info dict for the system block: drop chain metadata
        # (system blocks aren't part of any user's chain) and use the
        # synthetic session id so the dedup hash + session_path_cache
        # entry stay isolated from user-block sessions.
        sys_info = dict(info)
        sys_info["session_id"] = sys_session_id
        sys_info["base_session_id"] = ""
        sys_info["turn_index"] = -1
        sys_info["prev_hash"] = ""
        sys_info["parent_hash"] = ""

        # Token slice for BM25 / semantic indexing AND for detokenizing
        # the rendered system prompt text. Positions [0..cap_start) of
        # the post-phantom token stream.
        sys_token_ids: list[int] = []
        if full_prompt_ids and cap_start > 0:
            tid_start = total_phantom
            tid_end = min(total_phantom + cap_start, len(full_prompt_ids))
            if tid_end > tid_start:
                sys_token_ids = list(full_prompt_ids[tid_start:tid_end])

        # Detokenize so we can stash the rendered system prompt text in
        # the manifest's `conversation_text`. Useful for debugging and
        # any future code path that needs to recover the prompt without
        # rerunning a tokenizer.
        sys_text = ""
        try:
            from pri import retrieve as auto_memory
            tok = getattr(auto_memory, "_tokenizer", None)
            if tok is not None and sys_token_ids:
                sys_text = tok.decode(sys_token_ids, skip_special_tokens=True)
        except Exception:
            sys_text = ""

        # System blocks cover positions [0..cap_start) in the original
        # sequence. rope_start=0 in their manifest means "this memory
        # IS the system prefix" — the inject path is free to either
        # use it as a canonical system region or strip it (per
        # NLS_STRIP_INJECT_SYS_BLOCK_LEN) just like any other capture.
        sys_path, sys_block_hash, sys_real_tokens = self._save_block(
            info=sys_info,
            raw_attn_kv=raw_attn_kv,
            raw_mamba=raw_mamba,
            slice_start=0,
            slice_end=cap_start,
            rope_start=0,
            rope_end=cap_start,
            role="system",
            session_id=sys_session_id,
            prev_hash="",
            parent_hash="",
            mamba_layers_read=mamba_layers_read,
            total_bytes=total_bytes,
            req_id=req_id,
            filename_suffix="_sys",
            conversation_text_override=sys_text,
        )
        if sys_path is None:
            return

        self._register_block(
            path=sys_path,
            block_hash=sys_block_hash,
            real_tokens=sys_real_tokens,
            info=sys_info,
            role="system",
            session_id=sys_session_id,
            token_ids=sys_token_ids,
            mem_text_override=sys_text,
            rope_start=0,
        )
        logger.info(
            "NLS system-block self-warmed: hash=%s tokens=%d -> %s",
            sys_prompt_hash, sys_real_tokens, sys_path,
        )

    def _compound_merge_mamba(
        self,
        target_session_id: str,
        raw_mamba: dict,
        req_id: str,
    ) -> None:
        """KL #626 Pass-2: replace Mamba state in target session's existing
        .nls with the compounded state captured by this request's readback.

        No new memory entry is created. Mirrors the offline post-processing
        ``merge_mamba_into_kv`` from ``scripts/ingest_blockchain.py`` but
        runs in-process at the end of the Pass-2 readback.
        """
        target_path = self._session_path_cache.get(target_session_id, "")
        if not target_path:
            target_path = self._find_kv_path_by_session(target_session_id)
        if not target_path or not Path(target_path).exists():
            logger.warning(
                "NLS compound merge: target session=%s not found "
                "(cache=%s, disk_scan=%s)",
                target_session_id,
                bool(self._session_path_cache.get(target_session_id, "")),
                bool(target_path),
            )
            return

        try:
            from pri.format import (
                load_nls, save_nls, read_manifest,
            )
            target_manifest = read_manifest(target_path)
            target_data = load_nls(target_path)
        except Exception:
            logger.error(
                "NLS compound merge: failed to load %s",
                target_path, exc_info=True,
            )
            return

        replaced = 0
        for k, v in raw_mamba.items():
            if "_mamba_" in k:
                target_data[k] = v
                replaced += 1

        # Preserve all the manifest fields except those that save_nls
        # recomputes (per ingest_blockchain.merge_mamba_into_kv).
        extra = {
            k: v for k, v in (target_manifest or {}).items()
            if k not in (
                "version", "num_keys", "created_at", "block_hash",
                "attn_layers", "mamba_layers", "seq_len", "has_mamba",
            )
        }
        extra["two_pass_merged"] = True

        try:
            save_nls(target_data, target_path, extra_manifest=extra)
            logger.info(
                "NLS compound merge: req=%s, target_session=%s, "
                "replaced %d Mamba keys -> %s",
                req_id[:16], target_session_id, replaced, target_path,
            )
            # Drop snapshot cache so next load picks up the new Mamba.
            self._snapshot_cache.pop(target_path, None)
            # KL #736 piece 2: target_data + raw_mamba just went out of
            # scope (refs were inside this method's frame). Hint the
            # allocator so the dequant tensors don't linger as cached
            # blocks across the next agentic turn's prefill.
            self._release_cuda_cache_if_due(reason="compound_merge")
        except Exception:
            logger.error(
                "NLS compound merge save FAILED: %s",
                target_path, exc_info=True,
            )

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        # Process any pending post-completion captures first (prefix-cache safe)
        if _pending_readback_captures:
            self._process_pending_captures(forward_context)

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, NLSSnapshotMetadata) or not metadata.requests:
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("NLS snapshot: attn_metadata is None, skipping")
            return

        for req_meta in metadata.requests:
            t0 = time.perf_counter()

            # KL #625: DeltaNet-init only — seed Mamba, skip KV
            if req_meta.deltanet_init_path and req_meta.num_snapshot_tokens == 0:
                self._inject_deltanet_init_from_meta(
                    forward_context, req_meta
                )
                continue

            # Multi-snapshot: merge KV from multiple files
            if req_meta.multi_snapshots:
                _resume_layout = str(
                    getattr(req_meta, "inject_layout", "concat") or "concat"
                )
                snapshot = self._load_multi_snapshots(
                    req_meta.multi_snapshots,
                    mamba_delta_sum=req_meta.mamba_delta_sum,
                    rope_offset=req_meta.num_register,
                    inject_layout=_resume_layout,
                )
                if snapshot is None:
                    logger.error("NLS snapshot: failed to load multi-snapshots")
                    continue
            else:
                snapshot = self._load_snapshot(req_meta.snapshot_path)
                if snapshot is None:
                    logger.error(
                        "NLS snapshot: failed to load %s", req_meta.snapshot_path
                    )
                    continue
                # KL #652 follow-up: the multi-snapshot path applies
                # `rope_offset = num_register` so memory K vectors are
                # rotated to their physical position in the new request.
                # The single-snapshot fallback was historically missing
                # this — K vectors carried a `-num_register` phase bias
                # which manifested as token-loop degeneration past
                # ~1700 phantom tokens. Apply the same correction here.
                #
                # We mutate a shallow copy of the snapshot dict so the
                # _snapshot_cache stays clean for future requests with
                # different num_register values.
                _rope_offset = int(getattr(req_meta, "num_register", 0))
                if _rope_offset > 0:
                    rope_old = 0
                    if req_meta.snapshot_path.endswith(".nls"):
                        try:
                            from pri.format import (
                                read_manifest as _rm,
                            )
                            _man = _rm(req_meta.snapshot_path)
                            if _man:
                                rope_old = int(_man.get("rope_start", 0))
                        except Exception:
                            rope_old = 0
                    snapshot = self._rerotate_snapshot_keys(
                        snapshot,
                        old_start=rope_old,
                        new_start=_rope_offset,
                        seq_len=req_meta.num_snapshot_tokens,
                    )

            slot_mapping = req_meta.slot_mapping
            num_tokens = req_meta.num_snapshot_tokens
            layers_injected = 0
            first_shape_logged = False

            # Log ALL available layers and attn_metadata on first inject
            if not hasattr(self, '_diag_logged'):
                self._diag_logged = True
                layer_names = sorted(forward_context.no_compile_layers.keys())
                logger.info(
                    "NLS DIAG: %d layers in no_compile_layers: %s",
                    len(layer_names), layer_names[:6],
                )
                # Classify layers by type
                mamba_names = []
                attn_names = []
                for ln in layer_names:
                    lyr = forward_context.no_compile_layers[ln]
                    kv_a = getattr(lyr, "kv_cache", None)
                    if kv_a is None:
                        continue
                    inner = kv_a[0] if isinstance(kv_a, (list, tuple)) else kv_a
                    if isinstance(inner, (list, tuple)):
                        mamba_names.append(ln)
                    else:
                        attn_names.append(ln)
                logger.info(
                    "NLS DIAG layer types: %d mamba, %d attn",
                    len(mamba_names), len(attn_names),
                )
                if mamba_names:
                    lyr = forward_context.no_compile_layers[mamba_names[0]]
                    kv_a = lyr.kv_cache[0]
                    logger.info(
                        "NLS DIAG mamba[0] '%s': %d state tensors, shapes=%s",
                        mamba_names[0], len(kv_a),
                        [list(t.shape) for t in kv_a],
                    )
                if isinstance(attn_metadata, dict):
                    # Log mamba metadata if present
                    mamba_keys = [
                        k for k in attn_metadata
                        if "mamba" in k.lower() or "linear_attn" in k.lower()
                    ]
                    if mamba_keys:
                        mval = attn_metadata[mamba_keys[0]]
                        logger.info(
                            "NLS DIAG mamba_meta[%s]: has_initial_states_p=%s, "
                            "prep_initial_states=%s",
                            mamba_keys[0],
                            getattr(mval, 'has_initial_states_p', '?'),
                            getattr(mval, 'prep_initial_states', '?'),
                        )
                    attn_keys = [
                        k for k in attn_metadata
                        if "self_attn" in k or "layers.3" in k
                    ]
                    if not attn_keys:
                        attn_keys = sorted(attn_metadata.keys())[:2]
                    for mkey in attn_keys[:2]:
                        mval = attn_metadata[mkey]
                        attrs = {}
                        for a in dir(mval):
                            if a.startswith('_'):
                                continue
                            try:
                                v = getattr(mval, a)
                                if callable(v):
                                    continue
                                if isinstance(v, torch.Tensor):
                                    s = f"T{list(v.shape)}"
                                    if v.numel() <= 10:
                                        s += f"={v.tolist()}"
                                    elif v.numel() <= 50:
                                        s += f"={v[:10].tolist()}..."
                                    attrs[a] = s
                                elif isinstance(v, (int, float, bool, str)):
                                    attrs[a] = v
                            except Exception:
                                pass
                        logger.info(
                            "NLS DIAG attn_meta[%s]: %s", mkey, attrs
                        )

            # ── Neural scoring: build region metadata from multi-snapshot info ──
            _do_neural_scoring = (
                _neural is not None
                and _neural.is_enabled()
                and req_meta.neural_scoring
                and req_meta.multi_snapshots is not None
                and len(req_meta.multi_snapshots) > 1
            )
            _neural_regions: list[dict] = []
            if _do_neural_scoring:
                for snap_info in req_meta.multi_snapshots:
                    _neural_regions.append({
                        "start": snap_info["offset"],
                        "end": snap_info["offset"] + snap_info["num_tokens"],
                        "num_tokens": snap_info["num_tokens"],
                        "kv_path": snap_info["path"],
                        "ring_type": snap_info.get("ring", "general"),
                        "meta_score": snap_info.get("meta_score", 0.0),
                        "always_inject": snap_info.get("ring", "") in (
                            "identity", "behavioral", "consolidation",
                        ),
                    })
                _neural.begin_scoring(
                    regions=_neural_regions,
                    total_phantom=num_tokens,
                    num_kv_heads=self._num_kv_heads,
                    head_dim=self._rope_head_dim,
                )
                # Wire paged-cache handles so _try_suppress_now() can zero V
                # in the cache once scoring resolves. Opt-in via NLS_V_SUPPRESSION.
                if hasattr(_neural, "set_suppression_context"):
                    _neural.set_suppression_context(
                        forward_context=forward_context,
                        slot_mapping=slot_mapping,
                        composite_block_size=self._composite_block_size,
                    )

            mamba_layers_injected = 0
            mamba_first_logged = False

            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue

                layer_idx = _extract_layer_idx(layer_name)
                if layer_idx is None:
                    continue

                # vLLM 0.19+ dropped forward_context.virtual_engine; default to 0.
                _ve = getattr(forward_context, "virtual_engine", 0)
                kv_or_states = (
                    kv_cache_attr[_ve]
                    if isinstance(kv_cache_attr, (list, tuple))
                    else kv_cache_attr
                )

                # ── Mamba layer: inject recurrent state ──
                if isinstance(kv_or_states, (list, tuple)):
                    if len(kv_or_states) < 2:
                        continue
                    conv_key = f"layer_{layer_idx}_mamba_conv"
                    ssm_key = f"layer_{layer_idx}_mamba_ssm"
                    if conv_key not in snapshot or ssm_key not in snapshot:
                        continue
                    gidx = self._layer_to_group.get(layer_idx)
                    mbm = req_meta.mamba_block_map
                    if gidx is None or mbm is None or gidx not in mbm:
                        continue
                    block_list = mbm[gidx]
                    try:
                        conv_cache = kv_or_states[0]
                        ssm_cache = kv_or_states[1]
                        snap_conv = snapshot[conv_key].to(
                            device=conv_cache.device, dtype=conv_cache.dtype,
                        )
                        snap_ssm = snapshot[ssm_key].to(
                            device=ssm_cache.device, dtype=ssm_cache.dtype,
                        )
                        # Match captured blocks to allocated blocks
                        n_blocks = min(len(block_list), snap_conv.shape[0])
                        for bi in range(n_blocks):
                            conv_cache[block_list[bi]] = snap_conv[bi]
                            ssm_cache[block_list[bi]] = snap_ssm[bi]
                        mamba_layers_injected += 1
                        if not mamba_first_logged:
                            mamba_first_logged = True
                            logger.info(
                                "NLS mamba inject L%d: group=%d, "
                                "blocks=%d/%d, conv=%s->%s, ssm=%s->%s",
                                layer_idx, gidx, n_blocks, len(block_list),
                                list(snap_conv.shape),
                                list(conv_cache.shape),
                                list(snap_ssm.shape),
                                list(ssm_cache.shape),
                            )
                    except Exception as e:
                        logger.warning(
                            "NLS mamba inject L%d failed: %s",
                            layer_idx, e, exc_info=True,
                        )
                    continue

                # ── Attention layer: inject paged KV ──
                k_key = f"layer_{layer_idx}_k"
                v_key = f"layer_{layer_idx}_v"
                if k_key not in snapshot or v_key not in snapshot:
                    continue

                kv_cache_layer = kv_or_states
                k_snap = snapshot[k_key]
                v_snap = snapshot[v_key]

                # ── Neural scorer: cache per-region K slices ──
                if _do_neural_scoring and layer_idx in _neural.SCORE_LAYERS:
                    k_per_region = []
                    for region in _neural_regions:
                        start = region["start"]
                        end = region["end"]
                        k_per_region.append(k_snap[start:end].clone())
                    _neural.cache_injected_k(layer_idx, k_per_region)

                if not first_shape_logged:
                    logger.info(
                        "NLS snapshot inject debug: layer=%s (idx=%d), "
                        "kv_cache shape=%s dtype=%s, k_snap=%s, v_snap=%s, "
                        "slot_mapping=%s (min=%d max=%d)",
                        layer_name, layer_idx, list(kv_cache_layer.shape),
                        kv_cache_layer.dtype, list(k_snap.shape),
                        list(v_snap.shape), list(slot_mapping.shape),
                        slot_mapping.min().item(), slot_mapping.max().item(),
                    )
                    first_shape_logged = True

                self._inject_kv(
                    kv_cache_layer, k_snap, v_snap,
                    slot_mapping, num_tokens, attn_metadata, layer_name,
                )
                layers_injected += 1

                if layers_injected == 1:
                    try:
                        shape = kv_cache_layer.shape
                        dev_slots = slot_mapping[:5].to(
                            device=kv_cache_layer.device, dtype=torch.long,
                        )
                        if len(shape) == 5 and shape[0] == 2:
                            page_sz = shape[2]
                            pg = dev_slots // page_sz
                            off = dev_slots % page_sz
                            rb_k = kv_cache_layer[0, pg, off].float()
                            rb_v = kv_cache_layer[1, pg, off].float()
                            rb_k_flat = rb_k.reshape(5, -1)
                            rb_v_flat = rb_v.reshape(5, -1)
                            sn_k = k_snap[:5].float().to(rb_k.device)
                            sn_v = v_snap[:5].float().to(rb_v.device)
                            sn_k_flat = sn_k.reshape(5, -1)
                            sn_v_flat = sn_v.reshape(5, -1)
                            k_diff = (rb_k_flat - sn_k_flat).abs().max().item()
                            v_diff = (rb_v_flat - sn_v_flat).abs().max().item()
                            k_nz = (rb_k_flat.abs() > 1e-6).sum().item()
                            logger.info(
                                "NLS DIAG readback L%d: "
                                "k_maxdiff=%.4f v_maxdiff=%.4f "
                                "k_nonzero=%d/%d rb_k_norm=%.2f "
                                "snap_k_norm=%.2f page_sz=%d "
                                "pages=%s offsets=%s contig=%s",
                                layer_idx, k_diff, v_diff,
                                k_nz, rb_k_flat.numel(),
                                rb_k_flat.norm().item(),
                                sn_k_flat.norm().item(),
                                page_sz, pg.tolist(), off.tolist(),
                                kv_cache_layer.is_contiguous(),
                            )
                    except Exception as e:
                        logger.warning(
                            "NLS DIAG readback failed: %s", e,
                            exc_info=True,
                        )

            dt = time.perf_counter() - t0
            logger.info(
                "NLS snapshot injected: attn=%d mamba=%d layers, "
                "%d tokens, %.1f ms, path=%s",
                layers_injected, mamba_layers_injected,
                num_tokens, dt * 1000, req_meta.snapshot_path,
            )

    def _inject_deltanet_init_from_meta(
        self,
        forward_context: "ForwardContext",
        req_meta: "SnapshotReqMeta",
    ):
        """KL #625: Inject Mamba-only state for DeltaNet compounding.

        Loads conv/ssm states from a previous .nls file and writes them
        into the Mamba cache at the correct block positions. No KV injection,
        no phantom tokens — the model processes its prompt normally, but
        DeltaNet layers start from the accumulated state rather than zeros.
        """
        nls_path = req_meta.deltanet_init_path
        mbm = req_meta.mamba_block_map

        def _cleanup_registry():
            _orig = req_meta.deltanet_init_path
            for rid in list(_deltanet_init_registry.keys()):
                if _deltanet_init_registry.get(rid) == _orig:
                    _deltanet_init_registry.pop(rid, None)
                    break

        if not nls_path or not mbm:
            _cleanup_registry()
            return

        # Resolve session: references to actual file paths
        if nls_path.startswith("session:"):
            session_id = nls_path[len("session:"):]
            nls_path = self._resolve_latest_nls(session_id)
            if not nls_path:
                logger.warning(
                    "DeltaNet-init: could not resolve session '%s' to .nls file",
                    session_id,
                )
                _cleanup_registry()
                return

        try:
            from pri.format import load_nls
            snapshot = load_nls(nls_path)
        except Exception as e:
            logger.warning(
                "DeltaNet-init: failed to load %s: %s", nls_path, e
            )
            _cleanup_registry()
            return

        mamba_injected = 0
        for layer_name in forward_context.no_compile_layers:
            layer = forward_context.no_compile_layers[layer_name]
            kv_cache_attr = getattr(layer, "kv_cache", None)
            if kv_cache_attr is None:
                continue

            layer_idx = _extract_layer_idx(layer_name)
            if layer_idx is None:
                continue

            _ve2 = getattr(forward_context, "virtual_engine", 0)
            kv_or_states = (
                kv_cache_attr[_ve2]
                if isinstance(kv_cache_attr, (list, tuple))
                else kv_cache_attr
            )

            if not isinstance(kv_or_states, (list, tuple)):
                continue
            if len(kv_or_states) < 2:
                continue

            conv_key = f"layer_{layer_idx}_mamba_conv"
            ssm_key = f"layer_{layer_idx}_mamba_ssm"
            if conv_key not in snapshot or ssm_key not in snapshot:
                continue

            gidx = self._layer_to_group.get(layer_idx)
            if gidx is None or gidx not in mbm:
                continue
            block_list = mbm[gidx]

            try:
                conv_cache = kv_or_states[0]
                ssm_cache = kv_or_states[1]
                snap_conv = snapshot[conv_key].to(
                    device=conv_cache.device, dtype=conv_cache.dtype,
                )
                snap_ssm = snapshot[ssm_key].to(
                    device=ssm_cache.device, dtype=ssm_cache.dtype,
                )
                n_blocks = min(len(block_list), snap_conv.shape[0])
                for bi in range(n_blocks):
                    conv_cache[block_list[bi]] = snap_conv[bi]
                    ssm_cache[block_list[bi]] = snap_ssm[bi]
                mamba_injected += 1
            except Exception as e:
                logger.warning(
                    "DeltaNet-init L%d failed: %s", layer_idx, e
                )

        _cleanup_registry()

        # Mark seeded block indices so cache clearing is skipped during prefill.
        # Cap the set to prevent unbounded growth (blocks are likely consumed
        # before the clear function runs for a different request, making the
        # protection a safety net rather than the primary mechanism).
        if mamba_injected > 0:
            if len(_deltanet_seeded_blocks) > 256:
                _deltanet_seeded_blocks.clear()
            for gidx, block_list in mbm.items():
                for bi in block_list:
                    _deltanet_seeded_blocks.add(int(bi))

        logger.info(
            "DeltaNet-init: %d Mamba layers seeded from %s (protected=%d blocks)",
            mamba_injected, Path(nls_path).name,
            len(_deltanet_seeded_blocks),
        )

    def _inject_kv(
        self,
        dst_kv_cache: torch.Tensor,
        k_snap: torch.Tensor,
        v_snap: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_tokens: int,
        attn_metadata: AttentionMetadata,
        layer_name: str,
    ) -> None:
        device = dst_kv_cache.device
        dtype = dst_kv_cache.dtype

        k = k_snap[:num_tokens].to(device=device, dtype=dtype)
        v = v_snap[:num_tokens].to(device=device, dtype=dtype)

        # KL #630: Memory salience amplification
        if KV_K_SCALE != 1.0:
            k = k * KV_K_SCALE
        if KV_V_SCALE != 1.0:
            v = v * KV_V_SCALE

        slots = slot_mapping[:num_tokens].to(device=device, dtype=torch.long)
        shape = dst_kv_cache.shape

        if len(shape) == 5 and shape[0] == 2:
            # Layout: [2, num_pages, page_size, ...]
            # Flatten pages and offsets, keep K/V and feature dims.
            num_pages = shape[1]
            page_size = shape[2]
            kv_dim = 1
            for d in shape[3:]:
                kv_dim *= d
            total_slots = num_pages * page_size

            # Decompose flat slot index into page + offset
            page_idx = slots // page_size
            offset_idx = slots % page_size

            # Use direct indexing into original tensor (no reshape needed)
            k_flat = k.reshape(num_tokens, *shape[3:])
            v_flat = v.reshape(num_tokens, *shape[3:])
            dst_kv_cache[0, page_idx, offset_idx] = k_flat
            dst_kv_cache[1, page_idx, offset_idx] = v_flat

        elif len(shape) == 4:
            flat = dst_kv_cache.reshape(shape[0] * shape[1], -1)
            is_view = flat.data_ptr() == dst_kv_cache.data_ptr()
            if not is_view:
                logger.warning(
                    "NLS inject: 4D reshape is a COPY, using scatter"
                )
            kv_concat = torch.cat([
                k.reshape(num_tokens, -1),
                v.reshape(num_tokens, -1),
            ], dim=-1)
            if is_view:
                flat[slots] = kv_concat
            else:
                flat = dst_kv_cache.reshape(shape[0] * shape[1], -1)
                flat.scatter_(
                    0,
                    slots.unsqueeze(-1).expand(-1, kv_concat.shape[-1]),
                    kv_concat,
                )
        elif len(shape) == 5 and shape[1] == 2:
            # Block-major paged layout: [num_blocks, 2, page_size, n_heads, head_dim]
            block_idxs = slots // self._composite_block_size
            offsets = slots % self._composite_block_size
            tail = shape[3:]
            k_flat = k.reshape(num_tokens, *tail)
            v_flat = v.reshape(num_tokens, *tail)
            dst_kv_cache[block_idxs, 0, offsets] = k_flat
            dst_kv_cache[block_idxs, 1, offsets] = v_flat
        else:
            block_idxs = slots // self._composite_block_size
            offsets = slots % self._composite_block_size
            if len(shape) > 3:
                tail = shape[3:]
                k_flat = k.reshape(num_tokens, *tail)
                v_flat = v.reshape(num_tokens, *tail)
            else:
                k_flat = k.reshape(num_tokens, -1)
                v_flat = v.reshape(num_tokens, -1)
            dst_kv_cache[block_idxs, 0, offsets] = k_flat
            dst_kv_cache[block_idxs, 1, offsets] = v_flat

    def _rope_rerotate_k(
        self, k: torch.Tensor, old_start: int, new_start: int, seq_len: int,
    ) -> torch.Tensor:
        """Shift post-RoPE K vectors from old positions to new positions.

        RoPE is a rotation, so shifting from pos_old to pos_new is equivalent
        to applying RoPE(delta) where delta = new_start - old_start.
        K shape: [seq_len, num_kv_heads * head_dim]
        """
        if old_start == new_start:
            return k

        delta = new_start - old_start
        head_dim = self._rope_head_dim
        num_heads = self._num_kv_heads
        theta = self._rope_theta

        k_heads = k.view(seq_len, num_heads, head_dim).float()

        positions = torch.arange(old_start, old_start + seq_len, device=k.device)
        new_positions = positions + delta

        dim_pairs = head_dim // 2
        freq_exponents = torch.arange(0, head_dim, 2, device=k.device).float()
        freqs = 1.0 / (theta ** (freq_exponents / head_dim))

        old_angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
        new_angles = new_positions.unsqueeze(-1) * freqs.unsqueeze(0)
        delta_angles = new_angles - old_angles

        cos_d = torch.cos(delta_angles)  # [seq_len, dim_pairs]
        sin_d = torch.sin(delta_angles)  # [seq_len, dim_pairs]

        k_even = k_heads[:, :, 0::2]  # [seq_len, num_heads, dim_pairs]
        k_odd = k_heads[:, :, 1::2]

        cos_d = cos_d.unsqueeze(1)  # [seq_len, 1, dim_pairs]
        sin_d = sin_d.unsqueeze(1)

        k_new_even = k_even * cos_d - k_odd * sin_d
        k_new_odd = k_even * sin_d + k_odd * cos_d

        k_out = torch.empty_like(k_heads)
        k_out[:, :, 0::2] = k_new_even
        k_out[:, :, 1::2] = k_new_odd

        return k_out.to(k.dtype).view(seq_len, num_heads * head_dim)

    def _rerotate_snapshot_keys(
        self,
        snapshot: dict,
        old_start: int,
        new_start: int,
        seq_len: int,
    ) -> dict:
        """Apply RoPE re-rotation to every ``layer_X_k`` tensor in a
        snapshot dict. KL #652 + follow-up: the single-snapshot inject
        path needs the same `+num_register` rotation that the
        multi-snapshot path already applies inside `_load_multi_snapshots`.

        Returns a shallow-copied dict so the underlying ``_snapshot_cache``
        entry is left untouched (subsequent requests for the same path
        may have different ``num_register`` values, e.g. when the
        streaming scorer is disabled per-request).
        """
        if old_start == new_start or seq_len <= 0:
            return snapshot
        out = dict(snapshot)
        rotated = 0
        for k, v in snapshot.items():
            if not isinstance(k, str) or not k.endswith("_k"):
                continue
            if not isinstance(v, torch.Tensor) or v.dim() < 2:
                continue
            if v.shape[0] < seq_len:
                continue
            try:
                # Re-rotate only the leading `seq_len` rows; tail (if any)
                # is untouched. Matches `_load_multi_snapshots` semantics.
                head = self._rope_rerotate_k(
                    v[:seq_len], old_start=old_start,
                    new_start=new_start, seq_len=seq_len,
                )
                if v.shape[0] == seq_len:
                    out[k] = head
                else:
                    out[k] = torch.cat([head, v[seq_len:]], dim=0)
                rotated += 1
            except Exception as e:
                logger.warning(
                    "Single-snapshot rerotate L%s failed: %s", k, e,
                )
        if rotated:
            logger.info(
                "Single-snapshot RoPE re-rotated %d K layers: "
                "old_start=%d new_start=%d seq_len=%d",
                rotated, old_start, new_start, seq_len,
            )
        return out

    def _get_assistant_mask(self, nls_path: str, strip: int, n_tok: int):
        """Read manifest segments and build a boolean mask for assistant positions."""
        try:
            from pri.format import read_manifest
            manifest = read_manifest(nls_path)
            if manifest is None or "segments" not in manifest:
                return None

            mask = torch.zeros(n_tok, dtype=torch.bool)
            for seg in manifest["segments"]:
                if seg["role"] != "assistant":
                    continue
                # Segment positions are in the original (pre-strip) token space
                seg_start = max(seg["start"] - strip, 0)
                seg_end = max(seg["end"] - strip, 0)
                if seg_start >= n_tok or seg_end <= 0:
                    continue
                seg_end = min(seg_end, n_tok)
                mask[seg_start:seg_end] = True
            return mask if mask.any() else None
        except Exception:
            return None

    _system_mamba_cache: dict[str, torch.Tensor] | None = None

    def _get_system_mamba_states(self) -> dict[str, torch.Tensor] | None:
        """Load and cache the system prompt's Mamba states (genesis block).

        Used as the baseline for delta-stitching: each memory's Mamba state
        was computed from system_state → mem_tokens.  The delta
        (mem_state - system_state) isolates the memory's contribution.
        """
        if self._system_mamba_cache is not None:
            return self._system_mamba_cache
        try:
            from pri import retrieve as auto_memory
            if auto_memory._store is None:
                return None
            for mem in auto_memory._store._memories:
                if mem.role == "system" or mem.session_id == "__system__":
                    data = self._load_snapshot(mem.kv_path)
                    if data is None:
                        continue
                    states = {}
                    for k, t in data.items():
                        if "mamba_conv" in k or "mamba_ssm" in k:
                            states[k] = t.clone()
                    if states:
                        self.__class__._system_mamba_cache = states
                        logger.info(
                            "Mamba genesis block loaded: %d state tensors from %s",
                            len(states), mem.kv_path,
                        )
                        return states
        except Exception:
            logger.warning("Failed to load system Mamba states", exc_info=True)
        return None

    def _load_multi_snapshots(
        self,
        snapshot_list: list,
        mamba_delta_sum: int = 0,
        rope_offset: int = 0,
        inject_layout: str = "concat",
    ) -> dict | None:
        """Merge KV from multiple snapshot files into one combined snapshot.

        Attention K/V: concatenated with RoPE re-rotation (position-indexed).
        ``rope_offset`` is the logical position where the first memory token
        will physically live in the new request's sequence (i.e. the size of
        the register prefix from the streaming scorer). Memory N then sits at
        ``rope_offset + Σ_{i<N} num_tokens_i`` in the new request, and that's
        the ``new_start`` we must rotate each chunk to. Without this shift,
        memories rotated for ``new_start=0`` are perceived by the model as
        sitting `rope_offset` positions earlier than they actually are —
        which manifests as RoPE-phase corruption and decode-time degeneration.

        Mamba states: strategy depends on mamba_delta_sum:
          0 = genesis-only (conservative, original KL #614)
          1 = genesis + Σ(deltas) — approximate accumulated state
          2 = genesis + last delta only (most recent memory dominates)
        Conv state uses the last memory's value (sliding window of recent tokens).
        """
        try:
            if inject_layout == "resume":
                try:
                    from pri.inject_geometry_audit import (
                        log_geometry_audit,
                        summarize_geometry_audit,
                    )
                    _audit = summarize_geometry_audit(
                        snapshot_list,
                        rope_offset=rope_offset,
                        resume_mode=True,
                        mamba_delta_sum=mamba_delta_sum,
                    )
                    log_geometry_audit(_audit)
                except Exception as _audit_exc:
                    logger.warning(
                        "NLS inject geometry audit skipped: %s", _audit_exc,
                    )

            merged: dict[str, torch.Tensor] = {}
            total_tokens = 0
            asst_suppressed_tokens = 0

            # KL #614: load system genesis Mamba states for delta stitching
            sys_mamba = self._get_system_mamba_states()
            mamba_deltas: dict[str, list[torch.Tensor]] = {}
            last_mamba_conv: dict[str, torch.Tensor] = {}

            for snap_info in snapshot_list:
                data = self._load_snapshot(snap_info["path"])
                if data is None:
                    logger.error("Multi-snapshot: failed to load %s", snap_info["path"])
                    return None

                n_tok = snap_info["num_tokens"]
                strip = int(snap_info.get("strip_prefix", 0))
                # KL #708: physical layout is [register | mem_0 | mem_1 | ...]
                # so memory N sits at logical position
                # ``rope_offset + Σ_{i<N} num_tokens_i``.
                offset = rope_offset + total_tokens

                # KL #619: read manifest rope_start for correct RoPE re-rotation.
                # NLS v2 blockchain captures use capture_start to strip the system
                # prompt at capture time, so the .nls tensor starts at rope_start
                # (not 0). When strip_prefix is also 0, old_start must use
                # rope_start to avoid a +rope_start positional error.
                rope_start = int(snap_info.get("rope_start", 0) or 0)
                manifest_role = ""
                _manifest = None
                if snap_info["path"].endswith(".nls"):
                    from pri.format import read_manifest
                    _manifest = read_manifest(snap_info["path"])
                    if _manifest:
                        rope_start = int(_manifest.get("rope_start", rope_start) or 0)
                        manifest_role = str(_manifest.get("role", "") or "")

                # Resume turn-chain: manifest rope_start is per-request (after
                # system, typically cap_start). K was computed at absolute
                # position (phantom_prefix + rope_start). Retrieval/concat keeps
                # the legacy rope_start-only accounting.
                phantom_at_capture = 0
                if inject_layout == "resume":
                    if manifest_role == "turn":
                        # Pack cumulative offset is authoritative for turn
                        # captures; manifest capture_num_phantom is provenance.
                        phantom_at_capture = total_tokens
                    elif _manifest:
                        phantom_at_capture = int(
                            _manifest.get("capture_num_phantom", 0) or 0
                        )

                # KL #609: read manifest segments for V-suppression of assistant turns
                asst_mask = None
                if STRIP_ASSISTANT_KEEP_RATIO < 0:
                    asst_mask = self._get_assistant_mask(
                        snap_info["path"], strip, n_tok,
                    )
                    if asst_mask is not None:
                        asst_suppressed_tokens += int(asst_mask.sum().item())

                for key, tensor in data.items():
                    if key.startswith("_meta"):
                        continue

                    is_attn_kv = key.endswith("_k") or key.endswith("_v")
                    is_mamba_ssm = "mamba_ssm" in key
                    is_mamba_conv = "mamba_conv" in key

                    if is_attn_kv:
                        # KL #649: Qwen 3.6 hybrid (attn + DeltaNet) — only
                        # every 4th layer is real attention; the rest are
                        # DeltaNet linear layers whose kv_cache fallback path
                        # produces bogus (N, 1) placeholders. Detect those by
                        # shape and skip them so _rope_rerotate_k doesn't crash.
                        expected_kv_width = self._num_kv_heads * self._rope_head_dim
                        if tensor.dim() < 2 or tensor.shape[-1] != expected_kv_width:
                            continue
                        if tensor.shape[0] < strip + n_tok:
                            if strip > 0:
                                logger.warning(
                                    "Multi-snapshot: %s %s short for strip=%d "
                                    "n_tok=%d (seq=%d); disabling strip for this key",
                                    snap_info["path"], key, strip, n_tok,
                                    tensor.shape[0],
                                )
                            eff_strip = 0
                        else:
                            eff_strip = strip
                        chunk = tensor[eff_strip:eff_strip + n_tok]
                        if key.endswith("_k"):
                            if inject_layout == "resume":
                                rope_old = max(
                                    eff_strip, rope_start + phantom_at_capture,
                                )
                            else:
                                rope_old = max(eff_strip, rope_start)
                            chunk = self._rope_rerotate_k(
                                chunk, old_start=rope_old, new_start=offset,
                                seq_len=n_tok,
                            )
                        if key.endswith("_v") and asst_mask is not None:
                            chunk = chunk.clone()
                            chunk[asst_mask[:chunk.shape[0]]] = 0

                        if key in merged:
                            merged[key] = torch.cat([merged[key], chunk], dim=0)
                        else:
                            merged[key] = chunk.clone()

                    elif is_mamba_ssm and sys_mamba and key in sys_mamba:
                        # KL #614: accumulate delta (mem_state - system_state).
                        # KL #714: per-memory shape guard. When a single memory
                        # was captured under a different slot config than the
                        # current genesis cache, its
                        # mamba_ssm tensor's leading dim won't match the genesis.
                        # Skip THIS memory's contribution for THIS key but keep
                        # the genesis cache intact so other memories in the same
                        # merge still delta-stitch normally. Dedup the warning
                        # per-key per-process to avoid log flooding.
                        sys_t = sys_mamba[key]
                        if tensor.shape != sys_t.shape:
                            warned = getattr(
                                self.__class__,
                                "_mamba_shape_mismatch_warned",
                                None,
                            )
                            if warned is None:
                                warned = set()
                                self.__class__._mamba_shape_mismatch_warned = warned
                            if key not in warned:
                                logger.warning(
                                    "NLS mamba shape mismatch on key=%s: "
                                    "cap=%s sys=%s. Skipping this memory's "
                                    "delta contribution (stale slot config); "
                                    "other memories continue to delta-stitch.",
                                    key, tuple(tensor.shape), tuple(sys_t.shape),
                                )
                                warned.add(key)
                            continue
                        delta = tensor.float() - sys_t.float()
                        if key not in mamba_deltas:
                            mamba_deltas[key] = []
                        mamba_deltas[key].append(delta)

                    elif is_mamba_conv:
                        last_mamba_conv[key] = tensor.clone()

                    else:
                        # Fallback for unknown non-attention keys
                        chunk = tensor[:n_tok]
                        if key in merged:
                            merged[key] = torch.cat([merged[key], chunk], dim=0)
                        else:
                            merged[key] = chunk.clone()

                total_tokens += n_tok

            # KL #625: NLS_MAMBA_DELTA_SUM controls SSM injection strategy:
            #   0 (default) = genesis-only (original behavior)
            #   1 = genesis + Σ(deltas) — approximate accumulated state
            #   2 = genesis + last delta only (most recent memory dominates)
            #   3 = Fable C′ resume — verbatim Mamba from last chain block
            #   4 = Fable C′ resume — telescoping frame chain (= last block SSM)
            _delta_sum_mode = mamba_delta_sum or int(
                os.environ.get("NLS_MAMBA_DELTA_SUM", "0")
            )

            mamba_stitched = 0

            # Fable C′ resume: verbatim Mamba from the last chain block.
            if (
                _delta_sum_mode == 3
                and inject_layout == "resume"
                and snapshot_list
            ):
                last_path = snapshot_list[-1]["path"]
                last_data = self._load_snapshot(last_path)
                if last_data:
                    for key, tensor in last_data.items():
                        if "mamba_conv" in key or "mamba_ssm" in key:
                            merged[key] = tensor.clone()
                    mamba_stitched = sum(
                        1 for k in merged if "mamba_" in k and not k.startswith("_")
                    )
                    logger.info(
                        "Resume mamba: verbatim last block %s (%d mamba keys)",
                        last_path, mamba_stitched,
                    )
            elif (
                _delta_sum_mode == 4
                and inject_layout == "resume"
                and snapshot_list
            ):
                # Telescoping frame composition — numerically equals last-block SSM.
                for snap_info in snapshot_list:
                    block_data = self._load_snapshot(snap_info["path"])
                    if block_data is None:
                        continue
                    for key, tensor in block_data.items():
                        if "mamba_ssm" not in key:
                            continue
                        cur = tensor
                        if sys_mamba and key in sys_mamba:
                            merged[key] = cur.to(sys_mamba[key].dtype).clone()
                        else:
                            merged[key] = cur.clone()
                last_data = self._load_snapshot(snapshot_list[-1]["path"])
                if last_data:
                    for key, tensor in last_data.items():
                        if "mamba_conv" in key:
                            merged[key] = tensor.clone()
                mamba_stitched = sum(
                    1 for k in merged if "mamba_ssm" in k and not k.startswith("_")
                )
                logger.info(
                    "Resume mamba: telescoping chain %d blocks (%d ssm keys)",
                    len(snapshot_list), mamba_stitched,
                )
            elif sys_mamba:
                for key in list(mamba_deltas.keys()):
                    if key in sys_mamba:
                        if _delta_sum_mode == 1 and mamba_deltas[key]:
                            summed = sum(mamba_deltas[key])
                            merged[key] = (
                                sys_mamba[key].float() + summed
                            ).to(sys_mamba[key].dtype)
                            if inject_layout == "resume":
                                logger.debug(
                                    "Resume mamba L key=%s: genesis+sum(%d deltas)",
                                    key, len(mamba_deltas[key]),
                                )
                        elif _delta_sum_mode == 2 and mamba_deltas[key]:
                            merged[key] = (
                                sys_mamba[key].float() + mamba_deltas[key][-1]
                            ).to(sys_mamba[key].dtype)
                        else:
                            merged[key] = sys_mamba[key].clone()
                        mamba_stitched += 1
                for key in list(last_mamba_conv.keys()):
                    if key in sys_mamba:
                        if _delta_sum_mode >= 1:
                            merged[key] = last_mamba_conv[key].clone()
                        else:
                            merged[key] = sys_mamba[key].clone()
                    else:
                        merged[key] = last_mamba_conv[key]

                if inject_layout == "resume" and _delta_sum_mode == 1 and mamba_stitched:
                    logger.info(
                        "Resume mamba: genesis+delta_sum across %d blocks "
                        "(%d keys stitched)",
                        len(snapshot_list), mamba_stitched,
                    )

            else:
                # Fallback: old behavior (concatenate) if system state unavailable
                if not sys_mamba:
                    logger.warning(
                        "Mamba delta-stitch unavailable (no genesis block), "
                        "falling back to last-state"
                    )
                for snap_info in snapshot_list:
                    data = self._load_snapshot(snap_info["path"])
                    if data is None:
                        continue
                    for key, tensor in data.items():
                        if "mamba_conv" in key or "mamba_ssm" in key:
                            merged[key] = tensor.clone()

            merged["_meta_seq_len"] = torch.tensor([total_tokens])
            logger.info(
                "Multi-snapshot merged: %d files, %d total tokens, %d layer keys "
                "(RoPE re-rotated, asst_V_suppressed=%d, mamba_stitched=%d/%d, "
                "delta_sum_mode=%d)",
                len(snapshot_list), total_tokens,
                sum(1 for k in merged if not k.startswith("_")),
                asst_suppressed_tokens, mamba_stitched, len(mamba_deltas),
                _delta_sum_mode,
            )
            return merged
        except Exception:
            logger.error("Multi-snapshot merge failed", exc_info=True)
            return None

    def _resolve_latest_nls(self, session_id: str) -> str:
        """Find most recent .nls file whose manifest session_id matches exactly."""
        # Fast path: in-memory cache from recent captures
        if session_id in self._session_path_cache:
            cached = self._session_path_cache[session_id]
            if Path(cached).exists():
                return cached

        # Slow path: scan capture directory (populates cache for all found sessions)
        cap_dir = self._kv_snapshot_capture_dir
        if not cap_dir.is_dir():
            return ""
        candidates = sorted(cap_dir.glob("kv_snapshot_*.nls"), reverse=True)
        for p in candidates:
            try:
                from pri.format import read_manifest
                manifest = read_manifest(p)
                if manifest is None:
                    continue
                sid = manifest.get("session_id", "")
                if sid and sid not in self._session_path_cache:
                    self._session_path_cache[sid] = str(p)
                if sid == session_id:
                    return str(p)
            except Exception:
                continue
        return ""

    def _release_cuda_cache_if_due(self, reason: str = "") -> None:
        """KL #736 piece 2: hint the CUDA caching allocator to return
        blocks to the pool at transient-tensor lifecycle boundaries
        (post-readback / post-compound-merge / post-request-finish).
        Throttled by ``NLS_CUDA_RELEASE_INTERVAL_S`` (default 5s) so we
        never pay the per-call sync cost on every iteration of a busy
        agentic loop. No-op on CPU-only builds."""
        try:
            if not torch.cuda.is_available():
                return
            now_ns = time.monotonic_ns()
            min_gap_ns = int(self._cuda_release_interval_s * 1_000_000_000)
            if (now_ns - self._last_cuda_release_ns) < min_gap_ns:
                return
            self._last_cuda_release_ns = now_ns
            torch.cuda.empty_cache()
            if reason:
                logger.debug("NLS cuda empty_cache (reason=%s)", reason)
        except Exception:
            pass

    def _load_snapshot(self, path: str) -> dict | None:
        cached = self._snapshot_cache.get(path)
        if cached is not None:
            return cached
        try:
            if path.endswith(".nls"):
                from pri.format import load_nls
                data = load_nls(path)
            elif path.endswith(".kvz"):
                from pri.kv_compress import load_compressed
                data = load_compressed(path)
            else:
                data = torch.load(path, map_location="cpu", weights_only=True)
            self._snapshot_cache.put(path, data)
            stats = self._snapshot_cache.stats()
            logger.info(
                "NLS snapshot loaded from %s (%d keys) — cache %d/%d entries, %.1f/%.1f MiB, evicts=%d",
                path,
                len(data),
                stats["entries"],
                stats["max_entries"],
                stats["total_bytes"] / (1024 * 1024),
                stats["max_bytes"] / (1024 * 1024),
                stats["evictions"],
            )
            return data
        except Exception:
            logger.error("NLS snapshot load failed: %s", path, exc_info=True)
            return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self):
        return

    # ── SupportsHMA interface ────────────────────────────────────

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        try:
            logger.info(
                "NLS request_finished_all_groups CALLED: req=%s, groups=%d, "
                "block_ids_lens=%s, registry_has=%s",
                request.request_id[:16], len(block_ids),
                [len(b) for b in block_ids],
                request.request_id in _capture_registry,
            )
            reg_info = _capture_registry.pop(request.request_id, None)

            if reg_info is None:
                return False, None

            # KL #714: Pass-2 loopback requests carry both memory_off=1
            # (suppress retrieval) and memory_no_capture=1 (suppress new
            # memory creation), BUT they still need the readback to run
            # so `_compound_merge_mamba` can patch the target session's
            # compounded Mamba state into the existing .nls. Don't early-
            # return on either flag when compound_into is set — the
            # readback's compound branch handles it and returns before
            # the dual-emit save runs.
            _is_pass2 = bool(reg_info.get("compound_into", ""))
            if reg_info.get("memory_off", False) and not _is_pass2:
                logger.info("NLS request_finished: skipped (memory_off)")
                return False, None
            if reg_info.get("no_capture", False) and not _is_pass2:
                logger.info(
                    "NLS request_finished: skipped (memory_no_capture)",
                )
                return False, None
            if _is_pass2:
                logger.info(
                    "NLS request_finished: Pass-2 loopback (compound_into=%s) "
                    "— queuing readback for compound merge",
                    reg_info.get("compound_into", "")[:32],
                )
            uid = reg_info.get("user_id", "default")
            sid = reg_info.get("session_id", "")
            if uid == "default" and not sid:
                logger.info("NLS request_finished: skipped (no identity)")
                return False, None
            logger.info(
                "NLS request_finished: reg_info user=%s session=%s",
                uid, sid,
            )

            num_prompt = reg_info.get("expected_tokens", 0)
            num_phantom = reg_info.get("num_phantom", 0)
            if num_prompt <= 0:
                logger.info("NLS request_finished: skipped (no tokens)")
                return False, None

            # KL #626 two-pass + dual-emit (assistant capture):
            # Decode tokens occupy positions [num_prompt_tokens .. num_tokens)
            # in the same paged cache. We capture them via the same readback
            # so we can slice into separate user/tool (prefill) and assistant
            # (decode) memory blocks. `request.num_prompt_tokens` here was
            # bumped by `total_phantom` at injection time (line ~719), so the
            # decode count is `num_tokens - num_prompt_tokens`.
            try:
                num_decoded = max(
                    0, int(request.num_tokens) - int(request.num_prompt_tokens)
                )
            except Exception:
                num_decoded = 0

            # Capture decoded token ids so the assistant block can index its
            # own text in BM25/semantic without contaminating from the user
            # prompt slice. vLLM exposes these via `output_token_ids` on the
            # Request object.
            try:
                output_token_ids = list(
                    getattr(request, "output_token_ids", []) or []
                )
            except Exception:
                output_token_ids = []

            attn_group_idx = self._full_attn_group_idx
            if attn_group_idx < len(block_ids):
                attn_blocks = list(block_ids[attn_group_idx])
            else:
                attn_blocks = list(block_ids[0]) if block_ids else []

            if not attn_blocks:
                logger.info("NLS request_finished: skipped (no attn blocks)")
                return False, None

            mamba_block_ids = {}
            for gidx in self._mamba_group_indices:
                if gidx < len(block_ids) and block_ids[gidx]:
                    mamba_block_ids[gidx] = list(block_ids[gidx])

            _pending_readback_captures[request.request_id] = {
                "block_ids": attn_blocks,
                "mamba_block_ids": mamba_block_ids,
                # `num_prompt` is the boundary between prefill (user/tool block)
                # and decode (assistant block) in the post-strip readback.
                # `num_decoded` is the assistant-side length. Together they
                # define `num_total_real = num_prompt + num_decoded` which is
                # how many real positions the readback consumes (after the
                # leading `total_phantom` slots are stripped).
                "num_prompt": num_prompt,
                "num_decoded": num_decoded,
                "num_phantom": num_phantom,
                "num_register": reg_info.get("num_register", 0),
                "prompt_token_ids": reg_info.get("prompt_token_ids", []),
                "output_token_ids": output_token_ids,
                "user_id": reg_info.get("user_id", "default"),
                "session_id": reg_info.get("session_id", ""),
                "ring_type": reg_info.get("ring_type", "general"),
                # NLS v2 blockchain fields
                "capture_start": reg_info.get("capture_start", 0),
                "capture_end": reg_info.get("capture_end", 0),
                "block_role": reg_info.get("block_role", ""),
                "parent_hash": reg_info.get("parent_hash", ""),
                "prev_hash": reg_info.get("prev_hash", ""),
                "turn_index": reg_info.get("turn_index", -1),
                "base_session_id": reg_info.get("base_session_id", ""),
                "memory_text": reg_info.get("memory_text", ""),
                "sys_prompt_hash": reg_info.get("sys_prompt_hash", ""),
                # KL #626 Pass-2 compound: target session whose .nls should
                # have its Mamba state replaced by this request's readback.
                "compound_into": reg_info.get("compound_into", ""),
                # Optional explicit session id for the assistant slice.
                "asst_session_id": reg_info.get("asst_session_id", ""),
                "compaction_detected": reg_info.get("compaction_detected", False),
                "prefilled_capture": reg_info.get("prefilled_capture", False),
                "resume_stripped_sys": int(reg_info.get("resume_stripped_sys", 0) or 0),
                "inject_mode": str(reg_info.get("inject_mode", "") or ""),
                "resume_inject_aborted": bool(reg_info.get("resume_inject_aborted", False)),
            }
            logger.info(
                "NLS capture QUEUED: req=%s, attn_blocks=%d, mamba_blocks=%s, "
                "prompt=%d decoded=%d (phantom=%d, register=%d), session=%s, "
                "compound_into=%s, pending=%d",
                request.request_id[:16], len(attn_blocks), mamba_block_ids,
                num_prompt, num_decoded, num_phantom,
                reg_info.get("num_register", 0),
                reg_info.get("session_id", "")[:20],
                reg_info.get("compound_into", "")[:20] or "-",
                len(_pending_readback_captures),
            )
            # KL #736 piece 2: request scope is finished — any
            # streaming-slot tensors, retrieved snapshot copies the
            # multi-snap path made for this request, and Pass-1 K/V
            # buffers are now refless. Throttled empty_cache lets the
            # allocator coalesce before the next agent turn's prefill.
            self._release_cuda_cache_if_due(reason="request_finished")
            return True, None
        except Exception:
            logger.error(
                "NLS request_finished_all_groups EXCEPTION",
                exc_info=True,
            )
            return False, None
