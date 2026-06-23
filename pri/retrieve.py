"""Swiss retrieval — semantic search and ranking over stored ``.nls`` blocks.

Optional read path used when pure chain resume is insufficient (``resume_overflow``
profile) or when ``NLS_INJECT_MODE=swiss``. Not active on the v0.1 default
``resume`` profile (``NLS_NEURAL_SCORING=0``).

Flow:

  1. **Query** — embed the live user message (model or sentence-transformer).
  2. **Filter** — partition by ``memory_user``, ``NLS_ROLE_FILTER``, silo flags.
  3. **Rank** — BM25 + cosine similarity + recency + delta-fact probe scores.
  4. **Chain walk** — optional hop expansion along linked sessions.
  5. **Return** — top-K block paths for multi-snapshot inject via ``pri.connector``.

Capture side-effects (when enabled): after meaningful prefills, index new blocks
with ring labels and fingerprints. See ``docs/getting-started/concepts.md``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("nls_auto_memory")

_store: Optional["MemoryStore"] = None
_memory_dir: Optional[str] = None
_enabled: bool = False
_capture_counter: int = 0
_tokenizer = None

MIN_CAPTURE_TOKENS = 50
MIN_RETRIEVE_TOKENS = 10
RETRIEVE_TOP_K = 5

_recent_captures: dict[str, float] = {}
CAPTURE_DEDUP_SECONDS = 30

# Per-request state (set by snapshot_connector, read by capture hook)
_current_user_id: str = "default"
_current_project_id: str = ""
_current_ring_type: str = "general"
_current_description: str = ""
_current_session_id: str = ""
_current_memory_off: bool = False
_current_ingest_only: bool = False
_current_no_capture: bool = False

# Last retrieval info (for external inspection / demo API)
_last_retrieval: Optional[dict] = None

# Attention reranking: path to the last query's captured snapshot (with Q)
_last_query_snapshot: Optional[str] = None
RERANK_COARSE_K = 20  # coarse retrieval candidates for reranking


_model_path: str = ""
_embed_loaded: bool = False


def _load_embed_weights_lazy() -> bool:
    """Lazy-load the model's embed_tokens weight for semantic fingerprinting.

    Called on first capture/retrieve, NOT at init — this ensures vLLM has
    already allocated its KV cache before we consume ~970MB of unified
    memory for the embedding matrix (critical on GB10 shared-memory arch).
    """
    global _embed_loaded
    if _embed_loaded:
        return True

    mp = _model_path or os.environ.get("NLS_MODEL_PATH", "")
    if not mp:
        _embed_loaded = True  # no path = SimHash mode, don't retry
        return False

    try:
        import glob
        from safetensors import safe_open
        from pri.store import set_embed_weights

        shard_files = sorted(glob.glob(os.path.join(mp, "*.safetensors")))
        for sf in shard_files:
            with safe_open(sf, framework="pt") as f:
                for key in f.keys():
                    if "embed_tokens" in key and "visual" not in key:
                        import torch
                        t = f.get_tensor(key)       # bfloat16, CPU
                        weights = t.to(dtype=torch.float16).numpy()
                        del t
                        set_embed_weights(weights)
                        _embed_loaded = True
                        logger.info(
                            "Loaded model embeddings (lazy): %s shape=%s "
                            "dtype=%s (%.0f MB)",
                            key, weights.shape,
                            weights.dtype, weights.nbytes / 1024 / 1024,
                        )
                        return True
        logger.warning("embed_tokens weight not found in %s", mp)
        _embed_loaded = True
        return False
    except Exception as e:
        logger.warning("Could not load model embeddings: %s (SimHash fallback)", e)
        _embed_loaded = True
        return False


ARIADNE_CACHE_PATHS = [
    "/workspace/data/ariadne_cache_perturn.json",
    "/workspace/data/ariadne_cache.json",
    "/workspace/data/ariadne_cache_blockchain.jsonl",
]


def init(memory_dir: str, model_path: str = "") -> bool:
    global _store, _memory_dir, _enabled, _model_path
    try:
        from pri.store import MemoryStore

        _model_path = model_path or os.environ.get("NLS_MODEL_PATH", "")
        _memory_dir = memory_dir
        # snapshot_dir tells MemoryStore where .nls captures live so the
        # startup manifest reconciliation can rebuild the index authoritatively.
        snap_dir = os.environ.get("NLS_SNAPSHOT_DIR", "")
        _store = MemoryStore(memory_dir, snapshot_dir=snap_dir or None)
        _enabled = True
        logger.info(
            "Auto-memory v3 ENABLED: dir=%s, memories=%d, "
            "embeddings=deferred (model_path=%s)",
            memory_dir, _store.size,
            "set" if _model_path else "none",
        )

        # Ariadne cache loading is deferred to first retrieval
        # (_load_ariadne_deferred) because embed weights aren't available at
        # init time — they load lazily after KV cache allocation.

        return True
    except Exception as e:
        logger.error("Auto-memory init failed: %s", e, exc_info=True)
        _enabled = False
        return False


def is_enabled() -> bool:
    return _enabled and _store is not None


_ingestion_mode_activated: bool = False
_last_ingest_time: float = 0.0
_INGEST_IDLE_TIMEOUT: float = 120.0  # auto-disable ingestion mode after 2min idle
_ariadne_loaded: bool = False
_fingerprints_reseeded: bool = False


def _load_ariadne_deferred() -> None:
    """Load Ariadne cache on first retrieval when embed weights are available."""
    global _ariadne_loaded
    if _ariadne_loaded or _store is None:
        return
    _ariadne_loaded = True
    from pri.store import _embed_weights
    if _embed_weights is None:
        return
    for cache_path in ARIADNE_CACHE_PATHS:
        if os.path.exists(cache_path):
            try:
                updated = _store.load_ariadne_cache(cache_path)
                if updated:
                    logger.info(
                        "Deferred Ariadne cache load: %s (%d memories updated)",
                        cache_path, updated,
                    )
            except Exception:
                logger.debug("Ariadne cache load failed: %s", cache_path)


def _reseed_fingerprints_if_needed() -> None:
    """Auto-reseed or extend fingerprints when dim or count mismatches."""
    global _fingerprints_reseeded
    if _store is None:
        return
    from pri.store import _embed_weights, _embed_dim
    if _embed_weights is None:
        return
    fp = _store._fingerprints
    n_mems = len(_store._memories)
    if fp is None or fp.shape[0] == 0:
        _store.reseed_fingerprints()
        _fingerprints_reseeded = True
    elif fp.shape[1] != _embed_dim:
        logger.info("Fingerprint dim mismatch (%d vs embed=%d) — full reseeding", fp.shape[1], _embed_dim)
        _store.reseed_fingerprints()
        _fingerprints_reseeded = True
    elif fp.shape[0] < n_mems:
        _store.extend_fingerprints()
    elif not _fingerprints_reseeded:
        _fingerprints_reseeded = True


_semantic_loaded: bool = False


def _load_semantic_embeddings_lazy() -> None:
    """Load or compute sentence-transformer semantic embeddings once."""
    global _semantic_loaded
    if _semantic_loaded or _store is None:
        return
    _semantic_loaded = True
    _store.load_semantic_embeddings()


def set_request_context(
    user_id: str = "default",
    project_id: str = "",
    ring_type: str = "general",
    description: str = "",
    session_id: str = "",
    memory_off: bool = False,
    ingest_only: bool = False,
    no_capture: bool = False,
):
    """Set per-request labels. Called from snapshot_connector or logits processor.

    no_capture=True enables retrieve-only mode: injection still runs but the
    request's own KV is NOT captured back into the memory store. This avoids
    self-feedback loops (e.g. a benchmark auto-ingesting its own queries).
    """
    global _current_user_id, _current_project_id, _current_ring_type
    global _current_description, _current_session_id, _current_memory_off
    global _current_ingest_only, _current_no_capture
    global _ingestion_mode_activated, _last_ingest_time
    _current_user_id = user_id
    _current_project_id = project_id
    _current_ring_type = ring_type
    _current_description = description
    _current_session_id = session_id
    _current_memory_off = memory_off
    _current_ingest_only = ingest_only
    _current_no_capture = no_capture

    if ingest_only and _store is not None:
        import time as _time
        _last_ingest_time = _time.time()
        if not _ingestion_mode_activated:
            _store.enable_ingestion_mode(True)
            _ingestion_mode_activated = True
            logger.info("Auto-memory: ingestion mode auto-activated (first ingest request)")

    elif _ingestion_mode_activated and not ingest_only and _store is not None:
        import time as _time
        if _time.time() - _last_ingest_time > _INGEST_IDLE_TIMEOUT:
            _store.enable_ingestion_mode(False)
            _ingestion_mode_activated = False
            logger.info("Auto-memory: ingestion mode auto-disabled (idle %.0fs)",
                        _time.time() - _last_ingest_time)

    logger.info("set_request_context: user=%s ring=%s project=%s session=%s",
                user_id, ring_type, project_id, session_id or "-")


def enable_ingestion_mode(enabled: bool = True) -> None:
    """Enable/disable ingestion mode on the underlying MemoryStore.

    In ingestion mode, RAM-heavy caches (BM25, turn/Ariadne embeddings)
    are periodically flushed to disk, and dedup scans are skipped.
    """
    if _store is not None:
        _store.enable_ingestion_mode(enabled)
        logger.info("Auto-memory ingestion_mode=%s", enabled)


def compact() -> dict:
    """Compact on-disk state: dedup JSONL, dedup BM25, remove orphan NLS.

    Returns stats dict from MemoryStore.compact(). Safe to call any time.
    """
    if _store is None:
        return {"error": "store not initialized"}
    return _store.compact()


def capture(
    token_ids: list[int],
    kv_path: str,
    num_tokens: int,
    ariadne_question_ids: Optional[list[list[int]]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    ring_type: Optional[str] = None,
) -> Optional[str]:
    """Register a captured KV snapshot.

    Per-request parameters (session_id, user_id, ring_type) override globals
    when provided — this is the correct path for concurrent captures.
    """
    global _capture_counter
    if not is_enabled() or _store is None:
        return None

    _load_embed_weights_lazy()

    # Use per-request values if provided, fall back to globals for legacy callers
    eff_session_id = session_id if session_id is not None else _current_session_id
    eff_user_id = user_id if user_id is not None else _current_user_id
    eff_ring_type = ring_type if ring_type is not None else _current_ring_type

    if _current_memory_off or _current_no_capture:
        # Retrieve-only request: captured KV is not needed, delete the file
        # otherwise we accumulate thousands of orphan .nls files.
        try:
            Path(kv_path).unlink(missing_ok=True)
        except OSError:
            pass
        return None

    if len(token_ids) < MIN_CAPTURE_TOKENS:
        logger.debug("Auto-memory: skip capture (only %d tokens)", len(token_ids))
        try:
            Path(kv_path).unlink(missing_ok=True)
        except OSError:
            pass
        return None

    if not Path(kv_path).exists():
        logger.warning("Auto-memory: skip capture, file missing: %s", kv_path)
        return None

    # Dedup: include session_id so different sessions never dedup against each other
    dedup_key = f"{eff_session_id}:{eff_user_id}"
    hash_input = dedup_key.encode() + struct.pack(
        f"<{min(len(token_ids), 512)}I", *token_ids[:512]
    )
    prompt_hash = hashlib.sha256(hash_input).hexdigest()[:16]

    now = time.time()
    if prompt_hash in _recent_captures:
        if now - _recent_captures[prompt_hash] < CAPTURE_DEDUP_SECONDS:
            try:
                Path(kv_path).unlink(missing_ok=True)
            except OSError:
                pass
            return None
    _recent_captures[prompt_hash] = now
    if len(_recent_captures) > 200:
        _recent_captures.clear()

    mem_id = _store.add(
        token_ids=token_ids,
        kv_path=kv_path,
        num_tokens=num_tokens,
        ring_type=eff_ring_type,
        user_id=eff_user_id,
        project_id=_current_project_id,
        session_id=eff_session_id,
        source="auto_capture",
        description=_current_description,
    )
    _capture_counter += 1

    # Store BM25/turn data + optional Ariadne for Swiss Cheese retrieval.
    # Keyed by session_id (stable across restarts). Falls back to mem_id
    # for anonymous captures.
    try:
        turn_token_lists = _split_turns(token_ids)
        _store.add_bm25_data(
            key=eff_session_id or mem_id,
            token_ids=token_ids,
            turn_texts_token_ids=turn_token_lists,
            ariadne_question_ids=ariadne_question_ids,
        )
    except Exception:
        logger.debug("BM25 data storage failed (non-critical)", exc_info=True)

    logger.info(
        "Auto-memory CAPTURED: id=%s, ring=%s, user=%s, session=%s, "
        "tokens=%d, store=%d, ariadne=%s",
        mem_id, eff_ring_type, eff_user_id,
        eff_session_id or "-", num_tokens, _store.size,
        len(ariadne_question_ids) if ariadne_question_ids else 0,
    )

    return mem_id


_IM_START = 248045   # <|im_start|> in Qwen3.5
_IM_END = 248046     # <|im_end|>
_TOK_USER = 846      # "user"
_TOK_NEWLINE = 198   # "\n"


def _extract_last_user_turn(token_ids: list[int]) -> Optional[list[int]]:
    """Extract the last user message from a chat-template token sequence.

    The full prompt contains system prompt + chat template noise that hurts
    BM25/embedding retrieval. We only want the user's actual question.
    Looks for the last <|im_start|>user\\n ... <|im_end|> block.
    """
    starts = [i for i, t in enumerate(token_ids) if t == _IM_START]
    ends = [i for i, t in enumerate(token_ids) if t == _IM_END]

    if not starts:
        return None

    for s in reversed(starts):
        turn_end = len(token_ids)
        for e in ends:
            if e > s:
                turn_end = e
                break

        turn_tokens = token_ids[s + 1:turn_end]
        if not turn_tokens:
            continue

        if turn_tokens[0] != _TOK_USER:
            continue

        content_start = 0
        for j, t in enumerate(turn_tokens):
            if t == _TOK_NEWLINE:
                content_start = j + 1
                break
        content = turn_tokens[content_start:]
        if len(content) >= 3:
            return content

    return None


def _split_turns(token_ids: list[int]) -> list[list[int]]:
    """Split a chat-template token sequence into per-turn chunks."""
    turns: list[list[int]] = []
    current: list[int] = []
    for tid in token_ids:
        if tid == _IM_START:
            if current:
                turns.append(current)
            current = []
        elif tid == _IM_END:
            if current:
                turns.append(current)
            current = []
        else:
            current.append(tid)
    if current:
        turns.append(current)
    return [t for t in turns if len(t) >= 5]


CHAIN_WALK_DECAY = float(os.environ.get("NLS_CHAIN_DECAY", "0.90"))
CHAIN_WALK_ENABLED = os.environ.get("NLS_CHAIN_WALK", "1") != "0"
CHAIN_WALK_ROLES = set(os.environ.get("NLS_CHAIN_WALK_ROLES", "tool").split(","))
CHAIN_WALK_HOPS = int(os.environ.get("NLS_CHAIN_HOPS", "2"))


def _expand_chain_links(
    results: list[tuple["Memory", float]],
) -> list[tuple["Memory", float]]:
    """Expand retrieval results by walking blockchain links.

    Two expansion modes:
    1. Same-turn linking: when a user block is retrieved, pull linked tool
       blocks from the same turn (base_session_id + turn_index).
    2. Adjacent-turn expansion: walk up to CHAIN_WALK_HOPS turns forward/back
       within the same base_session_id, pulling user blocks with decaying score.
       This gives session-coherent context (e.g., multi-turn discussions).
    """
    if not CHAIN_WALK_ENABLED or _store is None:
        return results

    seen_ids: set[str] = {mem.id for mem, _ in results}
    expanded: list[tuple["Memory", float]] = list(results)

    turn_keys: dict[tuple[str, int], float] = {}
    session_scores: dict[str, list[tuple[int, float]]] = {}
    for mem, score in results:
        if not mem.base_session_id or mem.turn_index < 0:
            continue
        key = (mem.base_session_id, mem.turn_index)
        if key not in turn_keys or score > turn_keys[key]:
            turn_keys[key] = score
        ss = session_scores.setdefault(mem.base_session_id, [])
        ss.append((mem.turn_index, score))

    if not turn_keys and not session_scores:
        return results

    # Build adjacent-turn targets: for each hit session, expand ±CHAIN_WALK_HOPS
    adjacent_targets: dict[tuple[str, int], float] = {}
    for bsid, hits in session_scores.items():
        for turn_idx, score in hits:
            for hop in range(1, CHAIN_WALK_HOPS + 1):
                decay = CHAIN_WALK_DECAY ** (hop + 1)
                for adj_turn in (turn_idx - hop, turn_idx + hop):
                    if adj_turn < 0:
                        continue
                    adj_key = (bsid, adj_turn)
                    if adj_key in turn_keys:
                        continue
                    adj_score = score * decay
                    if adj_key not in adjacent_targets or adj_score > adjacent_targets[adj_key]:
                        adjacent_targets[adj_key] = adj_score

    linked_count = 0
    adjacent_count = 0
    for candidate in _store._memories:
        if candidate.id in seen_ids:
            continue
        if not candidate.base_session_id or candidate.turn_index < 0:
            continue
        key = (candidate.base_session_id, candidate.turn_index)

        # Mode 1: same-turn tool block linking
        if candidate.role in CHAIN_WALK_ROLES and key in turn_keys:
            parent_score = turn_keys[key]
            expanded.append((candidate, parent_score * CHAIN_WALK_DECAY))
            seen_ids.add(candidate.id)
            linked_count += 1
            continue

        # Mode 2: adjacent-turn user block expansion
        if candidate.role == "user" and key in adjacent_targets:
            expanded.append((candidate, adjacent_targets[key]))
            seen_ids.add(candidate.id)
            adjacent_count += 1

    if linked_count > 0 or adjacent_count > 0:
        logger.info("Chain-walk expansion: %d same-turn links, %d adjacent-turn blocks from %d sessions",
                     linked_count, adjacent_count, len(session_scores))

    return expanded


def retrieve(
    query_token_ids: list[int],
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    top_k: int = RETRIEVE_TOP_K,
    *,
    base_session_id: Optional[str] = None,
    boost_compaction_context: bool = False,
) -> Optional[list[tuple[str, int, float, str, float]]]:
    """Find matching memories using Swiss Cheese retrieval.

    Returns list of (kv_path, num_tokens, score, ring_type, meta_score) tuples
    ordered by ring priority (identity first, general last).

    Uses BM25 + sentence-transformer semantic fusion.
    Falls back to basic cosine search when no BM25 data is available.
    """
    global _last_retrieval, _tokenizer
    if not is_enabled() or _store is None:
        return None

    _load_embed_weights_lazy()
    _load_ariadne_deferred()
    _reseed_fingerprints_if_needed()
    _load_semantic_embeddings_lazy()

    if _current_memory_off or _current_ingest_only:
        return None

    if len(query_token_ids) < MIN_RETRIEVE_TOKENS:
        return None

    uid = user_id or _current_user_id
    pid = project_id or _current_project_id

    # Extract just the user's last turn for retrieval — the full prompt
    # includes chat template + system prompt which adds noise to BM25/embedding
    search_tokens = _extract_last_user_turn(query_token_ids) or query_token_ids
    logger.info("Retrieve: full_prompt=%d tokens, search_query=%d tokens",
                len(query_token_ids), len(search_tokens))

    # Decode search tokens to text for the sentence-transformer
    query_text = None
    if _tokenizer is None and _model_path:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(
                _model_path, trust_remote_code=True,
            )
            logger.info("Loaded tokenizer from %s for semantic search", _model_path)
        except Exception as e:
            logger.warning("Failed to load tokenizer: %s", e)
    if _tokenizer is not None:
        query_text = _tokenizer.decode(search_tokens, skip_special_tokens=True)

    # Prepend temporal anchor so relative time queries align with indexed dates
    from pri.store import TEMPORAL_INDEX_ENABLED, temporal_query_anchor
    if TEMPORAL_INDEX_ENABLED and query_text:
        query_text = temporal_query_anchor() + query_text

    # NLS v2: role filter — inject user + tool blocks.
    # Tool blocks carry the actual facts (IPs, configs, file contents).
    # Assistant blocks are excluded (noisy summaries degrade attention — KL #633).
    _role_filter_str = os.environ.get("NLS_ROLE_FILTER", "user,tool")
    _role_filter = set(_role_filter_str.split(",")) if _role_filter_str else None

    results = _store.search_swiss_cheese(
        search_tokens,
        top_k=top_k,
        user_id=uid if uid != "default" else None,
        project_id=pid or None,
        role_filter=_role_filter,
        query_text=query_text,
        base_session_id=base_session_id,
        boost_compaction_context=boost_compaction_context,
    )
    if not results:
        # Fallback to legacy cosine search
        results = _store.search(
            query_token_ids,
            top_k=top_k,
            user_id=uid if uid != "default" else None,
            project_id=pid or None,
        )
    if not results:
        _last_retrieval = None
        return None

    # Chain-walk expansion: when a user block is retrieved, carry over
    # linked tool/assistant blocks from the same turn (base_session_id + turn_index).
    results = _expand_chain_links(results)

    # Verify files exist and build return list. KL #655: when a kv_path is
    # missing on disk, the index is corrupt (a previous capture / dedup left
    # the row pointing at a deleted file). Self-heal by dropping the dead
    # entry from the in-memory store and re-persisting; this prevents the
    # same dead memory from being retrieved on every subsequent query and
    # eventually shrinks the pool back to the truth-on-disk.
    valid = []
    valid_mems = []
    dead_mem_ids: list[str] = []
    for mem, sim in results:
        if not Path(mem.kv_path).exists():
            logger.warning(
                "Auto-memory: file missing for %s: %s — quarantining",
                mem.id, mem.kv_path,
            )
            dead_mem_ids.append(mem.id)
            continue
        valid.append((mem.kv_path, mem.num_tokens, sim, mem.ring_type, mem.meta_score))
        valid_mems.append(mem)

    if dead_mem_ids and _store is not None:
        try:
            removed = _store.drop_by_ids(dead_mem_ids)
            if removed:
                logger.info(
                    "Auto-memory: self-heal dropped %d dead index entries",
                    removed,
                )
        except Exception:
            logger.debug("Self-heal drop_by_ids failed (non-critical)", exc_info=True)

    if not valid:
        _last_retrieval = None
        return None

    total_tokens = sum(v[1] for v in valid)

    _last_retrieval = {
        "count": len(valid),
        "total_tokens": total_tokens,
        "memories": [
            {
                "id": valid_mems[i].id,
                "ring": v[3],
                "tokens": v[1],
                "similarity": round(v[2], 3),
                "preview": (valid_mems[i].description or "")[:120],
                "role": getattr(valid_mems[i], "role", "user"),
                "meta_score": round(getattr(valid_mems[i], "meta_score", 0.0), 2),
            }
            for i, v in enumerate(valid)
        ],
        "user_id": uid,
        "project_id": pid,
        "timestamp": time.time(),
    }

    logger.info(
        "Auto-memory RETRIEVE: user=%s, project=%s, matches=%d, "
        "total_tokens=%d, rings=%s",
        uid, pid or "-", len(valid), total_tokens,
        [v[3] for v in valid],
    )

    return valid


def set_query_snapshot(path: Optional[str]) -> None:
    """Set the path to the current query's captured snapshot (with Q vectors).

    Called by the snapshot capture hook after a query's KV is captured.
    If Q vectors are present, subsequent retrieve() calls will use
    attention reranking for more precise memory selection.
    """
    global _last_query_snapshot
    _last_query_snapshot = path


def get_last_retrieval() -> Optional[dict]:
    """Get info about the most recent retrieval (for demo/debug API)."""
    return _last_retrieval


def get_stats() -> dict:
    if _store is None:
        return {"enabled": False}
    stats = _store.get_stats()
    stats["enabled"] = _enabled
    stats["total_captures"] = _capture_counter
    return stats


def delete_user(user_id: str) -> int:
    """Delete all memories for a user."""
    if _store is None:
        return 0
    count = _store.delete_user(user_id)
    logger.info("Auto-memory: deleted %d memories for user=%s", count, user_id)
    return count


# Auto-init from environment
_env_dir = os.environ.get("NLS_MEMORY_DIR")
if _env_dir:
    init(_env_dir, model_path=os.environ.get("NLS_MODEL_PATH", ""))
