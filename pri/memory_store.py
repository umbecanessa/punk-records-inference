"""
NLS Memory Store v3 — The Hippocampus of the Non-Stateless LLM.

Cryptex-inspired memory organization inside the inference server.
Each memory is tagged with ring_type, user_id, project_id, session_id,
and injection priority — mirroring the 13-ring Cryptex architecture.

Fingerprinting modes:
  - "model": Uses the LLM's own embedding layer (embed_tokens) for
    semantic fingerprints. Mean-pooled token embeddings from the model's
    learned representations. Zero external dependencies.
  - "simhash": Legacy fallback using SimHash on token IDs.

Ring types (from Cryptex, mapped to KV memory):
  - identity:       Soul axioms, agent personality  (always_inject)
  - behavioral:     Communication patterns, rules   (always_inject)
  - user_model:     User preferences, context
  - project_facts:  Project-specific knowledge
  - credentials:    API keys, connection strings
  - instructions:   Task-specific directives
  - orchestration:  Workflow state, plans
  - consolidation:  Long-term distilled knowledge   (always_inject)
  - general:        Unclassified conversation memory

Multi-tenant: memories are partitioned by user_id. Cross-read memories
(always_inject=True) are shared across all users within the same store.

Storage:
  {memory_dir}/
    index.json          — metadata for all memories
    fingerprints.npy    — dense matrix [N, fingerprint_dim]
    kv_*.pt             — KV snapshot files
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import struct
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("nls_memory_store")

# ── Sentence-transformer semantic embedder (singleton) ──────────────
# Loaded once on first use, runs on GPU, caches embeddings to disk.
_SEMANTIC_MODEL_NAME = os.environ.get(
    "NLS_SEMANTIC_MODEL", "BAAI/bge-base-en-v1.5"
)
_SEMANTIC_QUERY_PREFIX = os.environ.get(
    "NLS_SEMANTIC_QUERY_PREFIX", "Represent this sentence: "
)
_SEMANTIC_DEVICE = os.environ.get("NLS_SEMANTIC_DEVICE", "cuda")


class SentenceEmbedder:
    """Thin wrapper around sentence-transformers for semantic retrieval."""

    _instance: Optional["SentenceEmbedder"] = None

    def __init__(self):
        self._model = None
        self._dim: int = 0

    @classmethod
    def get(cls) -> "SentenceEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            t0 = time.time()
            self._model = SentenceTransformer(
                _SEMANTIC_MODEL_NAME, device=_SEMANTIC_DEVICE
            )
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Semantic embedder loaded: model=%s dim=%d device=%s (%.1fs)",
                _SEMANTIC_MODEL_NAME, self._dim, _SEMANTIC_DEVICE,
                time.time() - t0,
            )
            return True
        except Exception as e:
            logger.warning("Failed to load semantic embedder: %s", e)
            self._model = None
            return False

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return self._dim

    def encode_texts(self, texts: list[str], batch_size: int = 512) -> np.ndarray:
        if not self._ensure_loaded():
            return np.zeros((len(texts), 1), dtype=np.float32)
        return self._model.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        )

    def encode_query(self, text: str) -> np.ndarray:
        if not self._ensure_loaded():
            return np.zeros(self._dim or 1, dtype=np.float32)
        q = f"{_SEMANTIC_QUERY_PREFIX}{text}" if _SEMANTIC_QUERY_PREFIX else text
        return self._model.encode([q], normalize_embeddings=True)[0]

SIMHASH_DIM = 512
HASH_SEED_BASE = 0x4E4C534D  # "NLSM"
MAX_MEMORIES = int(os.environ.get("NLS_MAX_MEMORIES", "250000"))
SIMILARITY_THRESHOLD = 0.25
DEDUP_THRESHOLD = 0.92

# Swiss Cheese retrieval config (KL #457, tuned for full 11K haystack)
BM25_K1 = 1.2
BM25_B = 0.75
SC_LAYER_WEIGHTS = {
    "BM25": 0.35,
    "Semantic": 0.65,
}
SC_GATE_MULTIPLIER = 1.5
SC_IDF_SAMPLE_SIZE = 1000

# Assistant-as-funnel: disabled — semantic embeddings provide better coverage
# without artificially narrowing the candidate pool.
ASST_FUNNEL_ENABLED = os.environ.get("NLS_ASST_FUNNEL", "0") == "1"
ASST_FUNNEL_TOP_K = int(os.environ.get("NLS_ASST_FUNNEL_TOP_K", "100"))

# Recency: applied ONLY as a tiebreaker between content-colliding memories
# (caught by the embedding-dedup branch with sim > DEDUP_THRESHOLD), not as
# a global score multiplier. Two near-identical facts ("my wife is Lucia"
# at turn 3 vs "my wife is now Monica" at turn 7) → keep the newer one.
# Two unrelated memories ("Thanks" turn 6 vs "wife Monica turn 3") → recency
# has no business pushing one over the other; their semantic + meta-score
# signals must decide alone.
#
# RECENCY_ENABLED / RECENCY_DECAY / RECENCY_FLOOR control the legacy GLOBAL
# decay multiplier (each candidate × max(floor, 1 - decay × age_hours));
# default RECENCY_FLOOR=1.0 makes that path a no-op. Set RECENCY_FLOOR<1
# only to re-enable global decay for experiments. The collision-scoped
# tiebreaker lives in the dedup branch and is always on.
RECENCY_ENABLED = os.environ.get("NLS_RECENCY", "1") != "0"
RECENCY_DECAY = float(os.environ.get("NLS_RECENCY_DECAY", "0.002"))
RECENCY_FLOOR = float(os.environ.get("NLS_RECENCY_FLOOR", "1.0"))

# DeltaNet fact-quality boost: model-native Q-vs-F signal from DeltaNet SSM states.
# Memories with high "delta energy" (large SSM state change from genesis) are factual;
# memories with low delta energy are likely questions/reactions/meta.
# Boost = 1 + DELTA_FACT_BOOST * normalized_energy  (factual memories win retrieval).
DELTA_FACT_ENABLED = os.environ.get("NLS_DELTA_FACT", "1") != "0"
DELTA_FACT_BOOST = float(os.environ.get("NLS_DELTA_FACT_BOOST", "0.35"))
DELTA_FACT_PROBE_LAYERS = [2, 14, 26, 38]
# JL #19.2 Phase 5: structured delta fingerprint dimension. Per probed
# layer × 32 heads × 3 stats (Frobenius norm / mean / std). Persisted to
# `delta_fingerprints.npy` so a server restart doesn't have to re-derive
# every memory's fingerprint from its `.nls` snapshot.
DELTA_FACT_FP_DIM = len(DELTA_FACT_PROBE_LAYERS) * 32 * 3
# Cache schema version. Bump on layout changes (probe layers, stat list,
# dim ordering) so older caches invalidate and force a one-time rebuild.
DELTA_FP_CACHE_VERSION = 1

# JL #19.6 Step 3: Tier 2 FFN signature cache (R1 router-weight cosine,
# picked in JL #19.5 spike). For each memory we persist a flat
# ``[NUM_LAYERS × NUM_EXPERTS]`` vector: per-layer expert-firing
# frequency normalized over (n_tokens × top_k) and L2-normalized over
# the full vector. Layout matches ``r1_routing_weight_vector`` in
# ``scripts/spike_ffn_sig_score.py`` so production scoring is bit-
# compatible with the spike that selected this representation.
#
# Defaults track Qwen3-Next-80B-A3B-Instruct (40 MoE layers × 512
# experts). Env-overridable so the same plugin works against future
# checkpoints without a code change. Storage at the defaults is
# 40 × 512 × 2 = 40 KB per memory at float16 (~5 GB at 100K mems);
# float16 is precision-sufficient for the 0.4–0.55 cosine thresholds
# Tier 2 uses (verified empirically vs the float32 spike scorer).
FFN_SIG_NUM_LAYERS = int(os.environ.get("NLS_FFN_SIG_NUM_LAYERS", "40"))
FFN_SIG_NUM_EXPERTS = int(os.environ.get("NLS_FFN_SIG_NUM_EXPERTS", "512"))
FFN_SIG_DIM = FFN_SIG_NUM_LAYERS * FFN_SIG_NUM_EXPERTS
# Cache schema version. Bump on layout changes (dim, dtype, normalization).
FFN_SIG_CACHE_VERSION = 1
# Whether the persisted Tier 2 cache is loaded at boot. Off by default
# during cold-ship (Layer 6 is observation-only); flip on once Tier 2
# enters composition path. Independent of write path: incremental
# capture-time writes happen whenever the in-memory cache is populated.
FFN_SIG_LOAD_AT_BOOT = os.environ.get("NLS_FFN_SIG_LOAD_AT_BOOT", "1") != "0"

# Ring types with injection priority (lower = injected first = stronger attention)
RING_PRIORITIES: dict[str, int] = {
    "identity": 0,
    "behavioral": 1,
    "user_model": 2,
    "consolidation": 3,
    "instructions": 4,
    "project_facts": 5,
    "credentials": 6,
    "orchestration": 7,
    "general": 8,
}

ALWAYS_INJECT_RINGS = {"identity", "behavioral", "consolidation"}


# ── Temporal indexing ──────────────────────────────────────────────
# Bake human-readable timestamp into BM25/semantic text so time-based
# queries ("what did we do Monday?") match naturally without heuristics.
TEMPORAL_INDEX_ENABLED = os.environ.get("NLS_TEMPORAL_INDEX", "1") != "0"


def temporal_preamble(ts: float) -> str:
    """Turn a Unix timestamp into a searchable date preamble."""
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("[%Y-%m-%d %A %I:%M %p] ")


def temporal_query_anchor() -> str:
    """Current time anchor prepended to queries for relative date matching."""
    now = datetime.now()
    return f"[Current time: {now.strftime('%Y-%m-%d %A %I:%M %p')}] "


# ── KL #639: Language-agnostic meta-score ─────────────────────────
# Scores how likely a message is meta/reactive (1.0) vs factual (0.0).
# Uses only universal signals — no language-specific word lists.

_QUESTION_MARKS = set("?？؟⸮;")  # Latin, CJK, Arabic, irony, Greek question mark
_DATA_TOKEN_RE = re.compile(
    r"\d"               # any digit (age, PIN, date, phone, address number)
    r"|@"               # email pattern
    r"|://"             # URL pattern
    r"|[a-fA-F0-9]{16}" # hex string (API keys, hashes)
    r"|[A-Za-z0-9+/]{20,}={0,2}"  # base64-ish (credentials, tokens)
)
META_PENALTY_WEIGHT = float(os.environ.get("NLS_META_PENALTY_WEIGHT", "0.85"))

# KL #651: dual-centroid model-native Q-vs-F signal.
# DELTA_SIGNAL_ENABLED — flip to 0 to fall back to regex meta_score everywhere.
# DELTA_SIGNAL_SHARPNESS — sigmoid steepness applied to the signed signal:
#   high values mean a small separation already produces a confident penalty.
#   Default chosen so |signal|=0.05 → ~0.62 confidence; |signal|=0.20 → ~0.95.
DELTA_SIGNAL_ENABLED = os.environ.get("NLS_DELTA_SIGNAL_ENABLED", "1") == "1"
DELTA_SIGNAL_SHARPNESS = float(os.environ.get("NLS_DELTA_SIGNAL_SHARPNESS", "15.0"))


def _delta_signal_to_qness(signal: float) -> float:
    """Convert signed delta_signal ∈ [-1,1] to question-ness ∈ (0,1).

    KL #651: positive signal = fact-shaped → low question-ness → low penalty.
    Negative signal = question-shaped → high question-ness → strong penalty.
    Sigmoid keeps the response smooth and bounded.
    """
    import math
    z = max(-30.0, min(30.0, -signal * DELTA_SIGNAL_SHARPNESS))
    return 1.0 / (1.0 + math.exp(-z))


def compute_meta_score(text: str, role: str = "user") -> float:
    """Compute a 0..1 meta-score for a message. Language-agnostic.

    KL #639 invariant: *no keyword lists*. Detection must work across
    Italian, Dutch, English, etc. Question-shape detection in
    non-universal scripts/morphologies (e.g., imperative mood without
    a trailing "?") is the responsibility of the DeltaNet model-native
    qness signal (KL #651 dual-centroid), which uses the model's own
    embedding geometry rather than lexical rules. The combination
    "universal regex + DeltaNet" is the multilingual contract.

    0.0 = high-quality factual content (capture & rank normally)
    1.0 = pure meta/question/reaction (penalize at retrieval)

    Signals (all universal across languages):
    - has_data_tokens: digits, @, URLs, hex/base64 strings → always 0.0
    - has_question_mark: `?` / `？` / `؟` → soft signal toward meta
    - word_count: very short messages with no data → likely reactions
    - role: assistant messages are always 0.0 (they contain synthesis)
    """
    if role != "user":
        return 0.0

    text = text.strip()
    if not text:
        return 1.0

    if _DATA_TOKEN_RE.search(text):
        return 0.0

    words = text.split()
    word_count = len(words)

    has_qmark = bool(set(text) & _QUESTION_MARKS)
    ends_qmark = text[-1] in _QUESTION_MARKS if text else False

    score = 0.0

    if ends_qmark and word_count <= 12:
        score += 0.6
    elif has_qmark:
        score += 0.3

    if word_count <= 3 and not has_qmark:
        score += 0.5
    elif word_count <= 6:
        score += 0.2
    elif word_count <= 10:
        score += 0.1

    return min(score, 1.0)


# ── JL #6 / Layer 2: Provenance tags (NLS+ calibrated reasoning) ────
# Every memory carries a provenance tag declaring the *origin and trust
# class* of its content. The verifier (Layer 5, future) reads these tags
# to weight evidence; the model never sees them. Tags are plumbing only
# at this stage — capture path emits them, persistence round-trips them,
# nothing branches on them yet. Verifier wiring lands in JL #7+.
#
# Tag semantics (locked in JL #1 / refined in JL #5):
#   USER_PROVIDED   — direct input from the user or app (system prompts,
#                     user messages). Trustworthy as a record of what
#                     the user said; not reality-validated.
#   EXTERNAL_CLAIM  — external source (tool output, web search, third-
#                     party API). Default for tool-role memories; a
#                     future per-tool refinement may upgrade some to
#                     EXPERIENCED (e.g. dns_lookup queries actual DNS
#                     servers; file_read reads the actual filesystem).
#   EXPERIENCED     — firsthand observation by the agent of reality
#                     (sensor read, executed action's result). None
#                     emitted yet; reserved.
#   REASONED        — model's own generative output (assistant
#                     messages). Default-untrusted for fact extraction;
#                     verifier will downweight unless calibration
#                     history says this kind of reasoning succeeded.
#   IMAGINED        — deliberation rollout (Tier 1+ prediction phase,
#                     JL #3). High-volume, mostly transient. None
#                     emitted yet; will be tagged when deliberation
#                     phase lands.
#   CALIBRATED      — prediction whose outcome was later validated by
#                     reality. None emitted yet; promotion path lands
#                     with Layer 4 (calibration store).
#   SENSED          — raw sensor data / unprocessed observation.
#                     Reserved; same family as EXPERIENCED but pre-
#                     interpretation.
#   UNKNOWN         — provenance unclear (legacy memories captured
#                     pre-JL #6). Migration-on-load tags these by role
#                     where possible; truly unknown stay UNKNOWN.
PROVENANCE_USER_PROVIDED = "USER_PROVIDED"
PROVENANCE_EXTERNAL_CLAIM = "EXTERNAL_CLAIM"
PROVENANCE_EXPERIENCED = "EXPERIENCED"
PROVENANCE_REASONED = "REASONED"
PROVENANCE_IMAGINED = "IMAGINED"
PROVENANCE_CALIBRATED = "CALIBRATED"
PROVENANCE_SENSED = "SENSED"
PROVENANCE_UNKNOWN = "UNKNOWN"

PROVENANCE_VALUES = frozenset({
    PROVENANCE_USER_PROVIDED, PROVENANCE_EXTERNAL_CLAIM,
    PROVENANCE_EXPERIENCED, PROVENANCE_REASONED,
    PROVENANCE_IMAGINED, PROVENANCE_CALIBRATED,
    PROVENANCE_SENSED, PROVENANCE_UNKNOWN,
})

# Default role → provenance map for capture-time tagging. Conservative:
# we never default to CALIBRATED or EXPERIENCED — those require explicit
# evidence (reality validation / sensor origin) that the capture path
# can't infer. Per-tool refinement of EXTERNAL_CLAIM → EXPERIENCED is a
# future enhancement (see JL #6 limitations).
ROLE_TO_DEFAULT_PROVENANCE: dict[str, str] = {
    "user": PROVENANCE_USER_PROVIDED,
    "system": PROVENANCE_USER_PROVIDED,
    "tool": PROVENANCE_EXTERNAL_CLAIM,
    "assistant": PROVENANCE_REASONED,
}


def default_provenance_for_role(role: str) -> str:
    """Return the default provenance tag for a memory of the given role.

    Conservative mapping — anything we can't classify cleanly returns
    UNKNOWN rather than guessing. Callers can always override.
    """
    return ROLE_TO_DEFAULT_PROVENANCE.get(role or "", PROVENANCE_UNKNOWN)


@dataclass
class Memory:
    id: str
    kv_path: str
    num_tokens: int
    timestamp: float
    ring_type: str = "general"
    user_id: str = "default"
    project_id: str = ""
    session_id: str = ""
    always_inject: bool = False
    access_count: int = 0
    last_accessed: float = 0.0
    source: str = ""
    prompt_hash: str = ""
    description: str = ""
    # NLS v2 blockchain fields
    role: str = ""
    block_hash: str = ""
    parent_hash: str = ""
    prev_hash: str = ""
    turn_index: int = -1
    base_session_id: str = ""
    rope_start: int = 0
    # KL #639: language-agnostic meta-score for retrieval-time penalty.
    # 0.0 = high-quality factual memory, 1.0 = likely meta/question/reaction.
    meta_score: float = 0.0
    # KL #708 plumbing + KL #645 hardening: backend-provided 16-char SHA-256
    # prefix of the rendered system prompt. Enables system-block dedup
    # via `has_system_block_for_hash()`, plus future cross-request strip
    # validation (warn when a captured memory's rope_start doesn't match
    # the rendered system prompt's actual token count).
    sys_prompt_hash: str = ""
    # JL #6 / Layer 2: provenance tag (see PROVENANCE_* constants above).
    # Emitted at capture time by `default_provenance_for_role(role)` unless
    # an explicit override is supplied. Read by the verifier (Layer 5,
    # future) to weight evidence. Migration-on-load tags pre-JL #6
    # memories from their role; truly unclassifiable rows keep UNKNOWN.
    provenance: str = PROVENANCE_UNKNOWN
    # True when captured on a build-agent request where punk-records set
    # memory_compaction_detected (transcript shrink), i.e. the first pass
    # after OpenCode re-delivers the compacted summary as context.
    is_compaction_context: bool = False

    @property
    def injection_priority(self) -> int:
        return RING_PRIORITIES.get(self.ring_type, 99)


# ── Model-internal embedding fingerprints ────────────────────────────
# The LLM's own embed_tokens weight matrix, set once at startup via
# set_embed_weights(). Shape: [vocab_size, hidden_dim], float32 on CPU.
_embed_weights: Optional[np.ndarray] = None
_embed_dim: int = 0


def set_embed_weights(weights: np.ndarray) -> None:
    """Set the model's embedding matrix for fingerprinting.

    Called once at startup from auto_memory with the model's
    embed_tokens.weight as float16 numpy (unified memory friendly).
    """
    global _embed_weights, _embed_dim
    _embed_weights = weights
    _embed_dim = weights.shape[1]
    logger.info(
        "Model embedding fingerprints enabled: vocab=%d, dim=%d, "
        "dtype=%s, %.0f MB",
        weights.shape[0], _embed_dim, weights.dtype,
        weights.nbytes / 1024 / 1024,
    )


def get_fingerprint_dim() -> int:
    """Return the active fingerprint dimension."""
    if _embed_weights is not None:
        return _embed_dim
    return SIMHASH_DIM


def compute_fingerprint(token_ids: list[int]) -> np.ndarray:
    """Compute a semantic fingerprint from token IDs.

    Uses model-internal embeddings if available (mean-pooled embed_tokens),
    otherwise falls back to SimHash.
    """
    if _embed_weights is not None:
        return _compute_embedding_fingerprint(token_ids)
    return _compute_simhash_fingerprint(token_ids)


def _compute_embedding_fingerprint(token_ids: list[int]) -> np.ndarray:
    """Mean-pooled embedding from the model's own embed_tokens layer.

    Weights are stored as float16 to save unified memory on GB10.
    Mean pooling and normalization are done in float32 for precision.
    """
    if not token_ids:
        return np.zeros(_embed_dim, dtype=np.float32)

    valid_ids = np.array(
        [tid for tid in token_ids if 0 <= tid < _embed_weights.shape[0]],
        dtype=np.int64,
    )
    if len(valid_ids) == 0:
        return np.zeros(_embed_dim, dtype=np.float32)

    embeddings = _embed_weights[valid_ids]             # [N, hidden_dim] float16
    mean_vec = embeddings.astype(np.float32).mean(axis=0)  # [hidden_dim] float32

    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec /= norm
    return mean_vec


# ── SimHash fallback ─────────────────────────────────────────────────

def _token_hash_vector(token_id: int) -> np.ndarray:
    vec = np.zeros(SIMHASH_DIM, dtype=np.float32)
    for i in range(SIMHASH_DIM):
        h = hashlib.md5(
            struct.pack("<II", token_id ^ HASH_SEED_BASE, i)
        ).digest()
        val = struct.unpack("<i", h[:4])[0]
        vec[i] = 1.0 if val >= 0 else -1.0
    return vec


_hash_cache: dict[int, np.ndarray] = {}


def _get_hash_vector(token_id: int) -> np.ndarray:
    if token_id not in _hash_cache:
        _hash_cache[token_id] = _token_hash_vector(token_id)
        if len(_hash_cache) > 50000:
            oldest = list(_hash_cache.keys())[:10000]
            for k in oldest:
                del _hash_cache[k]
    return _hash_cache[token_id]


def _compute_simhash_fingerprint(token_ids: list[int]) -> np.ndarray:
    """SimHash fingerprint from token IDs (legacy fallback)."""
    if not token_ids:
        return np.zeros(SIMHASH_DIM, dtype=np.float32)

    acc = np.zeros(SIMHASH_DIM, dtype=np.float32)
    for tid in token_ids:
        acc += _get_hash_vector(tid)

    norm = np.linalg.norm(acc)
    if norm > 0:
        acc /= norm
    return acc


def batch_cosine_similarity(
    query: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    return matrix @ query


class MemoryStore:
    """Persistent labeled memory index with Cryptex-style ring organization.

    Data integrity contract
    -----------------------
    The filesystem is the single source of truth. Every .nls capture file
    carries its own session_id in its manifest header. At startup we rebuild
    the in-memory index by scanning manifests, not by reading index.jsonl
    (which is treated as a cache, not truth).

    All derived data (BM25 entries, turn embeddings, Ariadne embeddings) is
    keyed by session_id (str), never by list position. This makes every
    index stable across restarts/dedup/reingest and eliminates the entire
    "ghost entry" drift class of bugs.
    """

    INGEST_FLUSH_EVERY = 200  # flush caches every N adds in ingestion mode

    def __init__(
        self,
        memory_dir: str,
        max_memories: int = MAX_MEMORIES,
        *,
        snapshot_dir: Optional[str] = None,
        reconcile_from_manifests: bool = True,
        readonly: bool = False,
    ):
        self._dir = Path(memory_dir)
        if not readonly:
            self._dir.mkdir(parents=True, exist_ok=True)
        self._readonly = readonly
        self._max_memories = max_memories
        # captures dir: prefer explicit arg, then env, then persistent volume,
        # then the legacy /tmp location for BACKWARDS COMPAT on unmigrated hosts.
        snapshot_dir = snapshot_dir or os.getenv("NLS_SNAPSHOT_DIR", "")
        if snapshot_dir:
            self._capture_dir = Path(snapshot_dir) / "captures"
        elif (self._dir / "captures").exists():
            self._capture_dir = self._dir / "captures"
        else:
            # Legacy fallback — only used when migrating old data.
            self._capture_dir = Path("/tmp/nls_kv_snapshot/captures")

        self._memories: list[Memory] = []
        self._fingerprints: Optional[np.ndarray] = None

        self._index_path = self._dir / "index.json"
        self._fp_path = self._dir / "fingerprints.npy"
        sem_suffix = "_t1" if TEMPORAL_INDEX_ENABLED else ""
        self._sem_path = self._dir / f"semantic_embeddings{sem_suffix}.npy"

        self._ingestion_mode: bool = False
        self._ingest_add_count: int = 0
        self._ingest_fp_buffer: list[np.ndarray] = []
        self._last_flushed_idx: int = 0
        self._fp_dirty: bool = False
        self._fp_save_interval: int = 50
        self._adds_since_fp_save: int = 0
        self._hash_index: dict[str, int] = {}

        self._semantic_embs: Optional[np.ndarray] = None

        # KL #655: track in-place modifications to existing records (e.g.
        # dedup updating kv_path/num_tokens). _save() is normally append-only,
        # but when this set is non-empty we must atomically rewrite the JSONL
        # so the change actually lands on disk. Without this, dedup would
        # silently delete the old file while the index kept pointing at it.
        self._dirty_indices: set[int] = set()

        self._load()
        if reconcile_from_manifests and not readonly:
            self._reconcile_from_manifests()
        self._last_flushed_idx = len(self._memories)
        self._load_bm25()
        self._delta_energy: Optional[np.ndarray] = None
        # KL #651: dual-centroid model-native Q-vs-F signal.
        # `_delta_signal[i]` ∈ [-1, 1] is `cos(fp, fact_centroid) - cos(fp, question_centroid)`.
        # Positive = model-native fact-shape, negative = question-shape. Replaces
        # regex meta_score as the runtime signal everywhere (capture gate, retrieval
        # debuff, streaming probe). Falls back to meta_score when centroids are missing.
        self._delta_signal: Optional[np.ndarray] = None
        self._user_centroids: dict[str, np.ndarray] = {}      # fact centroid per user (normalized, fast read)
        self._user_q_centroids: dict[str, np.ndarray] = {}    # KL #651: question centroid per user (normalized)
        # JL #19.2 Phase 4: un-normalized cumulative fingerprint sums + counts
        # per user, for online running-mean centroid updates on capture.
        # The normalized centroid above is derived from `sum / count` and
        # cached for fast retrieval reads. Both fields are rebuilt from
        # disk at startup (`_rebuild_user_centroids` /
        # `_recompute_delta_fact_scores`); the running update keeps them
        # current between restarts so Tier 1 / KL #651 don't lock to a
        # stale centroid as users accumulate new conversations.
        self._user_fact_sums: dict[str, np.ndarray] = {}
        self._user_q_sums: dict[str, np.ndarray] = {}
        self._user_fact_counts: dict[str, int] = {}
        self._user_q_counts: dict[str, int] = {}
        self._user_delta_range: dict[str, tuple[float, float]] = {}
        # JL #19.2 Phase 5: structured delta fingerprint cache. Shape
        # (N, DELTA_FACT_FP_DIM), float32. Loaded from disk at boot
        # (validated against `genesis_hash` + memory id list); appended
        # in-place on capture; persisted on each capture (matching the
        # existing `delta_energy.npy` / `delta_signal.npy` rewrite
        # pattern). When populated, `_rebuild_user_centroids` and
        # `_recompute_delta_fact_scores` skip per-memory `.nls` loads.
        self._delta_fp_cache: Optional[np.ndarray] = None
        self._delta_fp_cache_genesis_hash: Optional[str] = None
        # JL #19.6 Step 3: Tier 2 FFN signature cache. Same parallel-array
        # pattern as the delta fingerprint cache: shape (N, FFN_SIG_DIM)
        # float16, indexed by memory position. Populated incrementally at
        # capture time by ``snapshot_connector._readback_and_save`` and
        # persisted alongside the existing delta_fingerprints.npy.
        self._ffn_sig_cache: Optional[np.ndarray] = None
        self._ffn_sig_cache_genesis_hash: Optional[str] = None
        if DELTA_FACT_ENABLED and not readonly:
            self._load_delta_fingerprints()
        if FFN_SIG_LOAD_AT_BOOT and not readonly:
            self._load_ffn_signatures()
        if not readonly:
            self._cleanup_orphans()

    def _load(self):
        jsonl_path = self._index_path.with_suffix(".jsonl")
        if jsonl_path.exists():
            try:
                # KL #655: dedup by memory ID (the unique-per-capture key).
                # The previous logic deduped by session_id, which destroyed
                # legitimate per-turn captures that legitimately share a
                # session — e.g. an agent loop with 6 turns from one session
                # would collapse to 1 memory at every restart.
                # Memory IDs are guaranteed unique (mem_<ts_ms>_<seq>) so
                # last-write-wins is only triggered by genuine append-only
                # duplicates from crash-resume scenarios.
                by_id: dict[str, Memory] = {}
                anonymous: list[Memory] = []
                total_lines = 0
                provenance_migrated = 0
                with open(jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        total_lines += 1
                        m = json.loads(line)
                        if "session_id" not in m:
                            m["session_id"] = ""
                        mem = Memory(**m)
                        # JL #6 migration-on-load: legacy memories captured
                        # before the provenance field existed default to
                        # UNKNOWN. If we know the role, upgrade them to the
                        # role-derived default. Idempotent across restarts:
                        # next load re-runs the same mapping if the file
                        # hasn't been rewritten yet.
                        if (
                            mem.provenance == PROVENANCE_UNKNOWN
                            and mem.role
                        ):
                            new_p = default_provenance_for_role(mem.role)
                            if new_p != PROVENANCE_UNKNOWN:
                                mem.provenance = new_p
                                provenance_migrated += 1
                        if mem.id:
                            by_id[mem.id] = mem
                        else:
                            anonymous.append(mem)

                self._memories = anonymous + list(by_id.values())
                if provenance_migrated > 0:
                    logger.info(
                        "JL #6 migration: tagged %d legacy memories with "
                        "role-derived provenance (UNKNOWN → role default).",
                        provenance_migrated,
                    )
                dups = total_lines - len(self._memories)
                if dups > 0:
                    logger.warning(
                        "Loaded memory index (JSONL): %d unique memories "
                        "(deduped %d duplicate lines from %d total)",
                        len(self._memories), dups, total_lines,
                    )
                    # Rewrite the file clean so we don't re-read duplicates
                    if not self._readonly:
                        self._compact_jsonl_on_load(jsonl_path)
                else:
                    logger.info("Loaded memory index (JSONL): %d memories", len(self._memories))
            except Exception:
                logger.warning("Failed to load memory index JSONL", exc_info=True)
                self._memories = []
        elif self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    data = json.load(f)
                memories = []
                migrated = 0
                for m in data.get("memories", []):
                    if "session_id" not in m:
                        m["session_id"] = ""
                    mem = Memory(**m)
                    if (
                        mem.provenance == PROVENANCE_UNKNOWN
                        and mem.role
                    ):
                        new_p = default_provenance_for_role(mem.role)
                        if new_p != PROVENANCE_UNKNOWN:
                            mem.provenance = new_p
                            migrated += 1
                    memories.append(mem)
                self._memories = memories
                if migrated > 0:
                    logger.info(
                        "JL #6 migration (legacy index.json): tagged %d "
                        "memories with role-derived provenance.", migrated,
                    )
            except Exception:
                logger.warning("Failed to load memory index", exc_info=True)
                self._memories = []

        if self._fp_path.exists() and self._memories:
            try:
                self._fingerprints = np.load(str(self._fp_path))
                if self._fingerprints.shape[0] != len(self._memories):
                    logger.warning(
                        "Fingerprint count mismatch (%d fps vs %d mems), resetting",
                        self._fingerprints.shape[0], len(self._memories),
                    )
                    self._fingerprints = None
                else:
                    nonzero = int(np.count_nonzero(
                        np.abs(self._fingerprints).sum(axis=1)
                    ))
                    logger.info(
                        "Loaded fingerprints: shape=%s, non-zero=%d/%d",
                        self._fingerprints.shape, nonzero,
                        self._fingerprints.shape[0],
                    )
            except Exception:
                self._fingerprints = None

        if self._fingerprints is None and self._memories:
            fp_dim = get_fingerprint_dim()
            self._fingerprints = np.zeros(
                (len(self._memories), fp_dim), dtype=np.float32
            )
            logger.info(
                "Initialized empty fingerprints: (%d, %d)",
                len(self._memories), fp_dim,
            )

        # Build prompt_hash → index for O(1) exact dedup
        self._hash_index = {}
        for i, m in enumerate(self._memories):
            if m.prompt_hash:
                self._hash_index[m.prompt_hash] = i

    def _save(self):
        try:
            jsonl_path = self._index_path.with_suffix(".jsonl")
            start = self._last_flushed_idx
            new_entries = self._memories[start:]

            # KL #655: when in-place updates are pending (dedup updated an
            # existing record's kv_path / num_tokens / description, or a
            # caller removed entries), pure append is wrong — the on-disk
            # JSONL still has the stale rows and a later restart would
            # resurrect paths to deleted files. Atomically rewrite the
            # entire JSONL to make the update durable.
            if self._dirty_indices:
                n_dirty = len(self._dirty_indices)
                self._dirty_indices.clear()
                self._rewrite_index_jsonl()
                logger.info(
                    "_save: atomic rewrite (in-place updates=%d, total=%d)",
                    n_dirty, len(self._memories),
                )
            elif new_entries:
                with open(jsonl_path, "a") as f:
                    for m in new_entries:
                        f.write(json.dumps(asdict(m), separators=(",", ":")) + "\n")
                self._last_flushed_idx = len(self._memories)

            # Save fingerprints periodically (not on every add)
            self._adds_since_fp_save += len(new_entries)
            if self._fp_dirty and self._adds_since_fp_save >= self._fp_save_interval:
                self._persist_fingerprints()
                self._adds_since_fp_save = 0
        except Exception:
            logger.error("_save() FAILED", exc_info=True)

    def _persist_fingerprints(self):
        """Write the used portion of fingerprints to disk."""
        if self._fingerprints is not None and len(self._memories) > 0:
            n = len(self._memories)
            np.save(str(self._fp_path), self._fingerprints[:n])
            self._fp_dirty = False

    def _fp_append(self, fingerprint: np.ndarray):
        """Append a fingerprint using pre-allocated capacity (amortized O(1))."""
        n = len(self._memories)  # memory was already appended
        fp_dim = fingerprint.shape[0]
        if self._fingerprints is None or self._fingerprints.shape[0] == 0:
            cap = max(256, n * 2)
            self._fingerprints = np.zeros((cap, fp_dim), dtype=np.float32)
        elif self._fingerprints.shape[1] != fp_dim:
            logger.warning(
                "Fingerprint dim migrating (%d -> %d), %d entries lose fp",
                self._fingerprints.shape[1], fp_dim, n - 1,
            )
            cap = max(self._fingerprints.shape[0], n * 2)
            self._fingerprints = np.zeros((cap, fp_dim), dtype=np.float32)
        elif n > self._fingerprints.shape[0]:
            new_cap = self._fingerprints.shape[0] * 2
            new_fp = np.zeros((new_cap, fp_dim), dtype=np.float32)
            new_fp[:self._fingerprints.shape[0]] = self._fingerprints
            self._fingerprints = new_fp
        self._fingerprints[n - 1] = fingerprint
        self._fp_dirty = True

    def _fp_set(self, idx: int, fingerprint: np.ndarray):
        """Overwrite fingerprint at a specific index (for dedup updates)."""
        if self._fingerprints is not None and idx < self._fingerprints.shape[0]:
            self._fingerprints[idx] = fingerprint
            self._fp_dirty = True

    # ── Ground-truth reconciliation ──────────────────────────────
    # The .nls files on disk ARE the memory store. Everything else is a
    # cache that must agree with the filesystem or be rebuilt from it.

    def reseed_fingerprints(self) -> int:
        """Recompute all fingerprints using embed_tokens (2048-dim).

        Called automatically when fingerprint dimension doesn't match the
        current embed_dim (e.g., after upgrading from SimHash to embedding
        fingerprints). Reads conversation_text from each memory's .nls
        manifest to re-tokenize and re-fingerprint.

        Returns the number of fingerprints reseeded.
        """
        if _embed_weights is None:
            logger.warning("reseed_fingerprints: embed_weights not loaded yet")
            return 0

        target_dim = _embed_dim
        current_dim = self._fingerprints.shape[1] if self._fingerprints is not None else 0
        if current_dim == target_dim and self._fingerprints is not None:
            nonzero = int(np.count_nonzero(
                np.abs(self._fingerprints).sum(axis=1)
            ))
            if nonzero == len(self._memories):
                logger.info(
                    "reseed_fingerprints: already at target dim=%d, all %d non-zero",
                    target_dim, nonzero,
                )
                return 0

        logger.info(
            "reseed_fingerprints: recomputing %d fingerprints (%d→%d dim)",
            len(self._memories), current_dim, target_dim,
        )

        from nls_vllm_plugin.nls_format import read_manifest

        new_fp = np.zeros((len(self._memories), target_dim), dtype=np.float32)
        reseeded = 0
        no_text = 0

        for i, mem in enumerate(self._memories):
            text = ""
            if mem.kv_path:
                manifest = read_manifest(mem.kv_path)
                if manifest:
                    text = manifest.get("conversation_text", "")

            if not text:
                text = mem.description or ""

            if text:
                token_ids = self._text_to_token_ids(text)
                if token_ids:
                    fp = compute_fingerprint(token_ids)
                    if fp.shape[0] == target_dim:
                        new_fp[i] = fp
                        reseeded += 1
                    else:
                        no_text += 1
                else:
                    no_text += 1
            else:
                no_text += 1

            if (i + 1) % 2000 == 0:
                logger.info(
                    "reseed_fingerprints: %d/%d (%.0f%%), reseeded=%d, no_text=%d",
                    i + 1, len(self._memories),
                    (i + 1) / len(self._memories) * 100,
                    reseeded, no_text,
                )

        self._fingerprints = new_fp
        np.save(str(self._fp_path), self._fingerprints)

        logger.info(
            "reseed_fingerprints: DONE — %d/%d reseeded, %d no_text, saved to %s",
            reseeded, len(self._memories), no_text, self._fp_path,
        )
        return reseeded

    def extend_fingerprints(self) -> int:
        """Incrementally add fingerprints for memories beyond current array size.

        Much faster than full reseed — only processes new entries instead of
        recomputing all 34K+. Called when memories grow (new captures) but
        existing fingerprints are still valid.
        """
        if _embed_weights is None:
            return 0

        target_dim = _embed_dim
        n_mems = len(self._memories)
        fp = self._fingerprints

        if fp is None or fp.shape[0] == 0 or fp.shape[1] != target_dim:
            return self.reseed_fingerprints()

        existing = fp.shape[0]
        if existing >= n_mems:
            return 0

        from nls_vllm_plugin.nls_format import read_manifest

        new_count = n_mems - existing
        extended = np.zeros((n_mems, target_dim), dtype=np.float32)
        extended[:existing] = fp[:existing]
        added = 0

        for i in range(existing, n_mems):
            mem = self._memories[i]
            text = ""
            if mem.kv_path:
                manifest = read_manifest(mem.kv_path)
                if manifest:
                    text = manifest.get("conversation_text", "")
            if not text:
                text = mem.description or ""
            if text:
                token_ids = self._text_to_token_ids(text)
                if token_ids:
                    fp_vec = compute_fingerprint(token_ids)
                    if fp_vec.shape[0] == target_dim:
                        extended[i] = fp_vec
                        added += 1

        self._fingerprints = extended
        np.save(str(self._fp_path), self._fingerprints)
        logger.info(
            "extend_fingerprints: added %d/%d new (total %d), saved to %s",
            added, new_count, n_mems, self._fp_path,
        )
        return added

    def _mem_key(self, mem_or_idx) -> str:
        """Return the stable string key for a memory's derived data.

        Accepts an int (position in _memories) or a Memory. Uses session_id
        when present, falling back to mem.id. Returns "" for out-of-range
        indices so callers can no-op gracefully.
        """
        if isinstance(mem_or_idx, Memory):
            return mem_or_idx.session_id or mem_or_idx.id or ""
        if isinstance(mem_or_idx, int):
            if 0 <= mem_or_idx < len(self._memories):
                m = self._memories[mem_or_idx]
                return m.session_id or m.id or ""
            return ""
        if isinstance(mem_or_idx, str):
            return mem_or_idx
        return ""

    def _reconcile_from_manifests(self) -> None:
        """Rebuild the authoritative memory list from .nls manifests.

        KL #655: Each .nls file is exactly one memory. Reconcile keys by
        kv_path (file path) so multiple captures from the same session —
        which are legitimate per-turn memories — are all preserved.

        Scans self._capture_dir, reads each file's manifest header. Any
        index.jsonl entry whose kv_path does not exist on disk is dropped.
        Any .nls file not represented in the index gets an index entry
        reconstructed from its manifest.

        After this runs, self._memories is guaranteed to match the
        filesystem 1-to-1 and index.jsonl is rewritten to match.
        """
        try:
            from nls_vllm_plugin.nls_format import read_manifest
        except Exception:
            logger.warning("Cannot import nls_format; skipping reconcile")
            return

        if not self._capture_dir.exists():
            logger.warning(
                "Capture dir %s does not exist yet; reconcile skipped",
                self._capture_dir,
            )
            return

        # 1. Scan filesystem → kv_path → (ts, manifest). One entry per file.
        fs_by_path: dict[str, dict] = {}
        n_scanned = 0
        n_no_manifest = 0
        for fp in self._capture_dir.iterdir():
            if not fp.name.endswith(".nls"):
                continue
            n_scanned += 1
            m = read_manifest(fp)
            if m is None:
                n_no_manifest += 1
                continue
            try:
                ts_ms = int(fp.stem.replace("kv_snapshot_", ""))
            except ValueError:
                ts_ms = int(fp.stat().st_mtime * 1000)
            fs_by_path[str(fp)] = {
                "path": str(fp),
                "ts_ms": ts_ms,
                "manifest": m,
            }

        # 2. Index existing memories by kv_path so we can preserve metadata
        # (user_id, description, meta_score, etc.) for known files.
        existing_by_path: dict[str, Memory] = {
            m.kv_path: m for m in self._memories if m.kv_path
        }

        new_memories: list[Memory] = []
        rescued_count = 0
        for path, info in fs_by_path.items():
            m = info["manifest"]
            # Manifest field is "seq_len" (set by nls_format._build_manifest
            # from _meta_seq_len tensor). Older / external writers may use
            # "num_tokens" directly. Accept either.
            num_tokens = int(
                m.get("num_tokens") or m.get("seq_len") or 0
            )
            prior = existing_by_path.get(path)
            if prior is not None:
                # Refresh from manifest (manifest is ground truth for tensors).
                if num_tokens and prior.num_tokens != num_tokens:
                    prior.num_tokens = num_tokens
                if not prior.role:
                    prior.role = str(m.get("role", "") or "")
                if not prior.block_hash:
                    prior.block_hash = str(m.get("block_hash", "") or "")
                if not prior.parent_hash:
                    prior.parent_hash = str(m.get("parent_hash", "") or "")
                if not prior.prev_hash:
                    prior.prev_hash = str(m.get("prev_hash", "") or "")
                if prior.turn_index < 0:
                    prior.turn_index = int(m.get("turn_index", -1))
                if not prior.base_session_id:
                    prior.base_session_id = str(m.get("base_session_id", "") or "")
                if not prior.rope_start:
                    prior.rope_start = int(m.get("rope_start", 0))
                if not prior.sys_prompt_hash:
                    prior.sys_prompt_hash = str(m.get("sys_prompt_hash", "") or "")
                # JL #6: refresh provenance from manifest if it carries
                # one (newer captures), otherwise fall back to role-based
                # default for legacy records that have role but no
                # provenance.
                manifest_prov = str(m.get("provenance", "") or "")
                if manifest_prov in PROVENANCE_VALUES:
                    prior.provenance = manifest_prov
                elif prior.provenance == PROVENANCE_UNKNOWN and prior.role:
                    prior.provenance = default_provenance_for_role(prior.role)
                new_memories.append(prior)
            else:
                rescued_count += 1
                new_memories.append(Memory(
                    id=f"mem_{info['ts_ms']}_{len(new_memories)}",
                    kv_path=path,
                    num_tokens=num_tokens,
                    timestamp=info["ts_ms"] / 1000.0,
                    ring_type=str(m.get("ring_type", "general") or "general"),
                    user_id=str(m.get("user_id", "default") or "default"),
                    project_id=str(m.get("project_id", "") or ""),
                    session_id=str(m.get("session_id", "") or ""),
                    source="manifest_recovered",
                    description="",
                    role=str(m.get("role", "") or ""),
                    block_hash=str(m.get("block_hash", "") or ""),
                    parent_hash=str(m.get("parent_hash", "") or ""),
                    prev_hash=str(m.get("prev_hash", "") or ""),
                    turn_index=int(m.get("turn_index", -1)),
                    base_session_id=str(m.get("base_session_id", "") or ""),
                    rope_start=int(m.get("rope_start", 0)),
                    sys_prompt_hash=str(m.get("sys_prompt_hash", "") or ""),
                    provenance=(
                        str(m.get("provenance", "") or "")
                        if str(m.get("provenance", "") or "") in PROVENANCE_VALUES
                        else default_provenance_for_role(
                            str(m.get("role", "") or "")
                        )
                    ),
                ))

        # Preserve any truly anonymous index entries (no kv_path). Rare.
        anonymous = [m for m in self._memories if not m.kv_path]

        # Count phantoms = in-memory index entries pointing to deleted files
        phantom_paths = set(existing_by_path) - set(fs_by_path)

        prev_total = len(self._memories)
        self._memories = anonymous + new_memories

        # 3. Reset fingerprints — position-indexed, must be rebuilt.
        if self._fingerprints is not None and \
                self._fingerprints.shape[0] != len(self._memories):
            self._fingerprints = None

        logger.info(
            "Reconcile: scanned %d .nls (%d unreadable), %d files on disk, "
            "%d rescued from manifest, %d phantom index entries dropped. "
            "Memory list size: %d (was %d).",
            n_scanned, n_no_manifest, len(fs_by_path),
            rescued_count, len(phantom_paths),
            len(self._memories), prev_total,
        )

        # 4. Rewrite index.jsonl to match the reconciled state exactly.
        self._rewrite_index_jsonl()

    def _rewrite_index_jsonl(self) -> None:
        """Atomically rewrite index.jsonl to match self._memories exactly."""
        jsonl_path = self._index_path.with_suffix(".jsonl")
        try:
            tmp = jsonl_path.with_suffix(".jsonl.rebuild")
            with open(tmp, "w", encoding="utf-8") as f:
                for m in self._memories:
                    f.write(json.dumps(asdict(m), separators=(",", ":")) + "\n")
            tmp.replace(jsonl_path)
            self._last_flushed_idx = len(self._memories)
            logger.info("Rewrote %s with %d entries",
                        jsonl_path, len(self._memories))
        except Exception:
            logger.warning("Failed to rewrite index.jsonl", exc_info=True)

    def _compact_jsonl_on_load(self, jsonl_path: Path) -> None:
        """Rewrite JSONL with one line per unique memory (current state).

        Called when _load() detects duplicate entries. Atomic via tmp+rename.
        """
        try:
            tmp = jsonl_path.with_suffix(".jsonl.compact")
            with open(tmp, "w", encoding="utf-8") as f:
                for m in self._memories:
                    f.write(json.dumps(asdict(m), separators=(",", ":")) + "\n")
            tmp.replace(jsonl_path)
            logger.info("Compacted JSONL on load: %d unique entries -> %s",
                        len(self._memories), jsonl_path)
        except Exception:
            logger.warning("JSONL compaction on load failed", exc_info=True)

    def _compact_bm25_on_load(self, jsonl_path: Path) -> None:
        """Rewrite BM25 JSONL with one line per unique entry (sid-keyed)."""
        try:
            tmp = jsonl_path.with_suffix(".jsonl.compact")
            with open(tmp, "w", encoding="utf-8") as f:
                for sid in sorted(self._bm25_entries.keys()):
                    f.write(json.dumps(
                        {sid: self._bm25_entries[sid]},
                        separators=(",", ":"),
                    ) + "\n")
            tmp.replace(jsonl_path)
            logger.info("Compacted BM25 JSONL on load: %d unique entries",
                        len(self._bm25_entries))
        except Exception:
            logger.warning("BM25 JSONL compaction on load failed", exc_info=True)

    def compact(self) -> dict:
        """Rewrite JSONL + BM25 to one entry per memory and sweep orphans.

        Use this after bulk ingestion or to recover from duplication.
        Returns stats dict.
        """
        stats = {
            "memories_before": len(self._memories),
            "jsonl_lines_removed": 0,
            "bm25_lines_removed": 0,
            "orphan_nls_detected": 0,   # DETECTED — never auto-deleted
        }

        # 1. Compact index.jsonl
        jsonl_path = self._index_path.with_suffix(".jsonl")
        if jsonl_path.exists():
            old_lines = sum(1 for _ in open(jsonl_path, encoding="utf-8"))
            tmp = jsonl_path.with_suffix(".jsonl.compact")
            with open(tmp, "w", encoding="utf-8") as f:
                for m in self._memories:
                    f.write(json.dumps(asdict(m), separators=(",", ":")) + "\n")
            tmp.replace(jsonl_path)
            stats["jsonl_lines_removed"] = old_lines - len(self._memories)
            self._last_flushed_idx = len(self._memories)

        # 2. Compact bm25_data.jsonl (sid-keyed)
        bm25_jsonl = self._dir / "bm25_data.jsonl"
        if bm25_jsonl.exists():
            old_lines = sum(1 for _ in open(bm25_jsonl, encoding="utf-8"))
            tmp = bm25_jsonl.with_suffix(".jsonl.compact")
            with open(tmp, "w", encoding="utf-8") as f:
                for sid in sorted(self._bm25_entries.keys()):
                    entry = {kk: vv for kk, vv in self._bm25_entries[sid].items()
                             if kk not in ("turn_embs", "aq_embs")}
                    f.write(json.dumps({sid: entry}, separators=(",", ":")) + "\n")
            tmp.replace(bm25_jsonl)
            stats["bm25_lines_removed"] = old_lines - len(self._bm25_entries)

        # 3. DETECT orphan NLS files but NEVER delete them.
        #
        # The previous implementation deleted any .nls file whose name wasn't
        # in self._memories. That is unsafe because the in-memory index can
        # be a stale subset of the real filesystem state (e.g. after a
        # dedup-on-load drop, or after a partial restart). Deleting on that
        # basis is what caused the mass golden-KV file loss on 2026-04-14.
        #
        # We now only REPORT orphans in stats. A human (or an explicit
        # sweep_orphans() admin method) must decide to delete.
        indexed_paths = {Path(m.kv_path).name for m in self._memories if m.kv_path}
        capture_dirs = [self._capture_dir, self._dir]
        seen: set[str] = set()
        for cap_dir in capture_dirs:
            if not cap_dir.exists() or str(cap_dir) in seen:
                continue
            seen.add(str(cap_dir))
            for nls_file in cap_dir.glob("kv_snapshot_*.nls"):
                if nls_file.name not in indexed_paths:
                    stats["orphan_nls_detected"] += 1

        logger.info(
            "Compact: jsonl -%d lines, bm25 -%d lines, orphan_nls_detected=%d "
            "(NOT deleted — use sweep_orphans() if intentional), memories=%d",
            stats["jsonl_lines_removed"], stats["bm25_lines_removed"],
            stats["orphan_nls_detected"], len(self._memories),
        )
        return stats

    def sweep_orphans(self, dry_run: bool = True) -> dict:
        """Explicit orphan sweep. Never called automatically.

        With dry_run=True (default) it only returns the list of candidates.
        With dry_run=False it actually deletes. Intended to be invoked
        manually by an operator after confirming the index is correct.
        """
        indexed_paths = {Path(m.kv_path).name for m in self._memories if m.kv_path}
        victims: list[str] = []
        seen: set[str] = set()
        for cap_dir in (self._capture_dir, self._dir):
            if not cap_dir.exists() or str(cap_dir) in seen:
                continue
            seen.add(str(cap_dir))
            for nls_file in cap_dir.glob("kv_snapshot_*.nls"):
                if nls_file.name not in indexed_paths:
                    victims.append(str(nls_file))
        if not dry_run:
            removed = 0
            for v in victims:
                try:
                    Path(v).unlink()
                    removed += 1
                except OSError:
                    pass
            logger.warning("sweep_orphans: DELETED %d files", removed)
            return {"removed": removed, "candidates": victims}
        logger.info("sweep_orphans dry-run: %d candidates", len(victims))
        return {"removed": 0, "candidates": victims}

    def _cleanup_orphans(self):
        """Remove .pt files not referenced by any memory in the index."""
        indexed_paths = {Path(m.kv_path).name for m in self._memories}
        removed = 0
        for pt_file in self._dir.glob("kv_snapshot_*.pt"):
            if pt_file.name not in indexed_paths:
                try:
                    pt_file.unlink()
                    removed += 1
                except OSError:
                    pass
        if removed:
            logger.info("Memory cleanup: removed %d orphan .pt files", removed)

    # ── Ingestion mode ─────────────────────────────────────────────
    # During bulk ingestion we capture thousands of memories. Keeping
    # all BM25 data, turn embeddings, and Ariadne embeddings in RAM
    # causes OOM on unified-memory machines (GB10). Ingestion mode
    # flushes these caches to disk periodically and only keeps a
    # sliding window of _memories + _fingerprints for dedup.

    def enable_ingestion_mode(self, enabled: bool = True) -> None:
        if self._ingestion_mode and not enabled:
            self._final_flush()
        self._ingestion_mode = enabled
        self._ingest_add_count = 0
        self._ingest_fp_buffer.clear()
        logger.info("MemoryStore ingestion_mode=%s", enabled)

    def _final_flush(self) -> None:
        """Flush any remaining in-memory caches to disk.

        Called when leaving ingestion mode or at shutdown to ensure
        no BM25/embedding data is lost.
        """
        unflushed_bm25 = len(self._bm25_entries)
        unflushed_turn = len(self._turn_embs)
        unflushed_ar = len(self._ariadne_embs)
        unflushed_fp = len(self._ingest_fp_buffer)
        if unflushed_bm25 or unflushed_turn or unflushed_ar or unflushed_fp:
            logger.info(
                "Final flush: %d bm25, %d turn, %d ariadne, %d fp entries to disk",
                unflushed_bm25, unflushed_turn, unflushed_ar, unflushed_fp,
            )
            self._save()
            self._save_bm25()
            self._flush_ingest_fingerprints()
            self._ingest_fp_buffer.clear()
            for sid in self._turn_embs:
                self._save_embeddings_for(sid)
            for sid in self._ariadne_embs:
                if sid not in self._turn_embs:
                    self._save_embeddings_for(sid)

    def __del__(self):
        if self._ingestion_mode:
            try:
                self._final_flush()
            except Exception:
                pass
        elif self._fp_dirty:
            try:
                self._persist_fingerprints()
            except Exception:
                pass

    def flush_ingest_caches(self) -> None:
        """Save all data to disk, then release RAM-heavy caches.

        After this call the on-disk files are the source of truth.
        Retrieval will NOT work until caches are reloaded (which
        happens automatically at next startup or by calling _load_bm25).
        """
        self._save()
        self._save_bm25()
        self._flush_ingest_fingerprints()

        # Save per-memory embeddings that were deferred in ingestion mode
        for sid in self._turn_embs:
            self._save_embeddings_for(sid)
        for sid in self._ariadne_embs:
            if sid not in self._turn_embs:
                self._save_embeddings_for(sid)

        n_bm25 = len(self._bm25_entries)
        n_turn = len(self._turn_embs)
        n_ariadne = len(self._ariadne_embs)
        n_fp = len(self._ingest_fp_buffer)

        self._bm25_entries.clear()
        self._turn_embs.clear()
        self._ariadne_embs.clear()
        self._ingest_fp_buffer.clear()
        self._idf.clear()
        self._bigram_idf.clear()
        self._idf_dirty = True

        import gc
        gc.collect()

        logger.info(
            "Ingest cache flush: released %d bm25, %d turn_emb, "
            "%d ariadne_emb, %d fp entries from RAM (store size=%d)",
            n_bm25, n_turn, n_ariadne, n_fp, len(self._memories),
        )

    def _flush_ingest_fingerprints(self) -> None:
        """Append buffered fingerprints to fingerprints.npy incrementally."""
        if not self._ingest_fp_buffer:
            return
        batch = np.array(self._ingest_fp_buffer, dtype=np.float32)
        if self._fp_path.exists():
            try:
                existing = np.load(str(self._fp_path))
                if existing.shape[1] == batch.shape[1]:
                    combined = np.vstack([existing, batch])
                else:
                    logger.warning(
                        "Fingerprint dim changed (%d→%d), starting fresh",
                        existing.shape[1], batch.shape[1],
                    )
                    combined = batch
            except Exception:
                combined = batch
        else:
            combined = batch
        np.save(str(self._fp_path), combined)
        logger.info(
            "Flushed %d fingerprints (dim=%d) → total %d on disk",
            len(batch), batch.shape[1], combined.shape[0],
        )

    def add(
        self,
        token_ids: list[int],
        kv_path: str,
        num_tokens: int,
        ring_type: str = "general",
        user_id: str = "default",
        project_id: str = "",
        session_id: str = "",
        always_inject: bool = False,
        source: str = "auto_capture",
        description: str = "",
        role: str = "",
        block_hash: str = "",
        parent_hash: str = "",
        prev_hash: str = "",
        turn_index: int = -1,
        base_session_id: str = "",
        rope_start: int = 0,
        meta_score: float = 0.0,
        sys_prompt_hash: str = "",
        provenance: str = "",
        is_compaction_context: bool = False,
    ) -> str:
        """Add a labeled memory. Deduplicates within the same ring+user+session.

        ``provenance`` (JL #6 / Layer 2): origin/trust class of this
        memory. Empty string defaults to ``default_provenance_for_role(role)``;
        callers may override (e.g. promotion of an IMAGINED rollout to
        CALIBRATED after reality validates it). Verifier reads this
        field; nothing branches on it yet at this stage.
        """
        if not provenance:
            provenance = default_provenance_for_role(role)
        elif provenance not in PROVENANCE_VALUES:
            logger.warning(
                "MemoryStore.add: unknown provenance %r, defaulting to "
                "role-based (%s)",
                provenance, default_provenance_for_role(role),
            )
            provenance = default_provenance_for_role(role)
        fingerprint = compute_fingerprint(token_ids)

        # KL #655: prompt_hash must be (a) scoped to user_id + session_id so
        # cross-user / cross-session captures cannot collide, and (b) computed
        # from the TAIL of the token sequence — the leading tokens include
        # phantom tokens from injected memories which would otherwise produce
        # spurious matches across unrelated captures that happened to retrieve
        # similar memories. The tail is dominated by user content.
        TAIL_TOKENS = 256
        tail = token_ids[-TAIL_TOKENS:] if len(token_ids) >= TAIL_TOKENS else list(token_ids)
        scope_prefix = (
            f"u={user_id}|s={session_id}|p={project_id}|".encode("utf-8")
        )
        prompt_hash = hashlib.sha256(
            scope_prefix + struct.pack(f"<{len(tail)}I", *tail)
        ).hexdigest()[:16]

        if always_inject is False and ring_type in ALWAYS_INJECT_RINGS:
            always_inject = True

        # O(1) exact dedup via prompt_hash (covers re-ingestion and retries
        # within the same user+session). With the scoped hash above, two
        # different sessions can never collide here.
        if not self._ingestion_mode and prompt_hash in self._hash_index:
            idx = self._hash_index[prompt_hash]
            if idx < len(self._memories):
                old = self._memories[idx]
                # Defensive checks: hash scoping already encodes these, but
                # we re-verify because the index could be stale across an
                # upgrade or partial reload.
                same_ring = old.ring_type == ring_type
                same_user = old.user_id == user_id
                same_session = bool(session_id) and old.session_id == session_id
                if same_ring and same_user and same_session:
                    # Atomic-ish dedup: only swap pointers if the NEW file
                    # is actually present. Without this guard, a half-written
                    # capture would cause us to delete a perfectly good old
                    # file and leave the index pointing at a missing one.
                    new_exists = bool(kv_path) and Path(kv_path).exists()
                    if not new_exists:
                        logger.warning(
                            "Dedup abort: new kv_path missing (id=%s, "
                            "old=%s, new=%s); keeping old record",
                            old.id, old.kv_path, kv_path,
                        )
                        return old.id

                    old_path = old.kv_path
                    old.kv_path = kv_path
                    old.num_tokens = num_tokens
                    old.timestamp = time.time()
                    old.prompt_hash = prompt_hash
                    old.description = description or old.description
                    old.project_id = project_id or old.project_id
                    self._fp_set(idx, fingerprint)
                    # Mark this index dirty so the in-place update is
                    # actually persisted (KL #655 atomic rewrite path).
                    self._dirty_indices.add(idx)
                    if old_path and old_path != kv_path:
                        try:
                            Path(old_path).unlink(missing_ok=True)
                        except OSError:
                            pass
                    self._save()
                    return old.id

        mem_id = f"mem_{int(time.time()*1000)}_{len(self._memories)}"

        memory = Memory(
            id=mem_id,
            kv_path=kv_path,
            num_tokens=num_tokens,
            timestamp=time.time(),
            ring_type=ring_type,
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            always_inject=always_inject,
            source=source,
            prompt_hash=prompt_hash,
            description=description,
            role=role,
            block_hash=block_hash,
            parent_hash=parent_hash,
            prev_hash=prev_hash,
            turn_index=turn_index,
            base_session_id=base_session_id,
            rope_start=rope_start,
            meta_score=meta_score,
            sys_prompt_hash=sys_prompt_hash,
            provenance=provenance,
            is_compaction_context=is_compaction_context,
        )

        self._memories.append(memory)
        self._hash_index[prompt_hash] = len(self._memories) - 1

        if not self._ingestion_mode:
            self._fp_append(fingerprint)
            if len(self._memories) > self._max_memories:
                self._evict()
            self._save()
        else:
            # Ingestion mode: accumulate fingerprints incrementally to disk
            # instead of building a huge in-memory matrix.
            self._ingest_fp_buffer.append(fingerprint)
            self._ingest_add_count += 1
            if self._ingest_add_count % self.INGEST_FLUSH_EVERY == 0:
                self.flush_ingest_caches()

        return mem_id

    def search(
        self,
        query_token_ids: list[int],
        top_k: int = 5,
        min_similarity: float = SIMILARITY_THRESHOLD,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        include_always_inject: bool = True,
    ) -> list[tuple[Memory, float]]:
        """Search memories with optional user/project filtering.

        Returns memories sorted by injection_priority (ring order),
        not by similarity. Similarity is only used for filtering.
        """
        if not self._memories or self._fingerprints is None:
            return []

        query_fp = compute_fingerprint(query_token_ids)

        if query_fp.shape[0] != self._fingerprints.shape[1]:
            logger.warning(
                "Fingerprint dim mismatch (query=%d, stored=%d), migrating",
                query_fp.shape[0], self._fingerprints.shape[1],
            )
            new_fp = np.zeros(
                (len(self._memories), query_fp.shape[0]), dtype=np.float32
            )
            self._fingerprints = new_fp
            self._save()

        sims = batch_cosine_similarity(query_fp, self._fingerprints[:len(self._memories)])

        now = time.time()
        results: list[tuple[Memory, float]] = []
        always_inject_results: list[tuple[Memory, float]] = []

        for i, mem in enumerate(self._memories):
            # User/project filter
            if user_id is not None and mem.user_id != user_id:
                if not (include_always_inject and mem.always_inject):
                    continue

            if project_id is not None and mem.project_id and mem.project_id != project_id:
                if not mem.always_inject:
                    continue

            # Recency + frequency boost
            age_hours = max((now - mem.timestamp) / 3600, 0.01)
            recency_boost = 0.03 * math.exp(-age_hours / 48)
            freq_boost = 0.01 * min(mem.access_count, 10)
            adjusted_sim = float(sims[i]) + recency_boost + freq_boost

            if mem.always_inject and include_always_inject:
                always_inject_results.append((mem, adjusted_sim))
            elif adjusted_sim >= min_similarity:
                results.append((mem, adjusted_sim))

        # Always-inject memories are included regardless of similarity
        # but still sorted by priority
        all_results = always_inject_results + results

        # Deduplicate by ring_type (keep highest similarity per ring)
        seen_rings: dict[str, tuple[Memory, float]] = {}
        for mem, sim in all_results:
            key = f"{mem.ring_type}:{mem.project_id}"
            if key not in seen_rings or sim > seen_rings[key][1]:
                seen_rings[key] = (mem, sim)

        deduped = list(seen_rings.values())

        # Sort by injection priority (ring order), not similarity
        deduped.sort(key=lambda x: x[0].injection_priority)

        # Limit to top_k
        deduped = deduped[:top_k]

        # Update access counts
        for mem, _ in deduped:
            mem.access_count += 1
            mem.last_accessed = now

        return deduped

    # ── DeltaNet delta-energy scoring ────────────────────────────
    # Model-native fact-vs-question signal extracted from DeltaNet SSM
    # recurrent states. We compute structured delta fingerprints (norm/mean/std
    # per head per layer) and then score each memory by cosine similarity
    # to the "factual centroid" — the average fingerprint of memories with
    # low meta_score (likely facts). Factual memories cluster together in
    # delta-space; questions diverge.

    def _load_delta_fingerprints(self) -> None:
        """Load or compute DeltaNet delta fingerprints + factual affinity scores."""
        if not self._memories:
            return

        cache_path = self._dir / "delta_energy.npy"
        n = len(self._memories)

        if cache_path.exists():
            try:
                cached = np.load(str(cache_path))
                if cached.shape[0] == n:
                    self._delta_energy = cached
                    nonzero = int(np.count_nonzero(cached))
                    logger.info(
                        "Loaded DeltaNet fact scores: %d memories, %d non-zero",
                        n, nonzero,
                    )
                    # KL #651: also try to load cached delta_signal (the new
                    # signed Q-vs-F signal). If shape mismatches or absent,
                    # _rebuild_user_centroids will repopulate it.
                    sig_path = self._dir / "delta_signal.npy"
                    if sig_path.exists():
                        try:
                            cached_sig = np.load(str(sig_path))
                            if cached_sig.shape[0] == n:
                                self._delta_signal = cached_sig
                                logger.info(
                                    "Loaded delta_signal cache (KL #651): %d entries",
                                    n,
                                )
                        except Exception as e:
                            logger.debug("delta_signal cache load failed: %s", e)
                    # KL #650: cache only stores final scores, not the centroids
                    # used to derive them. Rebuild centroids in the background
                    # so incremental capture updates can score against them.
                    try:
                        self._rebuild_user_centroids()
                    except Exception as e:
                        logger.debug("Centroid rebuild failed (non-fatal): %s", e)
                    return
                logger.info(
                    "Delta fact cache stale (%d vs %d), recomputing",
                    cached.shape[0], n,
                )
            except Exception as e:
                logger.warning("Failed to load delta fact cache: %s", e)

        self._recompute_delta_fact_scores()

    def _rebuild_user_centroids(self) -> None:
        """Rebuild per-user fact + question centroids and delta_signal from cache.

        KL #650 / KL #651: complementary to the cache load path. Loads `.nls`
        files for both fact (`meta<0.3`) and question (`meta>0.7`) memories,
        builds both centroids, and recomputes `_delta_signal` per memory so the
        API server process — which never bulk-recomputes — has the runtime
        signal available for capture-gate / retrieval debuff / streaming probe.

        JL #19.2 Phase 5: when `delta_fingerprints.npy` is present and
        valid, use cached fingerprints instead of re-loading every
        `.nls` file. Only memories beyond the cached prefix
        (append-only suffix) require an actual snapshot load. Boot
        cost goes from O(N × snapshot_load + N × delta_compute) to
        O(N × cache_load + suffix × snapshot_load).
        """
        from nls_vllm_plugin.nls_format import load_nls

        genesis_snap = self._get_genesis_snap_cached()
        if genesis_snap is None:
            return

        n = len(self._memories)
        if self._delta_signal is None or self._delta_signal.shape[0] != n:
            self._delta_signal = np.zeros(n, dtype=np.float32)

        # JL #19.2 Phase 5: try to load the persisted fingerprint cache.
        # On hit, we still need to compute fingerprints for any memories
        # appended after the cache was last written.
        cached_fps = self._try_load_delta_fp_cache(genesis_snap)
        if cached_fps is not None:
            self._delta_fp_cache = cached_fps
            self._ensure_delta_fp_cache_size(n)
        else:
            self._delta_fp_cache_genesis_hash = (
                self._compute_genesis_hash(genesis_snap)
            )
        # JL #20.5e Step 1: with the id-keyed cache the returned
        # array is already current_count rows tall; per-row hits are
        # detected via norm > 1e-8 instead of position < cached_count.
        # cache_recovered tracks the actual realigned-row count for
        # the log line below; it's NOT a "first N rows are valid"
        # boundary anymore.
        cache_recovered = 0
        if cached_fps is not None:
            for i in range(cached_fps.shape[0]):
                if float(np.linalg.norm(cached_fps[i])) > 1e-8:
                    cache_recovered += 1

        user_groups: dict[str, list[int]] = {}
        for i, mem in enumerate(self._memories):
            user_groups.setdefault(mem.user_id, []).append(i)

        n_fact = n_q = 0
        suffix_loaded = 0
        for uid, indices in user_groups.items():
            fact_fps: list[np.ndarray] = []
            q_fps: list[np.ndarray] = []
            user_fps: dict[int, np.ndarray] = {}
            for i in indices:
                mem = self._memories[i]
                # JL #19.2 Phase 5 / JL #20.5e Step 1: cache hit branch
                # — read the pre-computed (un-normalized) fingerprint
                # and only normalize. No `.nls` IO, no DeltaNet
                # recomputation. Norm-based hit detection (id-keyed):
                # zero rows mean "no cached fp for this position",
                # fall through to the recompute path below.
                if cached_fps is not None:
                    fp_raw = cached_fps[i]
                    norm = float(np.linalg.norm(fp_raw))
                    if norm > 1e-8:
                        fp_arr = (
                            (fp_raw / norm).astype(np.float32, copy=False)
                        )
                        user_fps[i] = fp_arr
                        if mem.meta_score < 0.3:
                            fact_fps.append(fp_arr)
                        elif mem.meta_score > 0.7:
                            q_fps.append(fp_arr)
                        continue

                if not mem.kv_path or not Path(mem.kv_path).exists():
                    continue
                try:
                    snap = load_nls(mem.kv_path)
                    fp = self._compute_delta_fingerprint(snap, genesis_snap)
                    if fp is None:
                        continue
                    suffix_loaded += 1
                    # JL #19.2 Phase 5: ALSO populate the cache for the
                    # suffix (append) memories so the next restart sees a
                    # complete cache.
                    self._ensure_delta_fp_cache_size(n)
                    self._delta_fp_cache[i] = np.asarray(
                        fp, dtype=np.float32,
                    )
                    fp_arr = np.array(fp, dtype=np.float32)
                    fp_arr = np.nan_to_num(fp_arr)
                    norm = np.linalg.norm(fp_arr)
                    if norm > 1e-8:
                        fp_arr /= norm
                    user_fps[i] = fp_arr
                    if mem.meta_score < 0.3:
                        fact_fps.append(fp_arr)
                    elif mem.meta_score > 0.7:
                        q_fps.append(fp_arr)
                except Exception:
                    continue
            if len(fact_fps) < 2:
                continue

            # JL #19.2 Phase 4: stash the un-normalized cumulative sum +
            # count alongside the normalized centroid so the per-add
            # running update can keep the centroid current without a full
            # rebuild. Re-deriving these on next restart is the consistency
            # anchor against any numerical drift accumulated by the online
            # running mean.
            fact_sum = np.sum(fact_fps, axis=0)
            self._user_fact_sums[uid] = fact_sum.astype(np.float32)
            self._user_fact_counts[uid] = len(fact_fps)
            cn = np.linalg.norm(fact_sum)
            centroid = fact_sum / cn if cn > 1e-8 else fact_sum
            self._user_centroids[uid] = centroid.astype(np.float32)
            n_fact += 1

            q_centroid = None
            if len(q_fps) >= 2:
                q_sum = np.sum(q_fps, axis=0)
                self._user_q_sums[uid] = q_sum.astype(np.float32)
                self._user_q_counts[uid] = len(q_fps)
                qn = np.linalg.norm(q_sum)
                q_centroid = q_sum / qn if qn > 1e-8 else q_sum
                q_centroid = q_centroid.astype(np.float32)
                self._user_q_centroids[uid] = q_centroid
                n_q += 1

            scores = []
            for i, fp in user_fps.items():
                fact_cos = float(np.dot(fp, centroid))
                scores.append(fact_cos)
                if q_centroid is not None:
                    self._delta_signal[i] = fact_cos - float(np.dot(fp, q_centroid))
                else:
                    self._delta_signal[i] = fact_cos
            if scores:
                self._user_delta_range[uid] = (min(scores), max(scores))

        # Persist refreshed delta_signal so subsequent restarts can skip rebuild
        try:
            np.save(str(self._dir / "delta_signal.npy"), self._delta_signal)
        except Exception:
            pass
        # JL #19.2 Phase 5: persist the refreshed fingerprint cache so
        # subsequent boots can hit the cache and skip per-memory `.nls`
        # loads entirely.
        if self._delta_fp_cache is not None:
            self._save_delta_fp_cache()
        logger.info(
            "Rebuilt centroids from cache: %d fact, %d question (KL #651) "
            "(fp_cache: %s, recovered=%d, suffix_loaded=%d)",
            n_fact, n_q,
            "hit" if cache_recovered > 0 else "miss",
            cache_recovered,
            suffix_loaded,
        )

    def _recompute_delta_fact_scores(self) -> None:
        """Compute factual affinity score per memory using DeltaNet delta fingerprints.

        1. Extract structured delta fingerprint per memory (delta = mem - genesis)
        2. Group by user_id, compute per-user factual centroid (meta_score < 0.3)
        3. Score each memory by cosine similarity to its user's factual centroid
        4. Min-max normalize per user to [0, 1]
        """
        try:
            from nls_vllm_plugin.nls_format import load_nls, read_manifest
        except ImportError:
            logger.warning("Cannot import nls_format; delta fact disabled")
            return

        import torch

        n = len(self._memories)

        genesis_snap = self._get_genesis_snap_cached()
        if genesis_snap is None:
            logger.warning("Delta fact: no reference snapshot, disabled")
            return

        t0 = time.time()
        # JL #19.2 Phase 5: try the persisted fingerprint cache before
        # the per-memory `.nls` load loop. On cache hit, only the suffix
        # (memories appended since the last save) needs a snapshot load.
        cached_fps = self._try_load_delta_fp_cache(genesis_snap)
        if cached_fps is not None:
            self._delta_fp_cache = cached_fps
            self._ensure_delta_fp_cache_size(n)
        else:
            self._delta_fp_cache_genesis_hash = (
                self._compute_genesis_hash(genesis_snap)
            )

        # Phase 1: extract all fingerprints
        all_fps: dict[int, np.ndarray] = {}  # index -> normalized fingerprint
        computed = 0
        skipped = 0
        cache_hits = 0

        for i, mem in enumerate(self._memories):
            # JL #19.2 Phase 5 / JL #20.5e Step 1: cache hit branch.
            # Norm-based hit detection (id-keyed); zero rows mean
            # "no cached fp for this position", fall through to
            # snapshot-load + recompute below.
            if cached_fps is not None:
                fp_raw = cached_fps[i]
                norm = float(np.linalg.norm(fp_raw))
                if norm > 1e-8:
                    fp_arr = (fp_raw / norm).astype(np.float32, copy=False)
                    all_fps[i] = fp_arr
                    computed += 1
                    cache_hits += 1
                    if (i + 1) % 500 == 0:
                        logger.info(
                            "Delta fingerprints: %d/%d (computed=%d, "
                            "cache_hits=%d, skipped=%d)",
                            i + 1, n, computed, cache_hits, skipped,
                        )
                    continue

            if not mem.kv_path or not Path(mem.kv_path).exists():
                skipped += 1
                continue
            try:
                snap = load_nls(mem.kv_path)
                fp = self._compute_delta_fingerprint(snap, genesis_snap)
                if fp is not None:
                    fp_unnorm = np.array(fp, dtype=np.float32)
                    fp_unnorm = np.nan_to_num(fp_unnorm)
                    # JL #19.2 Phase 5: populate cache with the
                    # un-normalized fingerprint so future loads can
                    # decide normalization at read time.
                    self._ensure_delta_fp_cache_size(n)
                    self._delta_fp_cache[i] = fp_unnorm
                    norm = np.linalg.norm(fp_unnorm)
                    fp_arr = fp_unnorm / norm if norm > 1e-8 else fp_unnorm
                    all_fps[i] = fp_arr
                    computed += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

            if (i + 1) % 500 == 0:
                logger.info(
                    "Delta fingerprints: %d/%d (computed=%d, "
                    "cache_hits=%d, skipped=%d)",
                    i + 1, n, computed, cache_hits, skipped,
                )

        if computed < 3:
            logger.warning("Delta fact: only %d fingerprints, disabled", computed)
            return

        # Phase 2: per-user dual centroids — KL #651
        # The fact centroid is the mean fp of `meta<0.3` memories (model state
        # when committing content). The question centroid is the mean fp of
        # `meta>0.7` memories (model state when querying content). The cleanest
        # Q-vs-F signal is the *difference* in cosine between the two — fully
        # model-native, language-agnostic, self-bootstrapping.
        user_groups: dict[str, list[int]] = {}
        for i in all_fps:
            uid = self._memories[i].user_id
            user_groups.setdefault(uid, []).append(i)

        user_centroids: dict[str, np.ndarray] = {}
        user_q_centroids: dict[str, np.ndarray] = {}
        # JL #19.2 Phase 4: stash un-normalized cumulative sums + counts so
        # the running update on capture can advance the centroid without a
        # full rebuild.
        user_fact_sums: dict[str, np.ndarray] = {}
        user_q_sums: dict[str, np.ndarray] = {}
        user_fact_counts: dict[str, int] = {}
        user_q_counts: dict[str, int] = {}
        for uid, indices in user_groups.items():
            fact_fps = []
            q_fps = []
            for i in indices:
                ms = self._memories[i].meta_score
                if ms < 0.3:
                    fact_fps.append(all_fps[i])
                elif ms > 0.7:
                    q_fps.append(all_fps[i])
            if len(fact_fps) >= 2:
                c_sum = np.sum(fact_fps, axis=0).astype(np.float32)
                user_fact_sums[uid] = c_sum
                user_fact_counts[uid] = len(fact_fps)
                cn = np.linalg.norm(c_sum)
                c = (c_sum / cn) if cn > 1e-8 else c_sum
                user_centroids[uid] = c.astype(np.float32)
            if len(q_fps) >= 2:
                q_sum = np.sum(q_fps, axis=0).astype(np.float32)
                user_q_sums[uid] = q_sum
                user_q_counts[uid] = len(q_fps)
                qn = np.linalg.norm(q_sum)
                qc = (q_sum / qn) if qn > 1e-8 else q_sum
                user_q_centroids[uid] = qc.astype(np.float32)

        # Phase 3: compute legacy delta_energy AND the new signed delta_signal.
        # delta_energy is kept for backwards-compat (still used by some callers)
        # but delta_signal is the canonical Q-vs-F signal going forward.
        self._delta_energy = np.zeros(n, dtype=np.float32)
        self._delta_signal = np.zeros(n, dtype=np.float32)
        scored_users = set()
        users_with_q = 0
        self._user_centroids: dict[str, np.ndarray] = {}
        self._user_q_centroids: dict[str, np.ndarray] = {}
        # JL #19.2 Phase 4: commit pre-derived sums/counts.
        self._user_fact_sums = dict(user_fact_sums)
        self._user_q_sums = dict(user_q_sums)
        self._user_fact_counts = dict(user_fact_counts)
        self._user_q_counts = dict(user_q_counts)
        self._user_delta_range: dict[str, tuple[float, float]] = {}
        for uid, indices in user_groups.items():
            if uid not in user_centroids:
                continue
            centroid = user_centroids[uid]
            q_centroid = user_q_centroids.get(uid)
            raw_scores = []
            for i in indices:
                fp = all_fps[i]
                fact_cos = float(np.dot(fp, centroid))
                raw_scores.append((i, fact_cos))
                # KL #651: signed signal — positive when fp is closer to fact
                # centroid than to question centroid. Falls back to fact_cos
                # alone when no question centroid (cold start for this user).
                if q_centroid is not None:
                    q_cos = float(np.dot(fp, q_centroid))
                    self._delta_signal[i] = fact_cos - q_cos
                else:
                    self._delta_signal[i] = fact_cos

            scores_only = [s for _, s in raw_scores]
            mn, mx = min(scores_only), max(scores_only)
            for i, score in raw_scores:
                if mx - mn > 1e-10:
                    self._delta_energy[i] = (score - mn) / (mx - mn)
                else:
                    self._delta_energy[i] = 0.5
            self._user_centroids[uid] = centroid
            if q_centroid is not None:
                self._user_q_centroids[uid] = q_centroid
                users_with_q += 1
            self._user_delta_range[uid] = (mn, mx)
            scored_users.add(uid)

        cache_path = self._dir / "delta_energy.npy"
        try:
            np.save(str(cache_path), self._delta_energy)
        except Exception:
            pass
        signal_path = self._dir / "delta_signal.npy"
        try:
            np.save(str(signal_path), self._delta_signal)
        except Exception:
            pass
        # JL #19.2 Phase 5: persist the (now-fully-populated) fingerprint
        # cache so subsequent boots can hit the cache and skip per-memory
        # `.nls` snapshot loads + DeltaNet recomputation entirely.
        if self._delta_fp_cache is not None:
            self._save_delta_fp_cache()

        elapsed = time.time() - t0
        # KL #651: log signal separation diagnostics — mean delta_signal for
        # meta<0.3 vs meta>0.7 should be clearly separated for the model-native
        # signal to be useful. If they're not, fall back to meta_score.
        fact_signals = [
            float(self._delta_signal[i])
            for i in range(n) if self._memories[i].meta_score < 0.3
        ]
        q_signals = [
            float(self._delta_signal[i])
            for i in range(n) if self._memories[i].meta_score > 0.7
        ]
        sep_str = ""
        if fact_signals and q_signals:
            sep_str = f", separation: facts={np.mean(fact_signals):+.3f} questions={np.mean(q_signals):+.3f}"
        logger.info(
            "Delta fact scores computed: %d/%d memories in %.1fs, "
            "%d users with fact centroids, %d with question centroids%s "
            "(JL #19.2 Phase 5: cache_hits=%d, suffix_recomputed=%d)",
            computed, n, elapsed, len(scored_users), users_with_q, sep_str,
            cache_hits, computed - cache_hits,
        )

    def _find_and_load_genesis(self):
        """Find and load the genesis (system prompt only) .nls file."""
        from nls_vllm_plugin.nls_format import load_nls, read_manifest

        genesis_env = os.environ.get("NLS_GENESIS_PATH", "")
        if genesis_env and Path(genesis_env).exists():
            try:
                return load_nls(genesis_env)
            except Exception:
                pass

        snap_root = self._capture_dir.parent if self._capture_dir.exists() else self._dir
        for pattern in ["genesis_*.nls", "system_*.nls"]:
            for p in snap_root.glob(pattern):
                try:
                    return load_nls(p)
                except Exception:
                    continue

        for p in snap_root.glob("*.nls"):
            try:
                m = read_manifest(p)
                if m and m.get("seq_len", 999) < 150:
                    return load_nls(p)
            except Exception:
                continue
        return None

    def _get_genesis_snap_cached(self):
        """JL #19.2 Phase 4 fix: shared genesis-snap accessor for delta-fp paths.

        The two boot rebuild paths (`_rebuild_user_centroids`,
        `_recompute_delta_fact_scores`) had a `_find_and_load_genesis() or
        _compute_mean_reference()` fallback that lets cold installs without an
        explicit `genesis_*.nls` still produce sensible fingerprints. The live
        per-capture path in `update_delta_energy` (added during Phase 5) was
        missing that fallback — it returned early on `None`, so no fingerprint
        was ever computed for new captures, no `_advance_running_centroid` ran,
        and Tier 1 stayed cold-partition for any user added after boot.

        This helper provides the same fallback to all three call sites and
        memoizes the result on the instance so we don't pay
        `_compute_mean_reference`'s O(50 .nls loads) cost twice at boot or
        again per capture. The cache is a process-lifetime memo: it does not
        invalidate as memories are added, because the mean reference's only
        role is to stand in for an unknown system-prompt baseline — the
        per-memory delta against this baseline is well-defined regardless of
        how the user partition grows.
        """
        cached = getattr(self, "_genesis_snap_cache", None)
        if cached is not None:
            return cached
        cached = self._find_and_load_genesis()
        if cached is None:
            cached = self._compute_mean_reference()
        if cached is not None:
            self._genesis_snap_cache = cached
        return cached

    def _compute_mean_reference(self):
        """Compute mean DeltaNet SSM state across a sample of memories.

        Used as a neutral reference when no genesis .nls exists. The mean
        ensures no single memory gets delta=0, preserving all information.
        """
        from nls_vllm_plugin.nls_format import load_nls
        import torch

        MAX_SAMPLE = 50
        states: dict[str, list] = {}
        loaded = 0

        for mem in self._memories:
            if loaded >= MAX_SAMPLE:
                break
            if not mem.kv_path or not Path(mem.kv_path).exists():
                continue
            try:
                snap = load_nls(mem.kv_path)
                for layer_idx in DELTA_FACT_PROBE_LAYERS:
                    key = f"layer_{layer_idx}_mamba_ssm"
                    if key not in snap:
                        key = f"layer_{layer_idx}_mamba_state"
                    if key in snap:
                        t = snap[key].float()
                        if t.dim() > 3:
                            t = t[0:1]
                        states.setdefault(f"layer_{layer_idx}_mamba_ssm", []).append(t)
                loaded += 1
            except Exception:
                continue

        if loaded < 3:
            return None

        mean_snap = {}
        for key, tensors in states.items():
            stacked = torch.stack(tensors, dim=0)
            mean_snap[key] = stacked.mean(dim=0)

        logger.info("Mean reference computed from %d memories", loaded)
        return mean_snap

    @staticmethod
    def _compute_genesis_hash(genesis_snap: dict) -> str:
        """JL #19.2 Phase 5: deterministic hash of the resolved genesis
        snapshot. Used to invalidate the persisted fingerprint cache when
        the genesis changes (e.g. a new genesis .nls is dropped in, or
        the auto-computed mean reference shifts because the first ~50
        memories changed).

        We hash the SSM tensors at the probed layers — the same data
        `_compute_delta_fingerprint` differences against — so the hash
        captures exactly the input that would invalidate cached
        fingerprints.
        """
        h = hashlib.sha256()
        for layer_idx in DELTA_FACT_PROBE_LAYERS:
            for key in (
                f"layer_{layer_idx}_mamba_ssm",
                f"layer_{layer_idx}_mamba_state",
            ):
                if key in genesis_snap:
                    t = genesis_snap[key]
                    try:
                        arr = t.detach().to("cpu").float().numpy()
                    except Exception:
                        try:
                            arr = np.asarray(t, dtype=np.float32)
                        except Exception:
                            continue
                    h.update(key.encode("utf-8"))
                    h.update(np.ascontiguousarray(arr).tobytes())
                    break
        return h.hexdigest()[:32]

    def _delta_fp_cache_paths(self) -> tuple[Path, Path]:
        return (
            self._dir / "delta_fingerprints.npy",
            self._dir / "delta_fingerprints_meta.json",
        )

    def _try_load_delta_fp_cache(
        self, genesis_snap: dict,
    ) -> Optional[np.ndarray]:
        """JL #19.2 Phase 5 + JL #20.5e Step 1: load + validate the
        persisted delta fingerprint cache, with id-keyed re-alignment.

        Returns a ``(current_count, DELTA_FACT_FP_DIM)`` ndarray
        positioned in the CURRENT ``_memories`` order. Cache rows
        whose ids no longer match a current memory are dropped
        silently. Current memories with no cached fingerprint get a
        zero row (callers detect via ``np.linalg.norm(row) > 1e-8``
        and recompute that single fp on the fly). Returns ``None``
        only when the cache is missing, schema-incompatible, or its
        genesis-hash mismatches.

        Why id-keyed (JL #20.5e Step 1): the original implementation
        bailed to a full ~30–50 minute rebuild on the first
        position-vs-id mismatch. New memories or any reordering of
        ``index.jsonl`` (e.g. JSONL compaction, manual edit, recovery
        from .nls manifests) tripped the rebuild path. Mirroring the
        FFN sig cache's re-alignment pattern (added in JL #19.6 Step
        3) makes the cache load-order-tolerant: only the genuinely
        missing fps are recomputed at boot, not the entire pool.

        Validation order:
          1. Files exist on disk → bail if missing (cold start).
          2. Schema version + fp_dim + probe_layers match → bail
             if mismatched (cache produced by a different build).
          3. Genesis hash matches → bail if mismatched (the
             reference snapshot the fps were computed against has
             shifted, all fps are stale).
          4. Re-align cached rows by id; recover what we can.
        """
        fp_path, meta_path = self._delta_fp_cache_paths()
        if not fp_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            logger.debug("delta_fp meta load failed: %s", e)
            return None

        if meta.get("version") != DELTA_FP_CACHE_VERSION:
            logger.info(
                "delta_fp cache: version mismatch (%s vs %s), rebuilding",
                meta.get("version"), DELTA_FP_CACHE_VERSION,
            )
            return None
        if meta.get("fp_dim") != DELTA_FACT_FP_DIM:
            logger.info(
                "delta_fp cache: fp_dim mismatch (%s vs %s), rebuilding",
                meta.get("fp_dim"), DELTA_FACT_FP_DIM,
            )
            return None
        if meta.get("probe_layers") != list(DELTA_FACT_PROBE_LAYERS):
            logger.info(
                "delta_fp cache: probe_layers mismatch, rebuilding",
            )
            return None

        genesis_hash = self._compute_genesis_hash(genesis_snap)
        if meta.get("genesis_hash") != genesis_hash:
            logger.info(
                "delta_fp cache: genesis hash changed, rebuilding",
            )
            return None

        cached_ids = meta.get("memory_ids") or []
        cached_count = len(cached_ids)
        current_count = len(self._memories)
        if cached_count == 0:
            return None

        try:
            saved_arr = np.load(str(fp_path))
        except Exception as e:
            logger.debug("delta_fp cache load failed: %s", e)
            return None
        if saved_arr.dtype != np.float32:
            saved_arr = saved_arr.astype(np.float32)
        if saved_arr.shape != (cached_count, DELTA_FACT_FP_DIM):
            logger.info(
                "delta_fp cache: array shape %s != expected (%d, %d), "
                "discarding",
                saved_arr.shape, cached_count, DELTA_FACT_FP_DIM,
            )
            return None

        # JL #20.5e Step 1: id-keyed re-alignment. Build the saved-id
        # → row mapping once (O(cached_count)) then position each
        # current memory's fp by id lookup (O(current_count)).
        # Memories present in the cache but no longer in _memories
        # are dropped; memories present in _memories but not in the
        # cache are zero-initialized so the caller's recompute
        # branch fires only for those specific positions.
        id_to_saved_row: dict[str, int] = {
            sid: i for i, sid in enumerate(cached_ids)
        }
        new_cache = np.zeros(
            (current_count, DELTA_FACT_FP_DIM), dtype=np.float32,
        )
        recovered = 0
        for i, mem in enumerate(self._memories):
            saved_row = id_to_saved_row.get(mem.id)
            if saved_row is None:
                continue
            new_cache[i] = saved_arr[saved_row]
            recovered += 1

        self._delta_fp_cache_genesis_hash = genesis_hash
        logger.info(
            "delta_fp cache: realigned %d/%d fingerprints from %d cached "
            "rows (id-keyed re-align; %d positions need recompute)",
            recovered, current_count, cached_count,
            current_count - recovered,
        )
        return new_cache

    def _save_delta_fp_cache(self) -> None:
        """JL #19.2 Phase 5: persist the in-memory fingerprint cache to
        disk. Matches the rewrite pattern of ``delta_energy.npy`` /
        ``delta_signal.npy`` — full rewrite on every save, atomic from
        the reader's POV via numpy's tofile semantics. Acceptable at
        current scale (988 mems × 1.5KB = ~1.5MB rewrites). When the
        per-add cost becomes a bottleneck, switch to an append-only
        sidecar with periodic compaction.
        """
        if self._delta_fp_cache is None:
            return
        if self._delta_fp_cache_genesis_hash is None:
            return
        fp_path, meta_path = self._delta_fp_cache_paths()
        n = self._delta_fp_cache.shape[0]
        try:
            np.save(str(fp_path), self._delta_fp_cache[:n])
        except Exception as e:
            logger.debug("delta_fp cache save failed: %s", e)
            return
        meta = {
            "version": DELTA_FP_CACHE_VERSION,
            "fp_dim": DELTA_FACT_FP_DIM,
            "probe_layers": list(DELTA_FACT_PROBE_LAYERS),
            "memory_count": n,
            "memory_ids": [self._memories[i].id for i in range(n)],
            "genesis_hash": self._delta_fp_cache_genesis_hash,
        }
        try:
            meta_path.write_text(json.dumps(meta))
        except Exception as e:
            logger.debug("delta_fp meta save failed: %s", e)

    def _ensure_delta_fp_cache_size(self, new_size: int) -> None:
        """Grow the fingerprint cache to at least `new_size` rows."""
        if self._delta_fp_cache is None:
            self._delta_fp_cache = np.zeros(
                (new_size, DELTA_FACT_FP_DIM), dtype=np.float32,
            )
            return
        if self._delta_fp_cache.shape[0] >= new_size:
            return
        new_arr = np.zeros((new_size, DELTA_FACT_FP_DIM), dtype=np.float32)
        new_arr[: self._delta_fp_cache.shape[0]] = self._delta_fp_cache
        self._delta_fp_cache = new_arr

    # ── JL #19.6 Step 3: Tier 2 FFN signature cache ─────────────────
    # Mirrors the delta_fp pattern verbatim. The cache is a parallel
    # (N, FFN_SIG_DIM) float16 array indexed by memory position; it is
    # populated incrementally at capture time by snapshot_connector and
    # persisted alongside delta_fingerprints.npy. Loaded at boot and
    # validated against the same genesis hash as the delta_fp cache so
    # both caches invalidate together when the genesis snapshot shifts.

    def _ffn_sig_cache_paths(self) -> tuple[Path, Path]:
        return (
            self._dir / "ffn_signatures.npy",
            self._dir / "ffn_signatures_meta.json",
        )

    def _load_ffn_signatures(self) -> None:
        """Try to load the persisted FFN signature cache at boot.

        On hit, the cache is available immediately for Tier 2; new
        memories captured via the snapshot_connector hook append in-
        place. On miss (no file, schema mismatch, dim mismatch, or id
        mismatch), the cache stays at ``None`` and Tier 2 returns
        ``INCONCLUSIVE`` for memories that don't have a signature.

        Unlike the delta_fp cache there is no recompute fallback —
        FFN signatures derive from per-token routing decisions which
        are not persisted in the .nls snapshot. A one-time replay pass
        would be required to backfill old memories; out of scope for
        the JL #19.6 cold ship which runs on a fresh partition
        (cal_v4_dense_milan).
        """
        if not self._memories:
            return

        genesis_snap = self._get_genesis_snap_cached()
        if genesis_snap is None:
            return
        cached = self._try_load_ffn_sig_cache(genesis_snap)
        if cached is not None:
            self._ffn_sig_cache = cached
            self._ensure_ffn_sig_cache_size(len(self._memories))
            logger.info(
                "FFN sig cache: loaded %d/%d signatures",
                cached.shape[0], len(self._memories),
            )
        else:
            self._ffn_sig_cache_genesis_hash = (
                self._compute_genesis_hash(genesis_snap)
            )
            logger.info(
                "FFN sig cache: cold-start (no valid cache on disk); "
                "Tier 2 will populate incrementally via capture hook",
            )

    def _try_load_ffn_sig_cache(
        self, genesis_snap: dict,
    ) -> Optional[np.ndarray]:
        """Load + validate the persisted FFN signature cache.

        Validation: schema version, dim, num_layers, num_experts, and
        genesis hash. Unlike the delta_fp cache (which assumes
        ``_memories`` is loaded in the same order it was saved in),
        FFN sigs are re-aligned to the current ``_memories`` order via
        the saved ``memory_ids → row`` mapping. This tolerates JSONL
        compaction and dict-ordering shifts during ``_load()`` that
        would otherwise force a full discard at boot.

        Memories that exist in the cache but are no longer in
        ``_memories`` (deleted, evicted) are dropped silently. Memories
        in ``_memories`` that have no cached signature get a zero row
        (Tier 2's ``get_ffn_sig`` returns None on near-zero norm, so
        the consumer correctly treats those as "no sig yet" and the
        capture path will fill them on next observation).
        """
        fp_path, meta_path = self._ffn_sig_cache_paths()
        if not fp_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            logger.debug("ffn_sig meta load failed: %s", e)
            return None

        if meta.get("version") != FFN_SIG_CACHE_VERSION:
            logger.info(
                "ffn_sig cache: version mismatch (%s vs %s), discarding",
                meta.get("version"), FFN_SIG_CACHE_VERSION,
            )
            return None
        if meta.get("dim") != FFN_SIG_DIM:
            logger.info(
                "ffn_sig cache: dim mismatch (%s vs %s), discarding",
                meta.get("dim"), FFN_SIG_DIM,
            )
            return None
        if meta.get("num_layers") != FFN_SIG_NUM_LAYERS:
            logger.info(
                "ffn_sig cache: num_layers mismatch (%s vs %s), discarding",
                meta.get("num_layers"), FFN_SIG_NUM_LAYERS,
            )
            return None
        if meta.get("num_experts") != FFN_SIG_NUM_EXPERTS:
            logger.info(
                "ffn_sig cache: num_experts mismatch (%s vs %s), discarding",
                meta.get("num_experts"), FFN_SIG_NUM_EXPERTS,
            )
            return None

        genesis_hash = self._compute_genesis_hash(genesis_snap)
        if meta.get("genesis_hash") != genesis_hash:
            logger.info(
                "ffn_sig cache: genesis hash changed, discarding",
            )
            return None

        cached_ids = meta.get("memory_ids") or []
        cached_count = len(cached_ids)
        current_count = len(self._memories)
        if cached_count == 0:
            return None

        try:
            saved_arr = np.load(str(fp_path))
        except Exception as e:
            logger.debug("ffn_sig cache load failed: %s", e)
            return None
        if saved_arr.dtype != np.float16:
            saved_arr = saved_arr.astype(np.float16)
        if saved_arr.shape != (cached_count, FFN_SIG_DIM):
            logger.info(
                "ffn_sig cache: array shape %s != expected (%d, %d), "
                "discarding",
                saved_arr.shape, cached_count, FFN_SIG_DIM,
            )
            return None

        id_to_saved_row: dict[str, int] = {
            sid: i for i, sid in enumerate(cached_ids)
        }
        new_cache = np.zeros(
            (current_count, FFN_SIG_DIM), dtype=np.float16,
        )
        recovered = 0
        for i, mem in enumerate(self._memories):
            saved_row = id_to_saved_row.get(mem.id)
            if saved_row is None:
                continue
            new_cache[i] = saved_arr[saved_row]
            recovered += 1
        logger.info(
            "ffn_sig cache: realigned %d/%d signatures from %d cached rows "
            "(reorder-tolerant load)",
            recovered, current_count, cached_count,
        )

        self._ffn_sig_cache_genesis_hash = genesis_hash
        return new_cache

    def _save_ffn_sig_cache(self) -> None:
        """Persist the in-memory FFN signature cache to disk.

        Mirrors ``_save_delta_fp_cache`` — full rewrite per save,
        atomic from the reader's POV via ``np.save``. Acceptable at
        production scale (40 KB × 100K = 4 GB rewrites are bounded by
        disk throughput, but only fire on genesis-hash recompute paths
        and capture-time appends). When the per-add cost matters,
        switch to an append-only sidecar with periodic compaction —
        same future-work pin as the delta_fp cache.
        """
        if self._ffn_sig_cache is None:
            return
        if self._ffn_sig_cache_genesis_hash is None:
            return
        fp_path, meta_path = self._ffn_sig_cache_paths()
        n = self._ffn_sig_cache.shape[0]
        try:
            np.save(str(fp_path), self._ffn_sig_cache[:n])
        except Exception as e:
            logger.debug("ffn_sig cache save failed: %s", e)
            return
        meta = {
            "version": FFN_SIG_CACHE_VERSION,
            "dim": FFN_SIG_DIM,
            "num_layers": FFN_SIG_NUM_LAYERS,
            "num_experts": FFN_SIG_NUM_EXPERTS,
            "memory_count": n,
            "memory_ids": [self._memories[i].id for i in range(n)],
            "genesis_hash": self._ffn_sig_cache_genesis_hash,
        }
        try:
            meta_path.write_text(json.dumps(meta))
        except Exception as e:
            logger.debug("ffn_sig meta save failed: %s", e)

    def _ensure_ffn_sig_cache_size(self, new_size: int) -> None:
        """Grow the FFN signature cache to at least ``new_size`` rows.

        Fresh rows initialize to all-zero — Tier 2 readers detect the
        absence of a signature via ``np.linalg.norm(row) < 1e-6`` and
        return ``INCONCLUSIVE`` rather than scoring against a zero
        vector (which would always cosine to 0 and falsely look like
        an UNGROUNDED hit on every memory).
        """
        if self._ffn_sig_cache is None:
            self._ffn_sig_cache = np.zeros(
                (new_size, FFN_SIG_DIM), dtype=np.float16,
            )
            return
        if self._ffn_sig_cache.shape[0] >= new_size:
            return
        new_arr = np.zeros((new_size, FFN_SIG_DIM), dtype=np.float16)
        new_arr[: self._ffn_sig_cache.shape[0]] = self._ffn_sig_cache
        self._ffn_sig_cache = new_arr

    def attach_ffn_sig(self, mem_id: str, sig: np.ndarray) -> bool:
        """Public API used by ``snapshot_connector._readback_and_save``.

        Stores the L2-normalized R1 signature for the memory record
        identified by ``mem_id`` into the cache and persists the cache
        to disk. Returns ``True`` on success, ``False`` if the memory
        id is unknown or the signature is the wrong shape (e.g. the
        forward pass had no real tokens to capture, producing a zero-
        norm projection — caller should not call us in that case).

        Idempotent: re-attaching for the same memory id overwrites
        the previous signature in place. Used by the dedup branch of
        ``add()`` when the same prompt is captured twice — the second
        capture's signature replaces the first.
        """
        if not isinstance(sig, np.ndarray) or sig.shape != (FFN_SIG_DIM,):
            return False
        norm = float(np.linalg.norm(sig))
        if norm < 1e-8:
            return False
        idx = -1
        for i, mem in enumerate(self._memories):
            if mem.id == mem_id:
                idx = i
                break
        if idx < 0:
            return False
        self._ensure_ffn_sig_cache_size(len(self._memories))
        if self._ffn_sig_cache_genesis_hash is None:
            genesis_snap = self._get_genesis_snap_cached()
            if genesis_snap is not None:
                self._ffn_sig_cache_genesis_hash = (
                    self._compute_genesis_hash(genesis_snap)
                )
        sig_norm = (sig / norm).astype(np.float16, copy=False)
        self._ffn_sig_cache[idx] = sig_norm
        self._save_ffn_sig_cache()
        return True

    def get_ffn_sig(self, mem_idx: int) -> Optional[np.ndarray]:
        """Public API used by ``layer6_tier2``.

        Returns the L2-normalized signature for memory at index
        ``mem_idx`` as a float32 ndarray (cosine compute precision),
        or ``None`` if no signature is recorded for that memory
        (cache miss → Tier 2 returns INCONCLUSIVE for ops that depend
        on this memory's signature).
        """
        if self._ffn_sig_cache is None:
            return None
        if mem_idx < 0 or mem_idx >= self._ffn_sig_cache.shape[0]:
            return None
        row = self._ffn_sig_cache[mem_idx]
        norm = float(np.linalg.norm(row))
        if norm < 1e-6:
            return None
        return row.astype(np.float32, copy=False)

    @staticmethod
    def _compute_delta_fingerprint(snap: dict, genesis_snap: dict) -> list:
        """Compute structured delta fingerprint for one memory.

        Per probed layer, per head: Frobenius norm, mean, std of the delta.
        Total dim = len(DELTA_FACT_PROBE_LAYERS) * 32_heads * 3_stats = 384.
        """
        import torch
        fp = []
        for layer_idx in DELTA_FACT_PROBE_LAYERS:
            ssm_key = f"layer_{layer_idx}_mamba_ssm"
            if ssm_key not in snap:
                ssm_key = f"layer_{layer_idx}_mamba_state"
            gen_key = f"layer_{layer_idx}_mamba_ssm"
            if gen_key not in genesis_snap:
                gen_key = f"layer_{layer_idx}_mamba_state"
            if ssm_key not in snap or gen_key not in genesis_snap:
                return None
            mem_ssm = snap[ssm_key].float()
            gen_ssm = genesis_snap[gen_key].float()
            if mem_ssm.dim() > 3:
                mem_ssm = mem_ssm[0]
            else:
                mem_ssm = mem_ssm.squeeze(0)
            if gen_ssm.dim() > 3:
                gen_ssm = gen_ssm[0]
            else:
                gen_ssm = gen_ssm.squeeze(0)
            delta = mem_ssm - gen_ssm                        # [32, 128, 128]
            per_head = delta.reshape(32, -1)                 # [32, 16384]
            fp.extend(per_head.norm(dim=1).numpy().tolist())
            fp.extend(per_head.mean(dim=1).numpy().tolist())
            fp.extend(per_head.std(dim=1).numpy().tolist())
        return fp

    @staticmethod
    def _frame_delta_norm(snap_curr: dict, snap_pre: dict) -> Optional[float]:
        """Sum of per-layer Frobenius norms of ``(curr_ssm − pre_ssm)``.

        JL #17 trajectory term. Unlike ``_compute_delta_fingerprint`` this
        does **not** subtract genesis — it measures the *frame-to-frame*
        SSM-state change between two consecutive captures, isolating the
        contribution of tokens added in the latter frame.

        The genesis-anchored variant is dominated by the user's
        accumulated history (chat-template + content baseline) and
        collapses consecutive captures to near-identical fingerprints
        (JL #16 finding: traj_step = 0.0 across every observed envelope).
        The frame-difference variant cancels the shared baseline by
        construction — what remains is the SSM-space magnitude of the
        latter frame's actual contribution.

        Returns the sum of per-layer Frobenius norms across
        ``DELTA_FACT_PROBE_LAYERS``, in raw SSM units (un-normalized).
        Returns ``None`` on missing/mismatched SSM keys; never raises.
        """
        try:
            import torch  # noqa: F401
        except Exception:
            return None
        total = 0.0
        for layer_idx in DELTA_FACT_PROBE_LAYERS:
            ssm_key = f"layer_{layer_idx}_mamba_ssm"
            if ssm_key not in snap_curr or ssm_key not in snap_pre:
                ssm_key = f"layer_{layer_idx}_mamba_state"
            if ssm_key not in snap_curr or ssm_key not in snap_pre:
                return None
            try:
                curr_ssm = snap_curr[ssm_key].float()
                pre_ssm = snap_pre[ssm_key].float()
                if curr_ssm.dim() > 3:
                    curr_ssm = curr_ssm[0]
                else:
                    curr_ssm = curr_ssm.squeeze(0)
                if pre_ssm.dim() > 3:
                    pre_ssm = pre_ssm[0]
                else:
                    pre_ssm = pre_ssm.squeeze(0)
                if curr_ssm.shape != pre_ssm.shape:
                    return None
                delta = curr_ssm - pre_ssm
                total += float(delta.norm())
            except Exception:
                return None
        return float(total)

    def _advance_running_centroid(
        self,
        uid: str,
        fp_arr: np.ndarray,
        *,
        is_fact: bool,
    ) -> None:
        """JL #19.2 Phase 4: extend the per-user fact or question centroid
        with one new (already-normalized) fingerprint, using a running mean
        over the un-normalized cumulative sum.

        Stored centroid is ``normalize(sum)``; storing the sum (not the
        normalized centroid) and the count together is what makes the
        update equivalent to the startup full-rebuild — running a
        normalize-of-normalized chain instead would accumulate
        direction bias because ``n*c_n`` (with normalized ``c_n``)
        differs from the un-normalized ``sum_n`` whenever the
        fingerprints aren't perfectly co-linear.

        First fingerprint for a user is treated as ``sum=fp``,
        ``count=1``, ``centroid=fp``. The minimum-population gate that
        the *startup* rebuild uses (``len(fact_fps) >= 2`` to call a
        centroid trustworthy) is intentionally *not* enforced online —
        if a user has only one fact memory, the gate's downstream
        consumers (Tier 1 / KL #651 signal) will see a single-sample
        centroid which still beats the prior "no centroid → neutral"
        path: a 1-sample centroid is a valid (if noisy) basis vector,
        and the next capture refines it.
        """
        target_sum_map = (
            self._user_fact_sums if is_fact else self._user_q_sums
        )
        target_count_map = (
            self._user_fact_counts if is_fact else self._user_q_counts
        )
        target_centroid_map = (
            self._user_centroids if is_fact else self._user_q_centroids
        )

        old_sum = target_sum_map.get(uid)
        old_count = target_count_map.get(uid, 0)
        if old_sum is None or old_sum.shape != fp_arr.shape:
            new_sum = fp_arr.astype(np.float32).copy()
            new_count = 1
        else:
            new_sum = (old_sum + fp_arr).astype(np.float32)
            new_count = old_count + 1

        target_sum_map[uid] = new_sum
        target_count_map[uid] = new_count
        new_norm = float(np.linalg.norm(new_sum))
        if new_norm > 1e-8:
            target_centroid_map[uid] = (new_sum / new_norm).astype(np.float32)
        else:
            target_centroid_map[uid] = new_sum.copy()

    def update_delta_energy(self, mem_id: str) -> None:
        """Incrementally compute delta fact score for a newly added memory.

        KL #650: prior version wrote a hard-coded 0.5 (neutral), so the
        delta-sharpened meta-debuff path never fired for fresh captures.
        Now we compute the actual cosine to the user's factual centroid
        and place it on the same min/max scale used by the bulk recompute.

        JL #19.2 Phase 4: this path now also performs the running-mean
        update of `_user_centroids` / `_user_q_centroids` (and the backing
        un-normalized `_user_fact_sums` / `_user_q_sums`). Prior to this
        fix the centroids were rebuilt only at startup, so any memory
        added during a server session was invisible to the centroid
        until the next restart — leaving cold-start users centroid-less
        and warm users locked to a stale centroid as their conversation
        evolved. The drift between rebuilds is bounded by the
        running-mean math; the next startup rebuild remains the
        consistency anchor.
        """
        if not DELTA_FACT_ENABLED or self._delta_energy is None:
            return

        idx = None
        for i, m in enumerate(self._memories):
            if m.id == mem_id:
                idx = i
                break
        if idx is None:
            return

        mem = self._memories[idx]
        if not mem.kv_path or not Path(mem.kv_path).exists():
            return

        try:
            from nls_vllm_plugin.nls_format import load_nls
            # JL #19.2 Phase 4 fix: use the cached helper which falls back to
            # `_compute_mean_reference()` when no `genesis_*.nls` exists.
            # Without this fallback, environments seeded post-boot (any new
            # user, including calibration personas) silently skipped the
            # entire centroid update on every capture, leaving Tier 1 cold.
            genesis_snap = self._get_genesis_snap_cached()
            if genesis_snap is None:
                return
            snap = load_nls(mem.kv_path)
            fp = self._compute_delta_fingerprint(snap, genesis_snap)
            if fp is None:
                return

            fp_unnorm = np.array(fp, dtype=np.float32)
            fp_unnorm = np.nan_to_num(fp_unnorm)
            norm = np.linalg.norm(fp_unnorm)
            fp_arr = fp_unnorm.copy()
            if norm > 1e-8:
                fp_arr /= norm

            n_mem = len(self._memories)
            if idx >= self._delta_energy.shape[0]:
                new_arr = np.zeros(n_mem, dtype=np.float32)
                new_arr[:self._delta_energy.shape[0]] = self._delta_energy
                self._delta_energy = new_arr
            # KL #651: keep _delta_signal sized in lockstep
            if self._delta_signal is None:
                self._delta_signal = np.zeros(n_mem, dtype=np.float32)
            elif idx >= self._delta_signal.shape[0]:
                new_sig = np.zeros(n_mem, dtype=np.float32)
                new_sig[:self._delta_signal.shape[0]] = self._delta_signal
                self._delta_signal = new_sig

            # JL #19.2 Phase 5: stash the un-normalized fingerprint into
            # the persisted cache. We store the un-normalized form so a
            # future load can decide normalization at read time and so
            # the centroid rebuild's `np.sum(fps)` remains the un-biased
            # vector mean (Phase 4 invariant).
            self._ensure_delta_fp_cache_size(n_mem)
            self._delta_fp_cache[idx] = fp_unnorm

            uid = mem.user_id

            # JL #19.2 Phase 4: running-mean centroid update before we
            # score this memory, so the new memory is scored against a
            # centroid that already reflects its own contribution
            # (matches the rebuild path's invariant where each memory is
            # scored against the centroid that includes it). Centroid
            # update only triggers for clearly-fact (meta < 0.3) or
            # clearly-question (meta > 0.7) memories — middle-band stays
            # out of both baselines, matching the rebuild's exclusion.
            ms = float(mem.meta_score)
            if ms < 0.3:
                self._advance_running_centroid(
                    uid, fp_arr, is_fact=True,
                )
            elif ms > 0.7:
                self._advance_running_centroid(
                    uid, fp_arr, is_fact=False,
                )

            centroid = self._user_centroids.get(uid)
            q_centroid = self._user_q_centroids.get(uid)
            if centroid is None or centroid.shape != fp_arr.shape:
                # No factual centroid for this user yet (e.g. cold cache load
                # or user with no meta<0.3 memories). Fall back to neutral so
                # the meta-debuff at least sees a populated entry.
                self._delta_energy[idx] = 0.5
                self._delta_signal[idx] = 0.0
            else:
                raw = float(np.dot(fp_arr, centroid))
                mn, mx = self._user_delta_range.get(uid, (raw, raw))
                # JL #19.2 Phase 4: extend the per-user delta_energy
                # range by the new raw cosine so the [0,1] normalization
                # doesn't clip against a stale max from startup. The
                # range is *monotonically* widened — never shrunk — and
                # any drift is corrected by the next startup rebuild.
                if raw < mn:
                    mn = raw
                if raw > mx:
                    mx = raw
                self._user_delta_range[uid] = (mn, mx)
                if mx - mn > 1e-10:
                    norm_score = (raw - mn) / (mx - mn)
                else:
                    norm_score = 0.5
                self._delta_energy[idx] = float(np.clip(norm_score, 0.0, 1.0))
                # KL #651: signed Q-vs-F signal (cos to fact - cos to question)
                if q_centroid is not None and q_centroid.shape == fp_arr.shape:
                    self._delta_signal[idx] = raw - float(np.dot(fp_arr, q_centroid))
                else:
                    self._delta_signal[idx] = raw  # cold start: use raw fact cos

            cache_path = self._dir / "delta_energy.npy"
            np.save(str(cache_path), self._delta_energy[:n_mem])
            try:
                np.save(str(self._dir / "delta_signal.npy"), self._delta_signal[:n_mem])
            except Exception:
                pass
            # JL #19.2 Phase 5: persist the fingerprint cache so the
            # next restart sees this just-captured fingerprint and can
            # skip the per-memory `.nls` snapshot load + DeltaNet
            # recomputation. Matches the rewrite pattern of the score
            # caches above. When the per-add rewrite cost matters at
            # scale (~10k+ memories), switch to an append-only sidecar
            # with periodic compaction.
            try:
                self._save_delta_fp_cache()
            except Exception as e:
                logger.debug("delta_fp cache persist failed: %s", e)
        except Exception as e:
            logger.debug("Delta fact update failed for %s: %s", mem_id, e)

    # ── Swiss Cheese retrieval (KL #457) ───────────────────────────

    def add_bm25_data(
        self,
        key,
        token_ids: list[int],
        turn_texts_token_ids: Optional[list[list[int]]] = None,
        ariadne_question_ids: Optional[list[list[int]]] = None,
    ) -> None:
        """Store BM25 / turn / Ariadne data for a memory.

        `key` accepts session_id (str, preferred), a Memory object, or a
        list index (int, legacy). All derived data is keyed by session_id
        (or mem.id fallback) so it survives restarts, dedup, reingest.

        Called from auto_memory.capture() right after MemoryStore.add().
        """
        sid = self._mem_key(key)
        if not sid:
            return  # nothing to key against

        tf = Counter(token_ids)
        dl = len(token_ids)
        bigrams = [
            (token_ids[i], token_ids[i + 1])
            for i in range(len(token_ids) - 1)
        ]
        bg_tf = Counter(bigrams)

        entry = {
            "tf": {str(k): v for k, v in tf.items()},
            "dl": dl,
            "bg": {f"{a},{b}": v for (a, b), v in bg_tf.items()},
        }
        self._bm25_entries[sid] = entry
        self._bm25_persisted.discard(sid)  # force re-persist on next flush
        self._idf_dirty = True
        if not self._ingestion_mode:
            self._save_bm25()

        # Per-turn embeddings → numpy binary
        turn_embs = []
        if turn_texts_token_ids:
            for ttids in turn_texts_token_ids:
                if len(ttids) >= 5:
                    turn_embs.append(self._idf_embed(ttids))
        if turn_embs:
            self._turn_embs[sid] = np.array(turn_embs, dtype=np.float16)
        if not self._ingestion_mode:
            self._save_embeddings_for(sid)

        # Ariadne question embeddings → numpy binary
        aq_embs = []
        if ariadne_question_ids:
            for aqids in ariadne_question_ids:
                if aqids:
                    aq_embs.append(self._idf_embed(aqids))
        if aq_embs:
            self._ariadne_embs[sid] = np.array(aq_embs, dtype=np.float16)
        if not self._ingestion_mode:
            self._save_embeddings_for(sid)

    def load_ariadne_cache(self, cache_path: str) -> int:
        """Bulk-load Ariadne question annotations from a JSON or JSONL file.

        JSON format: {session_id: [question_str, ...], ...}
        JSONL format: one {session_id: [question_str, ...]} per line.
        Matches memories by session_id, tokenizes questions with a simple
        whitespace approach (for IDF embed, exact tokenization not critical).
        Returns count of memories updated.
        """
        if not os.path.exists(cache_path):
            return 0

        cache: dict[str, list[str]] = {}
        try:
            if cache_path.endswith(".jsonl"):
                with open(cache_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            cache.update(entry)
                        except Exception:
                            continue
            else:
                with open(cache_path, encoding="utf-8") as f:
                    cache = json.load(f)
        except Exception:
            logger.warning("Failed to load Ariadne cache: %s", cache_path)
            return 0

        # Build the set of valid sids (memories we actually have).
        known_sids = {mem.session_id for mem in self._memories
                      if mem.session_id}

        updated = 0
        for sid, questions in cache.items():
            if sid not in known_sids or not questions:
                continue
            aq_embs = []
            for q_text in questions:
                tids = self._text_to_token_ids(q_text)
                if tids:
                    aq_embs.append(self._idf_embed(tids))
            if aq_embs:
                self._ariadne_embs[sid] = np.array(aq_embs, dtype=np.float16)
                self._save_embeddings_for(sid)
                updated += 1

        if updated:
            logger.info("Ariadne cache loaded: %d memories updated from %s",
                        updated, cache_path)
        return updated

    def _base_session(self, mem: "Memory") -> str:
        """Extract the base conversation session from a memory block.

        Used to match user blocks with assistant blocks from the same
        conversation turn, regardless of the _tN / _asst suffixes.
        """
        if mem.base_session_id:
            return mem.base_session_id
        sid = mem.session_id
        cleaned = sid.replace("_asst", "")
        if "_t" in cleaned:
            return cleaned.rsplit("_t", 1)[0]
        return cleaned

    def _assistant_funnel(
        self,
        query_token_ids: list[int],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Optional[set[str]]:
        """Coarse BM25 search over assistant blocks to find relevant base sessions.

        Returns a set of base_session_id strings whose paired user blocks
        should be included in the Swiss-Cheese candidate pool. Returns None
        if the funnel is disabled or has no assistant data.
        """
        if not ASST_FUNNEL_ENABLED:
            return None

        asst_indices = [
            i for i in self._get_candidate_indices(user_id, project_id, False)
            if self._memories[i].role == "assistant"
        ]
        if len(asst_indices) < 10:
            return None

        self._ensure_idf()

        scores = np.zeros(len(asst_indices), dtype=np.float32)
        for ci, idx in enumerate(asst_indices):
            scores[ci] = self._bm25_score(query_token_ids, idx)

        top_n = min(ASST_FUNNEL_TOP_K, len(asst_indices))
        top_ci = np.argsort(scores)[-top_n:][::-1]

        base_sessions: set[str] = set()
        for ci in top_ci:
            if scores[ci] <= 0:
                break
            mem = self._memories[asst_indices[ci]]
            base = self._base_session(mem)
            if base:
                base_sessions.add(base)

        if base_sessions:
            logger.info(
                "Assistant funnel: %d asst blocks scored, top-%d → %d unique sessions",
                len(asst_indices), top_n, len(base_sessions),
            )
        return base_sessions if base_sessions else None

    def search_swiss_cheese(
        self,
        query_token_ids: list[int],
        top_k: int = 5,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        include_always_inject: bool = True,
        role_filter: Optional[set] = None,
        query_text: Optional[str] = None,
        base_session_id: Optional[str] = None,
        boost_compaction_context: bool = False,
    ) -> list[tuple[Memory, float]]:
        """Swiss Cheese retrieval: BM25 + semantic embedding fusion.

        Layers: BM25 (lexical), Semantic (sentence-transformer dense vector)
        Falls back to BM25-only when semantic embeddings are not available.

        query_text: raw text of the user's query (avoids token→text roundtrip).
        role_filter: if set, only consider memories whose .role is in this set.
        """
        if not self._memories:
            return []

        # Candidate indices (filtered by user/project)
        candidates = self._get_candidate_indices(user_id, project_id,
                                                 include_always_inject)
        if not candidates:
            return []

        # NLS v2: filter by role if requested
        if role_filter:
            candidates = [i for i in candidates
                          if self._memories[i].role in role_filter]
            if not candidates:
                return []

        # ── Assistant-as-funnel pre-filter ──────────────────────────
        # Run coarse BM25 over assistant blocks to identify relevant
        # base sessions, then narrow user candidates to those sessions.
        # Full user pool is kept as fallback for Swiss-Cheese scoring.
        funnel_sessions: Optional[set[str]] = None
        if role_filter and "user" in role_filter and ASST_FUNNEL_ENABLED:
            funnel_sessions = self._assistant_funnel(
                query_token_ids, user_id, project_id,
            )
        if funnel_sessions:
            funnel_candidates = [
                i for i in candidates
                if self._base_session(self._memories[i]) in funnel_sessions
            ]
            logger.info(
                "Assistant funnel narrowed: %d → %d user candidates (%d sessions)",
                len(candidates), len(funnel_candidates), len(funnel_sessions),
            )
            if funnel_candidates:
                candidates = funnel_candidates

        # Ensure IDF is computed
        self._ensure_idf()

        n = len(candidates)

        # ── Layer 1: BM25 ─────────────────────────────────────────
        bm25_scores = np.zeros(n, dtype=np.float32)
        for ci, idx in enumerate(candidates):
            bm25_scores[ci] = self._bm25_score(query_token_ids, idx)

        # ── Layer 2: Semantic (sentence-transformer) ────────────────
        semantic_scores = np.zeros(n, dtype=np.float32)
        if self._semantic_embs is not None and query_text:
            embedder = SentenceEmbedder.get()
            if embedder._ensure_loaded():
                q_sem = embedder.encode_query(query_text)
                for ci, idx in enumerate(candidates):
                    semantic_scores[ci] = self._semantic_score(q_sem, idx)

        # ── Normalize each layer to [0,1] ─────────────────────────
        all_raw = {
            "BM25": bm25_scores,
            "Semantic": semantic_scores,
        }
        norm = {}
        for layer, raw in all_raw.items():
            mn, mx = raw.min(), raw.max()
            spread = mx - mn
            if spread > 1e-10:
                norm[layer] = (raw - mn) / spread
            else:
                norm[layer] = np.zeros(n, dtype=np.float32)

        # ── Confidence gates (disabled for 2-layer — both always ON) ──
        gates: dict[str, float] = {layer: 1.0 for layer in SC_LAYER_WEIGHTS}

        # ── Gated weighted fusion ─────────────────────────────────
        combined = np.zeros(n, dtype=np.float32)
        total_w = 0.0
        for layer in SC_LAYER_WEIGHTS:
            if gates.get(layer, 0) > 0:
                combined += SC_LAYER_WEIGHTS[layer] * norm[layer]
                total_w += SC_LAYER_WEIGHTS[layer]

        if total_w < 1e-8:
            for layer in SC_LAYER_WEIGHTS:
                combined += SC_LAYER_WEIGHTS[layer] * norm[layer]

        # ── Global recency decay (legacy / experimental) ──────────
        # Disabled by default: RECENCY_FLOOR=1.0 makes the multiplier a
        # no-op. Recency is now scoped to the embedding-dedup branch
        # below, where it acts as a tiebreaker between content-colliding
        # memories rather than a global age penalty. Kept reachable for
        # ablation studies via NLS_RECENCY_FLOOR<1.
        if RECENCY_ENABLED and RECENCY_FLOOR < 1.0:
            now = time.time()
            for ci, idx in enumerate(candidates):
                age_hours = max((now - self._memories[idx].timestamp) / 3600, 0.0)
                decay = max(RECENCY_FLOOR, 1.0 - RECENCY_DECAY * age_hours)
                combined[ci] *= decay

        # ── KL #639/#650: Meta-score penalty + DeltaNet fact sharpening ──
        # Demote memories tagged as meta/question/reaction so factual
        # memories win the limited FINAL_K slots. delta_energy is the
        # cosine similarity to the user's factual centroid (computed from
        # meta<0.3 memories). It tells us how vocabulary-aligned a memory
        # is with the user's facts.
        #
        # KL #650: original formula `ms * (1.3 - 0.8*de)` was BACKWARDS:
        # it cut the penalty in half for memories near the factual centroid,
        # but self-questions like "what's my dog's name?" land near the
        # centroid (they share vocabulary with "I have a golden retriever")
        # and were getting boosted instead of demoted. Inverted direction:
        # high de + meta>0 = self-question about a fact → amplified penalty.
        # KL #651: dual-centroid penalty. The model-native signed signal
        # (`_delta_signal`, in [-1,1]) is the canonical Q-vs-F signal when
        # available; we sigmoid it into a question-ness ∈ (0,1) and use the
        # MAX of (regex meta_score, model-native question_ness) so neither
        # signal can hide a question. Falls back cleanly to meta_score when
        # delta_signal is missing (cold-start, centroid-less user).
        if META_PENALTY_WEIGHT > 0:
            has_signal = (DELTA_SIGNAL_ENABLED and self._delta_signal is not None
                          and self._delta_signal.shape[0] >= len(self._memories))
            for ci, idx in enumerate(candidates):
                mem_i = self._memories[idx]
                ms = mem_i.meta_score
                qness = ms
                # KL #651 dual-centroid Q-vs-F signal targets USER content
                # where the line between "I had pasta" (fact) and "what
                # did I eat?" (question) can blur. Tool / system blocks
                # are explicitly external grounding signals (not user
                # discourse) so the dual-centroid penalty doesn't apply
                # — they trust the regex meta_score (typically 0.0 for
                # data-bearing tool responses).
                if has_signal and mem_i.role not in ("tool", "system"):
                    sig = float(self._delta_signal[idx])
                    if sig != 0.0:  # 0 = no centroid for this user → keep ms
                        model_q = _delta_signal_to_qness(sig)
                        qness = max(ms, model_q)
                if qness <= 0:
                    continue
                combined[ci] *= max(0.0, 1.0 - META_PENALTY_WEIGHT * qness)

        # ── Compaction-context boost (Arm D overflow) ───────────────
        # Memories captured on the first build-agent pass after
        # punk-records compaction_detected (transcript shrink). Scoped
        # to the same resume chain — not a global keyword heuristic.
        _compaction_boost = float(
            os.environ.get("NLS_COMPACTION_CONTEXT_BOOST", "0.25"),
        )
        if boost_compaction_context and base_session_id and _compaction_boost > 0:
            for ci, idx in enumerate(candidates):
                mem_i = self._memories[idx]
                if (
                    mem_i.is_compaction_context
                    and mem_i.base_session_id == base_session_id
                ):
                    combined[ci] += _compaction_boost

        # ── Turn-index bonus (legacy / experimental) ──────────────────
        # Disabled by default: NLS_TURN_RECENCY_BONUS=0.0. The original
        # Q6 IP-staleness fix relied on a global +turn_index bonus which
        # — like the global recency decay above — penalized older fact
        # memories vs newer fillers regardless of whether the two were
        # actually colliding. The Q6 case is now handled by the
        # collision-scoped tiebreaker in the embedding-dedup branch
        # below; the two IP memories are >95% cosine-similar so the
        # dedup loop catches them and prefers the newer one. Kept
        # reachable via env var for ablation.
        TURN_RECENCY_BONUS = float(
            os.environ.get("NLS_TURN_RECENCY_BONUS", "0.0")
        )
        if TURN_RECENCY_BONUS > 0:
            for ci, idx in enumerate(candidates):
                ti = self._memories[idx].turn_index
                if ti > 0:
                    bonus = TURN_RECENCY_BONUS * (
                        np.log1p(ti) / np.log1p(100)
                    )
                    combined[ci] += float(bonus)

        # ── Rank and return ───────────────────────────────────────
        ranked_ci = np.argsort(combined)[::-1]

        now = time.time()
        results: list[tuple[Memory, float]] = []
        seen_rings: dict[str, tuple[Memory, float]] = {}

        for ci in ranked_ci:
            idx = candidates[ci]
            mem = self._memories[idx]
            score = float(combined[ci])

            if mem.always_inject and include_always_inject:
                key = f"_ai_{mem.ring_type}:{mem.project_id}"
                if key not in seen_rings or score > seen_rings[key][1]:
                    seen_rings[key] = (mem, score)
            elif score > 0.001:
                results.append((mem, score, idx))

        # ── Cross-session semantic / collision-scoped recency dedup ─
        # Memories from different sessions can be near-duplicates
        # (e.g. "my wife is Lucia" vs "my wife is now Monica" after a
        # divorce, or "api.example.com → OLD_IP" vs "api.example.com →
        # NEW_IP" after a DNS change). Keep only the highest-scoring
        # variant when embeddings are > DEDUP_THRESHOLD.
        #
        # This is the ONLY place "newer wins" is honored. Two unrelated
        # memories never reach this branch (their cosine similarity
        # will be below DEDUP_THRESHOLD), so a recent filler like
        # "Thanks" cannot displace an older fact like "wife Monica".
        #
        # Sort key: (-score, -turn_index, -timestamp). turn_index is
        # the first-class freshness signal for V3-era captures; for
        # legacy memories with turn_index == -1 the term collapses
        # and timestamp acts as the fallback so the user's "wife
        # Lucia turn 3 → wife Monica turn 7" example resolves
        # correctly even on pre-V3 data.
        results.sort(key=lambda x: (-x[1], -x[0].turn_index, -x[0].timestamp))
        deduped: list[tuple[Memory, float]] = []
        deduped_idxs: list[int] = []

        for mem, score, idx in results:
            is_dup = False
            if self._semantic_embs is not None and idx < self._semantic_embs.shape[0]:
                mem_emb = self._semantic_embs[idx]
                for kept_idx in deduped_idxs:
                    if kept_idx < self._semantic_embs.shape[0]:
                        sim = float(np.dot(mem_emb, self._semantic_embs[kept_idx]))
                        if sim > DEDUP_THRESHOLD:
                            is_dup = True
                            break
            if not is_dup:
                deduped.append((mem, score))
                deduped_idxs.append(idx)

        # Add always-inject memories at the front
        for key, (mem, score) in seen_rings.items():
            deduped.insert(0, (mem, score))

        deduped.sort(key=lambda x: (-x[0].always_inject, x[0].injection_priority, -x[1]))
        deduped = deduped[:top_k]

        for mem, _ in deduped:
            mem.access_count += 1
            mem.last_accessed = now

        if deduped:
            gate_str = " ".join(
                f"{l}={'ON' if gates.get(l, 0) > 0 else 'off'}"
                for l in SC_LAYER_WEIGHTS
            )
            delta_str = "SHARP" if (DELTA_FACT_ENABLED and self._delta_energy is not None) else "off"
            mem_details = ", ".join(
                f"{m.session_id or m.id}({s:.2f},m={m.meta_score:.1f})"
                for m, s in deduped[:3]
            )
            logger.info(
                "Swiss-Cheese search: %d candidates, gates=[%s], delta=%s, "
                "top_score=%.3f, returning %d [%s]",
                n, gate_str, delta_str,
                float(combined[ranked_ci[0]]) if n else 0,
                len(deduped), mem_details,
            )

        return deduped

    def find_compaction_context_memories(
        self,
        user_id: str,
        base_session_id: str,
        *,
        exclude_paths: Optional[set[str]] = None,
        limit: int = 1,
    ) -> list[Memory]:
        """Latest compaction-tagged blocks on a chain (for Arm D overflow)."""
        if not user_id or not base_session_id or limit <= 0:
            return []
        excluded = exclude_paths or set()
        matches = [
            m for m in self._memories
            if m.user_id == user_id
            and m.base_session_id == base_session_id
            and m.is_compaction_context
            and m.kv_path
            and m.kv_path not in excluded
            and Path(m.kv_path).exists()
        ]
        matches.sort(key=lambda m: (m.turn_index, m.timestamp), reverse=True)
        return matches[:limit]

    # ── BM25 internals ────────────────────────────────────────────

    def _get_candidate_indices(
        self,
        user_id: Optional[str],
        project_id: Optional[str],
        include_always_inject: bool,
    ) -> list[int]:
        candidates = []
        for i, mem in enumerate(self._memories):
            if user_id is not None and mem.user_id != user_id:
                if not (include_always_inject and mem.always_inject):
                    continue
            if project_id is not None and mem.project_id and mem.project_id != project_id:
                if not mem.always_inject:
                    continue
            candidates.append(i)
        return candidates

    def _ensure_idf(self) -> None:
        if not self._idf_dirty and self._idf:
            return
        sample_size = min(SC_IDF_SAMPLE_SIZE, len(self._bm25_entries))
        if sample_size == 0:
            self._idf_dirty = False
            return

        doc_freq: Counter = Counter()
        bigram_doc_freq: Counter = Counter()
        total_dl = 0
        n_docs = 0

        keys = list(self._bm25_entries.keys())
        if len(keys) > SC_IDF_SAMPLE_SIZE:
            import random
            keys = random.sample(keys, SC_IDF_SAMPLE_SIZE)

        for k in keys:
            entry = self._bm25_entries[k]
            tf = entry.get("tf", {})
            for tok in tf:
                doc_freq[int(tok)] += 1
            bg = entry.get("bg", {})
            seen_bg: set = set()
            for bg_key in bg:
                if bg_key not in seen_bg:
                    bigram_doc_freq[bg_key] += 1
                    seen_bg.add(bg_key)
            total_dl += entry.get("dl", 0)
            n_docs += 1

        if n_docs == 0:
            self._idf_dirty = False
            return

        self._idf = {}
        for tid, df in doc_freq.items():
            self._idf[tid] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        self._default_idf = float(np.median(list(self._idf.values()))) if self._idf else 1.0

        self._bigram_idf = {}
        for bg_key, df in bigram_doc_freq.items():
            self._bigram_idf[bg_key] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

        self._avg_dl = total_dl / n_docs if n_docs else 1.0
        self._idf_dirty = False
        logger.info("IDF recomputed: %d token IDFs, %d bigram IDFs, avg_dl=%.0f",
                     len(self._idf), len(self._bigram_idf), self._avg_dl)

    def _idf_embed(self, token_ids: list[int]) -> np.ndarray:
        """IDF-weighted mean-pooled embedding using model embed_tokens."""
        if _embed_weights is None or not token_ids:
            return compute_fingerprint(token_ids)

        valid = [t for t in token_ids if 0 <= t < _embed_weights.shape[0]]
        if not valid:
            return np.zeros(_embed_dim, dtype=np.float32)

        rows = _embed_weights[valid].astype(np.float32)
        weights = np.array(
            [max(self._idf.get(t, self._default_idf), 0.0) for t in valid],
            dtype=np.float32,
        )
        ws = weights.sum()
        if ws < 1e-10:
            vec = rows.mean(axis=0)
        else:
            weights /= ws
            vec = (rows * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

    def _bm25_score(self, query_tokens: list[int], mem_idx: int) -> float:
        sid = self._mem_key(mem_idx)
        entry = self._bm25_entries.get(sid) if sid else None
        if not entry:
            return 0.0
        tf = entry.get("tf", {})
        dl = entry.get("dl", 1)
        score = 0.0
        for qt in set(query_tokens):
            f = tf.get(str(qt), 0)
            if f == 0:
                continue
            idf_val = self._idf.get(qt, 0.0)
            if idf_val <= 0:
                continue
            tf_norm = (f * (BM25_K1 + 1)) / (
                f + BM25_K1 * (1 - BM25_B + BM25_B * dl / self._avg_dl)
            )
            score += idf_val * tf_norm
        return score

    def _bigram_bm25_score(
        self, q_bigrams: list[tuple[int, int]], mem_idx: int
    ) -> float:
        sid = self._mem_key(mem_idx)
        entry = self._bm25_entries.get(sid) if sid else None
        if not entry:
            return 0.0
        bg = entry.get("bg", {})
        dl = sum(bg.values()) or 1
        score = 0.0
        for a, b in set(q_bigrams):
            key = f"{a},{b}"
            f = bg.get(key, 0)
            if f == 0:
                continue
            idf_val = self._bigram_idf.get(key, 0.0)
            if idf_val <= 0:
                continue
            tf_norm = (f * (BM25_K1 + 1)) / (
                f + BM25_K1 * (1 - BM25_B + BM25_B * dl / self._avg_dl)
            )
            score += idf_val * tf_norm
        return score

    def _turn_maxsim(self, q_emb: np.ndarray, mem_idx: int) -> float:
        sid = self._mem_key(mem_idx)
        te_arr = self._turn_embs.get(sid) if sid else None
        if te_arr is not None and len(te_arr) > 0:
            sims = te_arr.astype(np.float32) @ q_emb
            return max(float(sims.max()), 0.0)
        if self._fingerprints is not None and 0 <= mem_idx < self._fingerprints.shape[0]:
            fp = self._fingerprints[mem_idx]
            if fp.shape[0] == q_emb.shape[0]:
                return max(float(np.dot(q_emb, fp)), 0.0)
        return 0.0

    def _ariadne_score(self, q_emb: np.ndarray, mem_idx: int) -> float:
        sid = self._mem_key(mem_idx)
        ae_arr = self._ariadne_embs.get(sid) if sid else None
        if ae_arr is not None and len(ae_arr) > 0:
            sims = ae_arr.astype(np.float32) @ q_emb
            return max(float(sims.max()), 0.0)
        if self._fingerprints is not None and 0 <= mem_idx < self._fingerprints.shape[0]:
            fp = self._fingerprints[mem_idx]
            if fp.shape[0] == q_emb.shape[0]:
                return max(float(np.dot(q_emb, fp)) * 0.5, 0.0)
        return 0.0

    def _text_to_token_ids(self, text: str) -> list[int]:
        """Simple whitespace tokenization mapped through embed vocab.

        For IDF embedding the exact tokenizer doesn't matter much — we just
        need token IDs that index into embed_weights. Uses a fast hash mapping.
        """
        if _embed_weights is None:
            return []
        words = text.lower().split()
        ids = []
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
            tid = h % _embed_weights.shape[0]
            ids.append(tid)
        return ids

    # ── BM25 + embedding persistence ─────────────────────────────

    def _load_bm25(self) -> None:
        # All dicts keyed by session_id (str). Legacy int-keyed entries on
        # disk are silently ignored — they were unsafe (position-drift) and
        # the canonical path rebuilds from source text after migration.
        self._bm25_entries: dict[str, dict] = {}
        self._turn_embs: dict[str, np.ndarray] = {}
        self._ariadne_embs: dict[str, np.ndarray] = {}
        self._idf: dict[int, float] = {}
        self._bigram_idf: dict[str, float] = {}
        self._default_idf: float = 1.0
        self._avg_dl: float = 500.0
        self._idf_dirty: bool = True
        # Tracks which BM25 entries are already on disk, so ingestion-mode
        # _save_bm25() only appends NEW entries. Prevents Nx duplication
        # every flush after a restart.
        self._bm25_persisted: set[str] = set()

        # BM25 tf/bigram data — try JSONL first (ingestion output), then JSON
        bm25_jsonl = self._dir / "bm25_data.jsonl"
        bm25_path = self._dir / "bm25_data.json"
        dup_lines = 0
        legacy_int_keys = 0
        if bm25_jsonl.exists():
            try:
                total = 0
                with open(bm25_jsonl, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        total += 1
                        obj = json.loads(line)
                        for k, v in obj.items():
                            v.pop("turn_embs", None)
                            v.pop("aq_embs", None)
                            if k.lstrip("-").isdigit():
                                legacy_int_keys += 1
                                continue
                            if k in self._bm25_entries:
                                dup_lines += 1
                            self._bm25_entries[k] = v
                self._bm25_persisted = set(self._bm25_entries.keys())
                if legacy_int_keys > 0:
                    logger.warning(
                        "Ignored %d legacy int-keyed BM25 entries in %s — "
                        "rebuild required via migration script",
                        legacy_int_keys, bm25_jsonl,
                    )
                if dup_lines > 0:
                    logger.warning(
                        "Loaded BM25 data (JSONL): %d unique sid entries "
                        "(deduped %d duplicate lines from %d total)",
                        len(self._bm25_entries), dup_lines, total,
                    )
                    self._compact_bm25_on_load(bm25_jsonl)
                else:
                    logger.info(
                        "Loaded BM25 data (JSONL): %d entries",
                        len(self._bm25_entries),
                    )
            except Exception:
                logger.warning("Failed to load BM25 JSONL", exc_info=True)
        elif bm25_path.exists():
            try:
                with open(bm25_path, encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    v.pop("turn_embs", None)
                    v.pop("aq_embs", None)
                    if k.lstrip("-").isdigit():
                        legacy_int_keys += 1
                        continue
                    self._bm25_entries[k] = v
                self._bm25_persisted = set(self._bm25_entries.keys())
                if legacy_int_keys:
                    logger.warning(
                        "Ignored %d legacy int-keyed BM25 entries",
                        legacy_int_keys,
                    )
                logger.info("Loaded BM25 data: %d entries",
                            len(self._bm25_entries))
            except Exception:
                logger.warning("Failed to load BM25 data", exc_info=True)

        # Turn + Ariadne embeddings (numpy binary). Files named {sid}.npz
        # Legacy int-named files ({idx}.npz) are ignored.
        emb_dir = self._dir / "sc_embeddings"
        legacy_int_files = 0
        if emb_dir.exists():
            loaded_t, loaded_a = 0, 0
            for npz_file in emb_dir.glob("*.npz"):
                stem = npz_file.stem
                if stem.lstrip("-").isdigit():
                    legacy_int_files += 1
                    continue
                try:
                    data = np.load(str(npz_file))
                    if "turn" in data:
                        self._turn_embs[stem] = data["turn"]
                        loaded_t += 1
                    if "ariadne" in data:
                        self._ariadne_embs[stem] = data["ariadne"]
                        loaded_a += 1
                except Exception:
                    pass
            if legacy_int_files:
                logger.warning(
                    "Ignored %d legacy int-named sc_embeddings files — "
                    "rebuild required via migration script",
                    legacy_int_files,
                )
            if loaded_t or loaded_a:
                logger.info("Loaded SC embeddings: %d turn, %d ariadne",
                            loaded_t, loaded_a)

    def _save_bm25(self) -> None:
        bm25_path = self._dir / "bm25_data.json"
        try:
            # Always JSONL append — O(new_entries) per save
            jsonl_path = self._dir / "bm25_data.jsonl"
            new_keys = [k for k in self._bm25_entries
                        if k not in self._bm25_persisted]
            if not new_keys:
                return
            with open(jsonl_path, "a", encoding="utf-8") as f:
                for k in new_keys:
                    v = self._bm25_entries[k]
                    clean = {kk: vv for kk, vv in v.items()
                             if kk not in ("turn_embs", "aq_embs")}
                    f.write(json.dumps({k: clean}, separators=(",", ":")) + "\n")
                    self._bm25_persisted.add(k)
        except Exception:
            logger.warning("Failed to save BM25 data", exc_info=True)

    def _rewrite_bm25_jsonl(self) -> None:
        """JL #19.2 — atomic full rewrite of `bm25_data.jsonl`.

        `_save_bm25` is append-only (O(new) per save) which is correct
        for the hot-path capture loop but doesn't reflect deletions:
        after `delete_user` drops keys from `_bm25_entries`, the on-disk
        JSONL still has those rows. The boot loader is last-write-wins
        per key, so stale rows are eventually masked, but the file grows
        unbounded across many delete cycles. This rewrite collapses the
        log to exactly what's in `_bm25_entries`, atomically via a
        tmp-then-rename swap.
        """
        jsonl_path = self._dir / "bm25_data.jsonl"
        tmp_path = jsonl_path.with_suffix(".jsonl.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for k, v in self._bm25_entries.items():
                    clean = {kk: vv for kk, vv in v.items()
                             if kk not in ("turn_embs", "aq_embs")}
                    f.write(json.dumps({k: clean}, separators=(",", ":")) + "\n")
            tmp_path.replace(jsonl_path)
            self._bm25_persisted = set(self._bm25_entries.keys())
        except Exception:
            logger.warning("Failed to rewrite bm25_data.jsonl", exc_info=True)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def load_semantic_embeddings(self) -> None:
        """Load or compute sentence-transformer embeddings for all memories.

        Cached to semantic_embeddings.npy. Recomputed when the count
        doesn't match or the file is missing.
        """
        n = len(self._memories)
        if n == 0:
            return

        if self._sem_path.exists():
            try:
                cached = np.load(str(self._sem_path))
                if cached.shape[0] == n:
                    self._semantic_embs = cached
                    logger.info(
                        "Loaded semantic embeddings: shape=%s", cached.shape,
                    )
                    return
                logger.info(
                    "Semantic embeddings stale (%d vs %d memories), recomputing",
                    cached.shape[0], n,
                )
            except Exception as e:
                logger.warning("Failed to load semantic embeddings: %s", e)

        self._recompute_semantic_embeddings()

    def _recompute_semantic_embeddings(self) -> None:
        """Compute sentence-transformer embeddings for all memories."""
        from nls_vllm_plugin.nls_format import read_manifest

        embedder = SentenceEmbedder.get()
        if not embedder._ensure_loaded():
            logger.warning("Semantic embedder not available, skipping")
            return

        n = len(self._memories)
        texts: list[str] = []
        t0 = time.time()

        for m in self._memories:
            text = ""
            if m.kv_path:
                try:
                    manifest = read_manifest(m.kv_path)
                    text = manifest.get("conversation_text", "") or ""
                except Exception:
                    pass
            if not text:
                text = m.description or m.session_id or ""
            if TEMPORAL_INDEX_ENABLED and m.timestamp > 0:
                text = temporal_preamble(m.timestamp) + text
            texts.append(text[:512])

        logger.info("Computing semantic embeddings for %d memories...", n)
        self._semantic_embs = embedder.encode_texts(texts)
        np.save(str(self._sem_path), self._semantic_embs)
        logger.info(
            "Semantic embeddings computed: shape=%s in %.1fs, saved to %s",
            self._semantic_embs.shape, time.time() - t0, self._sem_path,
        )

    def _semantic_score(self, q_emb: np.ndarray, mem_idx: int) -> float:
        """Cosine similarity between query embedding and memory's semantic embedding."""
        if self._semantic_embs is None or mem_idx >= self._semantic_embs.shape[0]:
            return 0.0
        return max(float(np.dot(q_emb, self._semantic_embs[mem_idx])), 0.0)

    def update_semantic_embedding(self, mem_id: str, text: str) -> None:
        """Incrementally add or update semantic embedding for a memory."""
        embedder = SentenceEmbedder.get()
        if not embedder._ensure_loaded():
            return

        idx = None
        for i, m in enumerate(self._memories):
            if m.id == mem_id:
                idx = i
                break
        if idx is None:
            return

        if TEMPORAL_INDEX_ENABLED and self._memories[idx].timestamp > 0:
            text = temporal_preamble(self._memories[idx].timestamp) + text
        emb = embedder.encode_texts([text[:512]])[0]
        dim = emb.shape[0]

        if self._semantic_embs is None:
            self._semantic_embs = np.zeros(
                (len(self._memories), dim), dtype=np.float32,
            )

        rows_needed = idx + 1
        if rows_needed > self._semantic_embs.shape[0]:
            pad = np.zeros(
                (rows_needed - self._semantic_embs.shape[0], dim),
                dtype=np.float32,
            )
            self._semantic_embs = np.vstack([self._semantic_embs, pad])

        self._semantic_embs[idx] = emb
        try:
            np.save(str(self._sem_path), self._semantic_embs)
        except Exception:
            pass

    def _save_embeddings_for(self, key) -> None:
        """Persist turn + ariadne embeddings for a memory.

        `key` can be a session_id (preferred), a Memory, or a list index.
        Output file is {sid}.npz so keys are stable across restarts.
        """
        sid = self._mem_key(key)
        if not sid:
            return
        emb_dir = self._dir / "sc_embeddings"
        emb_dir.mkdir(exist_ok=True)
        # session_ids can contain / or other path chars in theory — sanitize.
        safe = sid.replace("/", "_").replace("\\", "_")
        npz_path = emb_dir / f"{safe}.npz"
        try:
            arrays = {}
            if sid in self._turn_embs:
                arrays["turn"] = self._turn_embs[sid]
            if sid in self._ariadne_embs:
                arrays["ariadne"] = self._ariadne_embs[sid]
            if arrays:
                np.savez_compressed(str(npz_path), **arrays)
        except Exception:
            logger.debug("Failed to save embeddings for sid %s", sid)

    def _evict(self):
        now = time.time()
        scores = []
        for mem in self._memories:
            if mem.always_inject:
                scores.append(float("inf"))
                continue
            age_hours = max((now - mem.timestamp) / 3600, 0.1)
            recency = 1.0 / age_hours
            frequency = mem.access_count + 1
            scores.append(recency * frequency)

        keep_indices = np.argsort(scores)[-self._max_memories:]
        keep_set = set(keep_indices)

        evicted = 0
        for i, mem in enumerate(self._memories):
            if i not in keep_set:
                evicted += 1
        logger.warning(
            "Eviction: dropping %d in-memory entries (files kept on disk). "
            "Raise NLS_MAX_MEMORIES to avoid this.", evicted,
        )

        self._memories = [self._memories[i] for i in sorted(keep_set)]
        if self._fingerprints is not None:
            self._fingerprints = self._fingerprints[sorted(keep_set)]

    def has_system_block_for_hash(self, sys_prompt_hash: str) -> bool:
        """KL #708 dedup helper: return True iff a memory with role='system'
        and matching `sys_prompt_hash` already exists in the store. Used by
        the connector's self-warm path to capture the system block once
        per content hash on first request after restart, avoiding both
        redundant captures and re-injection of a system prefix that's
        already represented by phantom slots elsewhere in the layout.
        """
        if not sys_prompt_hash:
            return False
        for mem in self._memories:
            if mem.role == "system" and mem.sys_prompt_hash == sys_prompt_hash:
                return True
        return False

    def _drop_indices_v4(self, drop_indices: set[int]) -> int:
        """JL #19.2 — V4-complete parallel-array purge.

        Removes the given memory positions and keeps every parallel
        per-memory data structure in lockstep:
          - `_memories` list (and the prompt_hash index that points into it)
          - `_fingerprints`         (semantic embeddings, KL #647)
          - `_delta_energy`         (KL #650 fact-vs-question score)
          - `_delta_signal`         (KL #651 raw signed signal)
          - `_delta_fp_cache`       (JL #19.2 Phase 5 persisted FP cache)

        Per-user state and on-disk artifacts (bm25, sc_embeddings, Phase 4
        running-mean centroids) are NOT touched here — `delete_user` is
        the only caller that knows the user identity needed for those, and
        it does that cleanup itself before/after this helper.

        Returns the number of memories removed.
        """
        if not drop_indices:
            return 0
        keep_indices = [
            i for i in range(len(self._memories)) if i not in drop_indices
        ]
        removed = len(self._memories) - len(keep_indices)
        if removed == 0:
            return 0

        self._memories = [self._memories[i] for i in keep_indices]

        def _subset(arr):
            if arr is None:
                return None
            if not keep_indices:
                return None
            try:
                return arr[keep_indices]
            except Exception:
                return None

        self._fingerprints = _subset(self._fingerprints)
        # KL #647 / Tier-1 semantic embeddings persisted to
        # `semantic_embeddings_t1.npy`. This is a per-memory parallel
        # array indexed by position, used by `_semantic_score`. Failing
        # to subset it leaks rows that line up with deleted memories,
        # silently corrupting Swiss-Cheese semantic ranking.
        if getattr(self, "_semantic_embs", None) is not None:
            self._semantic_embs = _subset(self._semantic_embs)
        if self._delta_energy is not None:
            new_de = _subset(self._delta_energy)
            self._delta_energy = (
                new_de if new_de is not None
                else np.zeros(len(self._memories), dtype=np.float32)
            )
        if self._delta_signal is not None:
            new_ds = _subset(self._delta_signal)
            self._delta_signal = (
                new_ds if new_ds is not None
                else np.zeros(len(self._memories), dtype=np.float32)
            )
        if self._delta_fp_cache is not None:
            self._delta_fp_cache = _subset(self._delta_fp_cache)
            # Cache size invariant: must match number of memories.
            if (
                self._delta_fp_cache is not None
                and self._delta_fp_cache.shape[0] != len(self._memories)
            ):
                # Force a Phase 5 cache rebuild on next boot.
                self._delta_fp_cache = None
                self._delta_fp_cache_genesis_hash = None

        self._hash_index = {}
        for i, mem in enumerate(self._memories):
            if mem.prompt_hash:
                self._hash_index[mem.prompt_hash] = i
        self._dirty_indices.update(range(len(self._memories)))
        self._save()

        # Persist the parallel arrays whose ownership lives outside `_save()`.
        try:
            np.save(
                str(self._dir / "delta_energy.npy"),
                self._delta_energy if self._delta_energy is not None
                else np.zeros(len(self._memories), dtype=np.float32),
            )
        except Exception:
            logger.debug("delete-path: failed to persist delta_energy.npy",
                         exc_info=True)
        try:
            np.save(
                str(self._dir / "delta_signal.npy"),
                self._delta_signal if self._delta_signal is not None
                else np.zeros(len(self._memories), dtype=np.float32),
            )
        except Exception:
            logger.debug("delete-path: failed to persist delta_signal.npy",
                         exc_info=True)
        try:
            self._save_delta_fp_cache()
        except Exception:
            logger.debug("delete-path: failed to persist delta fp cache",
                         exc_info=True)

        # JL #19.2 — `_save()`'s fingerprint flush is gated by an
        # add-counter (only triggers every `_fp_save_interval` adds).
        # On the delete path no adds happen, so the on-disk
        # `fingerprints.npy` retains the pre-delete shape until the
        # next add cycle ticks the counter. Force-flush here so the
        # file is consistent immediately.
        try:
            self._fp_dirty = True
            self._persist_fingerprints()
            self._adds_since_fp_save = 0
        except Exception:
            logger.debug("delete-path: failed to force-flush fingerprints",
                         exc_info=True)

        # Semantic embeddings (KL #647) are loaded lazily — if the array
        # isn't in memory yet, `_subset` skipped it and the on-disk file
        # is now mis-sized. Two cases:
        #   (a) loaded → subset already happened above; persist here.
        #   (b) not loaded → invalidate the on-disk file so the next
        #       `load_semantic_embeddings()` triggers a clean recompute
        #       against the surviving memories. Recompute is sized in
        #       seconds even for ~10k memories.
        sem_path = getattr(self, "_sem_path", None)
        if getattr(self, "_semantic_embs", None) is not None and \
                sem_path is not None:
            try:
                np.save(str(sem_path), self._semantic_embs)
            except Exception:
                logger.debug(
                    "delete-path: failed to persist semantic_embeddings",
                    exc_info=True,
                )
        elif sem_path is not None:
            try:
                Path(sem_path).unlink(missing_ok=True)
            except OSError:
                pass

        return removed

    def delete_user(self, user_id: str) -> int:
        """JL #19.2 — V4-complete user purge.

        Removes every memory for `user_id`, deletes the backing .nls files,
        rebuilds parallel per-memory arrays (semantic FPs, delta_energy /
        signal, Phase 5 delta FP cache), drops Phase 4 per-user
        running-mean centroid state, and best-effort cleans the BM25
        entries and `sc_embeddings/*.npz` files keyed by sessions that
        belonged to this user.

        What this does NOT touch (out of scope, separate ownership):
          - `chain_state` SessionChain entries — see `chain_state.py`
          - The DeltaNet KV / Mamba allocator slots (those drain on TTL)
          - Layer-5/6 envelope logs (immutable audit trail)

        Returns the number of memories removed.
        """
        # 1) Collect victim positions and delete .nls files first so the
        #    fingerprint cache rebuild on next boot doesn't try to load a
        #    deleted file.
        victim_indices: set[int] = set()
        victim_sessions: set[str] = set()
        for i, mem in enumerate(self._memories):
            if mem.user_id != user_id:
                continue
            victim_indices.add(i)
            sid = getattr(mem, "session_id", None) or mem.id
            if sid:
                victim_sessions.add(sid)
            try:
                if mem.kv_path:
                    Path(mem.kv_path).unlink(missing_ok=True)
            except OSError:
                pass

        if not victim_indices:
            return 0

        # 2) Phase 4 per-user state — must drop BEFORE _drop_indices_v4 so
        #    no stale running-mean lingers after the parallel arrays shrink.
        for d in (
            getattr(self, "_user_centroids", None),
            getattr(self, "_user_q_centroids", None),
            getattr(self, "_user_fact_sums", None),
            getattr(self, "_user_q_sums", None),
            getattr(self, "_user_fact_counts", None),
            getattr(self, "_user_q_counts", None),
            getattr(self, "_user_delta_range", None),
        ):
            if isinstance(d, dict):
                d.pop(user_id, None)

        # 3) Drop the parallel arrays and rewrite index.jsonl.
        removed = self._drop_indices_v4(victim_indices)

        # 4) Best-effort: BM25 entries + sc_embeddings files keyed by
        #    sessions that belonged to this user.
        bm25_dropped = 0
        for sid in victim_sessions:
            if sid in getattr(self, "_bm25_entries", {}):
                self._bm25_entries.pop(sid, None)
                bm25_dropped += 1
            self._bm25_persisted.discard(sid)
            self._turn_embs.pop(sid, None)
            self._ariadne_embs.pop(sid, None)
            try:
                safe = sid.replace("/", "_").replace("\\", "_")
                npz = self._dir / "sc_embeddings" / f"{safe}.npz"
                npz.unlink(missing_ok=True)
            except Exception:
                pass
        if bm25_dropped:
            self._idf_dirty = True
            # `_save_bm25` is append-only and would NOT remove the
            # dropped rows from `bm25_data.jsonl` — use the JL #19.2
            # atomic rewrite instead so the on-disk file matches the
            # in-memory `_bm25_entries` exactly. This keeps boot-time
            # IDF / BM25 reconstruction free of dead-user residue.
            try:
                self._rewrite_bm25_jsonl()
            except Exception:
                logger.debug("delete_user: bm25 rewrite failed",
                             exc_info=True)

        # 5) KL #717 chain state — drop SessionChain entries for this user
        #    so a freshly re-seeded persona starts at turn 1 with empty
        #    prev_hash. Without this, a new turn captured for the re-used
        #    user_id picks up the stale chain head and the Merkle link
        #    points to a now-deleted .nls file.
        chains_dropped = 0
        try:
            from nls_vllm_plugin.snapshot_connector import (
                _session_chains, _session_chains_lock,
            )
            with _session_chains_lock:
                victim_keys = [
                    k for k in list(_session_chains.keys())
                    if isinstance(k, tuple) and len(k) >= 1 and k[0] == user_id
                ]
                for k in victim_keys:
                    _session_chains.pop(k, None)
                chains_dropped = len(victim_keys)
        except Exception:
            logger.debug("delete_user: chain state purge failed",
                         exc_info=True)

        logger.info(
            "delete_user(%s): removed %d memories, %d bm25 entries, "
            "%d sessions touched, %d chain entries dropped, "
            "remaining_total=%d",
            user_id, removed, bm25_dropped, len(victim_sessions),
            chains_dropped, len(self._memories),
        )
        return removed

    def drop_by_ids(self, mem_ids: list[str]) -> int:
        """Remove the given memory IDs from the in-memory index and persist.

        Used by the retrieval-time self-heal in auto_memory: when a memory's
        kv_path is missing on disk, drop it so subsequent queries don't keep
        re-finding the dead pointer. We do NOT attempt to delete the kv_path
        file (it is already gone, that's the whole point) — we only repair
        the in-memory index and force an atomic JSONL rewrite.

        JL #19.2: now delegates to `_drop_indices_v4` so the Phase 4/5
        parallel arrays (delta_energy, delta_signal, delta_fp_cache) stay
        in lockstep with `_memories`. Prior to this fix, `drop_by_ids`
        only subset `_fingerprints`, leaving `delta_fp_cache` mis-sized
        and triggering Phase 5 cache invalidation on next boot.
        """
        if not mem_ids:
            return 0
        target = set(mem_ids)
        drop_indices: set[int] = set()
        for i, mem in enumerate(self._memories):
            if mem.id in target:
                drop_indices.add(i)
        if not drop_indices:
            return 0
        removed = self._drop_indices_v4(drop_indices)
        logger.info("drop_by_ids: removed %d dead entries", removed)
        return removed

    @property
    def size(self) -> int:
        return len(self._memories)

    def get_stats(self) -> dict:
        if not self._memories:
            return {"size": 0, "total_tokens": 0}
        total_tokens = sum(m.num_tokens for m in self._memories)
        users = set(m.user_id for m in self._memories)
        rings = {}
        for m in self._memories:
            rings[m.ring_type] = rings.get(m.ring_type, 0) + 1
        now = time.time()
        ages = [(now - m.timestamp) / 3600 for m in self._memories]
        return {
            "size": len(self._memories),
            "total_tokens": total_tokens,
            "avg_tokens": total_tokens / len(self._memories),
            "users": len(users),
            "user_ids": sorted(users),
            "rings": rings,
            "always_inject_count": sum(1 for m in self._memories if m.always_inject),
            "oldest_hours": max(ages),
            "newest_hours": min(ages),
            "total_accesses": sum(m.access_count for m in self._memories),
            "bm25_entries": len(self._bm25_entries),
            "capture_dir": str(self._capture_dir),
            "index_path": str(self._index_path),
        }

    def reload(self) -> dict:
        """Hot-reload the memory index and BM25 data from disk without restart.

        Returns a summary dict of before/after counts. This is the V2
        management primitive: after purging or editing index files externally,
        call reload() to pick up changes without bouncing the container.
        """
        before = len(self._memories)
        before_bm25 = len(self._bm25_entries)

        self._memories = []
        self._fingerprints = None
        self._bm25_entries = {}
        self._bm25_persisted = set()
        self._turn_embs = {}
        self._ariadne_embs = {}
        self._idf_dirty = True

        self._load()
        if not self._readonly:
            self._reconcile_from_manifests()
        self._last_flushed_idx = len(self._memories)
        self._load_bm25()

        after = len(self._memories)
        after_bm25 = len(self._bm25_entries)
        logger.info(
            "Memory store RELOADED: memories %d -> %d, bm25 %d -> %d",
            before, after, before_bm25, after_bm25,
        )
        return {
            "memories_before": before,
            "memories_after": after,
            "bm25_before": before_bm25,
            "bm25_after": after_bm25,
        }

    def delete_by_session_ids(self, session_ids: list[str]) -> dict:
        """Delete memories by session_id. Removes from index + BM25 + embeddings.

        Does NOT delete .nls files (those are the ground truth — reconcile will
        re-add them unless the caller also removes the files).
        """
        to_remove = set(session_ids)
        before = len(self._memories)
        self._memories = [m for m in self._memories if m.session_id not in to_remove]
        removed = before - len(self._memories)

        for sid in to_remove:
            self._bm25_entries.pop(sid, None)
            self._turn_embs.pop(sid, None)
            self._ariadne_embs.pop(sid, None)

        if removed > 0 and not self._readonly:
            self._rewrite_index_jsonl()
            self._idf_dirty = True

        logger.info("Deleted %d memories (requested %d sids)", removed, len(to_remove))
        return {"deleted": removed, "requested": len(to_remove), "remaining": len(self._memories)}
