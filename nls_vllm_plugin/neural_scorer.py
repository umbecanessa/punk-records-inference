"""
NLS Neural Scorer — Model-native memory reranking via in-context Q@K attention.

Zero-cost retrieval refinement: during the normal query prefill, computes the
model's actual attention scores between query tokens and each injected memory
region.  Memories that the model itself finds irrelevant are suppressed (V
values zeroed in-place) before generation begins.

How it works:
  1. Swiss Cheese returns top-N coarse candidates (N = COARSE_K, default 20)
  2. All N candidates are multi-injected into the KV cache
  3. During prefill, the attention hook computes Q_query @ K_memory^T
     per memory region, across all attention layers
  4. Scores are aggregated per-memory (max-sim × layer-mean)
  5. Low-scoring memories are suppressed (V zeroed in paged cache)
  6. Generation proceeds with only neural-confirmed memories active

Why this works when offline comparison failed:
  - Q and K are from the SAME forward pass (model is conditioned on full context)
  - This is literally what Transformer attention does — we just read the score
  - Offline comparison failed because Q and K came from different contexts with
    different sequence lengths, RoPE positions, and computational histories

Cost: ~0.2ms of GPU compute per query (negligible vs prefill cost)

Usage:
  Called automatically by auto_memory.retrieve() when scoring is enabled.
  Controlled by NLS_NEURAL_SCORING env var or per-request kv_transfer_params.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import torch

logger = logging.getLogger("nls_neural_scorer")

# ── Configuration ─────────────────────────────────────────────────────

COARSE_K = int(os.environ.get("NLS_NEURAL_COARSE_K", "20"))
FINAL_K = int(os.environ.get("NLS_NEURAL_FINAL_K", "5"))
SCORE_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
SUPPRESS_THRESHOLD = float(os.environ.get("NLS_NEURAL_SUPPRESS_THRESHOLD", "0.15"))
ENABLED = os.environ.get("NLS_NEURAL_SCORING", "1") == "1"
META_NEURAL_PENALTY = float(os.environ.get("NLS_META_NEURAL_PENALTY", "1.0"))

# V-suppression: zero V in paged cache for below-rank memories so subsequent
# attention (remainder of prefill + all decode) ignores them.
#   NLS_V_SUPPRESSION=1         → activate
#   NLS_V_SUPPRESSION_KEEP_K=N  → keep top-N (default FINAL_K)
#   NLS_V_SUPPRESSION_AT_LAYER  → suppress immediately after this scoring layer
#                                 (e.g. 11 = after 3 scoring layers aggregated,
#                                  layers 12..39 prefill + decode use clean V).
#                                 Default: -1 = wait for last scoring layer.
V_SUPPRESSION = os.environ.get("NLS_V_SUPPRESSION", "0") == "1"
V_SUPPRESSION_KEEP_K = int(os.environ.get("NLS_V_SUPPRESSION_KEEP_K", str(FINAL_K)))
V_SUPPRESSION_AT_LAYER = int(os.environ.get("NLS_V_SUPPRESSION_AT_LAYER", "-1"))

# ── Module state ──────────────────────────────────────────────────────

_scoring_active: bool = False

# Memory regions: [{path, start, end, num_tokens, mem_idx, ring_type}, ...]
# start/end are token positions within the phantom prefix
_memory_regions: list[dict] = []

# KL #639: paths of memories the neural scorer decided to KEEP.
# The streaming scorer checks this to avoid evicting neural-kept memories.
_neural_keep_paths: set[str] = set()

# Cached K tensors from injection: layer_idx -> list of K tensors per region
# Each K tensor is [num_tokens, num_kv_heads * head_dim] on CPU
_injected_k_by_layer: dict[int, list[torch.Tensor]] = {}

# Accumulated attention scores: region_idx -> list of per-layer scores
_region_layer_scores: dict[int, list[float]] = {}

# Total phantom tokens (sum of all injected memory tokens)
_total_phantom: int = 0

# Model geometry (set once during first scoring pass)
_num_kv_heads: int = 0
_head_dim: int = 0

# V-suppression context (set by snapshot_connector right after begin_scoring)
_suppress_forward_context = None
_suppress_slot_mapping: Optional[torch.Tensor] = None
_suppress_composite_block_size: int = 0
_suppression_applied: bool = False


def is_enabled() -> bool:
    return ENABLED


def is_scoring() -> bool:
    return _scoring_active


def begin_scoring(
    regions: list[dict],
    total_phantom: int,
    num_kv_heads: int = 2,
    head_dim: int = 256,
) -> None:
    """Start a neural scoring pass. Called by snapshot_connector after injection.

    Args:
        regions: list of dicts with keys:
            - start: token position in phantom prefix
            - end: token position in phantom prefix
            - num_tokens: number of tokens in this memory
            - mem_idx: index in auto_memory._store._memories
            - ring_type: ring classification
            - kv_path: path to .nls file
        total_phantom: total phantom tokens injected
        num_kv_heads: number of KV heads in the model
        head_dim: head dimension
    """
    global _scoring_active, _memory_regions, _total_phantom
    global _num_kv_heads, _head_dim, _region_layer_scores
    global _suppression_applied

    _scoring_active = True
    _memory_regions = regions
    _total_phantom = total_phantom
    _num_kv_heads = num_kv_heads
    _head_dim = head_dim
    _region_layer_scores = {i: [] for i in range(len(regions))}
    _injected_k_by_layer.clear()
    _suppression_applied = False

    logger.info(
        "Neural scoring BEGIN: %d regions, %d phantom tokens, "
        "heads=%d, head_dim=%d",
        len(regions), total_phantom, num_kv_heads, head_dim,
    )


def set_suppression_context(
    forward_context,
    slot_mapping: torch.Tensor,
    composite_block_size: int,
) -> None:
    """Wire the paged-cache handles needed for V-suppression.

    Called by snapshot_connector immediately after begin_scoring() so that
    _try_suppress_now() can execute once scoring resolves, without re-plumbing
    these through the scoring call sites.
    """
    global _suppress_forward_context, _suppress_slot_mapping
    global _suppress_composite_block_size
    _suppress_forward_context = forward_context
    _suppress_slot_mapping = slot_mapping
    _suppress_composite_block_size = composite_block_size


def _try_suppress_now(reason: str) -> None:
    """Apply V-suppression to the paged cache, if enabled and not already done."""
    global _suppression_applied
    if not V_SUPPRESSION or _suppression_applied:
        return
    if _suppress_forward_context is None or _suppress_slot_mapping is None:
        logger.warning(
            "V-suppression requested (%s) but context not wired — skipping", reason,
        )
        return
    try:
        suppress_regions = get_suppress_regions(keep_k=V_SUPPRESSION_KEEP_K)
        n = 0
        if suppress_regions:
            n = suppress_v_in_cache(
                _suppress_forward_context,
                suppress_regions,
                _suppress_slot_mapping,
                _suppress_composite_block_size,
            )
        logger.info(
            "V-SUPPRESSION APPLIED (%s): keep_k=%d, suppressed_regions=%d, "
            "suppressed_tokens=%d",
            reason, V_SUPPRESSION_KEEP_K, len(suppress_regions), n,
        )
    except Exception as e:
        logger.error(
            "V-suppression failed at %s: %s", reason, e, exc_info=True,
        )
    finally:
        # Never retry within the same query, even on failure.
        _suppression_applied = True


def _aggregate_partial_scores() -> None:
    """Write current per-layer means into _memory_regions[*]['neural_score'].

    Used for early V-suppression before all scoring layers have fired.
    """
    for region_idx, layer_scores in _region_layer_scores.items():
        if layer_scores:
            mean_score = sum(layer_scores) / len(layer_scores)
        else:
            mean_score = 0.0
        _memory_regions[region_idx]["neural_score"] = mean_score


def cache_injected_k(layer_idx: int, k_tensors: list[torch.Tensor]) -> None:
    """Cache K tensors during injection for later scoring.

    Called by snapshot_connector during start_load_kv for each attention layer.

    Args:
        layer_idx: model layer index (e.g. 3, 7, 11, ...)
        k_tensors: list of K tensors, one per memory region.
            Each tensor is [num_tokens, num_kv_heads * head_dim]
    """
    if layer_idx in SCORE_LAYERS:
        _injected_k_by_layer[layer_idx] = [
            k.detach().cpu().clone() for k in k_tensors
        ]


def score_layer(
    layer_idx: int,
    q: torch.Tensor,
    num_query_tokens: int,
) -> None:
    """Compute Q@K^T attention scores for this layer.

    Called from the _hippocampus_forward hook during prefill. Only processes
    attention layers in SCORE_LAYERS.

    Args:
        layer_idx: current layer index
        q: query Q vectors [batch_tokens, num_heads * head_dim] on GPU
            Only the last num_query_tokens are the actual query (rest is phantom)
        num_query_tokens: number of real query tokens
    """
    if not _scoring_active or layer_idx not in _injected_k_by_layer:
        return

    if layer_idx not in SCORE_LAYERS:
        return

    k_list = _injected_k_by_layer[layer_idx]
    if not k_list:
        return

    device = q.device
    num_heads = q.shape[-1] // _head_dim
    scale = 1.0 / math.sqrt(_head_dim)

    # Extract query Q (last num_query_tokens of the batch)
    q_query = q[-num_query_tokens:]  # [M, num_heads * head_dim]
    q_query = q_query.view(num_query_tokens, num_heads, _head_dim)  # [M, H, D]

    for region_idx, k_mem in enumerate(k_list):
        if k_mem is None:
            _region_layer_scores[region_idx].append(0.0)
            continue

        mem_len = k_mem.shape[0]
        k_gpu = k_mem.to(device=device, dtype=q.dtype)
        k_gpu = k_gpu.view(mem_len, _num_kv_heads, _head_dim)  # [N, Hkv, D]

        # GQA: repeat K heads to match Q heads
        gqa_ratio = num_heads // _num_kv_heads
        if gqa_ratio > 1:
            k_gpu = k_gpu.repeat_interleave(gqa_ratio, dim=1)  # [N, H, D]

        # Compute attention: [M, H, D] @ [N, H, D]^T → [H, M, N]
        # Efficient batch matmul
        q_t = q_query.permute(1, 0, 2)  # [H, M, D]
        k_t = k_gpu.permute(1, 0, 2)    # [H, N, D]
        attn = torch.bmm(q_t, k_t.transpose(1, 2)) * scale  # [H, M, N]

        # MaxSim per query token: for each query token, max attention over memory
        max_attn = attn.max(dim=-1).values  # [H, M]

        # Score = mean across heads and query tokens
        score = max_attn.mean().item()
        _region_layer_scores[region_idx].append(score)

    # ── Early V-suppression ────────────────────────────────────────────
    # If configured to suppress BEFORE the last scoring layer, flush partial
    # aggregates and suppress now so that the remaining prefill layers
    # (layer_idx+1 .. 39) see the cleaned V in the paged KV cache.
    if (
        V_SUPPRESSION
        and V_SUPPRESSION_AT_LAYER > 0
        and layer_idx == V_SUPPRESSION_AT_LAYER
        and not _suppression_applied
    ):
        _aggregate_partial_scores()
        _log_scores(prefix=f"early@L{layer_idx}")
        _try_suppress_now(reason=f"early@L{layer_idx}")

    if layer_idx == SCORE_LAYERS[-1]:
        _finalize_scores()
        # Late V-suppression: only fires if early suppression didn't already run.
        _try_suppress_now(reason="final@L39")


def _log_scores(prefix: str = "final") -> None:
    """Emit per-region score diagnostics."""
    for i, region in enumerate(_memory_regions):
        ns = region.get("neural_score", 0.0)
        ms = region.get("meta_score", 0.0)
        per_layer = _region_layer_scores.get(i, [])
        logger.info(
            "  [%s] Region %d: neural_score=%.4f, meta=%.2f, per_layer=%s, "
            "ring=%s, path=%s",
            prefix, i, ns, ms,
            [f"{s:.3f}" for s in per_layer],
            region.get("ring_type", "?"),
            region.get("kv_path", "?")[-40:],
        )


def _finalize_scores() -> None:
    """Aggregate per-layer scores into final per-memory scores."""
    logger.info(
        "Neural scoring FINALIZE: %d regions, layers_scored=%s",
        len(_memory_regions),
        {r: len(s) for r, s in _region_layer_scores.items()},
    )

    _aggregate_partial_scores()
    _log_scores(prefix="final")


def get_ranked_regions() -> list[dict]:
    """Return regions sorted by neural score, highest first.

    Each region dict now includes 'neural_score'.
    """
    if not _memory_regions:
        return []

    scored = list(_memory_regions)
    scored.sort(key=lambda r: r.get("neural_score", 0.0), reverse=True)
    return scored


def get_suppress_regions(keep_k: int = FINAL_K) -> list[dict]:
    """Return regions that should be suppressed (not in top-K).

    These regions will have their V values zeroed in the KV cache.
    Always-inject memories (identity/behavioral) are never suppressed.

    Meta-score penalty (KL #640): memories tagged as low-information
    (greetings, reactions, questions) have their neural_score discounted
    before ranking so factual memories win KEEP slots.
    """
    ranked = get_ranked_regions()

    # Apply meta_score penalty to produce effective_score for ranking
    if META_NEURAL_PENALTY > 0:
        for r in ranked:
            ms = r.get("meta_score", 0.0)
            ns = r.get("neural_score", 0.0)
            eff = ns * (1.0 - META_NEURAL_PENALTY * ms)
            r["effective_score"] = max(eff, 0.0)
        ranked.sort(key=lambda r: r.get("effective_score", 0.0), reverse=True)

    keep: list[dict] = []
    suppress: list[dict] = []

    for region in ranked:
        ns = region.get("effective_score", region.get("neural_score", 0.0))
        if region.get("always_inject", False):
            keep.append(region)
        elif ns < SUPPRESS_THRESHOLD:
            suppress.append(region)
        elif len(keep) < keep_k:
            keep.append(region)
        else:
            suppress.append(region)

    logger.info(
        "Neural scorer: KEEP %d, SUPPRESS %d (of %d total)",
        len(keep), len(suppress), len(ranked),
    )
    # KL #639: expose keep set so the streaming scorer can protect them
    _neural_keep_paths.clear()
    for r in keep:
        p = r.get("kv_path", "")
        if p:
            _neural_keep_paths.add(p)
    for r in keep:
        logger.info(
            "  KEEP: score=%.4f eff=%.4f meta=%.2f ring=%s path=%s",
            r.get("neural_score", 0.0),
            r.get("effective_score", r.get("neural_score", 0.0)),
            r.get("meta_score", 0.0),
            r.get("ring_type", "?"),
            r.get("kv_path", "?")[-40:],
        )
    for r in suppress:
        logger.info(
            "  SUPPRESS: score=%.4f eff=%.4f meta=%.2f ring=%s path=%s",
            r.get("neural_score", 0.0),
            r.get("effective_score", r.get("neural_score", 0.0)),
            r.get("meta_score", 0.0),
            r.get("ring_type", "?"),
            r.get("kv_path", "?")[-40:],
        )

    return suppress


def get_neural_keep_paths() -> set[str]:
    """Return the kv_paths of memories that the neural scorer decided to KEEP."""
    return _neural_keep_paths


def end_scoring() -> None:
    """Clean up scoring state after generation begins."""
    global _scoring_active, _memory_regions, _region_layer_scores
    global _suppress_forward_context, _suppress_slot_mapping
    global _suppress_composite_block_size, _suppression_applied
    _scoring_active = False
    _memory_regions = []
    _region_layer_scores = {}
    _injected_k_by_layer.clear()
    _suppress_forward_context = None
    _suppress_slot_mapping = None
    _suppress_composite_block_size = 0
    _suppression_applied = False
    _neural_keep_paths.clear()
    logger.info("Neural scoring END")


def suppress_v_in_cache(
    forward_context: "ForwardContext",
    suppress_regions: list[dict],
    slot_mapping: torch.Tensor,
    composite_block_size: int,
) -> int:
    """Zero out V values for suppressed memory regions in the paged KV cache.

    Called by snapshot_connector after scoring completes and before decode.
    Iterates over attention layers and zeros V at the suppressed positions.

    Returns number of tokens suppressed.
    """
    if not suppress_regions:
        return 0

    suppressed_positions = []
    for region in suppress_regions:
        start = region.get("start", 0)
        end = region.get("end", 0)
        suppressed_positions.extend(range(start, end))

    if not suppressed_positions:
        return 0

    pos_tensor = torch.tensor(suppressed_positions, dtype=torch.long)
    slots = slot_mapping[pos_tensor]

    layers_zeroed = 0
    for layer_name in forward_context.no_compile_layers:
        layer = forward_context.no_compile_layers[layer_name]
        kv_cache_attr = getattr(layer, "kv_cache", None)
        if kv_cache_attr is None:
            continue

        kv_or_states = (
            kv_cache_attr[0]
            if isinstance(kv_cache_attr, (list, tuple))
            else kv_cache_attr
        )

        # Only attention layers (not Mamba)
        if isinstance(kv_or_states, (list, tuple)):
            continue

        kv_cache = kv_or_states
        shape = kv_cache.shape
        device = kv_cache.device
        dev_slots = slots.to(device=device, dtype=torch.long)

        if len(shape) == 5 and shape[0] == 2:
            page_size = shape[2]
            page_idx = dev_slots // page_size
            offset_idx = dev_slots % page_size
            kv_cache[1, page_idx, offset_idx] = 0  # Zero V only
        elif len(shape) == 4:
            flat = kv_cache.reshape(shape[0] * shape[1], -1)
            feat = flat.shape[-1] // 2
            flat[dev_slots, feat:] = 0  # Zero V half
        else:
            block_idxs = dev_slots // composite_block_size
            offsets = dev_slots % composite_block_size
            kv_cache[block_idxs, 1, offsets] = 0  # Zero V

        layers_zeroed += 1

    logger.info(
        "V-suppression: %d tokens across %d layers zeroed (%d regions)",
        len(suppressed_positions), layers_zeroed, len(suppress_regions),
    )
    return len(suppressed_positions)
