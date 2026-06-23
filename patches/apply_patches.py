"""Hot-patch _save_gdn_block_states in qwen3_next.py inside the container."""
import os
import sys

try:
    import vllm

    _vllm_root = os.path.dirname(os.path.dirname(vllm.__file__))
    TARGET = os.path.join(
        _vllm_root, "model_executor", "models", "qwen3_next.py"
    )
except ImportError:
    TARGET = (
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
    )

OLD = """    # Save intermediate block states for each non-spec sequence
    for seq_idx in range(num_non_spec):
        first_scheduled = block_idx_first_scheduled[seq_idx].item()
        last_scheduled = block_idx_last_scheduled[seq_idx].item()
        n_blocks_to_fill = last_scheduled - first_scheduled

        if n_blocks_to_fill > 0:
            cache_blocks = state_indices[seq_idx, first_scheduled:last_scheduled]

            first_chunk = chunk_offsets[seq_idx].item()
            # GDN h[i] = state BEFORE chunk i, so state at end of first
            # block = h[first_chunk + chunk_stride] (state before the next
            # block's first chunk = state after the current block's last).
            first_aligned_chunk = first_chunk + chunk_stride

            num_unaligned_computed = context_lens[seq_idx].item() % mamba_block_size
            if num_unaligned_computed > 0:
                first_aligned_chunk -= num_unaligned_computed // GDN_CHUNK_SIZE

            from_where = h[
                0,
                first_aligned_chunk : first_aligned_chunk
                + n_blocks_to_fill * chunk_stride : chunk_stride,
            ]
            ssm_state[cache_blocks] = from_where.to(ssm_state.dtype)

        # Save final state to last scheduled block
        last_block_id = state_indices[seq_idx, last_scheduled]
        ssm_state[last_block_id] = final_state[seq_idx].to(ssm_state.dtype)"""

NEW = """    total_chunks_in_h = h.shape[1]

    # Save intermediate block states for each non-spec sequence
    for seq_idx in range(num_non_spec):
        first_scheduled = block_idx_first_scheduled[seq_idx].item()
        last_scheduled = block_idx_last_scheduled[seq_idx].item()
        n_blocks_to_fill = last_scheduled - first_scheduled

        if n_blocks_to_fill > 0:
            seq_num_chunks = chunk_counts[seq_idx].item()

            # h only contains states for the scheduled query tokens,
            # not the full context.  Intermediate block-boundary states
            # can only be extracted when the query spans at least
            # chunk_stride chunks (one full mamba block).
            if seq_num_chunks >= chunk_stride:
                cache_blocks = state_indices[seq_idx, first_scheduled:last_scheduled]

                first_chunk = chunk_offsets[seq_idx].item()
                first_aligned_chunk = first_chunk + chunk_stride

                num_unaligned_computed = context_lens[seq_idx].item() % mamba_block_size
                if num_unaligned_computed > 0:
                    first_aligned_chunk -= num_unaligned_computed // GDN_CHUNK_SIZE

                seq_end_chunk = chunk_offsets[seq_idx + 1].item()
                end_idx = min(
                    first_aligned_chunk + n_blocks_to_fill * chunk_stride,
                    seq_end_chunk,
                )
                if first_aligned_chunk < end_idx and first_aligned_chunk < total_chunks_in_h:
                    from_where = h[
                        0,
                        first_aligned_chunk : end_idx : chunk_stride,
                    ]
                    n_actual = from_where.shape[0]
                    if n_actual > 0:
                        ssm_state[cache_blocks[:n_actual]] = from_where.to(
                            ssm_state.dtype
                        )

        # Save final state to last scheduled block
        last_block_id = state_indices[seq_idx, last_scheduled]
        ssm_state[last_block_id] = final_state[seq_idx].to(ssm_state.dtype)"""

if not os.path.isfile(TARGET):
    print(f"TARGET NOT FOUND: {TARGET}")
    sys.exit(1)

with open(TARGET, "r", encoding="utf-8") as f:
    content = f.read()

if NEW in content:
    print("ALREADY PATCHED")
    sys.exit(0)

if OLD in content:
    content = content.replace(OLD, NEW)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"PATCHED OK: {TARGET}")
else:
    print(f"OLD PATTERN NOT FOUND in {TARGET} — may need patch update for this vLLM version")
    sys.exit(1)
