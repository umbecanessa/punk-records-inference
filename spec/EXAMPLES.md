# .nls manifest examples

Memory artifacts use binary `.nls` files. The JSON manifest is readable without
decompressing tensors.

## Minimal capture block

```json
{
  "version": 1,
  "seq_len": 128,
  "has_mamba": 1,
  "attn_layers": [0, 1, 2],
  "mamba_layers": [3, 4],
  "num_keys": 42,
  "created_at": 1719062400.5,
  "block_hash": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
  "user_id": "user_abc",
  "session_id": "oc_chain_t3_user",
  "ring_type": "general",
  "num_tokens": 128,
  "rope_start": 105,
  "rope_end": 233,
  "role": "turn",
  "turn_index": 3,
  "base_session_id": "oc_fingerprint123",
  "sys_prompt_hash": "deadbeefcafebabe"
}
```

## Agent turn with segments (V-suppression)

```json
{
  "version": 1,
  "seq_len": 512,
  "block_hash": "...",
  "role": "turn",
  "segments": [
    {"role": "user", "start": 105, "end": 180},
    {"role": "assistant", "start": 180, "end": 512}
  ]
}
```

Validate with:

```bash
python spec/validate.py /data/pri/snapshot/captures/
```

## GX10 manifest proof (KL #648, 2026-06-23)

Real turn-2 capture from `bench/opencode/manifest_proof.py` on stock Qwen3.5
(client sends `memory_capture_start` via `nls_kvp_helpers.enrich_kv_params`):

```json
{
  "session_id": "chain_manifest_kl648_t2_user",
  "turn_index": 2,
  "role": "turn",
  "rope_start": 24,
  "rope_end": 81,
  "num_tokens": 57,
  "sys_prompt_hash": "e9f5c29da63cc760",
  "base_session_id": "chain_manifest_kl648",
  "capture_num_phantom": 62,
  "conversation_text": "What backend port did I specify?"
}
```

Sidecar: `bench/results/manifest_opencode_t2.json`

`rope_start > 0` confirms capture sliced after the system-prompt boundary.
Direct marco_facts bench bypasses agent shim → `rope_start=0` is expected there.
OpenCode harness recall (6/6) uses agent shim for strip/inject; clients may also
send kvp directly (production Nest proxy pattern).
