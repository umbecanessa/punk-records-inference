# 1 — Methodology & measurement arms

## Environment

| Field | Value |
|-------|-------|
| Host | NVIDIA GPU ≥24 GB VRAM (Qwen3.5-35B-A3B-FP8 validated) |
| Model | `/model` (BYOC FP8 hybrid) |
| API | vLLM OpenAI-compatible :8000 + PRI plugin |
| Run folder | `bench/results/overnight_20260624_003614` |
| Git SHA | `9944c727779bfe722139b0851ce804149790c7d4` |

## Arms (every study)

| Arm | Client pattern | Purpose |
|-----|----------------|---------|
| **TEXT (local)** | Full inline history, `memory_off=1` | On-box baseline sharing GPU |
| **TEXT (OpenRouter)** | Cloud API, same model slug, no PRI | Isolated TEXT baseline |
| **RESUME** | Last message + chain KV inject | Primary value case |
| **RESUME_OVERFLOW** | RESUME + Swiss backfill on trim | Overflow stress |
| **ARM-D** | `resume_overflow` in turn sweep | Same as overflow in sweep harness |

## Harnesses

### Inject mode compare (`inject_mode_compare.py`)

Long12 run metadata:

| Field | Value |
|-------|-------|
| Plant turns | 15 (3 facts + 12 noise) |
| Garbled guard | `probe_then_neutral_fallback` |
| User / chain | `mode_cmp_c84a5e1dc5` / `chain_thread_c9cb6f8b1967` |

Five recall probes scored per arm. Metrics: `usage.prompt_tokens`, HTTP latency, pass/fail vs expected spans.

### Turn sweep (`turn_sweep.py`)

| Field | Value |
|-------|-------|
| Checkpoints | [20, 40, 60, 80] |
| User / chain | `turn_sweep_0b3f0e4cb1` / `chain_thread_eb71920e0d87` |
| Neutral fallbacks | 5 (garbled-capture hygiene) |

At each checkpoint: cumulative noise turns, then TEXT / RESUME / ARM-D recall @5.

### OpenCode long session

8 work turns + 6 strict recall probes (seed 42). PRI vs `memory_off` baseline.

## Isolation

Each harness run uses `fresh_chain_ids()` — unique `memory_user` + `memory_base_session`. No shared chain keys between studies (see `08_failure_modes.md`).

## Success criteria

1. **Value case (cp20–40, inject long12):** RESUME recall ≥ TEXT; prompt tokens ≪ TEXT.
2. **Agent case:** OpenCode RECALL ≥ baseline.
3. **Geometry:** RoPE pack delta_uniformity → 1.0 after phantom fix.
4. **Long-chain cliff:** Document RESUME degradation at cp60+ (not a bench failure).
