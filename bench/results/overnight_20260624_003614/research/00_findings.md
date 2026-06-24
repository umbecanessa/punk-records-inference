# 0 — Key findings & context

This page summarizes what the **2026-06-24 proof run** established, how it relates to the [Neural Ledger System (NLS)](https://github.com/umbecanessa/neural-ledger-system) architecture docs, and what remains open.

**Configuration:** Qwen3.5-35B-A3B-FP8 · NVIDIA ≥24 GB VRAM · default inject `resume`

---

## Executive summary

| Question | Answer |
|----------|--------|
| Does chain resume match inline TEXT recall on agent-length sessions? | **Yes** — 5/5 on long12, 6/6 OpenCode |
| How much prompt do you save? | **~98.9%** on long12 recall probes (~3701 tokens) |
| Latency impact? | **~46%** lower mean recall latency vs TEXT on long12 |
| Is RoPE pack geometry correct post-fix? | **Yes** — 100% delta_uniformity (83 blocks, Δ=−22) |
| Where does it break? | **cp60+** turn sweep (~17–23k inject tok) — RESUME garble, TEXT 5/5 |
| Is garble a RoPE bug? | **No** (post-fix audit passes); likely inject/decode at depth |
| Default inject mode? | **`resume`** — overflow does not fix the cliff |

---

## Context: NLS vs PRI

The public [NLS architecture](https://github.com/umbecanessa/neural-ledger-system/blob/main/ARCHITECTURE.md) describes a **five-phase pipeline**:

1. **Retrieval** — Swiss/BM25 fusion across many stored memories  
2. **Injection** — phantom KV positions + RoPE correction  
3. **Scoring** — neural V-suppression during prefill  
4. **Monitoring** — hot-swap during decode  
5. **Capture** — persist `.nls` blocks after each turn  

**This repository (PRI v0.1)** ships the reference implementation focused on what agent loops need first:

| NLS phase | PRI v0.1 status |
|-----------|-----------------|
| Capture (Phase 5) | ✅ Turn capture → `.nls` |
| Injection — chain resume | ✅ Default path (`memory_inject_mode=resume`) |
| Injection — RoPE re-rotation | ✅ Validated (geometry audit) |
| Retrieval (Phase 1) | ⚙️ Swiss available; not default |
| Scoring (Phase 3) | ⚙️ `pri/scorer.py` present; not default profile |
| Monitoring (Phase 4) | ❌ Not in v0.1 scope |

The NLS docs emphasize **retrieval across sessions**. Our benches prove **turn-to-turn chain resume** on a single agent session — complementary, and the right default for OpenCode-style loops.

---

## Finding 1 — Token and latency efficiency

On the long12 inject-compare chain (12 noise turns, 5 recall probes):

| Arm | Mean prompt tok | Mean latency | Recall |
|-----|----------------:|-------------:|--------|
| TEXT | 3743 | 2885 ms | 5/5 |
| RESUME | 42 | 1549 ms | 5/5 |
| OVERFLOW | 42 | 1473 ms | 5/5 |

See [Token efficiency](02_token_efficiency.md) and [Latency](03_latency_analysis.md).

**Takeaway:** When recall is equal, RESUME removes almost all prompt prefill cost on long chains.

---

## Finding 2 — RoPE geometry fix

Pre-fix microscope audits showed **98.8%** delta_uniformity (one turn-59 phantom outlier). After resume/connector fixes:

- **83/83** blocks at RoPE delta **−22**
- **100%** delta_uniformity — verdict **pass**

See [RoPE geometry](07_rope_geometry.md). Historical pre-fix notes: [`../internal/ROPE_DELTA_AUDIT.md`](../internal/ROPE_DELTA_AUDIT.md).

**Takeaway:** Pack layout and re-rotation math are consistent; remaining long-chain failures are elsewhere.

---

## Finding 3 — Long-chain cliff (open)

Turn sweep (Marco + cumulative noise, checkpoints 20/40/60/80):

| cp | inject tok | TEXT | RESUME |
|----|------------|------|--------|
| 20 | 6225 | 5/5 | 5/5 |
| 40 | 11981 | 5/5 | 5/5 |
| 60 | 17131 | 5/5 | 3/5 |
| 80 | 23543 | 5/5 | 0/5 |

Garble investigation (cp60) established:

1. TEXT **5/5** + RESUME fail → **inject-mediated decode**, not missing facts  
2. **Facts-only inject** (`max_blocks=3`) still garbles → not tail text pollution alone  
3. Hotel probe fails at ~17–18k inject tokens in isolation  

See [Turn sweep scaling](06_turn_sweep_scaling.md) · [Failure modes](08_failure_modes.md).

**Takeaway:** v0.1 is production-shaped for **short-to-medium** agent sessions; very long single-chain inject is a **known limitation**.

---

## Finding 4 — Harness and baseline lessons

| Issue | Resolution |
|-------|------------|
| OpenRouter TEXT with reasoning enabled | Distorts baseline — use `reasoning.effort=none` |
| Unguarded garbled plant turns | Fixed with garbled-turn capture guard in sweep |
| Session bleed between tests | Unique `memory_user` / `memory_base_session` per run |

Historical triage: [`../internal/FAILURE_AUDIT.md`](../internal/FAILURE_AUDIT.md) (superseded by postfix canonical artifacts).

---

## Finding 5 — Storage economics

Full turn-sweep session: **143 captures · ~648 MB** on disk (~4.5 MB/capture average in that workload).

See [Storage footprint](05_storage_footprint.md) · [Energy & cost estimates](09_energy_and_cost.md) (illustrative — wall power not metered).

**Takeaway:** Disk cost per turn is modest vs repeated multi-k-token GPU prefills.

---

## Recommended follow-ups

1. Profile vLLM decode under long KV inject (cp60+)  
2. MoE model matrix — same harness, compare cliff location  
3. Optional: wall-power metering on one recall arm  
4. Reconcile NLS retrieval-first docs with chain-resume default (this page + [docs/OVERVIEW.md](../../../docs/OVERVIEW.md))

---

## Data index

| Resource | Path |
|----------|------|
| Summary tables | [`../BENCHMARK_SUMMARY.md`](../BENCHMARK_SUMMARY.md) |
| Canonical JSON paths | [`../canonical_artifacts.json`](../canonical_artifacts.json) |
| Machine-readable metrics | [`research_data.json`](research_data.json) |
| Reproduce | [`../../README.md`](../../README.md) · [docs/BENCHMARKS.md](../../../docs/BENCHMARKS.md) |
