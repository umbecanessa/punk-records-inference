# Economics of stateful inference

All figures below come from the **reproducible June 2026 OSS proof run** ([`overnight_20260624_003614`](../bench/results/overnight_20260624_003614/)), unless noted as a labeled extrapolation. Configuration: **Qwen3.5-35B-A3B-FP8**, inject mode **`resume`**, TEXT vs RESUME arms in the same harness.

Extended tables and charts: [research/02_token_efficiency.md](../bench/results/overnight_20260624_003614/research/02_token_efficiency.md) · [research/09_energy_and_cost.md](../bench/results/overnight_20260624_003614/research/09_energy_and_cost.md) · [research/05_storage_footprint.md](../bench/results/overnight_20260624_003614/research/05_storage_footprint.md).

---

## The broken cost model

Today's default agent loop re-runs the **full transcript through the GPU on every turn**. Prefill cost grows with session length. That work is paid in:

- **GPU time** (latency, queue depth, HBM bandwidth)
- **Input tokens billed** (hosted APIs)
- **Energy** (inference is power-heavy at scale)

Text compression, summarization, and context compaction shrink *what you send* — but unless the runtime reuses prior computation, the model still **re-executes attention** over (compressed) history every turn.

PRI chain resume inverts part of that: prior turns live as **KV state on disk**. Turn ≥ 2 sends only the new user message (~42 prompt tokens in our long-chain bench) plus a disk read — the GPU does **not** re-prefill thousands of tokens of unchanged history.

---

## Compression vs compute persistence

| Approach | What it saves | GPU re-prefill on turn N | Self-hosted vLLM |
|----------|---------------|--------------------------|------------------|
| **Full inline history** | — | Re-reads entire transcript | Yes |
| **Text compression** (e.g. [Headroom](https://github.com/headroomlabs-ai/headroom)) | Tokens on the wire | Still runs attention over compressed text | Proxy / library |
| **Provider prompt cache** | Prefix hits at provider | Ephemeral; provider-bound | No |
| **Summarization / compaction** | Window size | Re-summarize + re-prefill cost | Varies |
| **PRI chain resume (v0.1)** | **Prefill compute** | **Skip** — inject stored KV | **Yes** |

**Headroom compresses what the agent reads. PRI reuses what the GPU already computed.** They stack: smaller prompts *and* skipped re-prefill.

---

## Measured prompt-token savings (June 2026)

RESUME sends only the latest user message on recall; prior context is injected KV, not re-tokenized prefill.

| Scenario | TEXT mean | RESUME mean | Tokens saved | Savings | Recall |
|----------|----------:|------------:|-------------:|--------:|:------:|
| Short chain (0 noise turns) | 446 | 42 | 404 | **90.5%** | 5/5 |
| Long12 agent chain | 3743 | 42 | **3701** | **98.9%** | 5/5 |
| Turn sweep cp20 (~6.2k inject tok) | 6209 | 42 | 6167 | **99.3%** | 5/5 |
| Turn sweep cp40 (~12k inject tok) | 11906 | 42 | 11864 | **99.6%** | 5/5 |
| Turn sweep cp60 (~17k inject tok) | 17003 | 42 | 16961 | **99.8%** | 3/5 |
| Turn sweep cp80 (~23k inject tok) | 23362 | 42 | 23320 | **99.8%** | 0/5 |

**Headline (agent-length sessions with parity):** **90–99% fewer prompt prefill tokens**, anchored at **98.9%** on the long12 chain with **5/5 recall** vs inline TEXT.

At cp60+, token savings stay above 99% but **recall degrades** — cheap wrong answers are not a net win. See [Limitations](LIMITATIONS.md).

Raw: `inject_mode_compare_*_long12_postfix.json`, `turn_sweep_cp20_80_v5.json` in the overnight run folder.

---

## Compute and API cost per recall (long12)

Using the bench compute proxy (`units ≈ prompt_tokens + 0.12 × completion_tokens`) and illustrative cloud rates ($0.18/1M input, $0.72/1M output — OpenRouter-class; see [09_energy_and_cost.md](../bench/results/overnight_20260624_003614/research/09_energy_and_cost.md)):

| Arm | Prompt tok | Compute units | Cloud API $ / recall |
|-----|----------:|--------------:|---------------------:|
| TEXT | 3743 | 3752 | $0.000728 |
| RESUME | 42 | 46 | $0.000031 |

| Saved vs TEXT | Amount | % |
|---------------|-------:|--:|
| Compute units | 3706 | **98.8%** |
| Cloud API $ (if TEXT ran on per-token billing) | $0.000697 | **95.7%** |

This is **real compute avoided**, not a smaller JSON payload re-processed on the GPU.

---

## Latency and GPU energy (long12, mean per recall)

Measured HTTP latency from the same run; energy estimated as `E(Wh) = P_gpu × latency / 3600` at **250 W** (not wall-metered — see assumptions in research/09).

| Arm | Latency | Est. GPU energy / recall |
|-----|--------:|-------------------------:|
| TEXT | 2885 ms | 0.200 Wh |
| RESUME | 1549 ms | 0.108 Wh |

**~46% lower latency** and **~46% less GPU energy per successful recall** on this workload.

Energy tracks latency here because both arms share the same GPU; the dominant win on self-hosted stacks is **shorter GPU busy time** → higher throughput per watt.

---

## Storage tradeoff (same run)

Chain resume writes one `.nls` capture per turn. From the full turn-sweep session in the overnight run:

| Metric | Value |
|--------|------:|
| Capture files | 143 |
| Total capture disk | **648 MB** |
| Mean per capture | ~4.5 MB |
| Mean per turn (83-turn chain) | ~7.8 MB |

| Resource | Cost per GB (illustrative) | Role |
|----------|---------------------------|------|
| HBM3e (GPU) | $30–40 | Recompute all context every turn |
| NVMe SSD | ~$0.10 | Store KV once; read on resume |

**648 MB on disk** replaces repeated **multi-thousand-token prefills** every recall. Storage energy is negligible vs a single inference pass (see research/05).

---

## Cumulative compute over a session

For an agent that would re-prefill the full chain each turn (inline TEXT), total prompt tokens processed through turn *N* scale as roughly 1 + 2 + … + *N* — quadratic in session depth.

For chain resume after turn 1, each turn pays ~**42 prompt tokens** (long12 mean) instead of re-reading the full transcript.

Example at **50 turns**, if inline prefill averaged ~3743 tokens per turn (long12-class depth):

| Model | Total prompt prefill tokens (approx.) | Reduction |
|-------|--------------------------------------:|----------:|
| Inline TEXT each turn | ~50 × 3743 ≈ **187,150** | — |
| RESUME from turn 2 | 3743 + 49 × 42 ≈ **5,801** | **~97%** |

Exact totals depend on chain length and per-turn growth; reproduce with `./bench/run_suite.sh --tier mode-compare`.

---

## Environmental impact (labeled extrapolation)

The overnight run did **not** meter wall power. The table below scales **measured long12 per-recall savings** (0.093 Wh GPU energy, 3701 input tokens avoided) to hypothetical daily recall volume.

Assumptions: every recall resembles the long12 inject-compare chain; 250 W GPU proxy; US grid ~0.41 kg CO₂/kWh.

| Recalls / day | GPU kWh saved / year | Input tokens avoided / year | Est. CO₂ avoided / year |
|--------------:|---------------------:|----------------------------:|------------------------:|
| 100 | 3.4 | 135 million | ~1.4 kg |
| 1,000 | 34 | 1.35 billion | ~14 kg |
| 10,000 | 339 | 13.5 billion | ~140 kg |

At **1 million users × 20 agent recalls/day** (same long12 profile): on the order of **~680 MWh/year** GPU energy and **~27 trillion input prefill tokens** not re-executed — roughly **~280 metric tons CO₂/year** at US grid intensity (~0.41 kg/kWh), before datacenter PUE, cooling, or non-agent workloads. Treat as **order-of-magnitude illustration**, not a forecast.

---

## Why this matters structurally

| | Inline history | PRI chain resume |
|---|----------------|------------------|
| Cost of prior context | GPU prefill every turn | Disk read + small inject |
| Prefill scaling | Grows with transcript length | **~flat** per turn (~42 tok in long12) |
| Energy per recall | Scales with history | **~46% lower** on measured long12 |
| Input-token billing | Full transcript re-billed | **~99% fewer** prompt tokens on recall |

The architectural point: **memory should be a storage problem, not a recompute problem.** Chain resume is a first slice of that fix on self-hosted vLLM — validated where recall matches TEXT, documented where it does not.

---

## Reproduce

```bash
./bench/run_suite.sh --tier mode-compare --seed 42 --base-url http://127.0.0.1:8000
./bench/run_suite.sh --tier sweep --base-url http://127.0.0.1:8000
python bench/build_research_reports.py --run-dir bench/results/overnight_20260624_003614
```

Historical cross-session numbers (April 2026 OpenCode demo) remain in [Benchmarks — April 2026](BENCHMARKS.md#historical-production-validation-april-2026); this page uses the **June reproducible run** as the canonical economics source.
