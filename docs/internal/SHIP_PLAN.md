> **Maintainer doc** — release checklist and scope. End users: [Documentation home](../index.md) · [Limitations](../LIMITATIONS.md).

# Punk Records Inference — Ship Plan

**Status:** Public release (v0.1.0 — 2026-06-24)  
**Source branch:** `exp/chain-of-latest`  
**Product name:** **Punk Records Inference**  
**Repo:** `punk-records-inference` · **Docker image:** `punkrecords/inference`

---

## Naming

| Surface | Name |
|---------|------|
| **Product (public)** | Punk Records Inference |
| **GitHub repo** | `punk-records-inference` |
| **Docker image** | `ghcr.io/punkrecords/inference:<tag>` |
| **Python package (extract)** | `pri` (`punk_records_inference` on PyPI if published) |
| **Memory format** | `.nls` (unchanged — file extension + manifest schema) |
| **Env vars (v0.1)** | Keep `NLS_*` prefix for migration from research branch; rename to `PRI_*` optional in v0.2 |

**Not the same thing:**

| Name | What it is |
|------|------------|
| **Punk Records Inference** | Open-source KV capture/resume stack (this plan) |
| **Punk Records** (hosted) | Commercial API at `api.punkrecords.live` — optional demo/control plane, not required to run Inference |

---

## 1. What we are shipping

**Punk Records Inference** is a **KV-state persistence layer** for vLLM: capture attention +
hybrid recurrent state after each turn, store it on disk (`.nls`), re-inject on the next
request so the model does not re-prefill full history.

| In scope | Out of scope (legacy — do not ship) |
|----------|-------------------------------------|
| KV capture (turn snapshots) | Custom MoE expert slots (256→320 expansion) |
| Chain resume inject + RoPE re-rotation | Router bias / thalamus / `router_bias_processor` |
| `.nls` format + memory index on disk | MoE router bias, legacy CAMM, streaming scorer |
| Optional Swiss retrieval (overflow profile) | Hosted Punk Records SaaS (separate product) |
| Agent middleware (strip + capture_start) | LoRA mining, blockchain ingest, 700+ lab scripts |
| Benchmarks + replication harnesses | Model weights (bring your own checkpoint) |

**One sentence:** Persistent inference state for long agent sessions — not text compression,
not MoE routing, not the hosted API product.

**Patent:** Provisional 64/050,345 covers the method; the public release is the **reference
implementation** with full RoPE/inject source under **PRC-1.0** (see LICENSE + LICENSING.md).
Commercial production use requires a separate written license.

---

## 2. Deliverables

| # | Deliverable | Channel | Notes |
|---|-------------|---------|-------|
| D1 | **`punk-records-inference` GitHub repo** | GitHub public | Clean extract: plugin, spec, middleware, bench, docs, tests |
| D2 | **`punkrecords/inference` Docker image** | GHCR / Docker Hub | Pinned vLLM + baked patches + plugin + defaults |
| D3 | **`.nls` format spec** | In repo `spec/` + optional HF dataset card | Memory artifact schema — not model weights |
| D4 | **Tier-1 benchmark suite** | `bench/` + published JSON artifacts | TEXT vs RESUME, reproducible one command |
| D5 | **Documentation** | `docs/` in repo | Architecture, client contract, supported models, limitations |

**Not deliverables for v0.1:** Hosted Punk Records API, checkpoint weights, MoE tooling.

---

## 3. Architecture (runtime)

```
  Client (OpenCode, curl, LangChain, …)
           │
           ▼
  ┌────────────────────────────────────────────┐
  │  Punk Records Inference                      │
  │  vLLM OpenAI API  :8000                      │
  │  ┌────────────────────────────────────────┐ │
  │  │ middleware: agent_shim                   │ │
  │  │  • strip transcript for resume           │ │
  │  │  • compute capture_start / sys hash      │ │
  │  │  • chain_id, turn_index, silo            │ │
  │  └────────────────────────────────────────┘ │
  │  ┌────────────────────────────────────────┐ │
  │  │ PRI SnapshotConnector (KV plugin)      │ │
  │  │  WRITE: capture → .nls                 │ │
  │  │  READ:  resume | swiss | overflow      │ │
  │  └────────────────────────────────────────┘ │
  └────────────────────────────────────────────┘
           │
           ▼
  Volume: /data/pri  (kv_snapshots + snapshot/captures/*.nls)
           │
           ▼
  Checkpoint: --model /model  (user-mounted, BYOC)
```

**Single container, single process** for the default path. Agent-shim logic lives in vLLM
middleware (extracted from hosted Punk Records `openai.service.ts`).

---

## 4. Subsystems (orthogonal — do not conflate)

| Subsystem | Direction | Config / trigger | v0.1 default |
|-----------|-----------|------------------|--------------|
| **Capture** | Write | `NLS_CHAIN_CAPTURE_MODE=turn` | **On** |
| **Resume** | Read | `memory_inject_mode=resume`, turn ≥ 2 | **On** |
| **Swiss** | Read | auto-retrieval when not resume / silo | Off (benchmark profile) |
| **Overflow** | Read compose | `memory_inject_mode=resume_overflow` | Opt-in profile |

Capture ≠ resume ≠ overflow. Document as four subsystems in `docs/ARCHITECTURE.md`.

---

## 5. Repository layout (`punk-records-inference`)

Extract from monorepo — **do not publish the NLS research tree as-is**.

```
punk-records-inference/
├── README.md                 # Punk Records Inference — hero + proof + docker run
├── LICENSE                   # Community license + patent notice
├── pyproject.toml            # package name: punk-records-inference
│
├── pri/                      # Python package (KV plugin)
│   ├── connector.py          # ← snapshot_connector.py (trimmed)
│   ├── resume.py             # ← chain_resume.py
│   ├── capture.py            # ← chain_capture.py
│   ├── store.py              # ← memory_store.py
│   ├── retrieve.py           # ← auto_memory.py (Swiss; optional)
│   ├── format.py             # ← nls_format.py
│   ├── scorer.py             # ← neural_scorer.py (inject V-suppression)
│   ├── text_quality.py
│   ├── admin.py              # ← nls_admin_api.py
│   └── middleware/
│       └── agent_shim.py     # ← hosted Punk Records strip + capture_start
│
├── patches/                  # vLLM source patches (build time)
│   ├── gdn_prefix_caching.patch
│   └── apply_patches.py
│
├── spec/                     # .nls memory format (not model weights)
│   ├── manifest.schema.json
│   ├── EXAMPLES.md
│   └── validate.py
│
├── docker/
│   ├── Dockerfile
│   ├── start.sh              # KV-only profile
│   └── compose.yaml
│
├── bench/
│   ├── tier1/
│   ├── opencode/
│   ├── run_suite.sh
│   └── results/
│
├── tests/
│   └── ...
│
└── docs/
    ├── ARCHITECTURE.md
    ├── CLIENT_CONTRACT.md
    ├── SUPPORTED_MODELS.md
    ├── LIMITATIONS.md
    ├── BENCHMARKS.md
    └── DOCKER.md
```

### Files explicitly excluded from extract

- `router_bias_processor.py`, `streaming_scorer.py`, `attention_reranker.py`
- `check_*`, `inspect_*`, `read_*`, monorepo `scripts/`
- `punk-records/` NestJS app (hosted product — separate repo)
- `docs/moe_research_log.md`

---

## 6. Docker image (`punkrecords/inference`)

### 6.1 Base and pinning

| Item | Policy |
|------|--------|
| Base | Official vLLM image at **pinned tag/commit** |
| Patches | Applied at **`RUN`** in Dockerfile |
| Plugin | `COPY pri/` → `/opt/pri`, `PYTHONPATH` |
| vLLM CLI flags | No `--logits-processors router_bias_processor` |
| Model | **Not baked in** — `MODEL_PATH` required (volume mount) |
| Volume | `/data/pri` → memory store paths |

### 6.2 Run contract

```bash
docker run --gpus all \
  -v pri-data:/data/pri \
  -v /path/to/checkpoint:/model:ro \
  -e MODEL_PATH=/model \
  -p 8000:8000 \
  ghcr.io/punkrecords/inference:0.1.0
```

---

## 7. Model compatibility (BYOC)

### Tier A — Supported

- **Architecture:** Qwen3 Next hybrid (FullAttention + Mamba/DeltaNet) on pinned vLLM
- **Checkpoint:** User-mounted at `MODEL_PATH` (stock Qwen3.5-A3B-FP8 target validation)

### Tier B — Experimental

- Qwen3.6 hybrid variants

### Tier C — Out of scope v0.1

- Dense-only transformers; non-vLLM engines

See `docs/SUPPORTED_MODELS.md`.

---

## 8. Client contract

Same `kv_transfer_params` contract as today — documented in `docs/CLIENT_CONTRACT.md`.
Middleware handles agent transcript strip when enabled.

---

## 9. Benchmarks and proof

```bash
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000
```

Tier 1: Marco facts TEXT vs RESUME, turn sweep cp20–80, OpenCode harness, geometry microscope.
Commit `bench/results/*.json` for README proof tables.

---

## 10. License and legal

PRC-1.0 community license + patent notice (64/050,345). Commercial use by
separate agreement — see [LICENSING.md](../LICENSING.md). RoPE/inject source
fully published for study and improvement.

---

## 11. Phased execution

### Phase 0 — KV-only cleanup (`exp/chain-of-latest`)

- [x] Remove router_bias logits processor from `start_vllm_v3.sh`
- [x] Parameterize `MODEL_PATH`
- [x] Resume bench green on GX10 (tier1 marco_facts 5/5 RESUME, stock Qwen3.5)

### Phase 1 — Extract `punk-records-inference` repo

- [x] Core modules → `pri/` package
- [x] `agent_shim` middleware
- [x] CI + tests (pytest 8/8 local)

### Phase 2 — Docker image `punkrecords/inference`

- [x] Dockerfile, compose, CI smoke (GX10 `pri-inference` running)

### Phase 3 — Docs + bench

- [x] Full `docs/` (filled)
- [x] Tier-1 results JSON on GX10 (`bench/results/tier1_marco_facts_42.json`)
- [x] OpenCode harness direct-vLLM + results JSON in repo (6/6, seed 42)
- [x] Image rebuild with baked fixes + vLLM digest pin (GX10)
- [x] Remove `nls_vllm_plugin/` shims — `pri/` only

### Phase 4 — Public release

- [ ] GitHub `punk-records-inference`
- [ ] `ghcr.io/punkrecords/inference:0.1.0`
- [ ] Announce as **Punk Records Inference**

---

## 12. Success criteria (v0.1)

1. [x] `docker run` + BYOC checkpoint → healthy (GX10 stock Qwen3.5)
2. [ ] Tier-1 RESUME recall ≥ TEXT at cp20–60; cp80 documented (turn_sweep JSON)
3. [x] OpenCode harness functional recall after 8+ turns (6/6)
4. [x] No MoE/thalamus/streaming_scorer in default image
5. [ ] Docs + license complete (docs done; LICENSE Phase 4)
6. [x] One-command bench replication (`bench/run_suite.sh` tier 1 + opencode)

---

## 13. References (internal monorepo)

- Branch: `exp/chain-of-latest`
- Research: `docs/research/TURN_RESUME_FABLE_C.md`
- Plugin source: `pri/` (extracted from NLS monorepo)
- Agent shim source: `punk-records/backend/src/openai/openai.service.ts`
- Patent: `patent/provisional_patent_application.md`
