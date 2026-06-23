> **Maintainer doc** — release planning and agent coordination. End users: [Documentation home](index.md).

# Agent Room Handoff — Punk Records Inference setup review

**Room code:** `7APVGK`  
**Server:** https://agent-room-mcp-production.up.railway.app  
**Repo:** `punk-records-inference` (private)  
**Author agent:** `pri-setup-agent`  
**Date:** 2026-06-23 (updated: final cleanup complete, commit-ready)

> For the reviewing agent: read this file first, then `docs/SHIP_PLAN.md`. Use Agent Room code **7APVGK** to ask follow-ups.

---

## Status (commit-ready, 2026-06-23)

| Phase | Status |
|-------|--------|
| 0–2 | Complete |
| 3 | Complete — tier1 5/5+5/5, OpenCode 6/6, manifest proof `rope_start=24` on GX10 |
| 4 | Not started (LICENSE, public GH, squash) |

- **`pri/` only** — no `nls_vllm_plugin/` shim tree
- **Middleware:** `BaseHTTPMiddleware` for agent shim + admin (ASGI stack reverted)
- **Bug fix kept:** `_ensure_kv_transfer_params` mutates `body["kv_transfer_params"]` in place
- **GX10 proof:** stock Qwen3.5-35B-A3B-FP8, container `pri-inference`
- **Final cleanup done** — dead env vars removed, docs updated, pytest 9/9
- **Awaiting user commit** — setup agent posted `git status` + suggested message to room

---

## Mission completed

Extract and ship **Punk Records Inference** (KV-only) from NLS monorepo branch `exp/chain-of-latest` @ `71a65774`. Phases 0–3 complete in-repo; Phase 4 (public release) **not** done.

**Git commit pending** user approval after `@pri-review-agent` sign-off.

---

## Phase 0 — KV-only cleanup ✅

### Single package: `pri/`

- **Canonical code:** `pri/` only (no compatibility shim tree)
- vLLM module paths: `pri.connector`, `pri.middleware.agent_shim`, `pri.admin`

### Legacy subsystems excluded

MoE router bias, CAMM, streaming-scorer, and thalamus modules are **not in this repo**.
No env gates — code paths simply absent.

### Agent middleware extracted

- **Source:** NLS `punk-records/backend/src/openai/openai.service.ts`
- **Target:** `pri/middleware/agent_shim.py`
- **Behavior:** strip agent transcript (turn ≥ 2), compute `memory_capture_start` via `/tokenize`, populate chain `kv_transfer_params`
- **Standalone diff:** turn index from user-message count (no Prisma DB)

### docker/start.sh

- `NLS_AGENT_SHIM=1`, `NLS_API_INJECT_MODE=resume`
- Middleware: `AgentShimMiddleware` + `NLSAdminMiddleware`
- KV connector: `pri.connector.NLSSnapshotConnector`

---

## Phase 1 — Package structure ✅

### Module renames (ship plan)

| Old | New (canonical) |
|-----|-----------------|
| `snapshot_connector.py` | `connector.py` |
| `chain_resume.py` | `resume.py` |
| `chain_capture.py` | `capture.py` |
| `memory_store.py` | `store.py` |
| `auto_memory.py` | `retrieve.py` |
| `nls_format.py` | `format.py` |
| `neural_scorer.py` | `scorer.py` |
| `nls_admin_api.py` | `admin.py` |

Unchanged: `text_quality.py`, `inject_geometry_audit.py`, `kv_compress.py`

### Spec

- `spec/manifest.schema.json`
- `spec/validate.py` — CLI: `python spec/validate.py path/to/*.nls`
- `spec/EXAMPLES.md` — includes real GX10 `rope_start=24` snippet

### Tests (local, no vLLM)

```powershell
$env:PYTHONPATH = (Get-Location).Path
pytest tests/ -q   # 9 passed
```

Files: `tests/test_format_roundtrip.py`, `test_agent_shim.py`, `test_capture_smoke.py`

---

## Phase 2 — Docker ✅

- `docker/Dockerfile` — base `vllm/vllm-openai:cu130-nightly`, applies `patches/apply_patches.py`
- `docker/compose.yaml` — `pri-data` volume + `${MODEL_PATH}:/model:ro`
- `patches/apply_patches.py` — dynamic vLLM path detection, idempotent

GX10 validated with `--entrypoint /opt/pri-repo/docker/start.sh` if image metadata is stale.

---

## Phase 3 — Docs + bench ✅

### Docs filled

- `docs/ARCHITECTURE.md`, `CLIENT_CONTRACT.md`, `SUPPORTED_MODELS.md`
- `docs/LIMITATIONS.md`, `DOCKER.md`, `BENCHMARKS.md`
- `README.md` updated

### Bench

- `bench/tier1/smoke_health.py` — `/health` + `/v1/models`
- `bench/tier1/marco_facts.py` — TEXT vs RESUME Marco recall
- `bench/opencode/manifest_proof.py` — KL #648 `rope_start > 0` proof
- `bench/run_suite.sh` — `./bench/run_suite.sh --tier 1 --base-url URL`

### Proof artifacts (committed)

| Bench | Result | Artifact |
|-------|--------|----------|
| tier1 marco_facts seed 42 | TEXT 5/5, RESUME 5/5 | `bench/results/tier1_marco_facts_42.json` |
| opencode long session seed 42 | RECALL 6/6 | `bench/results/opencode_long_session.json` |
| manifest proof turn 2 | `rope_start=24` | `bench/results/manifest_opencode_t2.json` |

---

## Review checklist for second agent

### Must verify on GPU box

1. **Docker build + run**
   ```bash
   export MODEL_PATH=/path/to/qwen-checkpoint
   docker compose -f docker/compose.yaml up --build
   curl http://127.0.0.1:8000/v1/models
   ```

2. **Tier-1 bench**
   ```bash
   ./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000
   ```
   Expect: `resume_pass >= text_pass` on Marco facts (3 plant turns).

3. **OpenCode harness** (after Marco green)
   ```bash
   ./bench/run_suite.sh --tier opencode --base-url http://127.0.0.1:8000 --seed 42
   ```

4. **Manifest proof** — turn 2 capture with `rope_start > 0`
   ```bash
   python bench/opencode/manifest_proof.py --base-url http://127.0.0.1:8000
   ```

### Code review hotspots

| Area | File | What to check |
|------|------|---------------|
| kvp mutation | `pri/middleware/agent_shim.py` | `_ensure_kv_transfer_params` mutates body in place |
| Middleware stack | `docker/start.sh` | Two `--middleware` lines, no `stack.py` |
| KV connector | `pri.connector.NLSSnapshotConnector` | vLLM `kv_connector_module_path=pri.connector` |
| Tokenize in middleware | `pri/middleware/agent_shim.py` | `/tokenize` loopback during request |
| Patch apply | `patches/apply_patches.py` | Matches vLLM nightly source or fails gracefully |

### Known gaps / not done

- Phase 4: license, squash history, public GitHub, GHCR publish
- `agent_shim` turn index ≠ hosted Punk Records DB bump (documented in LIMITATIONS.md)
- vLLM import required for full `connector` load — unit tests intentionally avoid it

### Explicitly excluded (do not add)

- `router_bias_processor`, `streaming_scorer`, `attention_reranker`
- MoE/thalamus/CAMM hot paths
- Hosted NestJS `punk-records/` app
- Model weights

---

## File tree (new/changed summary)

```
pri/
  connector.py, resume.py, capture.py, store.py, retrieve.py,
  format.py, scorer.py, admin.py
  middleware/agent_shim.py
docker/Dockerfile, compose.yaml, start.sh
spec/manifest.schema.json, validate.py, EXAMPLES.md
tests/test_*.py
bench/tier1/, bench/opencode/, run_suite.sh
bench/results/*.json
docs/*.md
```

---

## Agent Room setup for reviewer

Teammate agent should:

1. Open this repo in Cursor
2. Run (or ask agent): **Join agent room 7APVGK**
3. Restart Cursor after MCP reload
4. Read `.cursor/agent-room-transcript.md` for live thread
5. Reply in room with review findings

Install reference: https://github.com/umbecanessa/agent-room-mcp/blob/main/docs/INSTALL-IN-PROJECT.md

---

## Quick commands (copy-paste)

```powershell
# Unit tests (Windows)
cd C:\Users\umber\Documents\GitHub\punk-records-inference
$env:PYTHONPATH = (Get-Location).Path
pytest tests/ -q

# Validate a capture file
python spec/validate.py /data/pri/snapshot/captures/
```
