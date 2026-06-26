# Troubleshooting

Common issues when running Punk Records Inference and integrating agent clients.

---

## Server won't start

### `MODEL_PATH must point at a mounted checkpoint directory`

**Cause:** `MODEL_PATH` is unset or empty.

**Fix:**

```bash
export MODEL_PATH=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/<revision>
docker compose -f docker/compose.yaml up --build
```

### GPU not visible in container

**Cause:** NVIDIA Container Toolkit not installed or Docker not configured for GPU.

**Fix:** Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Verify with:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Weight file not found (symlink checkpoints)

**Cause:** Checkpoint symlinks point outside the mounted `/model` volume.

**Fix:** Use the HF cache overlay (`compose.gx10.yaml`):

```bash
export HF_CACHE=$HOME/.cache/huggingface
docker compose -f docker/compose.yaml -f docker/compose.gx10.yaml up --build
```

### OOM during startup

**Cause:** `MAX_MODEL_LEN` or `GPU_MEMORY_UTILIZATION` too aggressive for your GPU.

**Fix:**

```bash
export GPU_MEMORY_UTILIZATION=0.50
export MAX_MODEL_LEN=16384
```

---

## Capture issues

### Manifests show `rope_start=0`

**Cause:** Capture includes full system preamble — missing `memory_capture_start`.

**Fix:**

- Enable agent shim: `NLS_AGENT_SHIM=1` (default)
- Or send `memory_capture_start` + `memory_sys_prompt_hash` in KVP (see `bench/opencode/nls_kvp_helpers.py`)

Verify:

```bash
python bench/opencode/manifest_proof.py --base-url http://127.0.0.1:8000
```

### No `.nls` files written

**Checks:**

1. `NLS_CHAIN_CAPTURE_MODE=turn` (default)
2. Request did not set `memory_no_capture=1`
3. Decode produced enough tokens (`NLS_CAPTURE_MIN_DECODE_TOKENS`, default 4)
4. Output was not garbled (`NLS_TURN_STRIP_GARBLED_DECODE=1`)

Inspect admin:

```bash
curl -s http://127.0.0.1:8000/admin/memory/stats | jq .
curl -s http://127.0.0.1:8000/admin/memory/index | jq '.[:3]'
```

---

## Resume / recall issues

### RESUME arm fails but TEXT passes (bench)

**Cause:** Often poisoned captures from garbled decodes, wrong `rope_start`, or inject window exceeded.

**Diagnostics:**

```bash
# Turn sweep with garbled guard
./bench/run_suite.sh --tier sweep --base-url http://127.0.0.1:8000

# Geometry audit on sweep output
./bench/run_suite.sh --tier geometry --base-url http://127.0.0.1:8000 \
  --sweep-json bench/results/turn_sweep_cp20_80_clean.json
```

**Fixes:**

- Wipe memory volume and re-run: `docker compose down -v`
- Ensure `memory_capture_start` on all captures
- For long sessions: try `NLS_API_INJECT_MODE=resume_overflow`

### Resume inject aborted (RoPE fail)

**Cause:** Inconsistent geometry in chain blocks.

**Check:** `NLS_RESUME_ABORT_ON_ROPE_FAIL=1` (default) logs pack failures. Run geometry audit.

### Turn 1 cross-session bleed

**Cause:** Missing silo on first turn.

**Fix:** Set `memory_silo=1` on turn 1 (agent shim does this automatically).

---

## Agent middleware

### Shim not injecting KVP

**Checks:**

1. `NLS_AGENT_SHIM=1`
2. Middleware registered in `docker/start.sh` logs
3. Request is chat completions (not embeddings)

### Strip amputates user facts

**Cause:** Legacy `NLS_STRIP_INJECT_SYS_BLOCK_LEN` mismatch with live prompt.

**Fix:** Prefer `memory_capture_start` from tokenized system turn (shim or helpers) over static strip length.

---

## Benchmarks

### `Connection refused` on bench scripts

**Cause:** vLLM not running or wrong URL.

**Fix:**

```bash
curl -s http://127.0.0.1:8000/health
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000
```

### OpenRouter TEXT baseline fails

**Cause:** Missing API key.

**Fix:** Copy `bench/env.example` to `bench/.env` and set `OPENROUTER_API_KEY`. Never commit keys.

---

## Debug endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /admin/memory/stats` | Index size, capture counts |
| `GET /admin/memory/index` | Recent memory entries |
| `GET /admin/memory/block/{id}` | Block metadata |
| `POST /tokenize` | Token counts for capture_start |

Startup profile cache:

```bash
docker exec <container> cat /data/pri/profile.env
docker exec <container> cat /data/pri/model_profile.json
```

---

## Getting help

1. Collect: model id, GPU, `NLS_API_INJECT_MODE`, bench artifact JSON
2. Check [Limitations](../LIMITATIONS.md) for known v0.1 gaps
3. Open a [GitHub issue](https://github.com/umbecanessa/punk-records-inference/issues)
