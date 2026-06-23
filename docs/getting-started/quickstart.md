# Quickstart

Get Punk Records Inference running locally with Docker, verify the API, and run a smoke benchmark.

**Requirements:** See [Installation](installation.md) for GPU, Docker, and checkpoint setup.

---

## 1. Clone and configure

```bash
git clone https://github.com/umbecanessa/punk-records-inference.git
cd punk-records-inference
```

Set the model path to your local checkpoint (Qwen3.5 hybrid recommended for v0.1):

```bash
export MODEL_PATH=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/<revision>
```

See [Supported models](../SUPPORTED_MODELS.md) for topology requirements.

---

## 2. Build and run

```bash
docker compose -f docker/compose.yaml up --build
```

The container starts vLLM on port **8000** with the PRI plugin loaded via `--kv-transfer-config`. On first boot, `pri/startup_profile.py` probes `config.json` and writes layer env defaults to `/data/pri/profile.env`.

---

## 3. Health check

```bash
curl -s http://127.0.0.1:8000/v1/models | jq .
```

You should see your mounted checkpoint as the served model.

Admin memory endpoints (debug):

```bash
curl -s http://127.0.0.1:8000/admin/memory/stats
```

---

## 4. Run benchmarks (optional)

Host-side Python (server must be running):

```bash
pip install requests
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000
```

OpenCode long-session recall harness:

```bash
./bench/run_suite.sh --tier opencode --base-url http://127.0.0.1:8000 --seed 42
```

See [Benchmarks](../BENCHMARKS.md) for expected results and artifacts.

---

## 5. Unit tests (no GPU)

```bash
pip install pytest torch zstandard
pytest tests/ -q
```

---

## Next steps

- [Core concepts](concepts.md) — capture, resume, overflow
- [Client contract](../CLIENT_CONTRACT.md) — wire your agent client
- [Docker](../DOCKER.md) — env vars, volumes, inject modes
