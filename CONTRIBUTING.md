# Contributing

Thanks for your interest in Punk Records Inference.

## Development setup

```bash
git clone https://github.com/umbecanessa/punk-records-inference.git
cd punk-records-inference

# Unit tests (no GPU required)
pip install pytest torch zstandard
pytest tests/ -q
```

For integration work you need an NVIDIA GPU, Docker with GPU support, and a local model checkpoint. See [Quickstart](docs/getting-started/quickstart.md).

## Code conventions

- **`pri/`** is the sole Python package — keep imports at module top
- Env vars use `NLS_*` prefix until v0.2 (`PRI_*` rename planned)
- Capture, resume, and overflow are **orthogonal** — do not conflate in client or bench code
- Bench harnesses should mirror production `kv_transfer_params` (see `bench/opencode/nls_kvp_helpers.py`)

## Running benchmarks

```bash
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000
./bench/run_suite.sh --tier opencode --base-url http://127.0.0.1:8000 --seed 42
```

Document new results in `docs/BENCHMARKS.md` with artifact paths.

## Documentation

User-facing docs live under `docs/` with entry at [docs/index.md](docs/index.md). Maintainer planning docs live under [docs/internal/](docs/internal/) — keep them separate from getting-started guides.

When adding features, update:

1. Relevant reference doc (architecture, client contract, docker, env-vars)
2. [llms.txt](llms.txt) if entry points change
3. [Benchmarks](docs/BENCHMARKS.md) and README proof table when bench results change

## Pull requests

- Focused diffs — one concern per PR when possible
- All CI unit tests must pass (`pytest tests/ -q`)
- Do not commit secrets, API keys, or `.cursor/` local agent files

## Questions

Open a [GitHub issue](https://github.com/umbecanessa/punk-records-inference/issues) or start a Discussion. For security reports, see [SECURITY.md](SECURITY.md).

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
