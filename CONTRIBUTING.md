# Contributing

Thanks for your interest in Punk Records Inference.

## Development setup

```bash
git clone https://github.com/umbecanessa/punk-records-inference.git
cd punk-records-inference
pip install pytest torch zstandard
pytest tests/ -q
```

Integration work requires an NVIDIA GPU, Docker with GPU support, and a local checkpoint. See [Quickstart](docs/getting-started/quickstart.md).

## Code conventions

- **`pri/`** is the sole Python package — keep imports at module top
- Environment variables use the `NLS_*` prefix in v0.1
- Capture, resume, and overflow are orthogonal — do not conflate in client or bench code
- Bench harnesses should mirror production `kv_transfer_params` (see `bench/opencode/nls_kvp_helpers.py`)

## Benchmarks

```bash
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000
./bench/run_suite.sh --tier opencode --base-url http://127.0.0.1:8000 --seed 42
```

Document new results in `docs/BENCHMARKS.md` with artifact paths. See [`bench/README.md`](bench/README.md).

## Documentation

User-facing docs: [docs/index.md](docs/index.md). Maintainer planning: [docs/internal/](docs/internal/).

When adding features, update the relevant reference doc, [llms.txt](llms.txt) if entry points change, and [Benchmarks](docs/BENCHMARKS.md) when proof numbers change.

## Pull requests

- Focused diffs — one concern per PR when possible
- CI unit tests must pass (`pytest tests/ -q`)
- Do not commit secrets, API keys, or `.cursor/` local files

## Questions

[GitHub issues](https://github.com/umbecanessa/punk-records-inference/issues) · [Licensing](docs/LICENSING.md) · [Security](SECURITY.md) · [Code of Conduct](CODE_OF_CONDUCT.md)
