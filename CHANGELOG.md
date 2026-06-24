# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-06-24

### Changed

- Production documentation pass: user-facing tone, removed internal phase naming and hardware codenames
- [`docs/LICENSING.md`](docs/LICENSING.md) — authoritative PolyForm NC guide (no license comparisons)
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — streamlined publication tables and reproduce commands
- Renamed `PHASE_E_SUMMARY.md` → [`BENCHMARK_SUMMARY.md`](bench/results/overnight_20260624_003614/BENCHMARK_SUMMARY.md)
- Moved maintainer audits to `bench/results/.../internal/`; moved launch playbook to `docs/internal/`
- Added [`bench/README.md`](bench/README.md) and run folder [`README.md`](bench/results/overnight_20260624_003614/README.md)

## [0.1.2] - 2026-06-24

### Changed

- Adopted **[PolyForm Noncommercial License 1.0.0](LICENSE)** (SPDX: `PolyForm-Noncommercial-1.0.0`) — standard dual-license pattern; commercial use by separate agreement
- Updated [docs/LICENSING.md](docs/LICENSING.md) for PolyForm noncommercial vs commercial model

## [0.1.1] - 2026-06-24

### Changed

- Replaced Apache 2.0 with Punk Records Community License 1.0 (PRC-1.0) — superseded by 0.1.2
- Added docs/LICENSING.md explaining patent 64/050,345 and commercial licensing path

## [0.1.0] - 2026-06-24

First public release of **Punk Records Inference** — KV-state persistence for vLLM.

### Added

- vLLM plugin: turn capture → `.nls` manifests, chain resume inject, optional overflow profile
- Agent middleware (`NLS_AGENT_SHIM=1`) — transcript strip, `memory_capture_start`, KVP enrichment
- Docker image + compose for BYOC local inference
- `.nls` manifest schema and validator under `spec/`
- Tier-1 Marco facts, inject mode compare, turn sweep, and OpenCode long-session harnesses
- Phase E proof run artifacts and research analysis pages with reproducible charts
- RoPE geometry audit tooling and post-fix validation (100% delta uniformity)
- Documentation: installation, quickstart, client contract, architecture, benchmarks

### Known limitations

- Long-chain RESUME recall degrades at ~17–23k inject tokens (cp60+); TEXT baseline stays stable
- Qwen3.5 hybrid topology validated; other models need `startup_profile` tuning
- Single-node, BYOC only — no hosted SaaS in this repo

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) and [bench/results/overnight_20260624_003614/BENCHMARK_SUMMARY.md](bench/results/overnight_20260624_003614/BENCHMARK_SUMMARY.md).

[0.1.3]: https://github.com/umbecanessa/punk-records-inference/releases/tag/v0.1.3
[0.1.2]: https://github.com/umbecanessa/punk-records-inference/releases/tag/v0.1.2
[0.1.1]: https://github.com/umbecanessa/punk-records-inference/releases/tag/v0.1.1
[0.1.0]: https://github.com/umbecanessa/punk-records-inference/releases/tag/v0.1.0
