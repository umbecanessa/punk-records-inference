# Licensing

Punk Records Inference is **source-available** under the [Punk Records Community License 1.0](../LICENSE) (PRC-1.0).

The goal is simple: **get it out**, let people **run it, study it, improve it, and reproduce the benchmarks** — while keeping a clear path to **commercial licensing** for business use of the underlying method (U.S. Provisional Patent Application No. **64/050,345**).

This is intentionally **not** Apache 2.0 or MIT. Those licenses grant broad patent rights that would undermine the commercial licensing model described in the original ship plan.

---

## What you can do without asking

| Activity | Allowed |
|----------|---------|
| Clone, build, and run locally (Docker + BYOC) | Yes |
| Modify the code for your own learning or research | Yes |
| Self-host for personal or non-commercial workloads | Yes |
| Evaluate internally at a company (dev/staging, not production serving customers) | Yes |
| Contribute PRs, issues, docs, and bench results | Yes |
| Publish academic or technical write-ups that reference this repo | Yes |
| Fork and experiment (subject to this license on redistribution) | Yes |

---

## What requires a commercial license

**Commercial Use** (see LICENSE §2) includes production deployment for a for-profit organization, offering PRI or a PRI-based service to customers, or embedding it in a product you sell — unless we agree otherwise in writing.

If your use case is commercial, open a [GitHub issue with label `licensing`](https://github.com/umbecanessa/punk-records-inference/issues/new?labels=licensing) and describe:

- Organization and use case (product, internal platform, managed service, etc.)
- Deployment shape (self-hosted vLLM, SaaS, OEM embed, etc.)
- Expected scale (rough GPU count or request volume is enough)

We will respond with commercial terms or a clear pass/fail for your scenario.

---

## Patent

Provisional **64/050,345** covers the KV capture / chain resume / RoPE re-injection method. The community license grants **copyright** to use and improve this reference implementation; it does **not** grant a blanket **patent** license for commercial deployment.

That separation is what makes it possible to open the code today while reserving business licensing for later.

If the project gains traction, we may offer broader licenses (including more permissive open-source terms) for some components or tiers — but that is a future decision, not a promise in v0.1.

---

## Compared to Headroom

[Headroom](https://github.com/headroomlabs-ai/headroom) ships under **Apache 2.0** — maximum frictionless adoption for a compression library.

PRI sits closer to the **inference stack** and the **patented resume path**, so the license reflects a different trade-off: **reference implementation out in the open**, **commercial use by arrangement**.

The two projects complement each other (Headroom shrinks prompts; PRI skips re-prefill). Different layers, different license posture.

---

## For contributors

Contributions are welcome under the same [PRC-1.0](../LICENSE). By submitting a PR, you agree your contribution can be distributed under this license.

See [CONTRIBUTING.md](../CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md).

---

## Maintainer note

This document summarizes intent. Have counsel review LICENSE before large commercial deals or a license change. Internal scope notes: [internal/SHIP_PLAN.md](internal/SHIP_PLAN.md) §10.
