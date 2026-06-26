# Licensing

Punk Records Inference is licensed under the **[PolyForm Noncommercial License 1.0.0](../LICENSE)** (SPDX: `PolyForm-Noncommercial-1.0.0`).

---

## Summary

| Use | License |
|-----|---------|
| Personal projects, learning, hobby | PolyForm Noncommercial — no fee |
| Academic and public research | PolyForm Noncommercial — no fee |
| Charities, schools, government, public research orgs | PolyForm Noncommercial — no fee |
| Modify, fork, contribute PRs, reproduce benchmarks | PolyForm Noncommercial — no fee |
| **Commercial production** (for-profit deployment, SaaS, OEM embed) | **Separate commercial license** — contact below |

The full legal text is in [LICENSE](../LICENSE). This page is a plain-language guide only.

---

## Noncommercial use (permitted)

Under PolyForm Noncommercial you may:

- clone, build, and run the Docker image with your own model checkpoint;
- modify the source and run your changes locally;
- distribute forks and derivatives **under the same license** (see LICENSE §Distribution);
- publish research, blog posts, and benchmark reproductions;
- contribute improvements via pull requests.

PolyForm defines **noncommercial** in LICENSE §§ *Noncommercial Purposes*, *Personal Uses*, and *Noncommercial Organizations*.

---

## Commercial use (requires agreement)

Commercial use is **not** a permitted purpose under PolyForm Noncommercial. Examples:

- running PRI in **production** for a for-profit organization;
- offering PRI or a PRI-based inference API to **paying customers**;
- **embedding** PRI in a commercial product you sell or distribute.

To request a commercial license, open a [GitHub issue with label `licensing`](https://github.com/umbecanessa/punk-records-inference/issues/new?labels=licensing) and include:

1. Organization and use case
2. Deployment shape (self-hosted vLLM, managed service, OEM, etc.)
3. Rough scale (GPU count or request volume)

---

## Patent

U.S. Provisional Patent Application No. **64/050,345** covers methods related to KV-state capture, chain resume, and RoPE-aware re-injection implemented in this software.

PolyForm Noncommercial includes a **patent license for noncommercial use** of this software. **Commercial deployment** is outside that grant — negotiate patent scope as part of a commercial license.

See also [NOTICE](../NOTICE).

---

## Contributing

Contributions to this repository are licensed under PolyForm Noncommercial 1.0.0 for distribution here. See [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## Third-party components

vLLM, PyTorch, and other dependencies in the Docker image carry their own licenses. See [NOTICE](../NOTICE) and `docker/Dockerfile`.
