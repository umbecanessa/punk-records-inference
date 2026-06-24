# Licensing

Punk Records Inference is licensed under the **[PolyForm Noncommercial License 1.0.0](../LICENSE)** (SPDX: `PolyForm-Noncommercial-1.0.0`).

This is a well-known **source-available / dual-license** pattern: free for research and noncommercial use; **commercial use requires a separate agreement**. Many teams in AI and infra use PolyForm variants (Noncommercial, Small Business, Internal Use) or similar models when they want the code out without giving away production rights.

You're not alone in that trade-off — it's the same family of licenses used when "open for the community, commercial by arrangement" is the goal.

---

## What you can do for free (noncommercial)

PolyForm defines **noncommercial** broadly. In practice that includes:

| Activity | Typically allowed |
|----------|-------------------|
| Clone, build, run locally (Docker + BYOC) | Yes |
| Reproduce benchmark results and publish findings | Yes |
| Modify, fork, experiment, contribute PRs | Yes |
| Personal learning and hobby projects | Yes |
| Academic and public research | Yes |
| Use by charities, schools, government, public research orgs | Yes (even with funding) |

See LICENSE §§ *Noncommercial Purposes*, *Personal Uses*, *Noncommercial Organizations*.

---

## What requires a commercial license

Any **commercial** use — roughly, use that primarily supports a for-profit or revenue-generating activity — is **not** a permitted purpose under PolyForm Noncommercial.

Examples that need a **commercial license**:

- Running PRI in **production** for a for-profit company
- Offering PRI or a PRI-based inference service to **paying customers**
- **Embedding** PRI in a commercial product you sell or distribute

Open a [GitHub issue with label `licensing`](https://github.com/umbecanessa/punk-records-inference/issues/new?labels=licensing) with your org, use case, and deployment shape. We'll respond with terms or a clear yes/no.

---

## Dual license model

```
                    ┌─────────────────────────────┐
                    │   Punk Records Inference    │
                    └─────────────────────────────┘
                           │              │
              noncommercial│              │commercial
                           ▼              ▼
              PolyForm Noncommercial   Separate written
              (this repo, LICENSE)     commercial license
```

- **Community / research / eval** → use under PolyForm Noncommercial at no charge
- **Business / production** → contact for commercial terms (copyright + patent)

---

## Patent (64/050,345)

U.S. Provisional Patent Application No. **64/050,345** covers KV capture, chain resume, and RoPE-aware re-injection.

PolyForm Noncommercial includes a **patent license for noncommercial use** of this software. **Commercial deployment** is outside the PolyForm grant — negotiate a commercial license that covers the patent scope you need.

If the project grows, we may publish broader terms or tiered licensing; that's a future option, not a commitment in v0.1.

---

## Why not Apache 2.0?

[Headroom](https://github.com/headroomlabs-ai/headroom) and many libraries use Apache 2.0 for maximum adoption. PRI sits on a **patented inference path**; PolyForm Noncommercial matches the intent: **reference implementation in the open**, **commercial production by arrangement**.

Headroom compresses prompts; PRI skips re-prefill — complementary layers, different license posture.

---

## Other PolyForm licenses (if your use case differs)

| License | Best for |
|---------|----------|
| **Noncommercial** (this repo) | Research, hobby, academia, eval — commercial needs a deal |
| [Small Business](https://polyformproject.org/licenses/small-business/1.0.0) | Orgs under revenue/employee caps |
| [Internal Use](https://polyformproject.org/licenses/internal-use/1.0.0) | Employees only, no external distribution |

If you're a small startup pre-revenue, ask on a `licensing` issue — terms can be simpler than enterprise.

---

## Contributors

By submitting a PR, you agree your contribution is licensed under PolyForm Noncommercial 1.0.0 for distribution in this repository.

See [CONTRIBUTING.md](../CONTRIBUTING.md) · [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)

---

## Maintainer note

This page summarizes intent, not legal advice. Have counsel review before large commercial deals.
