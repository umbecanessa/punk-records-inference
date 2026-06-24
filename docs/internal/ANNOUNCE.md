# Public release announcement playbook

> Maintainer guide — how to launch PRI like [Headroom](https://github.com/headroomlabs-ai/headroom) did, and what to do on GitHub before you post.

Headroom (Tejas Chopra, Netflix) went from personal pain → open repo → proof table → social posts → conference talk → community. PRI can follow the same shape with a different wedge: **GPU re-prefill cost**, not prompt token bloat.

---

## What made Headroom land

| Tactic | What they did | PRI equivalent |
|--------|---------------|----------------|
| **Personal hook** | "90% of my Claude bill was tokens I didn't need" | "My agent re-read 20k tokens every turn while the GPU already had the state" |
| **One-liner product** | Context compression layer for AI agents | KV-state persistence layer for vLLM agents |
| **Proof in README** | Before/after table + benchmark suite command | Phase E table + `./bench/run_suite.sh` |
| **60-second start** | `pip install` + `headroom wrap claude` | `docker compose up` + curl health |
| **Compared to** | Table vs RTK, Compresr, provider compaction | Table vs Headroom, prompt cache, compaction |
| **llms.txt** | Agent-readable index at repo root | Done — `llms.txt` |
| **PolyForm NC** | Dual license: noncommercial free, commercial by agreement | Done — LICENSE + [LICENSING.md](LICENSING.md) |
| **Star ask** | "Give it a ⭐ if it saves you money" | "Star if skip-prefill helps your self-hosted stack" |
| **Deep dive** | Substack / blog post with architecture | Link `docs/ARCHITECTURE.md` + research pages |
| **Talk** | OSS conference session (YouTube) | Optional — same story arc: problem → architecture → demo |
| **Community** | Discord | GitHub Discussions or Discord when ready |

Reference posts:

- [LinkedIn launch post](https://www.linkedin.com/posts/chopratejas_i-looked-at-my-claude-bill-90-was-tokens-activity-7419774863344803840-S3Wd) — pain → build → OSS → star ask
- [Conference talk](https://www.youtube.com/watch?v=UOWSHg18cL0) — 30 min architecture + live narrative

---

## GitHub checklist (do before announcing)

### Repository settings

- [ ] **Description:** `KV-state persistence for vLLM — capture turn KV cache, resume on next request, skip re-prefill`
- [ ] **Website:** link to `docs/index.md` or future docs site
- [ ] **Topics:** `vllm`, `llm`, `ai-agents`, `kv-cache`, `inference`, `docker`, `opencode`, `local-first`, `context-window`, `qwen`
- [ ] **Social preview:** Settings → Social preview → upload `assets/social-preview.png`
- [ ] **Verify OG image:**
  ```bash
  gh api graphql -f query='query { repository(owner:"umbecanessa", name:"punk-records-inference") { openGraphImageUrl } }'
  ```

### Release artifact

- [ ] Tag **`v0.1.0`** on commit with CHANGELOG entry
  ```bash
  gh release create v0.1.0 --title "v0.1.0 — first public release" --notes-file CHANGELOG.md
  ```
- [ ] Enable **GitHub Discussions** (Q&A category) if you want Headroom-style community without Discord yet
- [ ] Enable **private vulnerability reporting** (see SECURITY.md)

### Hygiene

- [x] Remove pre-public / private-repo language from README and llms.txt
- [x] PolyForm Noncommercial LICENSE + [LICENSING.md](LICENSING.md) (patent 64/050,345; commercial dual-license)
- [x] Move maintainer plans to `docs/internal/` — not linked from hero README
- [ ] Confirm `bench/.env` is gitignored and no secrets in history

---

## Announcement templates

### Short post (X / LinkedIn / Mastodon)

```
I got tired of watching vLLM re-prefill the same 20k tokens on every agent turn
while the KV state was sitting right there on disk.

So I open-sourced Punk Records Inference (PRI):

→ Capture attention + hybrid recurrent state after each turn (.nls on disk)
→ Re-inject on the next request — skip the expensive re-prefill
→ vLLM plugin, Docker, BYOC, OpenAI-compatible API

Proof on Qwen3.5-35B-FP8: ~3700 prompt tokens saved per recall, ~46% lower latency vs inline TEXT on long sessions.

Pairs with text compression (Headroom shrinks what you send; PRI skips what you already computed).

⭐ https://github.com/umbecanessa/punk-records-inference
Docs + reproducible benches in the repo.
```

### Long post (blog / Substack)

Suggested outline (mirror Headroom's deep dive):

1. **The bill** — GPU hours or latency, not API tokens; agent turn N re-reads turns 1…N−1
2. **Why text compression isn't enough** — even a perfect prompt still re-runs attention unless you inject KV
3. **Architecture** — capture → `.nls` → resume inject → RoPE re-rotation (diagram from README)
4. **Proof** — link Phase E summary + one chart from research pages
5. **Honest limits** — cp60 cliff, BYOC, Qwen-validated
6. **Stack with Headroom** — complementary layers
7. **Call to action** — clone, run tier-1 bench, open issues, star

### Hacker News (Show HN)

Title: **Show HN: Punk Records Inference – persist vLLM KV state across agent turns**

Body: 3 paragraphs max — problem, what it is, proof link. Avoid patent/legal in first comment; point to NOTICE if asked.

### Reddit / Discord communities

Target: r/LocalLLaMA, vLLM Discord, agent-framework channels. Lead with reproducible `./bench/run_suite.sh` command and hardware requirements (≥24 GB VRAM for 35B FP8).

---

## Demo script (5 minutes)

1. `docker compose up` — show `/v1/models`
2. Run tier-1 bench — show TEXT vs RESUME token counts in JSON output
3. Open `research/02_token_efficiency.md` on GitHub — Mermaid chart renders inline
4. Optional: OpenCode harness `./bench/run_suite.sh --tier opencode --seed 42`

---

## After launch (week 1)

- Respond to GitHub issues within 48h
- Pin a Discussion: "What model / agent stack are you running?"
- Track reproduction reports — update BENCHMARKS.md with community hardware rows
- Do **not** over-promise hosted SaaS — scope stays BYOC in this repo

---

## What not to ship in public posts

- Internal monorepo URLs or agent-room coordination codes
- Unreleased product roadmaps beyond v0.1 scope table
- "Free commercial use" — production/commercial use requires a separate license; see [LICENSING.md](LICENSING.md)

See [internal/SHIP_PLAN.md](internal/SHIP_PLAN.md) for maintainer scope boundaries.
