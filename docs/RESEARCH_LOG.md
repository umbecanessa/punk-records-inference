# NLS Research Log (Curated)

*This is a curated selection from 650+ research entries spanning February–April 2026. Two months of exploration (LoRA, MoE routing), then 8 days from the KV injection breakthrough to production demo, followed by ongoing refinement of quality signals, agent-mode support, and Qwen 3.6 validation. Implementation details are redacted; the narrative, failures, and results are preserved.*

---

## Phase 1: LoRA-Based Memory (January 2026)

### The Hypothesis

If we chain LoRA adapters like a blockchain — each adapter encoding a "memory delta" with a cryptographic hash linking to its predecessor — we could create persistent identity without reprocessing context.

### What We Built

- A three-layer architecture: Genesis Block (base model), Delta Chain (linked LoRA adapters), Active Buffer (KV cache)
- Merkle-tree verification of adapter integrity
- Hot-loading of adapter chains per user

### What We Found

- LoRA adapters modify model weights, not inference states. You can't encode "the user's name is Umberto" as a weight change without running a training step.
- Training LoRA per-user per-memory doesn't scale — each update requires GPU compute comparable to fine-tuning.
- The architecture was elegant but solving the wrong problem: we needed episodic memory (specific facts), not parametric memory (behavioral tendencies).

### Key Takeaway

> "Don't modify how the model *thinks*. Modify what it *sees*."

---

## Phase 2: MoE Router Biasing (February–March 2026)

### The Hypothesis

Qwen3.5-MoE activates a subset of 256 experts per token. If specific experts correlate with specific memories, we could steer routing to "activate" relevant memories without changing the input.

### What We Built

- Per-request expert logit biases applied to the MoE routing layer
- A "suppression wall" blocking custom expert slots by default
- Prefill vs decode phase-specific routing scales
- Router capture forensics to analyze which experts fire for which content

### What We Found

- Expert activation does correlate with content type, but the mapping is indirect and noisy
- Biasing the router fights the model's training distribution — output quality degrades when routing deviates from learned patterns
- Factual recall was inconsistent: names worked, cities worked sometimes, specific details were unreliable
- Peak recall: ~67% on our benchmark, with unpredictable failure modes

### Key Experiments

- **Unified enriched narrative (89% agentic recall):** A single narrative combining personal warmth, factual repetition, and professional tool-use context, processed through all 30 DeltaNet layers as one compounded state. Proved that DeltaNet state IS compounding.
- **Domain-split failure:** Splitting different narrative types across different layer groups broke the causal chain — DeltaNet state compounds sequentially, like a blockchain. Splitting it is like grafting block 3 from one chain onto block 2 from another.

### Key Takeaway

> "DeltaNet state is compounded, not partitioned. And the model's internal routing is trained, not addressable."

---

## Phase 3: KV State Injection — The Pivot (April 2026)

### The Hypothesis

The model's KV cache IS its working memory. If we capture K/V tensors after processing text, store them, and re-inject them before the next query, the model should behave as if it just processed that text.

### Entry #426 — The Breakthrough

Captured K/V from all 10 full-attention layers after processing a set of facts. Injected into a fresh session as "phantom tokens." Asked questions about those facts.

**Result: 7 out of 7 correct.** The model recalled facts it had never seen in the current conversation.

This was the moment the project changed from "can this work?" to "how do we make this production-ready?"

### Entry #437 — Level 2: Direct Paged Cache Write

Moved from Python-level tensor manipulation to direct writes into vLLM's paged KV cache. The scheduler allocates physical pages for phantom positions; our connector writes stored K/V into those pages before prefill begins.

**Result: 8/8 recall, 82% token savings, <1ms injection latency.**

### The DeltaNet Problem (Entries #427–#449)

Qwen3.5-MoE has 30 DeltaNet (recurrent) layers in addition to 10 attention layers. Initial attempts to inject DeltaNet state alongside KV caused *worse* recall than KV-only.

**Root cause:** DeltaNet state from processing text with context doesn't match DeltaNet state at query time (which starts from zero). The mismatch corrupts the hidden states flowing into attention layers.

**Failed approaches:**

- Delta summation: norm explosion, 3/18 recall
- Mean-delta: blurs signals, 3-4/18
- Top-1 Mamba selection: mismatch with multi-memory KV, 4/18
- Genesis-only: ignores accumulated context, 4/18

---

## Phase 4: Production Engineering (April 2026)

### Full 40-Layer Capture (Entry #518)

Prior experiments only captured 10 out of 40 layers (only the full-attention group). Capturing ALL groups — including the 30 DeltaNet layers — yielded word-for-word parity with text-based processing.

**The missing 30 layers explained 100% of the text-vs-KV gap.**

### Two-Pass Capture Architecture (Entry #626)

The DeltaNet problem had a clean solution: two separate forward passes per turn during ingestion.

- **Pass 1:** Mamba from zeros → capture clean attention K/V
- **Pass 2:** Mamba seeded from previous turn → capture compounded DeltaNet state
- **Merge:** K/V from Pass 1 + Mamba from Pass 2 → one stored file

This decouples the capture regimes: attention K/V matches the query-time distribution (zero Mamba), while recurrent states carry accumulated narrative context.

### Per-Message Block Architecture (Entry #611)

Problem: every capture included the system prompt, wasting storage and creating duplication when injecting multiple memories.

Solution: content-addressed system block (captured once, hash-deduplicated) + clean user blocks (system prompt excluded via capture-range slicing). Any combination of user blocks can be composed with any system prompt.

### Retrieval: Swiss-Cheese Fusion (Entries #500–#637)

With hundreds of stored memories per user, retrieval quality becomes critical. We built a multi-signal fusion system:

- BM25 (lexical) catches exact name/number matches
- Sentence-transformer embeddings catch semantic paraphrases
- Temporal indexing enables time-relative queries
- Recency bias favors recent context
- Quality scoring penalizes low-information messages (greetings, reactions)
- Cross-session deduplication prevents redundant memories

**Key debugging story:** "How old am I?" failed to retrieve a memory containing "I'm 34 years old" because BM25 had zero overlap and semantic similarity was squashed by normalization with a small candidate pool. Fix: boost semantic weight to 65%, widen the candidate pool, lower the score floor.

### Neural Scoring: The Model Ranks Its Own Memories (Entries #579–#640)

Retrieval gives us candidates. But the model itself is the best judge of what's relevant. During prefill, we capture the model's query vectors and compute attention scores against each memory's stored keys. The model's own Q@K patterns rank the memories.

**Key discovery:** Raw neural scores systematically overweight identity-anchoring memories. "Hi, my name is Umberto" gets the highest attention score on almost every query because the name attracts attention regardless of topic — but it carries almost no factual content. A quality-aware penalty (computed from universal signals at capture time) discounts these, letting factual memories win the ranking.

**V-suppression:** Memories that score below threshold have their value vectors zeroed directly in the paged cache. The model can still "see" they exist (keys preserved) but gets no content signal. Applied at an early layer so the majority of the model processes clean context.

### Streaming Scorer: Live Memory Hot-Swap (Entries #608–#635)

The neural scorer selects memories at prefill time. But generation can span many tokens and the topic might shift. We built a streaming scorer that:

- Reserves "register slots" in the KV cache (positions with writable K/V)
- Probes the model's hidden state every N decode steps
- Compares against fingerprint vectors for all available memories
- Swaps memory contents in register slots mid-generation when relevance shifts
- Protects neural-scored "keep" memories from eviction

**Key bug:** The streaming scorer initialized before the neural scorer finished populating its "keep" set (timing issue during prefill). Result: it evicted memories the neural scorer wanted to keep, causing the model to "forget" mid-generation. Fix: lazy evaluation — check the keep set at swap time, not at setup time.

### The "Intro Message" Bug (Entry #641)

After all the above was working, the model still greeted users like strangers ("Nice to meet you!") even after 10 turns of conversation. Debugging revealed:

- The streaming scorer was swapping in "Hi, my name is Umberto" (a greeting from the first turn) at score 1.0 on every query
- This greeting memory evicted factual memories ("my wife Monica", "born in Rapallo")
- The neural scorer had ranked the greeting highly (name tokens attract attention) and the streaming scorer treated it as important

Fix: the quality-aware penalty (meta-score) in the neural scorer now demotes greetings and low-information messages, and the streaming scorer respects the neural scorer's keep/suppress decisions.

### Quality Signals: From Brittle to Robust

The first quality classifier we built used universal text features (data tokens, question marks, message length) to score messages on a fact-vs-question axis. It worked, but was brittle on edge cases like topical questions containing data tokens, or recall questions that share vocabulary with the facts they ask about.

A more robust approach emerged from observing how the model's internal state changes when processing different kinds of content. Factual statements produce large changes in state; questions produce minimal ones because they request retrieval rather than impart new information. We built a complementary quality signal from this internal-state geometry, language-agnostic by construction, and combined it with the original text-based signal so neither modality can mask a low-quality memory.

The combined signal now operates at three points: at capture time (junk never enters the pool), at retrieval (low-quality memories never out-rank facts), and during real-time monitoring of generation. Production verification: clean recall on the standard conversation benchmark, with stored facts reliably ranked above questions even when both share vocabulary.

### Agent Mode and Tool-Block Memory

A round of work added support for OpenAI-compatible function-calling agents. The system auto-detects agent flows and applies different memory composition than for conversational chat:

- **Chat mode:** captured memories are isolated from cross-turn linking, preserving pure semantic retrieval.
- **Agent mode:** tool result messages (shell outputs, API responses, file reads) are captured as linked memory blocks tied to the user message that triggered them. When the agent returns to a project after a context-compaction event, retrieval surfaces the user's prior memory and pulls back the linked tool blocks containing operational details that would otherwise have been lost.

This solves the agent-context-loss failure mode that motivated the entire project.

### Scale Validation Through a Real Agent

Late in the cycle, an internal scale issue with how injected states aligned positionally in the cache was identified and resolved. The fix expanded the system's safe operating range by an order of magnitude, validated end-to-end through a real coding-agent benchmark.

### OpenCode Cross-Session Recall (Final Validation)

The first end-to-end production benchmark with NLS as the model behind a real coding agent. OpenCode (a TUI coding agent, similar to Cursor or Aider) was configured to use NLS as its OpenAI-compatible backend, then driven through a multi-phase coding task:

- **Phase 1:** Scaffold a Next.js + NestJS monorepo, with explicit DECISIONs on frontend port, backend port, package manager, dev DB name, and pnpm version. The agent committed each DECISION as a code comment in the file where it was enforced.

- **Recall probes (4 questions across 2 cache states):**
  - Q1: Same TUI, fresh chat → "what backend port did we pick?" → "DECISION: Port 3001" ✓
  - Q2: Same chat as Q1 → "what's the project about?" → ICF Coaching Eval Tool, full description ✓
  - Q3: **Full TUI restart**, fresh chat → "what frontend port did we pick?" → "frontend port is 3000 (Next.js default)" ✓
  - Q4: Same chat as Q3 → "what's the project about?" → richer description than Q2 ✓

The frontend-vs-backend disambiguation (3000 vs 3001) is the strongest signal: a model just hallucinating Next.js defaults would get one right by coincidence, but it would not also get the other right with the agent's actually-chosen value. Both came back as the values OpenCode picked in Phase 1, in different sessions, with no chat-history bridging them.

Q3 in particular was the canonical demonstration: full TUI restart, truly cold inject, **18,751 phantom tokens** of stored context delivered for a **124-token** user prompt (**99.3% prompt-token savings**).

This is the canonical agentic-loop demo. The cross-session memory architecture was validated end-to-end behind a real coding agent — exactly the use case that motivated the entire project (the daily frustration of agents losing context after a few turns).

---

## Phase 5: Demo and Validation (April 2026)

### Punk Records Demo

Built a full-stack demo application to prove the system works end-to-end:

- Angular frontend with real-time memory panel (retrieval, scoring, swap events)
- NestJS backend sending only system prompt + current user message to vLLM
- API transparency log proving zero-history architecture
- Token savings bar showing real-time cost comparison

### What the Numbers Say

- 90%+ token savings on conversations beyond 10 turns
- Consistent factual recall across sessions (name, age, family, location, work)
- Zero recomputation of prior context
- <2ms injection latency regardless of memory count
- Model quality indistinguishable from full-context processing

---

## By the Numbers


| Metric                             | Value                                                |
| ---------------------------------- | ---------------------------------------------------- |
| Total research entries             | 653+                                                 |
| Calendar time                      | ~3 months (Feb–Apr 2026; KV injection phase: 8 days to demo, ~2 weeks to first agentic-loop validation) |
| Dead ends documented               | 50+                                                  |
| Architecture pivots                | 3 (LoRA → MoE routing → KV injection)                |
| Model architecture                 | Hybrid: 10 attention + 30 DeltaNet layers            |
| Hardware                           | Single NVIDIA GB10 desktop GPU                       |
| Lines of plugin code               | ~8,000                                               |
| Bugs that caused "KV doesn't work" | At least 12, all turned out to be something else     |


---

*The full unredacted research log remains private. This curated version preserves the intellectual journey while protecting implementation details.*