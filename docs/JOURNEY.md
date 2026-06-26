# The Journey: From Amnesia to Stateful Inference

*How NLS went from a frustrated idea to a working system over ~3 months, 650+ experiments, and more dead ends than I can count. The final architecture — KV state injection — went from first proof-of-concept to production demo in 8 days, then through ~2 more weeks of refinement to the first end-to-end agentic-loop validation behind a real coding agent on April 27, 2026.*

**Author:** Umberto Canessa Cerchi

---

## The Frustration

This started as a side project — late nights and weekends, fueled by frustration with one specific failure mode of AI coding agents.

For the past year, on personal time, I've been building end-to-end products with AI pair programmers as the orchestration layer — games, growth tools, internal CRMs, evaluation systems — each one more complex than the last. The goal was always the same: figure out how to make AI-driven development reliable enough to actually ship.

I got there in a lot of ways. But one thing drove me insane.

You're in your AI-pair editor of choice, you've been building for hours, the agent knows your entire stack. You say "deploy to the server, here's the SSH." It deploys. You do a few more iterations. You say "deploy again." And the agent starts spiraling — it forgot the server is remote, not local. It's trying to run commands on localhost. It burns 5,000 tokens going in circles before you notice. You paste the SSH address again. Three iterations later, it forgets the database password it retrieved 20 minutes ago.

This doesn't happen once. It happens *every single time*. Every long conversation. Every complex project. The agent drifts, loses context, hallucinates things it should know.

And yes — I know the "proper" way to handle this is to set up 10 specialized agents, each with a short focused context window, carefully orchestrated. But I'm a lazy ass. I just want one conversation that works. And I think most people are the same — nobody except the hardcore techies is going to architect a multi-agent pipeline with context partitioning. Normal people want to talk to one AI and have it remember.

That frustration — the *daily* frustration of context drift, of telling the agent the same thing for the fifteenth time, of watching it confidently do the wrong thing because it lost the thread — is what made me ask: why does the model have to forget? Why does every message reprocess the entire history? Why can't the understanding persist?

I figured there had to be a way to reuse the model's own computed representations. When you process text through a transformer, each layer builds rich numerical states — key-value tensors, attention patterns, recurrent states. Those states *are* the understanding. They're the reason the model can answer questions about what you told it. Why throw them away after every request and rebuild them from scratch?

That question launched a ~3-month after-hours research project.

---

## Phase 1: LoRA Adapters (January 2026)

My first idea was to use Low-Rank Adaptation (LoRA) — the standard technique for fine-tuning LLMs efficiently. The concept: treat each user's memory as a set of learned weight modifications, chain them together like a blockchain, and hot-swap them per request.

I built a working prototype: a "Neural Ledger" of LoRA adapters linked by cryptographic hashes, with a Genesis Block (base model), Delta Chain (accumulated adaptations), and Active Buffer (current conversation).

**What we actually achieved:**
We ran a 9-hour overnight endurance test: 581 inference turns, 23 consecutive LoRA training cycles (QLoRA micro-training with TIES merge), 88 facts taught, 36 contradiction overrides, 282 recall probes. Not on clean, distilled training data, but on chaotic real conversation with an autonomic sleep trigger.

The result: **zero degradation.** Overall drift across 581 turns was -0.48% (statistically insignificant). The model woke up *sharper* after each sleep cycle than the session average. Only 1 out of 581 turns showed potential cognitive dip — 99.8% resilience across 23 compounding weight mutations.

The industry assumption was that repeated LoRA fine-tuning causes catastrophic forgetting. We proved it doesn't have to — with careful merge strategy (TIES at 70% density) and autonomic training triggers.

**Why we dropped it:**
Training tool-calling patterns into the LoRA adapter degraded the model's existing tool capabilities — global weight modification affected ALL inference. Removing tool-call training swung the model toward excessive conversational behavior. There was no way to have both: memory AND tool calling in the same adapter. The model could remember your name but couldn't use an API anymore.

LoRA operates at the model *weight* level, not the inference *state* level. Every memory update requires actual training compute. That doesn't scale to thousands of users.

**But we still won something:** 23 compounding LoRA mutations over 9 hours without catastrophic forgetting. That's a non-trivial result that challenged conventional wisdom, even if we abandoned the approach.

**Entries: KL #1 - #150ish. Time spent: ~4 weeks (January 2026).**

---

## Phase 2: MoE Router Biasing (February–March 2026)

### The Hypothesis

Qwen3.5-MoE is a Mixture-of-Experts model with 256 routed experts per layer, 8 activated per token plus 1 shared expert. My second idea: what if I could bias the routing to activate specific experts that encode specific memories?

### The First Question: Can We Expand Without Breaking?

Before anything else, I needed to know if the expert pool was elastic. The model ships with 256 experts — could we add custom expert slots beyond 256 without breaking the routing? This was weeks of careful surgery: expanding tensor dimensions from 256 to 512, verifying the router still functions, testing that the base experts aren't degraded by the expansion. We eventually settled on serving 320 experts (256 base + 64 custom slots).

**It worked.** We could expand the expert pool and put new slots "behind a wall" — a suppression wall with a -1000.0 default bias, making custom slots completely invisible to the router unless a per-request bias explicitly lifts them.

### The Undercover Agent

This is where it got creative. We realized something: the model's attention layers just receive a stream of activations. They have no idea — and literally don't care — how that stream was generated. So what if we created a "fake" expert that wasn't a neural network at all, but a programmatic, heuristic-based agent? The router would think it's picking a normal expert, but it was actually an undercover agent injecting the facts we wanted in a predictable, deterministic way.

We called it the Programmatic Expert. And it worked — facts from our memory expert would appear in the model's outputs. We could see our expert pushing facts into the thinking process.

### The "Why Is Marco?" Problem

But then something fascinating and frustrating happened. We could see our programmatic expert successfully pushing facts into the model's processing. In the thinking trace, you could clearly see the injected fact appearing — the user's name, Marco, would surface during the model's reasoning. But the model had no context for *why* it knew that. It would encounter "Marco" in its own thinking and essentially go: *what is Marco? Why do I know this?* The fact was there but the coherence wasn't — there was no narrative context around the injected signal.

The recall was wildly inconsistent too. The same facts (Marco, Buddy, Milan, blue, pasta) would be either perfectly recalled (5/5) or completely rejected (0/5) depending on subtle context shifts. Common names like "Marco" failed while unusual names like "Buddy" succeeded — the base experts had stronger priors about common names that overwhelmed our single expert's signal.

That's what led us to the DeltaNet work: trying to give the model coherence by injecting state into the recurrent layers alongside the expert facts. But that opened a new front in the war against the router.

### Fighting the Router

Our memory expert was just 1 of 8 active experts per token — 12.5% influence on the output. As we documented: "One voice cannot override seven." The other 87.5% from base experts overwhelmed the factual signal.

We tried prefill vs decode scaling. We tried per-layer routing strategies. We tried everything. The fundamental problem: the router was trained on natural text distributions, and biasing it away from those distributions degraded the model's ability to generate coherent output. We were fighting the model's own training.

**The key insight:** I was trying to make the model remember by changing how it *thinks* (routing). I should have been changing what it *sees* (its input representations).

**But we still won something:** We proved that custom expert slots can be added without breaking the base model. We proved that programmatic experts can inject facts through the MoE architecture. And we proved that DeltaNet state compounds sequentially like a blockchain — an insight that became critical in Phase 3.

**Entries: KL #150 - #425. Time spent: ~8 weeks (February–March 2026).**

---

## Phase 3: KV State Injection — The Breakthrough (April 2026)

I'm not an AI researcher. I don't have a PhD in machine learning. But I'm a very curious guy with a lot of cross-domain knowledge and product intuition built from a year of shipping products with AI.

The pivotal idea didn't come from reading papers or staring at model internals. It came from a simple logical chain:

*When I send "Hi, my name is Umberto," the model doesn't actually read text. The text gets transformed into tokens, then computed through layers, and the model reasons about the resulting numbers. Those computational results — the model's understanding of my message — are what allow it to answer questions about me. And here's the thing: "Hi, my name is Umberto" will be encoded the same way regardless of how many times you pass it. The model's understanding is deterministic for a given input. So why do we throw those computational results away? What if we store them and re-inject them next time? The model would have the same understanding without re-doing the work.*

That was it. Not a theoretical insight from a paper. Just: the model computes understanding → we throw it away → we recompute it next time → that's stupid → what if we don't throw it away?

And crucially, the MoE phase had already taught us something that made this axiom click: **the model's internal layers don't care how they get their input.** From our work with the programmatic "undercover" expert, we'd proven that the attention layers just receive a stream of activations — they have no mechanism to distinguish whether that stream came from processing real text or was injected from somewhere else. They don't give a shit. As long as the numerical representations are correct, the model behaves identically.

That insight gave us both the confidence to try KV injection and a clear north star for validation: if text and KV injection produce the same representations, they should produce the same outputs. And they did — on the 18-Q LongMemEval benchmark, KV injection scored 8/18 and text scored 8/18. The remaining 10 questions that both missed weren't a KV problem — they were a model capacity limitation. Text couldn't answer them either.

### The First Experiment

I captured K/V tensors from all 10 full-attention layers after processing a set of facts. Stored them. Then started a new session, injected the stored K/V into the cache as "phantom tokens" (positions with real K/V but no actual text tokens), and asked the model about those facts.

**Result: 7/7 recall.** The model answered questions about facts it had never seen in the current conversation. The stored KV states were functionally indistinguishable from having just processed the text.

That was the moment I knew this was real.

### The Hard Parts

Making it work in a single experiment is one thing. Making it work reliably in production is another. The next two to three weeks were about solving every edge case the architecture demanded:

- **Positional alignment** when re-injecting states captured at one position into a different position in the cache.
- **Hybrid architectures** where attention layers and recurrent layers store fundamentally different kinds of state and need different handling.
- **Capture cleanliness** — a subtle interaction between recurrent and attention states meant naive capture produced subtly polluted representations. Solving this required rethinking the capture itself.
- **Relevance selection** — once you have hundreds of stored memories, you can't inject them all. I built a multi-stage pipeline that combines text-based retrieval with the model's own attention as the final ranker, plus a real-time monitoring layer that can swap memories mid-generation if the topic shifts.
- **Memory-pool quality** — not every message is worth remembering. Off-topic chatter, recall questions, and reactions pollute the pool. I built a language-agnostic quality classifier that filters captures before storage and demotes low-information memories during retrieval.
- **System prompt economics** — every captured turn would duplicate the system prompt, wasting storage. Solved by content-addressed deduplication of the system block and clean per-message capture of user content only.

Each of these took days of iteration to get right. The 650+ research log entries aren't padding — they're the actual path through the maze. Implementation specifics are kept in the patent disclosure rather than the public docs.

**Entries: KL #426 - #642 (and counting). Time spent: 8 days (April 14–22, 2026).** Eight days. The preceding ~2.5 months of LoRA and MoE exploration weren't wasted — they built the understanding of the model's internals that made the KV injection insight possible. But the actual architecture went from first proof-of-concept to text-vs-KV parity in 6 days, and from parity to production demo in 2 more.

---

## Where It Is Now

NLS runs in production on a single NVIDIA Grace Blackwell desktop GPU, powering two complementary demos:

**Punk Records (conversational):** Users can chat naturally, building up a conversation over many turns. Close the browser, come back hours or days later — the AI remembers everything. An API transparency log proves that each request sends only the current message. The memory panel shows which memories were retrieved, how the neural scorer ranked them, and whether the streaming scorer swapped anything during generation.

**OpenCode integration (agentic):** On April 27, 2026, NLS was validated end-to-end behind a real coding agent (OpenCode TUI) driving a multi-phase coding task. The agent scaffolded a Next.js + NestJS monorepo with explicit DECISIONs (frontend port, backend port, monorepo manager). Then in fully separate sessions (including a complete TUI restart), it was asked recall questions like "what frontend port did we pick?" and "what backend port did we pick?" — and answered correctly with disambiguating values (3000 and 3001), recovered from 18,751 tokens of injected KV state on a 124-token user prompt. **99.3% prompt-token savings on the recall path. 4/4 questions correct.**

That OpenCode validation was the moment everything came together. The frustration that started this project — agents losing context after a few turns, having to re-paste SSH addresses and database passwords — was finally solved end-to-end in the exact use case that motivated the work: a coding agent that remembers what it built across sessions.

The system handles multiple users concurrently with full memory isolation, persists through server restarts, self-heals its index from stored files on boot, and works on both Qwen 3.5-MoE and Qwen 3.6-MoE (cross-model validation showed Qwen 3.6 produces the highest LongMemEval scores ever achieved by the system: 9/18 vs the prior 8/18 best on Qwen 3.5).

---

## Phase 4: Open reference implementation (June 2026)

The production plugin behind [Punk Records Demo](https://punkrecords.live) remains the full five-phase stack. The **chain-resume slice** — what agent developers need first — is what this repository ships as OSS.

What landed here:

- **`pri/` package** — vLLM KV connector, turn capture, resume inject, `.nls` format
- **Docker image** — BYOC checkpoint, OpenAI-compatible API on :8000
- **Overnight proof run** — frozen artifacts, research analytics, RoPE geometry audit
- **Model matrix** — plug-and-play validation on Gemma, Llama (Tier B)

Key engineering fixes in this phase:

- **Resume pack ordering** — system block must precede turn KV in the phantom pack (fixed Tier B garble)
- **RoPE re-rotation** — 100% delta_uniformity on audited chains
- **Mamba delta-sum** — default for Qwen hybrid Tier A

See [research/00_findings.md](../bench/results/overnight_20260624_003614/research/00_findings.md) and [MODEL_MATRIX.md](MODEL_MATRIX.md).

---

## What I Learned

1. **Every "failure" moved the ball forward.** LoRA taught us catastrophic forgetting can be solved — 80+ compounding swaps on messy data. MoE routing proved programmatic experts can inject facts through the architecture, and that DeltaNet state compounds like a blockchain. Both of these insights were essential to making Phase 3 work. Nothing was wasted.
2. **The insight was simple. The engineering was not.** "Store the computed states and re-inject them" fits in one sentence. Making it actually work required solving a stack of compositional engineering problems — positional alignment, hybrid-architecture handling, capture cleanliness, relevance selection at scale, quality filtering, multi-tenant isolation, recovery on restart. Each took days. The 650+ research log entries aren't padding — they're the actual path through the maze.
3. **You don't need a PhD. You need curiosity and stubbornness.** I'm not an AI researcher. The key insight came from product logic ("why are we re-doing work we already did?"), not from reading attention mechanism papers. The MoE undercover agent came from thinking about the problem like a product — "the attention layer is a customer, it doesn't care who the supplier is." Domain expertise in AI research helps, but cross-domain intuition is what finds the non-obvious paths.
4. **The economics change everything.** Moving from "recompute on expensive HBM" to "read from cheap SSD" isn't just a cost optimization. It changes what's *possible*. Persistent AI relationships, long-term therapeutic memory, institutional knowledge that accumulates over years — these become economically viable when the cost of memory is storage, not compute.

---

## What's Next

The system works. The demo proves it. Now it's time to share it with the world and see what happens.

If you've read this far and you're thinking "this can't possibly work" — I had the same reaction at KL #426 when I first saw 7/7 recall from injected KV states. Try the [demo](https://punkrecords.live). The API log doesn't lie.

---

*The full research log, covering all 640+ experiments in detail (with implementation specifics redacted), is available in [RESEARCH_LOG.md](RESEARCH_LOG.md).*