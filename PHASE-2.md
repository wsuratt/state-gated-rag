Below is an **updated Phase 2 plan that matches exactly what you’ve built**, plus a few “gotchas” I’d fix before you invest time in training retrieval.

---

## What you already have (good)

* ✅ Trajectory collection (`data/collect_trajectories.py`)
* ✅ Next-action dataset and Phase 1 training (`data/dataset.py`, `training/train_next_action.py`)
* ✅ Encoder + GRU + action heads (`models/*`)
* ✅ `StateConditionedRetriever` with a contrastive loss function stub

So Phase 2 is mostly:

1. **chunking**
2. **a retrieval dataset**
3. **train the retriever/query projection**
4. **wire retrieval into agents + eval**
5. **run ablations**

---

# Phase 2 (Concrete) — “State-Gated RAG” on WebShop

## Phase 2A: End-to-end evaluation with *untrained* state-gated retrieval (fast sanity check)

Before training anything, do a “does this even run” version.

### A1) Implement chunking

**You already have a placeholder:** `data/chunk_observations.py`
Make it produce a list of chunk strings from an observation.

Minimum viable chunker:

* split by double newlines
* fallback to single newline
* hard cap chunk length (e.g. 800–1200 chars)
* filter tiny chunks

**Output:** `chunks: List[str]`

### A2) Implement chunk embedding function (reuse MiniLM)

You can reuse the SentenceTransformer already loaded inside `EventEncoder`. But note: your `EventEncoder` currently wraps `SentenceTransformer` inside a `nn.Module` and calls `.encode()` inside forward with `no_grad`. That’s fine.

Add a helper either:

* inside `EventEncoder`: `encode_texts(texts: List[str]) -> Tensor[n, text_dim]`
  or
* in a new file `models/chunk_encoder.py`

Then project chunk embeddings to `d_chunk` (pick `d_chunk = text_dim (384)` or make retriever use 384 so you don’t introduce another projection right away).

**Suggestion:** Set `d_chunk=384` for Phase 2A to simplify.

### A3) Wire retrieval into a new agent (no training yet)

In `agents/recurrent_agent.py`, add an agent variant:

**`recurrent_state_gated_zero_shot`**

* maintain event history as you already do
* every step:

  * compute `state = model.forward([events])["states"][0]`
  * chunk current obs → embed chunks → chunk_embs
  * use `StateConditionedRetriever.forward(state, chunk_embs, top_k)`
  * get top-k chunk texts
  * pass only those chunks (plus instruction) to actor prompt

You *will* need a query projection from state → chunk space:

* your retriever already has `query_proj(state)` which outputs `d_chunk`
* just make sure `d_chunk` matches your chunk embedding dim

So Phase 2A is literally: **does untrained retrieval gating not immediately kill performance**.

### A4) Evaluate against baselines

Run:

* baseline_react (rolling history)
* baseline_rag (query RAG)
* recurrent_state_gated_zero_shot

**Metrics:** success rate, avg steps, token usage.

If zero-shot state-gated is terrible: that’s fine—means you need Phase 2B training (expected).

---

## Phase 2B: Train the retriever (the real Phase 2)

This is the “learn W so that state queries retrieve the right chunk.”

### Key decision: How to label “right chunk” without manual annotation?

For WebShop you can do **cheap heuristic labeling** using the actor’s own click targets.

#### Intuition

If at time t the agent later clicks `click[Some Product Title]`, then the correct page chunk at time t should be the chunk that contains “Some Product Title” (or option text, “Buy Now”, etc.). That gives you positives for free.

### B1) Build a RetrievalDataset from trajectories

Create: `data/retrieval_dataset.py`

For each episode:
For each timestep t where you have an OBS:

* build `state_t` from event prefix up to obs_t (OBS/ACT only)
* chunk obs_t → chunks
* find next action string after this obs (you already do similar in `NextActionDataset`)
* use action text to choose positives:

  * if action is `click[XYZ]`, choose chunk(s) that contain substring `XYZ` (case-insensitive)
  * if action is `search[query]`, positive could be chunk that contains “Search” UI text OR skip search steps (totally fine)
  * if action is `click[Buy Now]`, choose chunk that contains “Buy Now”
  * if no match found, skip sample

Negatives:

* randomly sample N other chunks from same obs that do not contain `XYZ`

Store:

* events_prefix (or directly store precomputed state vectors if you want)
* chunk_texts
* pos_indices
* neg_indices

**Keep it simple**: train on click steps only (`click_*` classes). Search steps add noise.

### B2) Training objective

Use the InfoNCE / contrastive form:

* q = normalize(query_proj(state))
* pos = normalize(embed(chunk[pos]))
* negs = normalize(embed(chunk[neg_i]))

Loss:

* maximize dot(q, pos) vs negs

Your retriever already has `compute_retrieval_loss` (triplet-ish). That’s fine for MVP.

### B3) Implement `training/train_retrieval.py`

This should:

* load Phase 1 checkpoint into `RecurrentStateModelWithRetrieval` (or `RecurrentStateModel` + attach retriever)
* freeze encoder + GRU (optional but recommended at first)
* train **only** retriever parameters (`query_proj` and maybe `importance_head`)
* save `checkpoints/phase2/best_model.pt`

Config additions to `configs/training.yaml`:

```yaml
phase2:
  num_negatives: 16
  margin: 0.2
  top_k: 5
  freeze_state_model: true
  train_query_proj_only: true
```

### B4) Evaluate again

Now run:

* baseline_react
* baseline_rag (query based)
* recurrent_state_gated_trained (retriever trained)

If trained state-gated beats query-RAG (or achieves same success with fewer tokens), you’ve proven the core claim.

---

# Critical Fix / Gotcha in your current code

### Your printed “Baseline” during Phase 1 training is **NOT majority-class baseline**

In `training/train_next_action.py`, you print:

* `majority_baseline` (true majority baseline across dataset)
  but during batches you also compute:
* `baseline_logits = baseline_head(last_obs_emb)`
  and you report its accuracy as “Baseline”

So your “Baseline: 4.69%” in logs is the **learned last-observation-only head**, not “always predict most frequent action.”

That’s fine, just rename it in logs to avoid confusion:

* `baseline_last_obs_acc`
* and separately print `majority_class_baseline`

This matters when you write up results.

---

# What Claude Code should implement next (copy/paste task list)

### Phase 2A (1 day)

1. Implement `data/chunk_observations.py` with deterministic chunking.
2. Ensure chunk embeddings dimension matches retriever `d_chunk` (recommend `d_chunk=384`).
3. Add `agents/recurrent_agent.py` mode: `recurrent_state_gated_zero_shot`
4. Update `eval/run_webshop.py` to run it.

### Phase 2B (2–4 days)

5. Implement `data/retrieval_dataset.py` using click-target substring matching.
6. Implement `training/train_retrieval.py`:

   * load phase1 checkpoint
   * freeze encoder+GRU
   * train retriever query_proj with contrastive loss
7. Add new eval agent mode: `recurrent_state_gated_trained`
8. Add `scripts/train_phase2.sh` + `scripts/run_ablations.sh`
9. Add results table script that compares success/steps/tokens across 3 agents.

---

# Structuring the retrieval data (simple + scalable)

Use JSONL for retriever training too, something like:

```json
{
  "episode_id": "...",
  "t": 12,
  "events": [["OBS","..."],["ACT","..."], ...],   // prefix up to current OBS
  "obs_text": "...",
  "chunks": ["...", "...", "..."],
  "pos_idx": [3],
  "neg_idx": [0,1,2,4,5,6],
  "action": "click[Sony WH-1000XM4]"
}
```

This lets you:

* recompute state from prefix (slower but easy)
* or later cache state vectors to speed training
