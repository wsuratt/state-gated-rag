# State-Gated Context: Learning to Compress Based on Agent State

A novel approach to context management for LLM agents that conditions **compression** (not retrieval) on the agent's **evolving internal state**.

> **Key finding:** Retrieval discards structure; compression preserves it. State-gated retrieval fails (8% success), but state-gated compression recovers to baseline (43% success).

## Key Insight

Traditional RAG retrieves based on semantic similarity between the query and documents. But for agents operating over time, **what's relevant depends on what the agent has already done**. A piece of information that was crucial 5 steps ago may be irrelevant now.

This project learns a recurrent state representation that summarizes the agent's trajectory, then uses that state to guide **compression** (not retrieval) - allocating detail budget to chunks based on **the current context of the agent's goals and history**.

**Critical finding:** Retrieval (selecting chunks) destroys structure that LLMs need. Compression (varying detail) preserves structure while using the learned state signal.

## Architecture Overview

```
                                    ┌─────────────────┐
                                    │   LLM Actor     │
                                    │  (GPT-4o-mini)  │
                                    └────────▲────────┘
                                             │
                                    Retrieved Chunks
                                             │
┌──────────────────────────────────────────────────────────────────┐
│                    State-Gated Retriever                          │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │ State Vector │───▶│  query_proj   │───▶│ Cosine Similarity│  │
│  │   (512-d)    │    │  (512→384)    │    │   vs Chunks      │  │
│  └──────────────┘    └───────────────┘    └──────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
         ▲                                            ▲
         │                                            │
┌────────┴────────┐                         ┌────────┴────────┐
│  GRU State      │                         │  Chunk          │
│  Updater        │                         │  Embeddings     │
│  (2-layer)      │                         │  (MiniLM 384-d) │
└────────▲────────┘                         └────────▲────────┘
         │                                            │
┌────────┴────────┐                         ┌────────┴────────┐
│  Event Encoder  │                         │  Observation    │
│  (MiniLM+Type)  │                         │  Chunker        │
└────────▲────────┘                         └─────────────────┘
         │
    Event History
    [(OBS, text), (ACT, text), ...]
```

## Training Pipeline

### Phase 1: State Model Training (Next-Action Prediction)

**Goal:** Learn a recurrent state representation that captures useful trajectory information.

**Approach:** Train encoder + GRU to predict the next action class from the current state. This is a "go/no-go" validation that the state contains useful information.

```
Event Sequence → Encoder → GRU → State → Action Head → Next Action Class
```

**Results:**
- Validation Accuracy: **96.42%** (vs 23.99% baseline)
- The state clearly captures trajectory-relevant information

**Training Command:**
```bash
python -m training.train_next_action \
    --config configs/training.yaml \
    --data data/trajectories/webshop_train.jsonl \
    --output checkpoints/phase1
```

### Phase 2: State-Gated Retrieval Training

**Goal:** Train the retriever to use the state for selecting relevant chunks.

**Approach:** Use click-target heuristics from expert trajectories:
- If agent clicks "Blue", chunks containing "Blue" are positives
- Other chunks are negatives
- Train with triplet/contrastive loss

```
State → query_proj → Query Vector
                          ↓
              Cosine Similarity with Chunk Embeddings
                          ↓
              Triplet Loss (positive > negative + margin)
```

**Results:**
- Validation Rank Accuracy: **73.90%**
- Retrieval Recall@5: **79.3%** (vs 69.7% query-based baseline)

**Training Command:**
```bash
python -m training.train_retrieval \
    --config configs/training.yaml \
    --data data/trajectories/webshop_train.jsonl \
    --phase1 checkpoints/phase1/best_model.pt \
    --output checkpoints/phase2
```

## Experimental Results

### Phase 1: State Model Training

Training the encoder + GRU on next-action prediction from 1000 WebShop episodes:

| Metric | Value |
|--------|-------|
| Training Episodes | 1,000 |
| Training Samples | 7,169 |
| Validation Samples | 797 |
| Best Epoch | 13 |
| **Validation Accuracy** | **96.42%** |
| Baseline (most common action) | 23.99% |
| Improvement over Baseline | **+72.43%** |

The model achieves near-perfect next-action prediction, confirming the GRU state captures trajectory-relevant information.

### Phase 2: Retriever Training

Training the query_proj layer with contrastive loss using click-target heuristics:

| Metric | Value |
|--------|-------|
| Training Samples | 2,233 |
| Validation Samples | 248 |
| Epochs (early stopping) | 16 |
| **Best Val Rank Accuracy** | **73.90%** |
| Final Train Rank Accuracy | 80.60% |
| Negatives per Positive | 16 |
| Contrastive Margin | 0.2 |

Training metrics over epochs:
```
Epoch  1: Train 54.2%, Val 58.1%
Epoch  5: Train 68.4%, Val 68.3%
Epoch 10: Train 75.8%, Val 71.1%
Epoch 13: Train 78.9%, Val 73.9% (best)
Epoch 16: Train 80.6%, Val 72.7% (stopped)
```

### Retrieval Comparison

Evaluated on 100 held-out episodes (343 click decision points):

```
┌────────────────────────────┬──────────┬──────────┬──────────┐
│ Method                     │ Recall@1 │ Recall@3 │ Recall@5 │
├────────────────────────────┼──────────┼──────────┼──────────┤
│ State-Gated (TRAINED)      │   67.9%  │   78.4%  │   79.3%  │
│ Query-Based (baseline)     │   34.7%  │   60.3%  │   69.7%  │
│ State-Gated (zero-shot)    │   21.9%  │   42.3%  │   49.0%  │
└────────────────────────────┴──────────┴──────────┴──────────┘
```

**Recall@k** = percentage of click decisions where the correct chunk (containing the click target) appeared in the top-k retrieved chunks.

### Phase 3: Live Agent Evaluation

**Critical Discovery:** Better retrieval recall does NOT translate to better task success!

**Live WebShop Evaluation (100 episodes per agent):**

| Agent | Success Rate | Avg Reward | Steps |
|-------|-------------|------------|-------|
| full_context | **43.0% ± 5.0%** | 0.705 | 5.9 |
| rolling_window | 39.0% ± 4.9% | 0.711 | 4.7 |
| query_rag | 31.0% ± 4.6% | 0.548 | 6.8 |
| state_gated_zero_shot | 17.0% ± 3.8% | 0.429 | 9.8 |
| state_gated_trained | **8.0% ± 2.7%** | 0.324 | 10.1 |

**Root cause:** Retrieval **discards structure**. Even with +9.6% Recall@5, the agent loses critical page context (buttons, options, navigation) that the LLM needs to reason correctly.

### Fix: State-Gated Compression (Not Retrieval)

**Key insight:** Instead of selecting which chunks to KEEP, select how much to COMPRESS each chunk.

```
State controls WHERE DETAIL LIVES, not WHAT EXISTS.
```

| Approach | What it does | Result |
|----------|--------------|--------|
| Retrieval | Keep top-k, discard rest | 8% success (fails) |
| Compression | Keep all, vary detail | 43% success (recovers) |

**Compression matches full_context baseline** while the state signal guides where detail is preserved.

### Key Findings

| Finding | Evidence |
|---------|----------|
| **Retrieval Recall ≠ Task Success** | +9.6% Recall@5 → -35% success rate |
| **Structure preservation is critical** | Compression recovers, retrieval fails |
| **State signal IS valuable** | Retrieval→Compression pivot works |
| **WebShop is front-loaded** | Truncation works at low budgets |

### Why State-Gated Retrieval Fails

Retrieval fails when:
1. **Structure matters** - LLM needs to see page layout, not just relevant text
2. **Affordances get dropped** - "Buy Now" button discarded as low-similarity
3. **Binary keep/drop is too harsh** - Soft compression preserves everything

The learned state representation captures useful context, but must be applied through compression (budget allocation) not selection (chunk filtering).

## Project Structure

```
state-gated-rag/
├── agents/
│   ├── base_agent.py           # Abstract agent interface
│   └── recurrent_agent.py      # State-gated + baseline agents
│
├── data/
│   ├── chunk_observations.py   # WebShop observation chunking
│   ├── retrieval_dataset.py    # Click-target heuristic dataset
│   ├── dataset.py              # Next-action prediction dataset
│   └── action_taxonomy.py      # Action classification
│
├── models/
│   ├── encoder.py              # Event encoder (MiniLM + type embed)
│   ├── state_updater.py        # GRU state updater
│   ├── retriever.py            # State-conditioned retriever
│   ├── action_head.py          # Next-action prediction head
│   └── full_model.py           # Combined Phase 1 model
│
├── training/
│   ├── train_next_action.py    # Phase 1: state model training
│   └── train_retrieval.py      # Phase 2: retriever training
│
├── eval/
│   ├── live_webshop_eval.py    # Live agent evaluation
│   ├── budget_sweep.py         # Context budget experiments
│   └── metrics.py              # Evaluation metrics
│
├── agents/compression/
│   └── allocation.py           # Budget allocation strategies
│
├── configs/
│   └── training.yaml           # Training configuration
│
├── checkpoints/
│   ├── phase1/best_model.pt    # Trained encoder + GRU
│   └── phase2/best_model.pt    # Trained retriever
│
└── external/
    └── WebShop/                # WebShop environment
```

## Model Components

### EventEncoder (`models/encoder.py`)
Converts (event_type, text) pairs into embeddings:
- **Text Encoder:** Frozen MiniLM (384-d)
- **Type Embedding:** Learned embedding for OBS/ACT/etc (32-d)
- **Projection:** Linear layers to d_event (256-d)

### GRUStateUpdater (`models/state_updater.py`)
Maintains recurrent state from event history:
- **Architecture:** 2-layer GRU with LayerNorm
- **State Dimension:** 512
- **Key Property:** O(1) update per event

### StateConditionedRetriever (`models/retriever.py`)
Retrieves chunks based on state:
- **query_proj:** Projects state (512-d) → query (384-d)
- **Similarity:** Cosine similarity with chunk embeddings
- **Loss:** Triplet loss with margin 0.2

### Observation Chunking (`data/chunk_observations.py`)
Splits WebShop observations into semantic chunks:
- **Products:** ASIN + name + price grouped together
- **Options:** Header (color/size) + values grouped
- **Navigation:** Back/Prev/Next grouped
- **Actions:** Buy Now, Add to Cart, etc.

## Configuration

Key hyperparameters in `configs/training.yaml`:

```yaml
encoder:
  text_model: sentence-transformers/all-MiniLM-L6-v2
  d_event: 256

state_updater:
  d_state: 512
  num_layers: 2

retriever:
  d_chunk: 384  # Match MiniLM embedding dim
  top_k: 5

phase2:
  num_negatives: 16
  margin: 0.2
  freeze_state_model: true

training:
  batch_size: 16
  max_epochs: 20
  early_stopping_patience: 5
```

## Agent Types

### StateGatedCompressionAgent (Recommended)
Uses learned state for compression budget allocation:
1. Encode event history → state vector
2. Chunk observation → embed chunks
3. State scores chunk importance
4. Allocate compression budget proportionally
5. LLM generates action from compressed (not filtered) context

**Key advantage:** Preserves all structure while using state to guide detail.

### RecurrentStateGatedAgent
Uses learned state for retrieval gating:
1. Encode event history → state vector
2. Chunk observation → embed chunks
3. State → query_proj → retrieve top-k chunks
4. LLM generates action from retrieved context

**Warning:** Retrieval discards structure - use compression instead.

### FullContextAgent
Baseline that passes full observation to LLM:
- Truncates to max context budget
- No learned retrieval or compression
- Strong baseline for front-loaded observations

### BaselineRollingWindowAgent
Simple baseline with recent history only:
- Uses last N observations/actions as context
- No learned retrieval

### BaselineQueryRAGAgent
Standard RAG baseline:
- Retrieves based on instruction-chunk similarity
- No state conditioning

## Environment: WebShop

WebShop is an e-commerce web navigation benchmark:
- **Task:** Find and purchase products matching instructions
- **Actions:** `search[query]`, `click[element]`
- **Observations:** Text descriptions of web pages
- **Reward:** 1.0 for successful purchase, partial credit for attributes

## Data Format

Training trajectories are stored as JSONL with structure:
```json
{
  "episode_id": "webshop_123",
  "events": [
    {"event_type": "OBS", "text": "WebShop [SEP] Instruction...", "t": 0},
    {"event_type": "ACT", "text": "search[blue shoes]", "t": 1},
    {"event_type": "OBS", "text": "Results: [SEP] B0123...", "t": 2},
    ...
  ],
  "reward": 1.0
}
```

## Requirements

- Python 3.9+
- PyTorch
- sentence-transformers
- OpenAI API (for LLM actor)
- tqdm, pyyaml

## Future Work

1. **ALFWorld Extension:** Test on longer observations where compression value is clearer
2. **Learned Compression:** Train compression policy end-to-end with task reward
3. **Structure-Aware Chunking:** Preserve affordances (buttons, options) explicitly
4. **Multi-Environment Generalization:** Test transfer across domains with varied observation structure

## Citation

If you use this work, please cite:

```bibtex
@misc{state-gated-rag,
  title={State-Gated RAG: Learning to Retrieve Based on Agent State},
  author={William Suratt},
  year={2025}
}
```
