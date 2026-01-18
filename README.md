# State-Gated RAG: Learning to Retrieve Based on Agent State

A novel approach to retrieval-augmented generation (RAG) for LLM agents that conditions retrieval on the agent's **evolving internal state** rather than just the current query or observation.

## Key Insight

Traditional RAG retrieves based on semantic similarity between the query and documents. But for agents operating over time, **what's relevant depends on what the agent has already done**. A piece of information that was crucial 5 steps ago may be irrelevant now.

This project learns a recurrent state representation that summarizes the agent's trajectory, then uses that state to gate retrieval - retrieving information that's relevant **given the current context of the agent's goals and history**.

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

### Key Findings

| Finding | Evidence |
|---------|----------|
| **Training dramatically improves state-gated retrieval** | +30.3% Recall@5 (49.0% → 79.3%) |
| **Trained state-gated beats query-based** | 79.3% vs 69.7% Recall@5 (+9.6%) |
| **State captures trajectory context** | Zero-shot fails (21.9% R@1), trained succeeds (67.9% R@1) |
| **State model generalizes** | 96.42% val accuracy on unseen episodes |

### Why State-Gated Outperforms Query-Based

Query-based retrieval fails when:
1. **The instruction is ambiguous** - "find blue shoes" doesn't distinguish between selecting color vs clicking product
2. **Context matters** - After clicking a product, the agent needs options (color/size), not more products
3. **History affects relevance** - What was selected earlier constrains what's relevant now

The learned state representation captures this trajectory context, enabling more accurate retrieval.

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
│   ├── run_webshop.py          # WebShop evaluation
│   └── metrics.py              # Evaluation metrics
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

### RecurrentStateGatedAgent
Uses learned state for retrieval gating:
1. Encode event history → state vector
2. Chunk observation → embed chunks
3. State → query_proj → retrieve top-k chunks
4. LLM generates action from retrieved context

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

1. **ALFWorld Extension:** Apply to household task environment
2. **Mamba State Updater:** Replace GRU with selective state space model
3. **End-to-End Training:** Train retriever jointly with LLM feedback
4. **Multi-Environment Generalization:** Test transfer across domains

## Citation

If you use this work, please cite:

```bibtex
@misc{state-gated-rag,
  title={State-Gated RAG: Learning to Retrieve Based on Agent State},
  author={William Suratt},
  year={2025}
}
```
