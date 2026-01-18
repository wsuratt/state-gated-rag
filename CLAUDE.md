# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SSM-Agent: An SSM-inspired agent memory system that uses learned recurrent state + state-conditioned retrieval for agent tasks. The goal is to demonstrate this approach outperforms rolling-context and query-RAG baselines on WebShop and ALFWorld benchmarks.

## Commands

### Environment Setup
```bash
./scripts/setup_envs.sh                    # Install WebShop + ALFWorld from source
pip install -r requirements.txt            # Install Python dependencies
```

### Data Collection
```bash
./scripts/collect_data.sh 500              # Collect N WebShop episodes
python -m data.collect_trajectories --config configs/webshop.yaml --num_episodes 100
```

### Training
```bash
./scripts/train_phase1.sh                  # Phase 1: next-action prediction (go/no-go validation)
python -m training.train_next_action --config configs/training.yaml --data data/trajectories/webshop_baseline.jsonl --output checkpoints/phase1
```

### Testing
```bash
pytest tests/                              # Run all tests
pytest tests/test_encoder.py -v            # Run single test file
```

### Formatting
```bash
black .                                    # Format code
```

## Architecture

### Core Pipeline
```
Events → EventEncoder → GRU StateUpdater → ActionPredictionHead → Action logits
                              ↓
                    StateConditionedRetriever → Retrieved chunks
```

### Key Components

**data/**: Data schemas and collection
- `schemas.py`: `AgentEvent` and `Episode` dataclasses for trajectory logging
- `action_taxonomy.py`: Coarse action classification (7 WebShop classes, 14 ALFWorld classes)
- `collect_trajectories.py`: Run GPT-4o-mini actor on WebShop to collect training data
- `dataset.py`: `NextActionDataset` for training (prefix of events → next action class)

**models/**: Neural network components
- `encoder.py`: `EventEncoder` - frozen MiniLM text encoder + learned type embeddings → d_event vectors
- `state_updater.py`: `GRUStateUpdater` - O(1) per-event update, maintains recurrent hidden state
- `action_head.py`: `ActionPredictionHead` - state → action logits; `BaselineActionHead` - last obs only (ablation)
- `retriever.py`: `StateConditionedRetriever` - retrieves chunks based on state vector, not just query
- `full_model.py`: `RecurrentStateModel` combines all components

**training/**: Training scripts
- `train_next_action.py`: Phase 1 go/no-go validation - trains state model on next-action prediction

**agents/**: Agent implementations (to be built)
- `baseline_react.py`: Rolling window baseline
- `baseline_rag.py`: Query-based retrieval baseline
- `recurrent_agent.py`: Full architecture with state-conditioned retrieval

### Config Structure
- `configs/webshop.yaml`: Environment settings, data collection params, eval settings
- `configs/training.yaml`: Model dimensions (d_event=256, d_state=512), optimizer settings, training hyperparams

### Key Dimensions
- `d_event`: 256 (event embedding size)
- `d_state`: 512 (GRU hidden state size)
- `d_chunk`: 256 (retrieval chunk embedding size)
- WebShop actions: 7 classes (search, click_product, click_option, click_buy, click_back, click_nav, unknown)
- ALFWorld actions: 14 classes (go, open, close, take, put, toggle, heat, cool, clean, examine, look, inventory, use, unknown)

## Development Phases

1. **Phase 1**: Next-action prediction training (validates that recurrent state captures useful info)
2. **Phase 2**: Add state-conditioned retrieval with contrastive loss
3. **Phase 3**: Full agent evaluation with ablations
4. **Phase 4**: Port to ALFWorld
