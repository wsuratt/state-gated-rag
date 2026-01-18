# Full Implementation Plan: SSM-Inspired Agent Memory System

## Overview

**Goal:** Prove that learned recurrent state + state-conditioned retrieval beats rolling-context and query-RAG on WebShop/ALFWorld benchmarks.

**Timeline:** 6 weeks to paper-grade results

**Budget:** ~$50-100 total (API costs + compute)

**Stack:** Python 3.11, PyTorch, HuggingFace, GPT-4o-mini for actor

---

## Repository Structure

```
ssm-agent/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example                      # API keys template
│
├── configs/
│   ├── webshop.yaml                  # Environment config
│   ├── alfworld.yaml
│   ├── training.yaml                 # Hyperparameters
│   └── eval.yaml                     # Evaluation settings
│
├── data/
│   ├── __init__.py
│   ├── schemas.py                    # AgentEvent, Episode dataclasses
│   ├── action_taxonomy.py            # Coarse action classification
│   ├── collect_trajectories.py       # Run baseline agent + log
│   ├── build_action_vocab.py         # Build action clusters (Phase 2)
│   ├── chunk_observations.py         # Split obs into retrievable chunks
│   └── dataset.py                    # PyTorch Dataset classes
│
├── models/
│   ├── __init__.py
│   ├── encoder.py                    # EventEncoder (MiniLM + type embed)
│   ├── state_updater.py              # GRU StateUpdater
│   ├── action_head.py                # Next-action prediction head
│   ├── retriever.py                  # StateConditionedRetriever
│   └── full_model.py                 # Combined model for training
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py                 # Abstract agent interface
│   ├── baseline_react.py             # Rolling window baseline
│   ├── baseline_rag.py               # Query-based retrieval baseline
│   └── recurrent_agent.py            # Your full architecture
│
├── training/
│   ├── __init__.py
│   ├── train_next_action.py          # Phase 1: go/no-go validation
│   ├── train_retrieval.py            # Phase 2: retrieval gating
│   ├── losses.py                     # Loss functions
│   └── utils.py                      # Training utilities
│
├── eval/
│   ├── __init__.py
│   ├── run_webshop.py                # Evaluate on WebShop
│   ├── run_alfworld.py               # Evaluate on ALFWorld
│   ├── metrics.py                    # Success rate, steps, tokens
│   └── ablations.py                  # Run ablation matrix
│
├── scripts/
│   ├── setup_envs.sh                 # Install WebShop + ALFWorld
│   ├── collect_data.sh               # Data collection pipeline
│   ├── train_phase1.sh               # Week 1 training
│   ├── train_phase2.sh               # Week 2-3 training
│   ├── run_ablations.sh              # Full ablation sweep
│   └── generate_paper_tables.py      # Format results for paper
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Inspect trajectories
│   ├── 02_training_analysis.ipynb    # Loss curves, accuracy
│   ├── 03_state_visualization.ipynb  # t-SNE of state vectors
│   └── 04_retrieval_analysis.ipynb   # What does retrieval select?
│
└── tests/
    ├── test_encoder.py
    ├── test_state_updater.py
    └── test_action_taxonomy.py
```

---

## File-by-File Implementation

### 1. Setup Files

**requirements.txt**
```
# Core
torch>=2.0
transformers>=4.35
sentence-transformers>=2.2

# LLM APIs
openai>=1.0
anthropic>=0.18

# Environments
gymnasium>=0.29
# webshop - install from source
# alfworld - install from source

# Data
jsonlines>=4.0
pandas>=2.0
numpy>=1.24

# Training
wandb>=0.16
tqdm>=4.65
pyyaml>=6.0

# Analysis
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
umap-learn>=0.5

# Dev
pytest>=7.0
black>=23.0
ipykernel>=6.0
```

**.env.example**
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
WANDB_API_KEY=...
WANDB_PROJECT=ssm-agent
```

**configs/webshop.yaml**
```yaml
env:
  name: webshop
  max_steps: 50
  observation_max_length: 4096

data_collection:
  num_episodes: 1000
  actor_model: gpt-4o-mini
  actor_temperature: 0
  output_path: data/trajectories/webshop_baseline.jsonl

training:
  batch_size: 16
  learning_rate: 1e-4
  epochs: 20
  gradient_clip: 1.0
  
model:
  d_event: 256
  d_state: 512
  gru_layers: 2
  num_action_classes: 7
  
eval:
  num_episodes: 200
  max_steps: 50
```

**configs/training.yaml**
```yaml
# Shared training config
seed: 42
device: cuda  # or cpu

encoder:
  text_model: sentence-transformers/all-MiniLM-L6-v2
  freeze_text_encoder: true
  type_embed_dim: 32
  d_event: 256

state_updater:
  type: gru  # or mamba (later)
  d_state: 512
  num_layers: 2
  dropout: 0.1

action_head:
  hidden_dim: 256
  dropout: 0.2

retriever:
  d_query: 256
  top_k: 5

optimizer:
  type: adamw
  lr: 1e-4
  weight_decay: 0.01
  warmup_steps: 100

training:
  batch_size: 16
  max_epochs: 20
  early_stopping_patience: 5
  gradient_clip: 1.0
  log_every: 50
  eval_every: 500
```

---

### 2. Data Module

**data/schemas.py**
```python
"""Core data structures for episodes and events."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json
from datetime import datetime


@dataclass
class AgentEvent:
    """A single event in an agent's trajectory."""
    t: int                              # Step number
    event_type: str                     # OBS, ACT, REWARD, DONE, ERROR
    text: str                           # Observation or action text
    reward: Optional[float] = None      # Reward signal (if any)
    done: Optional[bool] = None         # Episode termination flag
    meta: Dict[str, Any] = field(default_factory=dict)  # Extra info
    timestamp: Optional[float] = None   # Wall clock time
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'AgentEvent':
        return cls(**d)


@dataclass
class Episode:
    """A complete episode (trajectory) from start to termination."""
    episode_id: str
    env: str                            # webshop, alfworld
    instruction: str                    # Task instruction
    events: List[AgentEvent]            # Sequence of events
    success: bool                       # Did the task succeed?
    total_steps: int                    # Number of steps taken
    total_reward: float = 0.0           # Cumulative reward
    token_usage: Dict[str, int] = field(default_factory=dict)  # API tokens
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d)
    
    @classmethod
    def from_json(cls, s: str) -> 'Episode':
        d = json.loads(s)
        d['events'] = [AgentEvent.from_dict(e) for e in d['events']]
        return cls(**d)
    
    def get_observations(self) -> List[str]:
        """Extract all observation texts."""
        return [e.text for e in self.events if e.event_type == 'OBS']
    
    def get_actions(self) -> List[str]:
        """Extract all action texts."""
        return [e.text for e in self.events if e.event_type == 'ACT']
    
    def get_obs_action_pairs(self) -> List[tuple]:
        """Get (observation, next_action) pairs for training."""
        pairs = []
        for i, event in enumerate(self.events):
            if event.event_type == 'OBS':
                # Find next action
                for j in range(i + 1, len(self.events)):
                    if self.events[j].event_type == 'ACT':
                        pairs.append((event.text, self.events[j].text))
                        break
        return pairs
```

**data/action_taxonomy.py**
```python
"""Coarse action classification for tractable next-action prediction."""

from typing import Dict, Tuple
import re


# ============================================================================
# WebShop Action Taxonomy
# ============================================================================

WEBSHOP_ACTIONS = {
    'search': 0,
    'click_product': 1,
    'click_option': 2,
    'click_buy': 3,
    'click_back': 4,
    'click_nav': 5,
    'unknown': 6,
}

WEBSHOP_ACTION_NAMES = {v: k for k, v in WEBSHOP_ACTIONS.items()}


def classify_webshop_action(action_str: str) -> Tuple[int, str]:
    """
    Map raw WebShop action string to coarse action class.
    
    Returns:
        (action_id, action_name)
    """
    action_str_lower = action_str.lower().strip()
    
    # Search actions
    if action_str_lower.startswith('search['):
        return WEBSHOP_ACTIONS['search'], 'search'
    
    # Must be a click action
    if not action_str_lower.startswith('click['):
        return WEBSHOP_ACTIONS['unknown'], 'unknown'
    
    # Extract click target
    match = re.match(r'click\[(.+)\]', action_str, re.IGNORECASE)
    if not match:
        return WEBSHOP_ACTIONS['unknown'], 'unknown'
    
    target = match.group(1).lower()
    
    # Buy/cart actions
    buy_keywords = ['buy now', 'add to cart', 'purchase', 'checkout']
    if any(kw in target for kw in buy_keywords):
        return WEBSHOP_ACTIONS['click_buy'], 'click_buy'
    
    # Back/navigation
    back_keywords = ['back', 'return', '< prev', 'previous']
    if any(kw in target for kw in back_keywords):
        return WEBSHOP_ACTIONS['click_back'], 'click_back'
    
    # Pagination
    nav_keywords = ['next', 'page', '>', 'more results']
    if any(kw in target for kw in nav_keywords):
        return WEBSHOP_ACTIONS['click_nav'], 'click_nav'
    
    # Options are typically short (colors, sizes)
    # Products are longer (full product names)
    if len(target) < 25:
        # Check if it looks like an option (color, size, etc.)
        option_patterns = [
            r'^(small|medium|large|xl|xxl|xs)$',
            r'^(red|blue|green|black|white|pink|purple|yellow|orange|brown|gray|grey)$',
            r'^\d+(\.\d+)?\s*(oz|ml|gb|tb|inch|in|cm|mm)$',
            r'^size\s*\d+',
            r'^\d+\s*pack$',
        ]
        for pattern in option_patterns:
            if re.match(pattern, target):
                return WEBSHOP_ACTIONS['click_option'], 'click_option'
        
        # Short but not obviously an option - probably still an option
        return WEBSHOP_ACTIONS['click_option'], 'click_option'
    
    # Longer text = product name
    return WEBSHOP_ACTIONS['click_product'], 'click_product'


# ============================================================================
# ALFWorld Action Taxonomy
# ============================================================================

ALFWORLD_ACTIONS = {
    'go': 0,
    'open': 1,
    'close': 2,
    'take': 3,
    'put': 4,
    'toggle': 5,
    'heat': 6,
    'cool': 7,
    'clean': 8,
    'examine': 9,
    'look': 10,
    'inventory': 11,
    'use': 12,
    'unknown': 13,
}

ALFWORLD_ACTION_NAMES = {v: k for k, v in ALFWORLD_ACTIONS.items()}


def classify_alfworld_action(action_str: str) -> Tuple[int, str]:
    """
    Map raw ALFWorld action string to coarse action class.
    
    Returns:
        (action_id, action_name)
    """
    action_str_lower = action_str.lower().strip()
    
    # Match action prefix
    for action_name, action_id in ALFWORLD_ACTIONS.items():
        if action_name == 'unknown':
            continue
        if action_str_lower.startswith(action_name):
            return action_id, action_name
    
    return ALFWORLD_ACTIONS['unknown'], 'unknown'


# ============================================================================
# Unified Interface
# ============================================================================

def classify_action(action_str: str, env: str) -> Tuple[int, str]:
    """Classify action for any supported environment."""
    if env == 'webshop':
        return classify_webshop_action(action_str)
    elif env == 'alfworld':
        return classify_alfworld_action(action_str)
    else:
        raise ValueError(f"Unknown environment: {env}")


def get_num_actions(env: str) -> int:
    """Get number of action classes for an environment."""
    if env == 'webshop':
        return len(WEBSHOP_ACTIONS)
    elif env == 'alfworld':
        return len(ALFWORLD_ACTIONS)
    else:
        raise ValueError(f"Unknown environment: {env}")


def get_action_name(action_id: int, env: str) -> str:
    """Get action name from ID."""
    if env == 'webshop':
        return WEBSHOP_ACTION_NAMES.get(action_id, 'unknown')
    elif env == 'alfworld':
        return ALFWORLD_ACTION_NAMES.get(action_id, 'unknown')
    else:
        raise ValueError(f"Unknown environment: {env}")
```

**data/collect_trajectories.py**
```python
"""
Collect trajectories by running a baseline agent on WebShop/ALFWorld.
"""

import os
import uuid
import argparse
from typing import Optional
from tqdm import tqdm
import jsonlines
from openai import OpenAI
import yaml

from data.schemas import AgentEvent, Episode


# ============================================================================
# Actor LLM
# ============================================================================

class ActorLLM:
    """Wrapper for LLM that generates actions."""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.total_tokens = {"prompt": 0, "completion": 0}
    
    def get_action(self, instruction: str, observation: str, history: str = "") -> str:
        """Generate next action given current state."""
        
        system_prompt = """You are an agent completing a shopping task. Given an instruction and the current webpage, output a single action.

Valid actions:
- search[query] - search for products
- click[element] - click on a button/link (use exact text from the page)

Rules:
1. Output ONLY the action, nothing else
2. Use exact text from the page for click targets
3. When you find a matching product, click on it
4. Select all required options before adding to cart
5. Click "Buy Now" when ready to purchase

Example outputs:
search[wireless bluetooth headphones]
click[Sony WH-1000XM4]
click[Black]
click[Buy Now]"""

        user_content = f"Instruction: {instruction}\n\n"
        if history:
            user_content += f"Recent actions:\n{history}\n\n"
        user_content += f"Current page:\n{observation}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=100,
            temperature=self.temperature
        )
        
        # Track token usage
        self.total_tokens["prompt"] += response.usage.prompt_tokens
        self.total_tokens["completion"] += response.usage.completion_tokens
        
        return response.choices[0].message.content.strip()


# ============================================================================
# WebShop Environment Wrapper
# ============================================================================

class WebShopEnvWrapper:
    """Wrapper for WebShop environment with logging."""
    
    def __init__(self):
        # Import here to avoid dependency issues
        try:
            from webshop.web_agent_site.envs import WebAgentTextEnv
            self.env = WebAgentTextEnv(
                observation_mode="text",
                num_products=None,  # Use all products
            )
        except ImportError:
            raise ImportError(
                "WebShop not installed. Run: "
                "git clone https://github.com/princeton-nlp/WebShop && "
                "cd WebShop && pip install -e ."
            )
    
    def reset(self, instruction: Optional[str] = None):
        """Reset environment and return initial observation."""
        obs, info = self.env.reset()
        if instruction:
            self.env.instruction = instruction
        return obs, {"instruction": self.env.instruction_text}
    
    def step(self, action: str):
        """Execute action and return (obs, reward, done, info)."""
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
    
    def get_instruction(self) -> str:
        """Get current task instruction."""
        return self.env.instruction_text


# ============================================================================
# Episode Collection
# ============================================================================

def collect_episode(
    env: WebShopEnvWrapper,
    actor: ActorLLM,
    max_steps: int = 50,
    verbose: bool = False
) -> Episode:
    """Run one episode and return logged Episode."""
    
    events = []
    obs, info = env.reset()
    instruction = info["instruction"]
    
    done = False
    t = 0
    total_reward = 0
    history_actions = []
    
    while not done and t < max_steps:
        # Log observation
        events.append(AgentEvent(
            t=t,
            event_type='OBS',
            text=obs[:4096],  # Truncate very long observations
            meta={"step": t}
        ))
        
        # Get action from actor
        history_str = "\n".join(history_actions[-3:]) if history_actions else ""
        action = actor.get_action(instruction, obs, history_str)
        
        # Log action
        events.append(AgentEvent(
            t=t,
            event_type='ACT',
            text=action,
            meta={"step": t}
        ))
        history_actions.append(action)
        
        if verbose:
            print(f"Step {t}: {action}")
        
        # Execute action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Log reward if non-zero
        if reward != 0:
            events.append(AgentEvent(
                t=t,
                event_type='REWARD',
                text="",
                reward=reward,
                meta={"step": t}
            ))
        
        t += 1
    
    # Log episode end
    events.append(AgentEvent(
        t=t,
        event_type='DONE',
        text="",
        done=True,
        meta={"final_reward": total_reward}
    ))
    
    # Determine success (WebShop: reward >= 1.0 is success, partial otherwise)
    success = total_reward >= 1.0
    
    return Episode(
        episode_id=str(uuid.uuid4()),
        env="webshop",
        instruction=instruction,
        events=events,
        success=success,
        total_steps=t,
        total_reward=total_reward,
        token_usage=actor.total_tokens.copy()
    )


def collect_trajectories(
    output_path: str,
    num_episodes: int = 1000,
    max_steps: int = 50,
    model: str = "gpt-4o-mini",
    resume: bool = True
):
    """Collect multiple episodes and save to JSONL."""
    
    # Setup
    env = WebShopEnvWrapper()
    actor = ActorLLM(model=model)
    
    # Resume from existing file if present
    start_idx = 0
    if resume and os.path.exists(output_path):
        with jsonlines.open(output_path) as reader:
            start_idx = sum(1 for _ in reader)
        print(f"Resuming from episode {start_idx}")
    
    # Collect episodes
    mode = 'a' if resume and start_idx > 0 else 'w'
    
    success_count = 0
    total_steps_list = []
    
    with jsonlines.open(output_path, mode=mode) as writer:
        for i in tqdm(range(start_idx, num_episodes), desc="Collecting episodes"):
            try:
                episode = collect_episode(env, actor, max_steps)
                writer.write(episode.to_json())
                
                if episode.success:
                    success_count += 1
                total_steps_list.append(episode.total_steps)
                
                # Print progress every 100 episodes
                if (i + 1) % 100 == 0:
                    success_rate = success_count / (i + 1 - start_idx)
                    avg_steps = sum(total_steps_list) / len(total_steps_list)
                    print(f"\nProgress: {i+1}/{num_episodes}")
                    print(f"Success rate: {success_rate:.2%}")
                    print(f"Avg steps: {avg_steps:.1f}")
                    print(f"Total tokens: {actor.total_tokens}")
                    
            except Exception as e:
                print(f"Error in episode {i}: {e}")
                continue
    
    # Final stats
    print(f"\n=== Collection Complete ===")
    print(f"Total episodes: {num_episodes}")
    print(f"Success rate: {success_count / (num_episodes - start_idx):.2%}")
    print(f"Avg steps: {sum(total_steps_list) / len(total_steps_list):.1f}")
    print(f"Total tokens used: {actor.total_tokens}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/webshop.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override with CLI args
    output_path = args.output or config["data_collection"]["output_path"]
    num_episodes = args.num_episodes or config["data_collection"]["num_episodes"]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Collect trajectories
    collect_trajectories(
        output_path=output_path,
        num_episodes=num_episodes,
        max_steps=config["env"]["max_steps"],
        model=config["data_collection"]["actor_model"]
    )
```

**data/dataset.py**
```python
"""PyTorch Datasets for training the state model."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import jsonlines
from tqdm import tqdm

from data.schemas import Episode, AgentEvent
from data.action_taxonomy import classify_action, get_num_actions


class NextActionDataset(Dataset):
    """
    Dataset for next-action prediction training.
    
    Each sample is a prefix of events (obs, act, obs, act, ..., obs)
    and the target is the next action class.
    """
    
    def __init__(
        self,
        episodes_path: str,
        env: str = "webshop",
        max_episodes: Optional[int] = None,
        min_steps: int = 2,
    ):
        self.env = env
        self.num_actions = get_num_actions(env)
        self.samples = []  # List of (episode_idx, prefix_end_idx, target_action_id)
        
        # Load episodes
        self.episodes: List[Episode] = []
        with jsonlines.open(episodes_path) as reader:
            for i, line in enumerate(tqdm(reader, desc="Loading episodes")):
                if max_episodes and i >= max_episodes:
                    break
                episode = Episode.from_json(line)
                self.episodes.append(episode)
        
        # Build training samples
        # For each observation, predict the next action
        for ep_idx, episode in enumerate(tqdm(self.episodes, desc="Building samples")):
            events = episode.events
            
            for i, event in enumerate(events):
                if event.event_type != 'OBS':
                    continue
                
                # Find the next action after this observation
                next_action = None
                for j in range(i + 1, len(events)):
                    if events[j].event_type == 'ACT':
                        next_action = events[j].text
                        break
                
                if next_action is None:
                    continue
                
                # Classify the action
                action_id, _ = classify_action(next_action, env)
                
                # Store sample: (episode_idx, event_idx_of_obs, target_action)
                self.samples.append((ep_idx, i, action_id))
        
        print(f"Created {len(self.samples)} training samples from {len(self.episodes)} episodes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dict with:
        - events: list of (event_type, text) tuples up to and including current obs
        - instruction: task instruction
        - target_action: action class ID
        """
        ep_idx, obs_idx, target_action = self.samples[idx]
        episode = self.episodes[ep_idx]
        
        # Get all events up to and including this observation
        events = []
        for i in range(obs_idx + 1):
            e = episode.events[i]
            if e.event_type in ['OBS', 'ACT']:
                events.append((e.event_type, e.text))
        
        return {
            'events': events,
            'instruction': episode.instruction,
            'target_action': target_action,
            'episode_id': episode.episode_id,
        }


def collate_episodes(batch: List[Dict]) -> Dict:
    """
    Custom collate function that handles variable-length event sequences.
    
    Returns batched tensors where sequences are NOT padded (we process per-sample).
    """
    return {
        'events': [sample['events'] for sample in batch],
        'instructions': [sample['instruction'] for sample in batch],
        'target_actions': torch.tensor([sample['target_action'] for sample in batch]),
        'episode_ids': [sample['episode_id'] for sample in batch],
    }


def create_dataloader(
    episodes_path: str,
    env: str = "webshop",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    max_episodes: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader for next-action prediction."""
    
    dataset = NextActionDataset(
        episodes_path=episodes_path,
        env=env,
        max_episodes=max_episodes,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_episodes,
    )
```

---

### 3. Models Module

**models/encoder.py**
```python
"""Event encoder: converts (event_type, text) -> embedding."""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class EventEncoder(nn.Module):
    """
    Encodes agent events into fixed-size embeddings.
    
    Components:
    - Frozen text encoder (MiniLM) for observation/action text
    - Learned type embedding for event types
    - Linear projection to combine them
    """
    
    EVENT_TYPES = ['OBS', 'ACT', 'REWARD', 'DONE', 'ERROR', 'PAD']
    
    def __init__(
        self,
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        d_event: int = 256,
        type_embed_dim: int = 32,
        freeze_text_encoder: bool = True,
    ):
        super().__init__()
        
        self.d_event = d_event
        
        # Text encoder (frozen)
        self.text_encoder = SentenceTransformer(text_model)
        self.text_dim = self.text_encoder.get_sentence_embedding_dimension()
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Event type embedding (learned)
        self.type_to_id = {t: i for i, t in enumerate(self.EVENT_TYPES)}
        self.type_embed = nn.Embedding(len(self.EVENT_TYPES), type_embed_dim)
        
        # Projection layer
        self.proj = nn.Sequential(
            nn.Linear(self.text_dim + type_embed_dim, d_event),
            nn.ReLU(),
            nn.Linear(d_event, d_event),
        )
    
    def forward(
        self,
        event_types: List[str],
        texts: List[str],
    ) -> torch.Tensor:
        """
        Encode a batch of events.
        
        Args:
            event_types: List of event type strings ['OBS', 'ACT', ...]
            texts: List of text strings
            
        Returns:
            Tensor of shape (batch_size, d_event)
        """
        device = next(self.parameters()).device
        
        # Get text embeddings (frozen)
        with torch.no_grad():
            text_embs = self.text_encoder.encode(
                texts,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            )
        
        # Get type embeddings (learned)
        type_ids = torch.tensor(
            [self.type_to_id.get(t, self.type_to_id['PAD']) for t in event_types],
            device=device
        )
        type_embs = self.type_embed(type_ids)
        
        # Combine and project
        combined = torch.cat([text_embs, type_embs], dim=-1)
        return self.proj(combined)
    
    def encode_single(self, event_type: str, text: str) -> torch.Tensor:
        """Encode a single event. Returns (d_event,) tensor."""
        return self.forward([event_type], [text]).squeeze(0)
    
    def encode_sequence(
        self,
        events: List[Tuple[str, str]],
    ) -> torch.Tensor:
        """
        Encode a sequence of events.
        
        Args:
            events: List of (event_type, text) tuples
            
        Returns:
            Tensor of shape (seq_len, d_event)
        """
        if len(events) == 0:
            device = next(self.parameters()).device
            return torch.zeros(0, self.d_event, device=device)
        
        event_types = [e[0] for e in events]
        texts = [e[1] for e in events]
        return self.forward(event_types, texts)
```

**models/state_updater.py**
```python
"""Recurrent state updater: processes event sequence into hidden state."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GRUStateUpdater(nn.Module):
    """
    GRU-based state updater.
    
    Maintains a hidden state that summarizes the event history.
    O(1) update per event (the key efficiency property).
    """
    
    def __init__(
        self,
        d_event: int = 256,
        d_state: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_event = d_event
        self.d_state = d_state
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=d_event,
            hidden_size=d_state,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(d_state)
    
    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.d_state, device=device)
    
    def forward(
        self,
        event_embeddings: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence of event embeddings.
        
        Args:
            event_embeddings: (batch, seq_len, d_event)
            h: Optional initial hidden state (num_layers, batch, d_state)
            
        Returns:
            outputs: (batch, seq_len, d_state) - state at each timestep
            h_final: (num_layers, batch, d_state) - final hidden state
        """
        batch_size = event_embeddings.size(0)
        
        if h is None:
            h = self.init_state(batch_size)
        
        outputs, h_final = self.gru(event_embeddings, h)
        outputs = self.layer_norm(outputs)
        
        return outputs, h_final
    
    def step(
        self,
        event_embedding: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process a single event (online/incremental update).
        
        Args:
            event_embedding: (d_event,) or (batch, d_event)
            h: (num_layers, batch, d_state)
            
        Returns:
            h_new: (num_layers, batch, d_state)
        """
        # Handle unbatched input
        if event_embedding.dim() == 1:
            event_embedding = event_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, d_event)
        elif event_embedding.dim() == 2:
            event_embedding = event_embedding.unsqueeze(1)  # (batch, 1, d_event)
        
        _, h_new = self.gru(event_embedding, h)
        return h_new
    
    def get_state_vector(self, h: torch.Tensor) -> torch.Tensor:
        """
        Extract the state vector from hidden state.
        
        Args:
            h: (num_layers, batch, d_state)
            
        Returns:
            state: (batch, d_state) - final layer's hidden state
        """
        return self.layer_norm(h[-1])  # Use last layer
```

**models/action_head.py**
```python
"""Action prediction head: state -> action logits."""

import torch
import torch.nn as nn


class ActionPredictionHead(nn.Module):
    """
    Predicts next action class from state vector.
    
    Simple MLP: state -> hidden -> logits
    """
    
    def __init__(
        self,
        d_state: int = 512,
        hidden_dim: int = 256,
        num_actions: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(d_state, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, d_state) or (d_state,)
            
        Returns:
            logits: (batch, num_actions) or (num_actions,)
        """
        return self.head(state)


class BaselineActionHead(nn.Module):
    """
    Baseline: predict action from last observation only (no recurrence).
    
    Used to verify that the GRU state actually helps.
    """
    
    def __init__(
        self,
        d_event: int = 256,
        hidden_dim: int = 256,
        num_actions: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(d_event, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(self, last_obs_embedding: torch.Tensor) -> torch.Tensor:
        """Predict from last observation embedding only."""
        return self.head(last_obs_embedding)
```

**models/retriever.py**
```python
"""State-conditioned retriever: select relevant chunks based on state."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class StateConditionedRetriever(nn.Module):
    """
    Retrieves relevant chunks based on the current state vector.
    
    The key insight: we retrieve based on the AGENT'S EVOLVING STATE,
    not just the current query/observation.
    """
    
    def __init__(
        self,
        d_state: int = 512,
        d_chunk: int = 256,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.d_chunk = d_chunk
        self.temperature = temperature
        
        # Project state to query space
        self.query_proj = nn.Sequential(
            nn.Linear(d_state, d_chunk),
            nn.ReLU(),
            nn.Linear(d_chunk, d_chunk),
        )
        
        # Optional: learned importance weighting
        self.importance_head = nn.Linear(d_chunk, 1)
    
    def forward(
        self,
        state: torch.Tensor,
        chunk_embeddings: torch.Tensor,
        top_k: int = 5,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieve top-k chunks based on state.
        
        Args:
            state: (batch, d_state) or (d_state,)
            chunk_embeddings: (num_chunks, d_chunk)
            top_k: number of chunks to retrieve
            return_scores: whether to return similarity scores
            
        Returns:
            indices: (batch, top_k) or (top_k,) - indices of selected chunks
            scores: optional (batch, top_k) - similarity scores
        """
        single_query = state.dim() == 1
        if single_query:
            state = state.unsqueeze(0)  # (1, d_state)
        
        # Project state to query
        query = self.query_proj(state)  # (batch, d_chunk)
        
        # Normalize for cosine similarity
        query = F.normalize(query, dim=-1)
        chunk_embeddings = F.normalize(chunk_embeddings, dim=-1)
        
        # Compute similarities
        scores = torch.matmul(query, chunk_embeddings.T)  # (batch, num_chunks)
        scores = scores / self.temperature
        
        # Get top-k
        top_k = min(top_k, chunk_embeddings.size(0))
        topk_scores, topk_indices = scores.topk(top_k, dim=-1)
        
        if single_query:
            topk_indices = topk_indices.squeeze(0)
            topk_scores = topk_scores.squeeze(0)
        
        if return_scores:
            return topk_indices, topk_scores
        return topk_indices, None
    
    def compute_retrieval_loss(
        self,
        state: torch.Tensor,
        positive_chunks: torch.Tensor,
        negative_chunks: torch.Tensor,
        margin: float = 0.2,
    ) -> torch.Tensor:
        """
        Contrastive loss for training the retriever.
        
        Args:
            state: (batch, d_state)
            positive_chunks: (batch, d_chunk) - chunks that should be retrieved
            negative_chunks: (batch, num_neg, d_chunk) - chunks that shouldn't
            margin: margin for triplet loss
            
        Returns:
            loss: scalar
        """
        query = self.query_proj(state)  # (batch, d_chunk)
        query = F.normalize(query, dim=-1)
        
        positive_chunks = F.normalize(positive_chunks, dim=-1)
        negative_chunks = F.normalize(negative_chunks, dim=-1)
        
        # Positive similarity
        pos_sim = (query * positive_chunks).sum(dim=-1)  # (batch,)
        
        # Negative similarities
        neg_sim = torch.matmul(
            query.unsqueeze(1),  # (batch, 1, d_chunk)
            negative_chunks.transpose(-1, -2)  # (batch, d_chunk, num_neg)
        ).squeeze(1)  # (batch, num_neg)
        
        # Triplet loss: positive should be closer than negative by margin
        loss = F.relu(margin - pos_sim.unsqueeze(-1) + neg_sim).mean()
        
        return loss
```

**models/full_model.py**
```python
"""Full model combining encoder, state updater, and action head."""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional

from models.encoder import EventEncoder
from models.state_updater import GRUStateUpdater
from models.action_head import ActionPredictionHead, BaselineActionHead
from models.retriever import StateConditionedRetriever


class RecurrentStateModel(nn.Module):
    """
    Full model for next-action prediction with recurrent state.
    
    Architecture:
    events -> encoder -> GRU state updater -> action head -> logits
    """
    
    def __init__(
        self,
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        d_event: int = 256,
        d_state: int = 512,
        num_gru_layers: int = 2,
        num_actions: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = EventEncoder(
            text_model=text_model,
            d_event=d_event,
            freeze_text_encoder=True,
        )
        
        self.state_updater = GRUStateUpdater(
            d_event=d_event,
            d_state=d_state,
            num_layers=num_gru_layers,
            dropout=dropout,
        )
        
        self.action_head = ActionPredictionHead(
            d_state=d_state,
            num_actions=num_actions,
            dropout=dropout,
        )
        
        # Baseline head for comparison (no recurrence)
        self.baseline_head = BaselineActionHead(
            d_event=d_event,
            num_actions=num_actions,
            dropout=dropout,
        )
    
    def forward(
        self,
        events_batch: List[List[Tuple[str, str]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a batch of event sequences.
        
        Args:
            events_batch: List of event sequences, each is List[(event_type, text)]
            
        Returns:
            dict with 'logits', 'baseline_logits', 'states'
        """
        device = next(self.parameters()).device
        batch_size = len(events_batch)
        
        all_logits = []
        all_baseline_logits = []
        all_states = []
        
        for events in events_batch:
            if len(events) == 0:
                # Empty sequence - use zero state
                h = self.state_updater.init_state(1)
                state = self.state_updater.get_state_vector(h)
                logits = self.action_head(state)
                baseline_logits = torch.zeros_like(logits)
            else:
                # Encode all events
                event_embs = self.encoder.encode_sequence(events)  # (seq_len, d_event)
                event_embs = event_embs.unsqueeze(0)  # (1, seq_len, d_event)
                
                # Get states at each timestep
                outputs, h_final = self.state_updater(event_embs)  # outputs: (1, seq_len, d_state)
                
                # Use final state for action prediction
                state = outputs[0, -1]  # (d_state,)
                logits = self.action_head(state)
                
                # Baseline: just use last observation embedding
                last_obs_emb = event_embs[0, -1]  # (d_event,)
                baseline_logits = self.baseline_head(last_obs_emb)
            
            all_logits.append(logits)
            all_baseline_logits.append(baseline_logits)
            all_states.append(state)
        
        return {
            'logits': torch.stack(all_logits),  # (batch, num_actions)
            'baseline_logits': torch.stack(all_baseline_logits),
            'states': torch.stack(all_states),  # (batch, d_state)
        }
    
    def predict(self, events: List[Tuple[str, str]]) -> int:
        """Predict action class for a single event sequence."""
        self.eval()
        with torch.no_grad():
            output = self.forward([events])
            return output['logits'][0].argmax().item()


class RecurrentStateModelWithRetrieval(RecurrentStateModel):
    """Extended model with state-conditioned retrieval."""
    
    def __init__(self, *args, d_chunk: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.retriever = StateConditionedRetriever(
            d_state=kwargs.get('d_state', 512),
            d_chunk=d_chunk,
        )
```

---

### 4. Training Module

**training/train_next_action.py**
```python
"""
Phase 1 Training: Next-action prediction (go/no-go validation).

This script trains the state model to predict the next action class
from the current state, validating that the recurrent state captures
useful information.
"""

import os
import argparse
from typing import Dict, Optional
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb

from data.dataset import NextActionDataset, collate_episodes
from data.action_taxonomy import get_num_actions, get_action_name
from models.full_model import RecurrentStateModel


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def compute_baseline_accuracy(dataset: NextActionDataset) -> float:
    """Compute accuracy of always predicting most common action."""
    from collections import Counter
    action_counts = Counter()
    for i in range(len(dataset)):
        sample = dataset[i]
        action_counts[sample['target_action']] += 1
    
    most_common = action_counts.most_common(1)[0][1]
    return most_common / len(dataset)


def train_epoch(
    model: RecurrentStateModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_acc = 0
    total_baseline_acc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch['events'])
        logits = output['logits']
        baseline_logits = output['baseline_logits']
        
        targets = batch['target_actions'].to(device)
        
        # Compute losses
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Metrics
        acc = compute_accuracy(logits, targets)
        baseline_acc = compute_accuracy(baseline_logits, targets)
        
        total_loss += loss.item()
        total_acc += acc
        total_baseline_acc += baseline_acc
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{acc:.2%}',
            'baseline': f'{baseline_acc:.2%}'
        })
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'baseline_accuracy': total_baseline_acc / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: RecurrentStateModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0
    total_acc = 0
    total_baseline_acc = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        output = model(batch['events'])
        logits = output['logits']
        baseline_logits = output['baseline_logits']
        
        targets = batch['target_actions'].to(device)
        
        loss = criterion(logits, targets)
        acc = compute_accuracy(logits, targets)
        baseline_acc = compute_accuracy(baseline_logits, targets)
        
        total_loss += loss.item()
        total_acc += acc
        total_baseline_acc += baseline_acc
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'baseline_accuracy': total_baseline_acc / num_batches,
    }


def train(
    config_path: str,
    data_path: str,
    output_dir: str,
    use_wandb: bool = True,
):
    """Main training function."""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'ssm-agent'),
            config=config,
            name=f"phase1_next_action",
        )
    
    # Load data
    env = config.get('env', 'webshop')
    num_actions = get_num_actions(env)
    
    full_dataset = NextActionDataset(
        episodes_path=data_path,
        env=env,
        max_episodes=config.get('max_episodes', None),
    )
    
    # Train/val split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_episodes,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_episodes,
    )
    
    # Compute baselines
    majority_baseline = compute_baseline_accuracy(full_dataset)
    print(f"Majority class baseline: {majority_baseline:.2%}")
    
    # Initialize model
    model = RecurrentStateModel(
        text_model=config['encoder']['text_model'],
        d_event=config['encoder']['d_event'],
        d_state=config['state_updater']['d_state'],
        num_gru_layers=config['state_updater']['num_layers'],
        num_actions=num_actions,
        dropout=config['state_updater']['dropout'],
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay'],
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['max_epochs'],
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config['training']['max_epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['training']['max_epochs']} ===")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip=config['training']['gradient_clip'],
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.2%}, "
              f"Baseline: {train_metrics['baseline_accuracy']:.2%}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.2%}, "
              f"Baseline: {val_metrics['baseline_accuracy']:.2%}")
        
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/baseline_accuracy': train_metrics['baseline_accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/baseline_accuracy': val_metrics['baseline_accuracy'],
                'lr': scheduler.get_last_lr()[0],
            })
        
        # Checkpointing
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'config': config,
            }, os.path.join(output_dir, 'best_model.pt'))
            
            print(f"  -> New best model saved (val acc: {best_val_acc:.2%})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping after {epoch + 1} epochs")
            break
    
    # Final summary
    print("\n=== Training Complete ===")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Majority baseline: {majority_baseline:.2%}")
    print(f"Improvement over baseline: {best_val_acc - majority_baseline:.2%}")
    
    # Go/no-go check
    if best_val_acc > majority_baseline + 0.05:  # At least 5% above majority
        print("\n✓ GO: Model shows meaningful improvement over baseline!")
    else:
        print("\n✗ NO-GO: Model does not sufficiently beat baseline. Debug needed.")
    
    if use_wandb:
        wandb.finish()
    
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to trajectories JSONL")
    parser.add_argument("--output", type=str, default="checkpoints/phase1")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        use_wandb=not args.no_wandb,
    )
```

---

### 5. Scripts

**scripts/setup_envs.sh**
```bash
#!/bin/bash
# Setup script for WebShop and ALFWorld environments

set -e

echo "=== Setting up SSM-Agent environments ==="

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "ssm-agent"; then
    echo "Creating conda environment..."
    conda create -n ssm-agent python=3.11 -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ssm-agent

# Install base requirements
echo "Installing base requirements..."
pip install -r requirements.txt

# Install WebShop
echo "Installing WebShop..."
if [ ! -d "external/WebShop" ]; then
    mkdir -p external
    cd external
    git clone https://github.com/princeton-nlp/WebShop.git
    cd WebShop
    pip install -e .
    cd ../..
fi

# Install ALFWorld
echo "Installing ALFWorld..."
if [ ! -d "external/alfworld" ]; then
    mkdir -p external
    cd external
    git clone https://github.com/alfworld/alfworld.git
    cd alfworld
    pip install -e .
    # Download game files
    alfworld-download
    cd ../..
fi

echo "=== Setup complete ==="
echo "Activate with: conda activate ssm-agent"
```

**scripts/collect_data.sh**
```bash
#!/bin/bash
# Collect trajectories from WebShop

set -e

# Configuration
NUM_EPISODES=${1:-1000}
OUTPUT_DIR="data/trajectories"
CONFIG="configs/webshop.yaml"

echo "=== Collecting $NUM_EPISODES WebShop episodes ==="

# Create output directory
mkdir -p $OUTPUT_DIR

# Run collection
python -m data.collect_trajectories \
    --config $CONFIG \
    --output "$OUTPUT_DIR/webshop_baseline.jsonl" \
    --num_episodes $NUM_EPISODES

echo "=== Collection complete ==="
echo "Trajectories saved to: $OUTPUT_DIR/webshop_baseline.jsonl"

# Print stats
echo ""
echo "=== Dataset stats ==="
python -c "
import jsonlines
from collections import Counter

success = 0
total = 0
steps = []

with jsonlines.open('$OUTPUT_DIR/webshop_baseline.jsonl') as reader:
    for ep in reader:
        total += 1
        if ep.get('success', False):
            success += 1
        steps.append(ep.get('total_steps', 0))

print(f'Total episodes: {total}')
print(f'Success rate: {success/total:.2%}')
print(f'Avg steps: {sum(steps)/len(steps):.1f}')
"
```

**scripts/train_phase1.sh**
```bash
#!/bin/bash
# Phase 1: Train next-action prediction model (go/no-go validation)

set -e

# Configuration
DATA_PATH=${1:-"data/trajectories/webshop_baseline.jsonl"}
OUTPUT_DIR=${2:-"checkpoints/phase1"}
CONFIG="configs/training.yaml"

echo "=== Phase 1: Next-Action Prediction Training ==="
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"

# Check data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    echo "Run ./scripts/collect_data.sh first"
    exit 1
fi

# Run training
python -m training.train_next_action \
    --config $CONFIG \
    --data $DATA_PATH \
    --output $OUTPUT_DIR

echo "=== Training complete ==="
echo "Model saved to: $OUTPUT_DIR/best_model.pt"
```

---

### 6. Evaluation Scripts

**eval/metrics.py**
```python
"""Evaluation metrics for agent performance."""

from typing import Dict, List
from dataclasses import dataclass
import numpy as np


@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""
    success_rate: float
    avg_steps: float
    avg_reward: float
    avg_tokens: float
    num_episodes: int
    
    def to_dict(self) -> Dict:
        return {
            'success_rate': self.success_rate,
            'avg_steps': self.avg_steps,
            'avg_reward': self.avg_reward,
            'avg_tokens': self.avg_tokens,
            'num_episodes': self.num_episodes,
        }
    
    def __str__(self) -> str:
        return (
            f"Success Rate: {self.success_rate:.2%}\n"
            f"Avg Steps: {self.avg_steps:.1f}\n"
            f"Avg Reward: {self.avg_reward:.3f}\n"
            f"Avg Tokens: {self.avg_tokens:.0f}\n"
            f"Num Episodes: {self.num_episodes}"
        )


def compute_metrics(episodes: List[Dict]) -> EvalMetrics:
    """Compute metrics from a list of episode results."""
    
    successes = [ep.get('success', False) for ep in episodes]
    steps = [ep.get('total_steps', 0) for ep in episodes]
    rewards = [ep.get('total_reward', 0) for ep in episodes]
    tokens = [
        ep.get('token_usage', {}).get('prompt', 0) + 
        ep.get('token_usage', {}).get('completion', 0)
        for ep in episodes
    ]
    
    return EvalMetrics(
        success_rate=np.mean(successes),
        avg_steps=np.mean(steps),
        avg_reward=np.mean(rewards),
        avg_tokens=np.mean(tokens),
        num_episodes=len(episodes),
    )
```

---

## Week-by-Week Execution Plan

### Week 1: Go/No-Go Validation

**Days 1-2: Environment Setup**
```bash
# Clone repo, setup environment
git clone <your-repo>
cd ssm-agent
./scripts/setup_envs.sh

# Verify WebShop works
python -c "from data.collect_trajectories import WebShopEnvWrapper; env = WebShopEnvWrapper(); print('WebShop OK')"
```

**Days 3-4: Data Collection**
```bash
# Set API key
export OPENAI_API_KEY=sk-...

# Collect trajectories (start with 500, expand later)
./scripts/collect_data.sh 500

# Inspect data
python -c "
import jsonlines
with jsonlines.open('data/trajectories/webshop_baseline.jsonl') as r:
    ep = next(iter(r))
    print('Instruction:', ep['instruction'][:100])
    print('Events:', len(ep['events']))
    print('Success:', ep['success'])
"
```

**Days 5-7: Training + Go/No-Go**
```bash
# Train the model
./scripts/train_phase1.sh

# Expected output at end of training:
# ✓ GO: Model shows meaningful improvement over baseline!
# 
# If you see:
# ✗ NO-GO: Model does not sufficiently beat baseline.
# 
# Then debug:
# 1. Check data quality (are actions correctly classified?)
# 2. Check encoder (are embeddings reasonable?)
# 3. Try larger model / more data
```

**Week 1 Success Criteria:**
- [ ] WebShop running locally
- [ ] 500+ episodes collected
- [ ] Baseline success rate computed (~20-40% typical)
- [ ] Model accuracy > majority baseline + 5%
- [ ] Model accuracy > "last obs only" baseline

---

### Week 2-3: Retrieval Gating

After Phase 1 validates, add state-conditioned retrieval:

1. Implement observation chunking
2. Train retrieval head with contrastive loss
3. Integrate into agent loop
4. Compare: retrieval-by-state vs retrieval-by-query

---

### Week 4-5: Full Agent + Ablations

1. Implement all three agents (baseline_react, baseline_rag, recurrent_agent)
2. Run ablation matrix
3. Generate results tables

---

### Week 6: ALFWorld + Paper

1. Port to ALFWorld
2. Run same ablations
3. Write up results

---

## Cost Estimate

| Item | Cost |
|------|------|
| Data collection (1000 episodes × GPT-4o-mini) | ~$15 |
| Phase 1 training (T4, ~5 hours) | ~$5 or free (Colab) |
| Ablation runs (500 episodes × 5 conditions) | ~$40 |
| ALFWorld experiments | ~$20 |
| **Total** | **~$80** |
