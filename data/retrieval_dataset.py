"""
Retrieval dataset for training state-conditioned retriever.

Uses click-target heuristics: if the agent clicks on "XYZ", then chunks
containing "XYZ" are positive examples, others are negatives.
"""

import re
import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from data.chunk_observations import chunk_webshop_observation


@dataclass
class RetrievalSample:
    """A single retrieval training sample."""
    episode_id: str
    step: int
    events: List[Tuple[str, str]]  # Event history up to current obs
    chunks: List[str]              # All chunks from current observation
    pos_indices: List[int]         # Indices of positive chunks
    neg_indices: List[int]         # Indices of negative chunks
    action: str                    # The action taken (for debugging)
    click_target: str              # The extracted click target


class RetrievalDataset(Dataset):
    """
    Dataset for training the state-conditioned retriever.

    For each click action, we identify:
    - Positive chunks: contain the click target text
    - Negative chunks: don't contain the click target

    The retriever should learn to rank positives higher than negatives
    based on the current state.
    """

    def __init__(
        self,
        trajectories_path: str,
        max_episodes: Optional[int] = None,
        num_negatives: int = 8,
        min_chunks: int = 3,
        skip_search: bool = True,
    ):
        """
        Args:
            trajectories_path: Path to JSONL file with episodes
            max_episodes: Limit number of episodes (for debugging)
            num_negatives: Number of negative chunks to sample per positive
            min_chunks: Minimum chunks required to create a sample
            skip_search: Skip search actions (they don't have clear targets)
        """
        self.num_negatives = num_negatives
        self.min_chunks = min_chunks
        self.samples: List[RetrievalSample] = []

        # Load and process episodes
        with open(trajectories_path) as f:
            episodes = []
            for i, line in enumerate(f):
                if max_episodes and i >= max_episodes:
                    break
                episodes.append(json.loads(line))

        print(f"Processing {len(episodes)} episodes for retrieval training...")

        for ep in tqdm(episodes, desc="Building retrieval samples"):
            self._process_episode(ep, skip_search)

        print(f"Created {len(self.samples)} retrieval training samples")

        # Stats
        pos_counts = [len(s.pos_indices) for s in self.samples]
        neg_counts = [len(s.neg_indices) for s in self.samples]
        print(f"Avg positives per sample: {sum(pos_counts)/len(pos_counts):.1f}")
        print(f"Avg negatives per sample: {sum(neg_counts)/len(neg_counts):.1f}")

    def _process_episode(self, episode: Dict, skip_search: bool):
        """Extract retrieval samples from an episode."""
        events = []

        for i, event in enumerate(episode['events']):
            if event['event_type'] == 'OBS':
                obs = event['text']

                # Find next action
                next_action = None
                for j in range(i + 1, len(episode['events'])):
                    if episode['events'][j]['event_type'] == 'ACT':
                        next_action = episode['events'][j]['text']
                        break

                if next_action is None:
                    events.append(('OBS', obs))
                    continue

                # Skip search actions if requested
                if skip_search and next_action.lower().startswith('search['):
                    events.append(('OBS', obs))
                    continue

                # Only process click actions
                if not next_action.lower().startswith('click['):
                    events.append(('OBS', obs))
                    continue

                # Extract click target
                match = re.match(r'click\[(.+)\]', next_action, re.IGNORECASE)
                if not match:
                    events.append(('OBS', obs))
                    continue

                click_target = match.group(1).lower().strip()

                # Chunk the observation
                chunks = chunk_webshop_observation(obs)

                if len(chunks) < self.min_chunks:
                    events.append(('OBS', obs))
                    continue

                # Find positive and negative chunks
                pos_indices = []
                neg_indices = []

                for idx, chunk in enumerate(chunks):
                    chunk_lower = chunk.lower()
                    if click_target in chunk_lower:
                        pos_indices.append(idx)
                    else:
                        neg_indices.append(idx)

                # Skip if no positives found
                if len(pos_indices) == 0:
                    events.append(('OBS', obs))
                    continue

                # Skip if no negatives
                if len(neg_indices) == 0:
                    events.append(('OBS', obs))
                    continue

                # Create sample
                sample = RetrievalSample(
                    episode_id=episode['episode_id'],
                    step=event['t'],
                    events=list(events),  # Copy current event history
                    chunks=chunks,
                    pos_indices=pos_indices,
                    neg_indices=neg_indices,
                    action=next_action,
                    click_target=click_target,
                )
                self.samples.append(sample)

                # Add observation to history
                events.append(('OBS', obs))

            elif event['event_type'] == 'ACT':
                events.append(('ACT', event['text']))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training sample.

        Returns:
            dict with:
            - events: List of (event_type, text) tuples
            - chunks: List of chunk texts
            - pos_idx: Index of positive chunk (randomly sampled if multiple)
            - neg_indices: List of negative chunk indices
        """
        sample = self.samples[idx]

        # Randomly sample one positive if multiple
        pos_idx = random.choice(sample.pos_indices)

        # Sample negatives (up to num_negatives)
        neg_indices = random.sample(
            sample.neg_indices,
            min(self.num_negatives, len(sample.neg_indices))
        )

        return {
            'events': sample.events,
            'chunks': sample.chunks,
            'pos_idx': pos_idx,
            'neg_indices': neg_indices,
            'episode_id': sample.episode_id,
            'click_target': sample.click_target,
        }


def collate_retrieval_batch(batch: List[Dict]) -> Dict:
    """
    Custom collate function for retrieval batches.

    Since sequences have variable length, we don't pad them.
    The training loop processes each sample individually for state computation.
    """
    return {
        'events': [s['events'] for s in batch],
        'chunks': [s['chunks'] for s in batch],
        'pos_indices': [s['pos_idx'] for s in batch],
        'neg_indices': [s['neg_indices'] for s in batch],
        'episode_ids': [s['episode_id'] for s in batch],
        'click_targets': [s['click_target'] for s in batch],
    }


def create_retrieval_dataloader(
    trajectories_path: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    max_episodes: Optional[int] = None,
    num_negatives: int = 8,
) -> DataLoader:
    """Create a DataLoader for retrieval training."""

    dataset = RetrievalDataset(
        trajectories_path=trajectories_path,
        max_episodes=max_episodes,
        num_negatives=num_negatives,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_retrieval_batch,
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test dataset creation
    dataset = RetrievalDataset(
        trajectories_path="/Users/williamsuratt/Desktop/state-gated-rag/data/trajectories/webshop_train.jsonl",
        max_episodes=100,
        num_negatives=8,
    )

    print(f"\n=== Sample Examples ===")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Events: {len(sample['events'])} events")
        print(f"  Chunks: {len(sample['chunks'])} chunks")
        print(f"  Click target: '{sample['click_target']}'")
        print(f"  Positive chunk [{sample['pos_idx']}]: {sample['chunks'][sample['pos_idx']][:60]}...")
        print(f"  Negative chunks: {len(sample['neg_indices'])} chunks")
