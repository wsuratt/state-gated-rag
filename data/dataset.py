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
                episode = Episode.from_dict(line)
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
