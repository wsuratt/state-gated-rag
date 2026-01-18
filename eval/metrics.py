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
