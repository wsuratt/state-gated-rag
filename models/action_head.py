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
