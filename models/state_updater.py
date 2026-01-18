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
