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
