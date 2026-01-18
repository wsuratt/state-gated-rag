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

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode raw text strings (for chunk embeddings).

        This returns the raw MiniLM embeddings without type embedding
        or projection, suitable for retrieval.

        Args:
            texts: List of text strings

        Returns:
            Tensor of shape (n, text_dim) where text_dim=384 for MiniLM
        """
        if len(texts) == 0:
            device = next(self.parameters()).device
            return torch.zeros(0, self.text_dim, device=device)

        device = next(self.parameters()).device

        with torch.no_grad():
            embeddings = self.text_encoder.encode(
                texts,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            )

        return embeddings

    def get_text_dim(self) -> int:
        """Return the raw text embedding dimension (384 for MiniLM)."""
        return self.text_dim
