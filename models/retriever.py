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
        d_chunk: int = 384,  # Match MiniLM embedding dim for simplicity
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
