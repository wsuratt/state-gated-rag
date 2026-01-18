"""
Recurrent agent with state-gated retrieval.

This agent uses a learned recurrent state to condition retrieval
of relevant observation chunks, rather than using the raw query
or the entire observation history.
"""

import torch
from typing import List, Tuple, Optional, Dict, Any
from openai import OpenAI

from agents.base_agent import BaseAgent, AgentConfig
from models.encoder import EventEncoder
from models.state_updater import GRUStateUpdater
from models.retriever import StateConditionedRetriever
from data.chunk_observations import chunk_webshop_observation


class RecurrentStateGatedAgent(BaseAgent):
    """
    Agent that uses recurrent state for retrieval gating.

    Architecture:
    1. Encode events (obs/act pairs) into embeddings
    2. Update recurrent state via GRU
    3. Use state to retrieve relevant observation chunks
    4. Pass retrieved chunks to LLM actor
    """

    def __init__(
        self,
        config: AgentConfig,
        encoder: EventEncoder,
        state_updater: GRUStateUpdater,
        retriever: StateConditionedRetriever,
        device: torch.device = None,
    ):
        super().__init__(config)

        self.encoder = encoder
        self.state_updater = state_updater
        self.retriever = retriever
        self.device = device or torch.device('cpu')

        # Move models to device
        self.encoder.to(self.device)
        self.state_updater.to(self.device)
        self.retriever.to(self.device)

        # Set to eval mode
        self.encoder.eval()
        self.state_updater.eval()
        self.retriever.eval()

        # LLM client
        self.client = OpenAI()

        # Internal state
        self.events: List[Tuple[str, str]] = []  # (event_type, text)
        self.hidden_state: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.history = []
        self.events = []
        self.step_count = 0
        self.hidden_state = self.state_updater.init_state(batch_size=1)

    def get_action(self, instruction: str, observation: str) -> str:
        """
        Generate next action using state-gated retrieval.

        Steps:
        1. Log observation event
        2. Update recurrent state
        3. Chunk observation and embed
        4. Retrieve top-k chunks based on state
        5. Generate action from LLM using retrieved chunks
        """
        # Log observation
        self.events.append(('OBS', observation))
        self.log_event('OBS', observation)

        # Get current state
        state = self._compute_current_state()

        # Chunk and embed observation
        chunks = chunk_webshop_observation(observation)
        if len(chunks) == 0:
            chunks = [observation[:500]]  # Fallback

        chunk_embeddings = self.encoder.encode_texts(chunks)

        # Retrieve top-k chunks based on state
        with torch.no_grad():
            indices, scores = self.retriever(
                state,
                chunk_embeddings,
                top_k=self.config.top_k,
                return_scores=True
            )

        # Get retrieved chunk texts
        retrieved_chunks = [chunks[i] for i in indices.tolist()]

        # Generate action using LLM
        action = self._generate_action(instruction, retrieved_chunks)

        # Log action and update state
        self.events.append(('ACT', action))
        self.log_event('ACT', action, retrieved_chunks=retrieved_chunks)
        self.step_count += 1

        return action

    def _compute_current_state(self) -> torch.Tensor:
        """Compute current state from event history."""
        if len(self.events) == 0:
            return self.state_updater.get_state_vector(self.hidden_state).squeeze(0)

        # Encode all events
        with torch.no_grad():
            event_embs = self.encoder.encode_sequence(self.events)
            event_embs = event_embs.unsqueeze(0)  # (1, seq_len, d_event)

            # Get state from GRU
            outputs, self.hidden_state = self.state_updater(event_embs)

            # Return final state vector
            state = outputs[0, -1]  # (d_state,)

        return state

    def _generate_action(
        self,
        instruction: str,
        retrieved_chunks: List[str],
    ) -> str:
        """Generate action using LLM with retrieved chunks."""

        system_prompt = """You are an agent completing a shopping task. Given an instruction and relevant page information, output a single action.

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

        # Format retrieved chunks
        context = "\n".join(f"- {chunk}" for chunk in retrieved_chunks)

        user_content = f"""Instruction: {instruction}

Relevant page information:
{context}

What action should you take?"""

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=100,
            temperature=self.config.temperature
        )

        return response.choices[0].message.content.strip()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: AgentConfig = None,
        device: torch.device = None,
    ) -> 'RecurrentStateGatedAgent':
        """
        Load agent from a training checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            config: Agent configuration
            device: Device to load model to

        Returns:
            Initialized agent
        """
        import yaml

        if config is None:
            config = AgentConfig()

        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available()
                                  else 'cuda' if torch.cuda.is_available()
                                  else 'cpu')

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = checkpoint.get('config', {})

        # Initialize components
        encoder = EventEncoder(
            text_model=model_config.get('encoder', {}).get(
                'text_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            d_event=model_config.get('encoder', {}).get('d_event', 256),
        )

        state_updater = GRUStateUpdater(
            d_event=model_config.get('encoder', {}).get('d_event', 256),
            d_state=model_config.get('state_updater', {}).get('d_state', 512),
            num_layers=model_config.get('state_updater', {}).get('num_layers', 2),
        )

        retriever = StateConditionedRetriever(
            d_state=model_config.get('state_updater', {}).get('d_state', 512),
            d_chunk=model_config.get('retriever', {}).get('d_chunk', 384),
        )

        # Load state dict
        # The checkpoint has the full model, we need to extract relevant parts
        state_dict = checkpoint['model_state_dict']

        # Extract encoder weights
        encoder_state = {k.replace('encoder.', ''): v
                        for k, v in state_dict.items() if k.startswith('encoder.')}
        encoder.load_state_dict(encoder_state, strict=False)

        # Extract state updater weights
        updater_state = {k.replace('state_updater.', ''): v
                        for k, v in state_dict.items() if k.startswith('state_updater.')}
        state_updater.load_state_dict(updater_state, strict=False)

        # Retriever is untrained for zero-shot (Phase 2A)
        # Will be loaded after Phase 2B training

        return cls(
            config=config,
            encoder=encoder,
            state_updater=state_updater,
            retriever=retriever,
            device=device,
        )


class BaselineRollingWindowAgent(BaseAgent):
    """
    Baseline agent that uses a rolling window of recent history.

    This provides a comparison point - no learned state, just recent context.
    """

    def __init__(self, config: AgentConfig, window_size: int = 3):
        super().__init__(config)
        self.window_size = window_size
        self.client = OpenAI()

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.history = []
        self.step_count = 0

    def get_action(self, instruction: str, observation: str) -> str:
        """Generate action using rolling window of recent history."""

        # Log observation
        self.log_event('OBS', observation)

        # Build context from recent history
        recent = self.history[-self.window_size * 2:]  # obs+act pairs
        history_str = ""
        for event in recent:
            if event['event_type'] == 'ACT':
                history_str += f"Action: {event['text']}\n"

        system_prompt = """You are an agent completing a shopping task. Given an instruction and the current webpage, output a single action.

Valid actions:
- search[query] - search for products
- click[element] - click on a button/link (use exact text from the page)

Rules:
1. Output ONLY the action, nothing else
2. Use exact text from the page for click targets
3. When you find a matching product, click on it
4. Select all required options before adding to cart
5. Click "Buy Now" when ready to purchase"""

        user_content = f"Instruction: {instruction}\n\n"
        if history_str:
            user_content += f"Recent actions:\n{history_str}\n"
        user_content += f"Current page:\n{observation}"

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=100,
            temperature=self.config.temperature
        )

        action = response.choices[0].message.content.strip()

        # Log action
        self.log_event('ACT', action)
        self.step_count += 1

        return action


class BaselineQueryRAGAgent(BaseAgent):
    """
    Baseline RAG agent that retrieves based on the current query/instruction.

    Uses semantic similarity between instruction and chunks, without
    considering agent state or history.
    """

    def __init__(
        self,
        config: AgentConfig,
        encoder: EventEncoder,
        device: torch.device = None,
    ):
        super().__init__(config)
        self.encoder = encoder
        self.device = device or torch.device('cpu')
        self.encoder.to(self.device)
        self.encoder.eval()
        self.client = OpenAI()

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.history = []
        self.step_count = 0

    def get_action(self, instruction: str, observation: str) -> str:
        """Generate action using query-based retrieval."""

        # Log observation
        self.log_event('OBS', observation)

        # Chunk observation
        chunks = chunk_webshop_observation(observation)
        if len(chunks) == 0:
            chunks = [observation[:500]]

        # Embed chunks and query
        with torch.no_grad():
            chunk_embs = self.encoder.encode_texts(chunks)
            query_emb = self.encoder.encode_texts([instruction])

            # Normalize for cosine similarity
            chunk_embs = torch.nn.functional.normalize(chunk_embs, dim=-1)
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

            # Compute similarities
            similarities = torch.matmul(query_emb, chunk_embs.T).squeeze(0)

            # Get top-k
            top_k = min(self.config.top_k, len(chunks))
            _, indices = similarities.topk(top_k)

        # Get retrieved chunks
        retrieved_chunks = [chunks[i] for i in indices.tolist()]

        # Generate action
        action = self._generate_action(instruction, retrieved_chunks)

        # Log action
        self.log_event('ACT', action, retrieved_chunks=retrieved_chunks)
        self.step_count += 1

        return action

    def _generate_action(
        self,
        instruction: str,
        retrieved_chunks: List[str],
    ) -> str:
        """Generate action using LLM with retrieved chunks."""

        system_prompt = """You are an agent completing a shopping task. Given an instruction and relevant page information, output a single action.

Valid actions:
- search[query] - search for products
- click[element] - click on a button/link (use exact text from the page)

Rules:
1. Output ONLY the action, nothing else
2. Use exact text from the page for click targets
3. When you find a matching product, click on it
4. Select all required options before adding to cart
5. Click "Buy Now" when ready to purchase"""

        context = "\n".join(f"- {chunk}" for chunk in retrieved_chunks)

        user_content = f"""Instruction: {instruction}

Relevant page information:
{context}

What action should you take?"""

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=100,
            temperature=self.config.temperature
        )

        return response.choices[0].message.content.strip()
