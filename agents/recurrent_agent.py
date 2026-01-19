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
from agents.compression.allocation import allocate_budget, build_compressed_context, LOW_BUDGET_THRESHOLD
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
- search[query] - search for products (use simple keywords)
- click[element] - click on a button, product ID, or option

Page types:
1. SEARCH PAGE: Use search[keywords] with 2-4 simple keywords
2. RESULTS PAGE: Click on the product ID (e.g., b00zdedvbi) to view it
3. PRODUCT PAGE: Click options (color/size in lowercase) then click[buy now]

Rules:
1. Output ONLY the action, nothing else
2. On results: click the product ID (starts with 'b0'), NOT the name
3. On product page: click option values in lowercase, then click[buy now]
4. All click values must be LOWERCASE
5. NEVER repeat an action you already took - check your recent actions!

Examples:
search[women hoodies]
click[b00zdedvbi]
click[blue]
click[large]
click[buy now]"""

        # Format retrieved chunks
        context = "\n".join(f"- {chunk}" for chunk in retrieved_chunks)

        # Build recent action history
        recent_actions = []
        for event in self.events[-6:]:  # Last 3 action/obs pairs
            if event[0] == 'ACT':
                recent_actions.append(event[1])

        action_history = ""
        if recent_actions:
            action_history = f"\nYour recent actions: {', '.join(recent_actions)}\n"

        user_content = f"""Instruction: {instruction}
{action_history}
Relevant page information:
{context}

What action should you take? Remember: click product IDs (like b00xxxxx) on results pages. Don't repeat actions you already took."""

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
        load_retriever: bool = False,
    ) -> 'RecurrentStateGatedAgent':
        """
        Load agent from a training checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            config: Agent configuration
            device: Device to load model to
            load_retriever: If True, load retriever weights (for Phase 2 trained model)

        Returns:
            Initialized agent
        """
        if config is None:
            config = AgentConfig()

        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available()
                                  else 'cuda' if torch.cuda.is_available()
                                  else 'cpu')

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

        # Handle different checkpoint formats (Phase 1 vs Phase 2)
        if 'model_state_dict' in checkpoint:
            # Phase 1 format: single combined state dict
            state_dict = checkpoint['model_state_dict']

            # Extract encoder weights
            encoder_state = {k.replace('encoder.', ''): v
                            for k, v in state_dict.items() if k.startswith('encoder.')}
            encoder.load_state_dict(encoder_state, strict=False)

            # Extract state updater weights
            updater_state = {k.replace('state_updater.', ''): v
                            for k, v in state_dict.items() if k.startswith('state_updater.')}
            state_updater.load_state_dict(updater_state, strict=False)

            # Optionally load retriever weights
            if load_retriever:
                retriever_state = {k.replace('retriever.', ''): v
                                  for k, v in state_dict.items() if k.startswith('retriever.')}
                if retriever_state:
                    retriever.load_state_dict(retriever_state, strict=False)
        else:
            # Phase 2 format: separate state dicts for each component
            if 'encoder_state_dict' in checkpoint:
                encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
            if 'state_updater_state_dict' in checkpoint:
                state_updater.load_state_dict(checkpoint['state_updater_state_dict'], strict=False)
            if load_retriever and 'retriever_state_dict' in checkpoint:
                retriever.load_state_dict(checkpoint['retriever_state_dict'], strict=False)

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

        system_prompt = """You are a shopping agent. Output ONE action per turn.

Actions: search[keywords] or click[element]

Flow:
1. SEARCH PAGE (shows "Search"): search[2-4 keywords]
2. RESULTS PAGE (shows product IDs like B09XXX): click[b09xxx] (lowercase ID)
3. PRODUCT PAGE (shows options): click each required option, then click[buy now]

CRITICAL: On product page, if you already clicked an option, move to the NEXT required option or click[buy now]. Don't repeat the same option.

All click values MUST be lowercase.

Examples: search[blue hoodie] / click[b00zdedvbi] / click[blue] / click[large] / click[buy now]"""

        user_content = f"Instruction: {instruction}\n\n"
        if history_str:
            user_content += f"Recent actions:\n{history_str}\n"
        user_content += f"Current page:\n{observation}\n\nAction:"

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
        self.recent_actions: List[str] = []  # Track actions for context

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.history = []
        self.step_count = 0
        self.recent_actions = []

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

        # Log action and track for context
        self.log_event('ACT', action, retrieved_chunks=retrieved_chunks)
        self.recent_actions.append(action)
        self.step_count += 1

        return action

    def _generate_action(
        self,
        instruction: str,
        retrieved_chunks: List[str],
    ) -> str:
        """Generate action using LLM with retrieved chunks."""

        system_prompt = """You are a shopping agent. Output ONE action per turn.

Actions: search[keywords] or click[element]

Flow:
1. SEARCH PAGE (shows "Search"): search[2-4 keywords]
2. RESULTS PAGE (shows product IDs like B09XXX): click[b09xxx] (lowercase ID)
3. PRODUCT PAGE (shows options): click each required option, then click[buy now]

Rules:
- All click values MUST be lowercase
- NEVER repeat an action you already took - check your recent actions!
- After clicking all required options, click[buy now]

Examples: search[blue hoodie] / click[b00zdedvbi] / click[blue] / click[large] / click[buy now]"""

        context = "\n".join(f"- {chunk}" for chunk in retrieved_chunks)

        # Build action history
        action_history = ""
        if self.recent_actions:
            action_history = f"\nYour recent actions: {', '.join(self.recent_actions[-5:])}\n"

        user_content = f"""Instruction: {instruction}
{action_history}
Page info:
{context}

What action should you take next? Don't repeat previous actions."""

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


class FullContextAgent(BaseAgent):
    """
    Upper bound baseline that passes the full observation to the LLM.

    No retrieval - just gives the LLM everything. This establishes
    an upper bound on performance (limited by context window / cost).
    """

    def __init__(self, config: AgentConfig, max_obs_chars: int = 4000):
        super().__init__(config)
        self.max_obs_chars = max_obs_chars
        self.client = OpenAI()
        self.recent_actions: List[str] = []

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.history = []
        self.step_count = 0
        self.recent_actions = []
        self.total_context_chars = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def get_action(self, instruction: str, observation: str) -> str:
        """Generate action using full observation context."""

        # Log observation
        self.log_event('OBS', observation)

        # Use context_budget if set, otherwise max_obs_chars
        budget = self.config.context_budget if self.config.context_budget > 0 else self.max_obs_chars
        obs_text = observation[:budget]

        # Track context usage
        self.total_context_chars += len(obs_text)

        system_prompt = """You are a shopping agent. Output ONE action per turn.

Actions: search[keywords] or click[element]

Flow:
1. SEARCH PAGE (shows "Search"): search[2-4 keywords]
2. RESULTS PAGE (shows product IDs like B09XXX): click[b09xxx] (lowercase ID)
3. PRODUCT PAGE (shows options): click each required option, then click[buy now]

Rules:
- All click values MUST be lowercase
- NEVER repeat an action you already took - check your recent actions!
- After clicking all required options, click[buy now]

Examples: search[blue hoodie] / click[b00zdedvbi] / click[blue] / click[large] / click[buy now]"""

        # Build action history
        action_history = ""
        if self.recent_actions:
            action_history = f"\nYour recent actions: {', '.join(self.recent_actions[-5:])}\n"

        user_content = f"Instruction: {instruction}\n{action_history}\nCurrent page:\n{obs_text}\n\nWhat action should you take next? Don't repeat previous actions."

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=100,
            temperature=self.config.temperature
        )

        # Track token usage from API
        if hasattr(response, 'usage') and response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

        action = response.choices[0].message.content.strip()

        # Log action and track
        self.log_event('ACT', action)
        self.recent_actions.append(action)
        self.step_count += 1

        return action


class StateGatedCompressionAgent(BaseAgent):
    """
    Agent that uses state-gated compression instead of retrieval.

    Key insight: Instead of selecting which chunks to KEEP (losing context),
    we select how much to COMPRESS each chunk (preserving structure).

    Architecture:
    1. Encode events (obs/act pairs) into embeddings
    2. Update recurrent state via GRU
    3. Use state to score chunks by importance
    4. Allocate token budget proportionally via softmax
    5. Compress each chunk to its budget
    6. Pass full compressed observation to LLM
    """

    def __init__(
        self,
        config: AgentConfig,
        encoder: EventEncoder,
        state_updater: GRUStateUpdater,
        retriever: StateConditionedRetriever,
        device: torch.device = None,
        total_budget: int = 2000,  # Total characters for compressed observation
        temperature: float = 1.0,  # Softmax temperature for budget allocation
        min_chunk_budget: int = 50,  # Minimum chars per chunk
    ):
        super().__init__(config)

        self.encoder = encoder
        self.state_updater = state_updater
        self.retriever = retriever
        self.device = device or torch.device('cpu')

        # Compression parameters
        self.total_budget = total_budget
        self.softmax_temperature = temperature
        self.min_chunk_budget = min_chunk_budget

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
        self.events: List[Tuple[str, str]] = []
        self.hidden_state: Optional[torch.Tensor] = None
        self.recent_actions: List[str] = []

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.history = []
        self.events = []
        self.step_count = 0
        self.hidden_state = self.state_updater.init_state(batch_size=1)
        self.recent_actions = []
        self.total_context_chars = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def get_effective_budget(self) -> int:
        """Get the effective compression budget (config overrides default)."""
        if self.config.context_budget > 0:
            return self.config.context_budget
        return self.total_budget

    def get_action(self, instruction: str, observation: str) -> str:
        """
        Generate next action using state-gated compression.

        Steps:
        1. Log observation event
        2. Update recurrent state
        3. Chunk observation and score with state
        4. Allocate budget proportionally
        5. Compress chunks and concatenate
        6. Generate action from LLM
        """
        # Log observation
        self.events.append(('OBS', observation))
        self.log_event('OBS', observation)

        # Get current state
        state = self._compute_current_state()

        # Chunk observation
        chunks = chunk_webshop_observation(observation)
        if len(chunks) == 0:
            chunks = [observation[:500]]

        # Compress observation using state-guided budget allocation
        compressed_obs = self._compress_observation(state, chunks, observation)

        # Track context usage
        self.total_context_chars += len(compressed_obs)

        # Generate action using LLM
        action = self._generate_action(instruction, compressed_obs)

        # Log action and update state
        self.events.append(('ACT', action))
        self.log_event('ACT', action)
        self.recent_actions.append(action)
        self.step_count += 1

        return action

    def _compute_current_state(self) -> torch.Tensor:
        """Compute current state from event history."""
        if len(self.events) == 0:
            return self.state_updater.get_state_vector(self.hidden_state).squeeze(0)

        with torch.no_grad():
            event_embs = self.encoder.encode_sequence(self.events)
            event_embs = event_embs.unsqueeze(0)

            outputs, self.hidden_state = self.state_updater(event_embs)
            state = outputs[0, -1]

        return state

    def _compress_observation(
        self,
        state: torch.Tensor,
        chunks: List[str],
        raw_observation: str = None,
    ) -> str:
        """
        Compress chunks using state-guided budget allocation.

        Uses the allocation module which switches between:
        - Proportional allocation at high budgets
        - Front buffer + anchors + top-k at low budgets

        Args:
            state: Current agent state vector
            chunks: List of observation chunks
            raw_observation: Original observation (for front buffer)

        Returns:
            Compressed observation string
        """
        effective_budget = self.get_effective_budget()

        # At low-to-medium budgets, use simple truncation
        # WebShop is front-loaded, so truncation is efficient here
        # Compression adds chunking overhead that only pays off at very high budgets
        TRUNCATION_THRESHOLD = 1500
        if effective_budget <= TRUNCATION_THRESHOLD and raw_observation:
            return raw_observation[:effective_budget]

        if len(chunks) == 1:
            # Single chunk - just truncate to budget
            return chunks[0][:effective_budget]

        # Embed chunks and get importance scores
        with torch.no_grad():
            chunk_embeddings = self.encoder.encode_texts(chunks)
            scores = self.retriever.score_chunks(state, chunk_embeddings)

        # Use the allocation module for smart budget distribution
        budgets = allocate_budget(
            scores=scores,
            chunks=chunks,
            total_budget=effective_budget,
            mode="auto",  # Auto-switches based on budget
            temperature=self.softmax_temperature,
            min_chunk_budget=self.min_chunk_budget,
        )

        # Standard compression
        compressed_parts = []
        for chunk, budget in zip(chunks, budgets):
            compressed = self._compress_chunk(chunk, budget)
            if compressed:
                compressed_parts.append(compressed)
        return "\n".join(compressed_parts)

    def _compress_chunk(self, chunk: str, budget: int) -> str:
        """
        Compress a single chunk to fit within budget.

        Uses extractive compression (keep first N chars).
        Could be extended to use LLM-based summarization.
        """
        if len(chunk) <= budget:
            return chunk

        # Extractive: keep first N chars, try to end at word boundary
        compressed = chunk[:budget]

        # Try to end at a word boundary
        last_space = compressed.rfind(' ')
        if last_space > budget * 0.7:  # Don't cut too much
            compressed = compressed[:last_space]

        return compressed.strip()

    def _generate_action(
        self,
        instruction: str,
        compressed_obs: str,
    ) -> str:
        """Generate action using LLM with compressed observation."""

        system_prompt = """You are a shopping agent. Output ONE action per turn.

Actions: search[keywords] or click[element]

Flow:
1. SEARCH PAGE (shows "Search"): search[2-4 keywords]
2. RESULTS PAGE (shows product IDs like B09XXX): click[b09xxx] (lowercase ID)
3. PRODUCT PAGE (shows options): click each required option, then click[buy now]

Rules:
- All click values MUST be lowercase
- NEVER repeat an action you already took - check your recent actions!
- After clicking all required options, click[buy now]

Examples: search[blue hoodie] / click[b00zdedvbi] / click[blue] / click[large] / click[buy now]"""

        # Build action history
        action_history = ""
        if self.recent_actions:
            action_history = f"\nYour recent actions: {', '.join(self.recent_actions[-5:])}\n"

        user_content = f"Instruction: {instruction}\n{action_history}\nCurrent page:\n{compressed_obs}\n\nWhat action should you take next? Don't repeat previous actions."

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=100,
            temperature=self.config.temperature
        )

        # Track token usage from API
        if hasattr(response, 'usage') and response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

        return response.choices[0].message.content.strip()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: AgentConfig = None,
        device: torch.device = None,
        total_budget: int = 2000,
        temperature: float = 1.0,
        min_chunk_budget: int = 50,
    ) -> 'StateGatedCompressionAgent':
        """Load agent from a training checkpoint."""
        if config is None:
            config = AgentConfig()

        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available()
                                  else 'cuda' if torch.cuda.is_available()
                                  else 'cpu')

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

        # Load state dicts (handle both Phase 1 and Phase 2 formats)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            encoder_state = {k.replace('encoder.', ''): v
                            for k, v in state_dict.items() if k.startswith('encoder.')}
            encoder.load_state_dict(encoder_state, strict=False)

            updater_state = {k.replace('state_updater.', ''): v
                            for k, v in state_dict.items() if k.startswith('state_updater.')}
            state_updater.load_state_dict(updater_state, strict=False)

            retriever_state = {k.replace('retriever.', ''): v
                              for k, v in state_dict.items() if k.startswith('retriever.')}
            if retriever_state:
                retriever.load_state_dict(retriever_state, strict=False)
        else:
            if 'encoder_state_dict' in checkpoint:
                encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
            if 'state_updater_state_dict' in checkpoint:
                state_updater.load_state_dict(checkpoint['state_updater_state_dict'], strict=False)
            if 'retriever_state_dict' in checkpoint:
                retriever.load_state_dict(checkpoint['retriever_state_dict'], strict=False)

        return cls(
            config=config,
            encoder=encoder,
            state_updater=state_updater,
            retriever=retriever,
            device=device,
            total_budget=total_budget,
            temperature=temperature,
            min_chunk_budget=min_chunk_budget,
        )
