"""Base agent interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    max_steps: int = 50
    temperature: float = 0.0
    model: str = "gpt-4o-mini"
    top_k: int = 5  # For retrieval agents
    verbose: bool = False
    context_budget: int = 0  # Max context chars (0 = unlimited)


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.history: List[Dict[str, Any]] = []
        self.step_count = 0
        self.total_context_chars = 0  # Track total context sent to LLM
        self.total_prompt_tokens = 0  # Track actual prompt tokens from API
        self.total_completion_tokens = 0  # Track completion tokens from API

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for new episode."""
        pass

    @abstractmethod
    def get_action(self, instruction: str, observation: str) -> str:
        """
        Generate next action given current state.

        Args:
            instruction: Task instruction
            observation: Current observation from environment

        Returns:
            Action string to execute
        """
        pass

    def log_event(self, event_type: str, text: str, **kwargs) -> None:
        """Log an event to history."""
        self.history.append({
            'step': self.step_count,
            'event_type': event_type,
            'text': text,
            **kwargs
        })
