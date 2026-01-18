"""Core data structures for episodes and events."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json
from datetime import datetime


@dataclass
class AgentEvent:
    """A single event in an agent's trajectory."""
    t: int                              # Step number
    event_type: str                     # OBS, ACT, REWARD, DONE, ERROR
    text: str                           # Observation or action text
    reward: Optional[float] = None      # Reward signal (if any)
    done: Optional[bool] = None         # Episode termination flag
    meta: Dict[str, Any] = field(default_factory=dict)  # Extra info
    timestamp: Optional[float] = None   # Wall clock time

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'AgentEvent':
        return cls(**d)


@dataclass
class Episode:
    """A complete episode (trajectory) from start to termination."""
    episode_id: str
    env: str                            # webshop, alfworld
    instruction: str                    # Task instruction
    events: List[AgentEvent]            # Sequence of events
    success: bool                       # Did the task succeed?
    total_steps: int                    # Number of steps taken
    total_reward: float = 0.0           # Cumulative reward
    token_usage: Dict[str, int] = field(default_factory=dict)  # API tokens
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> 'Episode':
        d = json.loads(s)
        d['events'] = [AgentEvent.from_dict(e) for e in d['events']]
        return cls(**d)

    @classmethod
    def from_dict(cls, d: dict) -> 'Episode':
        """Create Episode from a dict (e.g., from jsonlines)."""
        d = d.copy()  # Don't modify original
        d['events'] = [AgentEvent.from_dict(e) for e in d['events']]
        return cls(**d)

    def get_observations(self) -> List[str]:
        """Extract all observation texts."""
        return [e.text for e in self.events if e.event_type == 'OBS']

    def get_actions(self) -> List[str]:
        """Extract all action texts."""
        return [e.text for e in self.events if e.event_type == 'ACT']

    def get_obs_action_pairs(self) -> List[tuple]:
        """Get (observation, next_action) pairs for training."""
        pairs = []
        for i, event in enumerate(self.events):
            if event.event_type == 'OBS':
                # Find next action
                for j in range(i + 1, len(self.events)):
                    if self.events[j].event_type == 'ACT':
                        pairs.append((event.text, self.events[j].text))
                        break
        return pairs
