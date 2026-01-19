"""
Phase 3 A1: Live WebShop Evaluation

Runs all 5 agent types on live WebShop environment and measures:
- Task success rate (reward = 1.0)
- Average reward (partial credit)
- Steps to completion (mean and median)
- Token efficiency (approx tokens sent to LLM per episode)
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add WebShop to path
WEBSHOP_PATH = os.path.join(os.path.dirname(__file__), '..', 'external', 'WebShop')
sys.path.insert(0, WEBSHOP_PATH)

# Set Java paths for pyserini
os.environ.setdefault('JAVA_HOME', '/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home')
os.environ.setdefault('JVM_PATH', '/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib')

import torch
from web_agent_site.envs import WebAgentTextEnv

from agents.base_agent import AgentConfig
from agents.recurrent_agent import (
    RecurrentStateGatedAgent,
    BaselineRollingWindowAgent,
    BaselineQueryRAGAgent,
    FullContextAgent,
    StateGatedCompressionAgent,
)
from models.encoder import EventEncoder


@dataclass
class EpisodeResult:
    """Results from a single episode."""
    episode_id: int
    agent_type: str
    seed: int
    reward: float
    success: bool  # reward == 1.0
    steps: int
    tokens_approx: int  # Approximate tokens sent to LLM
    context_chars: int  # Total context characters sent (observation portion)
    prompt_tokens: int  # Actual prompt tokens from API
    completion_tokens: int  # Actual completion tokens from API
    instruction: str
    final_action: str


@dataclass
class AgentResults:
    """Aggregated results for an agent type."""
    agent_type: str
    num_episodes: int
    success_rate: float
    success_rate_std: float
    avg_reward: float
    reward_std: float
    mean_steps: float
    median_steps: float
    steps_std: float
    mean_tokens: float
    tokens_std: float
    mean_context_chars: float
    context_chars_std: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    episodes: List[EpisodeResult]


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (1 token ≈ 4 chars)."""
    return len(text) // 4


def run_episode(
    env: WebAgentTextEnv,
    agent: Any,
    instruction: str,
    max_steps: int = 15,
    verbose: bool = False,
) -> EpisodeResult:
    """Run a single episode and return results."""

    agent.reset()
    total_tokens = 0

    obs = env.observation
    done = False
    step = 0
    final_action = ""

    while not done and step < max_steps:
        # Get action from agent
        action = agent.get_action(instruction, obs)
        final_action = action

        # Estimate tokens (instruction + observation context)
        # This is approximate - actual depends on agent type
        total_tokens += estimate_tokens(instruction) + estimate_tokens(obs[:2000])

        if verbose:
            print(f"  Step {step}: {action[:50]}...")

        # Take action in environment
        obs, reward, done, info = env.step(action)
        step += 1

    # Get tracking stats from agent
    context_chars = getattr(agent, 'total_context_chars', 0)
    prompt_tokens = getattr(agent, 'total_prompt_tokens', 0)
    completion_tokens = getattr(agent, 'total_completion_tokens', 0)

    return EpisodeResult(
        episode_id=0,  # Set by caller
        agent_type=agent.__class__.__name__,
        seed=0,  # Set by caller
        reward=reward,
        success=(reward == 1.0),
        steps=step,
        tokens_approx=total_tokens,
        context_chars=context_chars,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        instruction=instruction,
        final_action=final_action,
    )


def create_agent(
    agent_type: str,
    config: AgentConfig,
    checkpoint_path: Optional[str] = None,
    device: torch.device = None,
) -> Any:
    """Create an agent of the specified type."""

    if agent_type == "state_gated_trained":
        if checkpoint_path is None:
            checkpoint_path = "checkpoints/phase2/best_model.pt"
        return RecurrentStateGatedAgent.from_checkpoint(
            checkpoint_path, config=config, device=device, load_retriever=True
        )

    elif agent_type == "state_gated_zero_shot":
        if checkpoint_path is None:
            checkpoint_path = "checkpoints/phase1/best_model.pt"
        return RecurrentStateGatedAgent.from_checkpoint(
            checkpoint_path, config=config, device=device, load_retriever=False
        )

    elif agent_type == "query_rag":
        encoder = EventEncoder()
        return BaselineQueryRAGAgent(config=config, encoder=encoder, device=device)

    elif agent_type == "rolling_window":
        return BaselineRollingWindowAgent(config=config, window_size=3)

    elif agent_type == "full_context":
        return FullContextAgent(config=config, max_obs_chars=4000)

    elif agent_type == "state_gated_compression":
        if checkpoint_path is None:
            checkpoint_path = "checkpoints/phase2/best_model.pt"
        return StateGatedCompressionAgent.from_checkpoint(
            checkpoint_path, config=config, device=device,
            total_budget=2000, temperature=1.0
        )

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def evaluate_agent(
    agent_type: str,
    num_episodes: int,
    seeds: List[int],
    config: AgentConfig,
    checkpoint_path: Optional[str] = None,
    verbose: bool = False,
) -> AgentResults:
    """Evaluate an agent on multiple episodes."""

    print(f"\n{'='*60}")
    print(f"Evaluating: {agent_type}")
    print(f"{'='*60}")

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')

    # Create agent
    agent = create_agent(agent_type, config, checkpoint_path, device)

    # Create environment
    env = WebAgentTextEnv(observation_mode='text', num_products=1000)

    episodes = []

    for i, seed in enumerate(tqdm(seeds[:num_episodes], desc=f"{agent_type}")):
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Reset environment with session (used as seed for goal selection)
        env.reset(session=seed)
        instruction = env.instruction_text

        if verbose:
            print(f"\nEpisode {i+1}/{num_episodes} (seed={seed})")
            print(f"  Instruction: {instruction[:80]}...")

        # Run episode
        result = run_episode(
            env, agent, instruction,
            max_steps=config.max_steps,
            verbose=verbose
        )
        result.episode_id = i
        result.seed = seed

        episodes.append(result)

        if verbose:
            print(f"  Reward: {result.reward:.2f}, Steps: {result.steps}")

    # Compute aggregated statistics
    rewards = [e.reward for e in episodes]
    successes = [e.success for e in episodes]
    steps = [e.steps for e in episodes]
    tokens = [e.tokens_approx for e in episodes]
    context_chars = [e.context_chars for e in episodes]
    prompt_tokens = [e.prompt_tokens for e in episodes]
    completion_tokens = [e.completion_tokens for e in episodes]

    return AgentResults(
        agent_type=agent_type,
        num_episodes=len(episodes),
        success_rate=np.mean(successes),
        success_rate_std=np.std(successes) / np.sqrt(len(successes)),  # Standard error
        avg_reward=np.mean(rewards),
        reward_std=np.std(rewards) / np.sqrt(len(rewards)),
        mean_steps=np.mean(steps),
        median_steps=np.median(steps),
        steps_std=np.std(steps),
        mean_tokens=np.mean(tokens),
        tokens_std=np.std(tokens),
        mean_context_chars=np.mean(context_chars),
        context_chars_std=np.std(context_chars),
        mean_prompt_tokens=np.mean(prompt_tokens),
        mean_completion_tokens=np.mean(completion_tokens),
        episodes=episodes,
    )


def print_results_table(results: List[AgentResults]):
    """Print a formatted results table."""

    print("\n" + "="*120)
    print("LIVE WEBSHOP EVALUATION RESULTS")
    print("="*120)

    # Header
    print(f"{'Agent':<25} {'Success':<16} {'Reward':<12} {'Steps':<8} {'Prompt Tok':<12} {'Compl Tok':<10}")
    print("-"*120)

    # Sort by success rate
    results_sorted = sorted(results, key=lambda x: x.success_rate, reverse=True)

    for r in results_sorted:
        success_str = f"{r.success_rate*100:.1f}% ± {r.success_rate_std*100:.1f}%"
        reward_str = f"{r.avg_reward:.3f}"
        steps_str = f"{r.mean_steps:.1f}"
        prompt_str = f"{r.mean_prompt_tokens:.0f}"
        compl_str = f"{r.mean_completion_tokens:.0f}"

        print(f"{r.agent_type:<25} {success_str:<16} {reward_str:<12} {steps_str:<8} {prompt_str:<12} {compl_str:<10}")

    print("="*120)

    # Key comparisons
    trained = next((r for r in results if r.agent_type == "state_gated_trained"), None)
    query_rag = next((r for r in results if r.agent_type == "query_rag"), None)

    if trained and query_rag:
        diff = (trained.success_rate - query_rag.success_rate) * 100
        print(f"\nState-Gated (trained) vs Query-RAG: {'+' if diff > 0 else ''}{diff:.1f}% success rate")


def main():
    parser = argparse.ArgumentParser(description="Live WebShop evaluation")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes per agent")
    parser.add_argument("--agents", type=str, nargs="+",
                        default=["state_gated_trained", "state_gated_zero_shot",
                                 "query_rag", "rolling_window", "full_context",
                                 "state_gated_compression"],
                        help="Agent types to evaluate")
    parser.add_argument("--max_steps", type=int, default=15,
                        help="Max steps per episode")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM temperature")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-k for retrieval agents")
    parser.add_argument("--output", type=str, default="results/live_eval",
                        help="Output directory for results")
    parser.add_argument("--seed_start", type=int, default=42,
                        help="Starting seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    parser.add_argument("--context_budget", type=int, default=0,
                        help="Max context chars for observation (0 = unlimited)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Generate seeds (same for all agents)
    seeds = list(range(args.seed_start, args.seed_start + args.num_episodes))

    # Agent config
    config = AgentConfig(
        max_steps=args.max_steps,
        temperature=args.temperature,
        model=args.model,
        top_k=args.top_k,
        verbose=args.verbose,
        context_budget=args.context_budget,
    )

    print(f"Running live WebShop evaluation")
    print(f"  Episodes per agent: {args.num_episodes}")
    print(f"  Agents: {args.agents}")
    if args.context_budget > 0:
        print(f"  Context budget: {args.context_budget} chars")
    print(f"  Model: {args.model}")
    print(f"  Seeds: {seeds[0]} to {seeds[-1]}")

    # Evaluate each agent
    all_results = []

    for agent_type in args.agents:
        try:
            result = evaluate_agent(
                agent_type=agent_type,
                num_episodes=args.num_episodes,
                seeds=seeds,
                config=config,
                verbose=args.verbose,
            )
            all_results.append(result)

            # Save intermediate results
            result_dict = asdict(result)
            result_dict['episodes'] = [asdict(e) for e in result.episodes]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(args.output, f"{agent_type}_{timestamp}.json")
            with open(output_file, 'w') as f:
                json.dump(result_dict, f, indent=2)

        except Exception as e:
            print(f"Error evaluating {agent_type}: {e}")
            import traceback
            traceback.print_exc()

    # Print final results table
    print_results_table(all_results)

    # Save combined results
    combined = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_episodes': args.num_episodes,
            'max_steps': args.max_steps,
            'model': args.model,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'seed_start': args.seed_start,
        },
        'results': [
            {k: v for k, v in asdict(r).items() if k != 'episodes'}
            for r in all_results
        ]
    }

    combined_file = os.path.join(args.output, f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(combined_file, 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
