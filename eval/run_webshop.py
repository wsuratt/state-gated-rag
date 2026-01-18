"""
Evaluate agents on WebShop environment.

Usage:
    python -m eval.run_webshop --agent recurrent --checkpoint checkpoints/phase1/best_model.pt --num_episodes 100
"""

import os
import sys
import argparse

# Add WebShop to path
WEBSHOP_PATH = os.path.join(os.path.dirname(__file__), '..', 'external', 'WebShop')
if os.path.exists(WEBSHOP_PATH):
    sys.path.insert(0, WEBSHOP_PATH)
import json
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm
import torch

from agents.base_agent import AgentConfig
from agents.recurrent_agent import (
    RecurrentStateGatedAgent,
    BaselineRollingWindowAgent,
    BaselineQueryRAGAgent,
)
from models.encoder import EventEncoder
from eval.metrics import compute_metrics, EvalMetrics


def create_agent(
    agent_type: str,
    checkpoint_path: Optional[str] = None,
    config: AgentConfig = None,
    device: torch.device = None,
):
    """
    Create an agent based on type.

    Args:
        agent_type: One of 'recurrent', 'rolling', 'query_rag'
        checkpoint_path: Path to model checkpoint (for recurrent agent)
        config: Agent configuration
        device: Device to use

    Returns:
        Initialized agent
    """
    if config is None:
        config = AgentConfig()

    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available()
                              else 'cuda' if torch.cuda.is_available()
                              else 'cpu')

    if agent_type == 'recurrent':
        if checkpoint_path is None:
            raise ValueError("checkpoint_path required for recurrent agent")
        return RecurrentStateGatedAgent.from_checkpoint(checkpoint_path, config, device)

    elif agent_type == 'rolling':
        return BaselineRollingWindowAgent(config)

    elif agent_type == 'query_rag':
        encoder = EventEncoder().to(device)
        return BaselineQueryRAGAgent(config, encoder, device)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_episode(
    env,
    agent,
    instruction: str,
    max_steps: int = 50,
    verbose: bool = False,
) -> Dict:
    """
    Run a single episode.

    Args:
        env: WebShop environment
        agent: Agent to evaluate
        instruction: Task instruction
        max_steps: Maximum steps per episode
        verbose: Print debug output

    Returns:
        Episode result dict
    """
    agent.reset()

    obs, info = env.reset()
    instruction = info.get('instruction', instruction)

    done = False
    step = 0
    total_reward = 0
    actions = []

    while not done and step < max_steps:
        try:
            action = agent.get_action(instruction, obs)
            actions.append(action)

            if verbose:
                print(f"Step {step}: {action}")

            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

        except Exception as e:
            print(f"Error at step {step}: {e}")
            break

    success = total_reward >= 1.0

    return {
        'instruction': instruction,
        'success': success,
        'total_reward': total_reward,
        'total_steps': step,
        'actions': actions,
        'done': done,
    }


def evaluate_agent(
    agent_type: str,
    checkpoint_path: Optional[str] = None,
    num_episodes: int = 100,
    max_steps: int = 50,
    top_k: int = 5,
    model: str = "gpt-4o-mini",
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> EvalMetrics:
    """
    Evaluate an agent on WebShop.

    Args:
        agent_type: Type of agent to evaluate
        checkpoint_path: Path to checkpoint (for recurrent agent)
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        top_k: Number of chunks to retrieve
        model: LLM model to use
        output_path: Path to save results
        verbose: Print debug output

    Returns:
        Evaluation metrics
    """
    # Import WebShop
    try:
        from web_agent_site.envs import WebAgentTextEnv
    except ImportError:
        raise ImportError(
            "WebShop not installed. Run: "
            "cd external/WebShop && pip install -e ."
        )

    # Create environment
    env = WebAgentTextEnv(
        observation_mode="text",
        num_products=None,
    )

    # Create agent
    config = AgentConfig(
        max_steps=max_steps,
        top_k=top_k,
        model=model,
        verbose=verbose,
    )

    agent = create_agent(agent_type, checkpoint_path, config)
    print(f"Created {agent_type} agent")

    # Run episodes
    results = []
    for i in tqdm(range(num_episodes), desc=f"Evaluating {agent_type}"):
        try:
            result = run_episode(
                env,
                agent,
                instruction="",  # Will be set by env
                max_steps=max_steps,
                verbose=verbose,
            )
            results.append(result)

            if verbose or (i + 1) % 10 == 0:
                current_metrics = compute_metrics(results)
                print(f"\nProgress ({i+1}/{num_episodes}): "
                      f"Success={current_metrics.success_rate:.2%}, "
                      f"Steps={current_metrics.avg_steps:.1f}")

        except Exception as e:
            print(f"Episode {i} failed: {e}")
            continue

    # Compute final metrics
    metrics = compute_metrics(results)

    print(f"\n=== Final Results for {agent_type} ===")
    print(metrics)

    # Save results
    if output_path:
        output_data = {
            'agent_type': agent_type,
            'checkpoint_path': checkpoint_path,
            'num_episodes': num_episodes,
            'config': {
                'max_steps': max_steps,
                'top_k': top_k,
                'model': model,
            },
            'metrics': metrics.to_dict(),
            'episodes': results,
            'timestamp': datetime.now().isoformat(),
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate agents on WebShop")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=['recurrent', 'rolling', 'query_rag'],
        help="Agent type to evaluate"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (required for recurrent agent)"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug output"
    )
    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = f"results/{args.agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    evaluate_agent(
        agent_type=args.agent,
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        top_k=args.top_k,
        model=args.model,
        output_path=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
