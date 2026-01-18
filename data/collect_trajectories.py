"""
Collect trajectories by running a baseline agent on WebShop/ALFWorld.
"""

import os
import uuid
import json
import argparse
from typing import Optional
from tqdm import tqdm
import jsonlines
from openai import OpenAI
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from data.schemas import AgentEvent, Episode


# ============================================================================
# Actor LLM
# ============================================================================

class ActorLLM:
    """Wrapper for LLM that generates actions."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.total_tokens = {"prompt": 0, "completion": 0}

    def get_action(self, instruction: str, observation: str, history: str = "") -> str:
        """Generate next action given current state."""

        system_prompt = """You are an agent completing a shopping task on WebShop. Given an instruction and the current webpage, output a single action.

Valid actions:
- search[query] - search for products
- click[element] - click on a button, link, option, or product ASIN

CRITICAL WORKFLOW:
1. Search page: search for products
2. Results page: click a product ASIN (e.g., click[B09ABC123])
3. Product page: You MUST select ALL required options (color, size) from the instruction BEFORE clicking Buy Now
4. After selecting all options: click[Buy Now]

The instruction specifies required attributes like "color: green stripe" and "size: large". On the product page, you'll see options listed after [SEP] markers. Select EACH required option by clicking its exact name.

Rules:
- Output ONLY the action, nothing else
- On product pages, ALWAYS select size and color options that match the instruction BEFORE buying
- Look at the instruction to see what color/size is required

Example for instruction "color: blue, size: medium":
search[wireless headphones]
click[B09XYZ456]
click[blue]
click[medium]
click[Buy Now]"""

        user_content = f"Instruction: {instruction}\n\n"
        if history:
            user_content += f"Recent actions:\n{history}\n\n"
        user_content += f"Current page:\n{observation}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=100,
            temperature=self.temperature
        )

        # Track token usage
        self.total_tokens["prompt"] += response.usage.prompt_tokens
        self.total_tokens["completion"] += response.usage.completion_tokens

        return response.choices[0].message.content.strip()


# ============================================================================
# WebShop Environment Wrapper
# ============================================================================

class WebShopEnvWrapper:
    """Wrapper for WebShop environment with logging."""

    def __init__(self, num_products: int = 1000):
        # Import here to avoid dependency issues
        try:
            from web_agent_site.envs import WebAgentTextEnv
            self.env = WebAgentTextEnv(
                observation_mode="text",
                num_products=num_products,
            )
        except ImportError:
            raise ImportError(
                "WebShop not installed. Make sure PYTHONPATH includes "
                "external/WebShop and run: source activate_env.sh"
            )

    def reset(self, instruction: Optional[str] = None):
        """Reset environment and return initial observation."""
        result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        if instruction:
            self.env.instruction = instruction
        return obs, {"instruction": self.env.instruction_text}

    def step(self, action: str):
        """Execute action and return (obs, reward, done, info)."""
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def get_instruction(self) -> str:
        """Get current task instruction."""
        return self.env.instruction_text


# ============================================================================
# Episode Collection
# ============================================================================

def collect_episode(
    env: WebShopEnvWrapper,
    actor: ActorLLM,
    max_steps: int = 50,
    verbose: bool = False
) -> Episode:
    """Run one episode and return logged Episode."""

    events = []
    obs, info = env.reset()
    instruction = info["instruction"]

    done = False
    t = 0
    total_reward = 0
    history_actions = []

    while not done and t < max_steps:
        # Log observation
        events.append(AgentEvent(
            t=t,
            event_type='OBS',
            text=obs[:4096],  # Truncate very long observations
            meta={"step": t}
        ))

        # Get action from actor
        history_str = "\n".join(history_actions[-3:]) if history_actions else ""
        action = actor.get_action(instruction, obs, history_str)

        # Log action
        events.append(AgentEvent(
            t=t,
            event_type='ACT',
            text=action,
            meta={"step": t}
        ))
        history_actions.append(action)

        if verbose:
            print(f"Step {t}: {action}")

        # Execute action
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Log reward if non-zero
        if reward != 0:
            events.append(AgentEvent(
                t=t,
                event_type='REWARD',
                text="",
                reward=reward,
                meta={"step": t}
            ))

        t += 1

    # Log episode end
    events.append(AgentEvent(
        t=t,
        event_type='DONE',
        text="",
        done=True,
        meta={"final_reward": total_reward}
    ))

    # Determine success (WebShop: reward >= 1.0 is success, partial otherwise)
    success = total_reward >= 1.0

    return Episode(
        episode_id=str(uuid.uuid4()),
        env="webshop",
        instruction=instruction,
        events=events,
        success=success,
        total_steps=t,
        total_reward=total_reward,
        token_usage=actor.total_tokens.copy()
    )


def collect_trajectories(
    output_path: str,
    num_episodes: int = 1000,
    max_steps: int = 50,
    model: str = "gpt-4o-mini",
    resume: bool = True
):
    """Collect multiple episodes and save to JSONL."""

    # Setup
    env = WebShopEnvWrapper()
    actor = ActorLLM(model=model)

    # Resume from existing file if present
    start_idx = 0
    if resume and os.path.exists(output_path):
        with jsonlines.open(output_path) as reader:
            start_idx = sum(1 for _ in reader)
        print(f"Resuming from episode {start_idx}")

    # Collect episodes
    mode = 'a' if resume and start_idx > 0 else 'w'

    success_count = 0
    total_steps_list = []

    with jsonlines.open(output_path, mode=mode) as writer:
        for i in tqdm(range(start_idx, num_episodes), desc="Collecting episodes"):
            try:
                episode = collect_episode(env, actor, max_steps)
                # Write as dict (jsonlines handles serialization)
                from dataclasses import asdict
                writer.write(asdict(episode))

                if episode.success:
                    success_count += 1
                total_steps_list.append(episode.total_steps)

                # Print progress every 100 episodes
                if (i + 1) % 100 == 0:
                    success_rate = success_count / (i + 1 - start_idx)
                    avg_steps = sum(total_steps_list) / len(total_steps_list)
                    print(f"\nProgress: {i+1}/{num_episodes}")
                    print(f"Success rate: {success_rate:.2%}")
                    print(f"Avg steps: {avg_steps:.1f}")
                    print(f"Total tokens: {actor.total_tokens}")

            except Exception as e:
                print(f"Error in episode {i}: {e}")
                continue

    # Final stats
    print(f"\n=== Collection Complete ===")
    print(f"Total episodes: {num_episodes}")
    print(f"Success rate: {success_count / (num_episodes - start_idx):.2%}")
    print(f"Avg steps: {sum(total_steps_list) / len(total_steps_list):.1f}")
    print(f"Total tokens used: {actor.total_tokens}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/webshop.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override with CLI args
    output_path = args.output or config["data_collection"]["output_path"]
    num_episodes = args.num_episodes or config["data_collection"]["num_episodes"]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect trajectories
    collect_trajectories(
        output_path=output_path,
        num_episodes=num_episodes,
        max_steps=config["env"]["max_steps"],
        model=config["data_collection"]["actor_model"]
    )
