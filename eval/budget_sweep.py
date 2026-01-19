#!/usr/bin/env python3
"""
Budget Sweep Experiment: Success vs Context Budget

Compares full_context (truncation) vs state_gated_compression across
different context budgets to demonstrate efficiency gains.

Expected outcome:
- At high budgets: tie
- At low budgets: compression wins
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from typing import List, Dict
import argparse

# Budget levels to test (in characters, ~4 chars = 1 token)
DEFAULT_BUDGETS = [512, 768, 1024, 1536, 2048, 4096]


def run_evaluation(
    agent: str,
    budget: int,
    num_episodes: int = 100,
    seed_start: int = 42,
) -> Dict:
    """Run evaluation for a single agent at a single budget."""

    cmd = [
        sys.executable,
        "eval/live_webshop_eval.py",
        "--agents", agent,
        "--num_episodes", str(num_episodes),
        "--context_budget", str(budget),
        "--seed_start", str(seed_start),
        "--output", f"results/budget_sweep/{agent}_{budget}",
    ]

    print(f"\n{'='*60}")
    print(f"Running: {agent} @ budget={budget} chars")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse results from output
    output = result.stdout + result.stderr
    print(output[-2000:] if len(output) > 2000 else output)

    # Extract metrics from output (basic parsing)
    metrics = {
        "agent": agent,
        "budget": budget,
        "num_episodes": num_episodes,
        "success_rate": None,
        "avg_reward": None,
        "mean_steps": None,
        "mean_context_chars": None,
    }

    for line in output.split("\n"):
        if agent in line and "%" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if "%" in part:
                    try:
                        metrics["success_rate"] = float(part.replace("%", "")) / 100
                    except:
                        pass
                    break

    return metrics


def run_budget_sweep(
    budgets: List[int],
    agents: List[str],
    num_episodes: int,
    seed_start: int,
    output_dir: str,
):
    """Run full budget sweep experiment."""

    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for budget in budgets:
        for agent in agents:
            try:
                result = run_evaluation(
                    agent=agent,
                    budget=budget,
                    num_episodes=num_episodes,
                    seed_start=seed_start,
                )
                all_results.append(result)

                # Save intermediate results
                with open(f"{output_dir}/sweep_results.json", "w") as f:
                    json.dump(all_results, f, indent=2)

            except Exception as e:
                print(f"Error running {agent} @ {budget}: {e}")

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = f"{output_dir}/sweep_{timestamp}.json"
    with open(final_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "budgets": budgets,
            "agents": agents,
            "num_episodes": num_episodes,
            "seed_start": seed_start,
            "results": all_results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("BUDGET SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {final_file}")

    # Print summary table
    print(f"\n{'Budget':<10} {'full_context':<15} {'compression':<15} {'Δ':<10}")
    print("-" * 50)

    for budget in budgets:
        fc = next((r for r in all_results if r["agent"] == "full_context" and r["budget"] == budget), None)
        comp = next((r for r in all_results if r["agent"] == "state_gated_compression" and r["budget"] == budget), None)

        fc_str = f"{fc['success_rate']*100:.1f}%" if fc and fc['success_rate'] else "N/A"
        comp_str = f"{comp['success_rate']*100:.1f}%" if comp and comp['success_rate'] else "N/A"

        if fc and comp and fc['success_rate'] and comp['success_rate']:
            delta = (comp['success_rate'] - fc['success_rate']) * 100
            delta_str = f"{'+' if delta > 0 else ''}{delta:.1f}%"
        else:
            delta_str = "N/A"

        print(f"{budget:<10} {fc_str:<15} {comp_str:<15} {delta_str:<10}")


def main():
    parser = argparse.ArgumentParser(description="Budget sweep experiment")
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
                        help="Budget levels to test (in chars)")
    parser.add_argument("--agents", type=str, nargs="+",
                        default=["full_context", "state_gated_compression"],
                        help="Agents to compare")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Episodes per condition")
    parser.add_argument("--seed_start", type=int, default=42,
                        help="Starting seed")
    parser.add_argument("--output", type=str, default="results/budget_sweep",
                        help="Output directory")

    args = parser.parse_args()

    run_budget_sweep(
        budgets=args.budgets,
        agents=args.agents,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
