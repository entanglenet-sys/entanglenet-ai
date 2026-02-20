#!/usr/bin/env python3
"""
PHILOS — Evaluation Script.

Runs benchmark evaluation on trained policies.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_policy.pt
    python scripts/evaluate.py --env pour_task --episodes 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from philos.evaluation.benchmarks import BenchmarkRunner, BenchmarkConfig
from philos.evaluation.metrics import PhilosMetrics
from philos.learning.policies.whole_body import WholeBodyPolicy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("philos.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHILOS Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--env", type=str, default="pour_task", choices=["pour_task", "fetch_sort_task"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/evaluation.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-trajectories", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logger.info("PHILOS Evaluation")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Environment: {args.env}")
    logger.info(f"  Episodes: {args.episodes}")

    # Create environment
    if args.env == "pour_task":
        from philos.simulation.environments.pour_task import PourTaskEnv
        env = PourTaskEnv()
    else:
        from philos.simulation.environments.fetch_sort_task import FetchSortTaskEnv
        env = FetchSortTaskEnv()

    # Load policy
    obs_dim = env._config.obs_dim if hasattr(env, "_config") else 48
    action_dim = env._config.action_dim if hasattr(env, "_config") else 10

    policy = WholeBodyPolicy(
        state_dim=obs_dim - 18,
        context_dim=18,
        action_dim=action_dim,
    )
    policy.load(args.checkpoint)
    logger.info("Policy loaded.")

    # Run benchmark
    config = BenchmarkConfig(
        name=f"philos_{args.env}",
        num_episodes=args.episodes,
        seed=args.seed,
        save_trajectories=args.save_trajectories,
    )

    runner = BenchmarkRunner()
    result = runner.run(env, policy, config)

    # Print summary
    logger.info("=" * 50)
    logger.info(f"Benchmark: {result.name}")
    logger.info(f"  Success rate: {result.success_rate:.1%}")
    logger.info(f"  Mean reward:  {result.mean_reward:.2f} ± {result.std_reward:.2f}")
    logger.info(f"  Mean steps:   {result.mean_steps:.0f}")
    logger.info(f"  Mean time:    {result.mean_duration_s:.3f}s")

    # Check KPIs
    kpi_results = PhilosMetrics.check_kpis(result.metrics)
    if kpi_results:
        logger.info("  KPI Results:")
        for name, met in kpi_results.items():
            status = "PASS" if met else "FAIL"
            logger.info(f"    {name}: [{status}]")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "name": result.name,
        "env": result.env_name,
        "num_episodes": result.num_episodes,
        "success_rate": result.success_rate,
        "mean_reward": result.mean_reward,
        "std_reward": result.std_reward,
        "mean_steps": result.mean_steps,
        "metrics": result.metrics,
        "kpi_results": kpi_results,
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
