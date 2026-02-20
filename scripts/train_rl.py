#!/usr/bin/env python3
"""
PHILOS — RL Training Script.

Trains a whole-body policy for mobile manipulation using PPO
in NVIDIA Isaac Sim (or stub environment for development).

Usage:
    python scripts/train_rl.py
    python scripts/train_rl.py --config configs/training/ppo_config.yaml
    python scripts/train_rl.py --num-envs 32 --total-timesteps 5000000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from philos.core.config import load_config, PhilosConfig
from philos.core.registry import ComponentRegistry
from philos.core.context_vector import ContextVector, ManipulationMode
from philos.learning.policies.whole_body import WholeBodyPolicy
from philos.learning.reward_functions import DynamicRewardFunction
from philos.learning.domain_randomization import DomainRandomizer
from philos.control.safety_shield import SafetyShield
from philos.simulation.isaac_env import EnvConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("philos.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHILOS RL Training")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--env", type=str, default="pour_task", choices=["pour_task", "fetch_sort_task"])
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    return parser.parse_args()


def create_environment(env_name: str, num_envs: int = 1):
    """Create the training environment."""
    registry = ComponentRegistry()

    if registry.has("simulation", env_name):
        env = registry.create("simulation", env_name)
        logger.info(f"Created environment '{env_name}' from registry.")
        return env

    # Fallback: direct import
    if env_name == "pour_task":
        from philos.simulation.environments.pour_task import PourTaskEnv, PourTaskConfig
        config = PourTaskConfig(num_envs=num_envs)
        return PourTaskEnv(config)
    elif env_name == "fetch_sort_task":
        from philos.simulation.environments.fetch_sort_task import FetchSortTaskEnv, FetchSortConfig
        config = FetchSortConfig(num_envs=num_envs)
        return FetchSortTaskEnv(config)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def train(args: argparse.Namespace) -> None:
    """Main training loop."""
    logger.info("=" * 60)
    logger.info("PHILOS — Physical Intelligence Training")
    logger.info("=" * 60)

    # Seed
    np.random.seed(args.seed)

    # Create environment
    env = create_environment(args.env, args.num_envs)
    logger.info(f"Environment: {args.env}")

    # Determine dimensions
    obs_dim = env._config.obs_dim if hasattr(env, "_config") else 48
    action_dim = env._config.action_dim if hasattr(env, "_config") else 10

    # Create policy
    policy = WholeBodyPolicy(
        state_dim=obs_dim - 18,  # Subtract context dim
        context_dim=18,
        action_dim=action_dim,
        hidden_dims=[512, 256, 128],
        lr=args.lr,
    )
    logger.info(f"Policy: WholeBodyPolicy (obs={obs_dim}, act={action_dim})")

    # Create reward function & domain randomizer
    reward_fn = DynamicRewardFunction()
    dr = DomainRandomizer(seed=args.seed)

    # Safety shield (runs even during training)
    safety = SafetyShield()

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    if args.wandb:
        try:
            import wandb
            wandb.init(project="philos", config=vars(args))
        except ImportError:
            logger.warning("wandb not installed — skipping.")

    # Training loop
    logger.info(f"Training for {args.total_timesteps:,} timesteps...")
    total_steps = 0
    episode = 0
    best_reward = -float("inf")

    t_start = time.time()

    while total_steps < args.total_timesteps:
        episode += 1
        obs, info = env.reset()

        # Apply domain randomization
        dr_params = dr.sample()
        dr.apply_to_env(env, dr_params)

        episode_reward = 0.0
        episode_steps = 0
        done = False

        # Collect rollout
        rollout_obs = []
        rollout_actions = []
        rollout_rewards = []
        rollout_dones = []
        rollout_log_probs = []
        rollout_values = []

        while not done and episode_steps < (env._config.max_episode_steps if hasattr(env, "_config") else 500):
            # Policy forward pass
            action, log_prob, value = policy.predict_with_value(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            rollout_obs.append(obs)
            rollout_actions.append(action)
            rollout_rewards.append(reward)
            rollout_dones.append(done)
            rollout_log_probs.append(log_prob)
            rollout_values.append(value)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

        # Train on collected rollout
        if len(rollout_obs) >= args.batch_size // 10:  # Minimum batch
            batch = {
                "observations": np.array(rollout_obs),
                "actions": np.array(rollout_actions),
                "rewards": np.array(rollout_rewards),
                "dones": np.array(rollout_dones),
                "log_probs": np.array(rollout_log_probs),
                "values": np.array(rollout_values),
            }
            train_info = policy.train_step(batch)

        # Logging
        if episode % 10 == 0:
            elapsed = time.time() - t_start
            fps = total_steps / elapsed if elapsed > 0 else 0
            logger.info(
                f"Episode {episode:>5d} | Steps {total_steps:>8,} / {args.total_timesteps:,} | "
                f"Reward: {episode_reward:>8.2f} | FPS: {fps:>6.0f}"
            )

            if args.wandb:
                try:
                    import wandb
                    wandb.log({
                        "episode": episode,
                        "total_steps": total_steps,
                        "episode_reward": episode_reward,
                        "episode_steps": episode_steps,
                        "fps": fps,
                    })
                except Exception:
                    pass

        # Save checkpoint
        if total_steps % args.save_interval < episode_steps:
            ckpt_path = ckpt_dir / f"policy_step_{total_steps}.pt"
            policy.save(str(ckpt_path))
            logger.info(f"Checkpoint saved: {ckpt_path}")

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_path = ckpt_dir / "best_policy.pt"
                policy.save(str(best_path))
                logger.info(f"New best model (reward={best_reward:.2f})")

    # Final save
    final_path = ckpt_dir / "final_policy.pt"
    policy.save(str(final_path))
    logger.info(f"Training complete. Final model saved to {final_path}")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
