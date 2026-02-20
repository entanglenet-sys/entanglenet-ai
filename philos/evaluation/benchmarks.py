"""
Benchmark runner for PHILOS evaluation campaigns.

Runs a set of episodes across environments and collects metrics,
producing structured BenchmarkResult objects.

Maps to WP4 T4.1 — Evaluation Protocol Design.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from philos.evaluation.metrics import PhilosMetrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str = "default_benchmark"
    num_episodes: int = 100
    max_steps_per_episode: int = 500
    seed: int = 42
    environments: list[str] = field(default_factory=lambda: ["pour_task", "fetch_sort_task"])
    save_trajectories: bool = False
    output_dir: str = "results/benchmarks"


@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""

    env_name: str
    episode_idx: int
    success: bool
    total_reward: float
    steps: int
    duration_s: float
    metrics: dict[str, float] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] | None = None


@dataclass
class BenchmarkResult:
    """Aggregated results of a benchmark run."""

    name: str
    env_name: str
    num_episodes: int
    success_rate: float
    mean_reward: float
    std_reward: float
    mean_steps: float
    mean_duration_s: float
    metrics: dict[str, float] = field(default_factory=dict)
    episodes: list[EpisodeResult] = field(default_factory=list)


class BenchmarkRunner:
    """Runs evaluation benchmarks across PHILOS environments.

    Usage:
        runner = BenchmarkRunner()
        results = runner.run(env, policy, config)
    """

    def __init__(self) -> None:
        self._metrics = PhilosMetrics()

    def run(
        self,
        env: Any,
        policy: Any,
        config: BenchmarkConfig | None = None,
    ) -> BenchmarkResult:
        """Run a full benchmark campaign.

        Args:
            env: Gymnasium-compatible environment.
            policy: Policy with predict(obs, context, deterministic=True).
            config: Benchmark parameters.

        Returns:
            Aggregated BenchmarkResult.
        """
        config = config or BenchmarkConfig()
        episode_results: list[EpisodeResult] = []

        logger.info(f"Starting benchmark '{config.name}': {config.num_episodes} episodes")

        for ep in range(config.num_episodes):
            result = self._run_episode(env, policy, ep, config)
            episode_results.append(result)

            if (ep + 1) % 10 == 0:
                successes = sum(1 for r in episode_results if r.success)
                logger.info(
                    f"  Episode {ep + 1}/{config.num_episodes} — "
                    f"success rate: {successes / (ep + 1):.1%}"
                )

        # Aggregate
        return self._aggregate(config.name, episode_results)

    def _run_episode(
        self,
        env: Any,
        policy: Any,
        episode_idx: int,
        config: BenchmarkConfig,
    ) -> EpisodeResult:
        """Run a single evaluation episode."""
        t_start = time.perf_counter()
        obs, info = env.reset(seed=config.seed + episode_idx)
        total_reward = 0.0
        trajectory: list[dict[str, Any]] = [] if config.save_trajectories else []

        self._metrics.reset_episode()

        for step in range(config.max_steps_per_episode):
            # Policy inference
            action = policy.predict(obs, context=None, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Collect per-step metrics
            self._metrics.record_step(
                action=action,
                reward=reward,
                info=info,
            )

            if config.save_trajectories:
                trajectory.append({
                    "step": step,
                    "obs": obs.tolist(),
                    "action": action.tolist() if hasattr(action, "tolist") else action,
                    "reward": reward,
                })

            if terminated or truncated:
                break

        duration = time.perf_counter() - t_start
        ep_metrics = self._metrics.get_episode_metrics()

        return EpisodeResult(
            env_name=getattr(env, "__class__", type(env)).__name__,
            episode_idx=episode_idx,
            success=info.get("success", terminated and not truncated),
            total_reward=total_reward,
            steps=step + 1,
            duration_s=duration,
            metrics=ep_metrics,
            trajectory=trajectory if config.save_trajectories else None,
        )

    @staticmethod
    def _aggregate(name: str, episodes: list[EpisodeResult]) -> BenchmarkResult:
        """Aggregate episode results into a benchmark summary."""
        rewards = [e.total_reward for e in episodes]
        successes = [e.success for e in episodes]

        # Aggregate per-metric
        all_metric_keys: set[str] = set()
        for e in episodes:
            all_metric_keys.update(e.metrics.keys())

        agg_metrics: dict[str, float] = {}
        for key in all_metric_keys:
            vals = [e.metrics[key] for e in episodes if key in e.metrics]
            if vals:
                agg_metrics[f"{key}_mean"] = float(np.mean(vals))
                agg_metrics[f"{key}_std"] = float(np.std(vals))

        return BenchmarkResult(
            name=name,
            env_name=episodes[0].env_name if episodes else "unknown",
            num_episodes=len(episodes),
            success_rate=float(np.mean(successes)),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_steps=float(np.mean([e.steps for e in episodes])),
            mean_duration_s=float(np.mean([e.duration_s for e in episodes])),
            metrics=agg_metrics,
            episodes=episodes,
        )
