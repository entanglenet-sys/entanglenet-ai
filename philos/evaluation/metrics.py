"""
PHILOS Metrics — KPI tracking for the evaluation protocol.

Tracks all key performance indicators from the PHILOS proposal:
    - Spill rate (< 5 % target)
    - Grasp success rate (> 90 % target)
    - Object identification accuracy (> 95 % target)
    - Placement accuracy (< 2 cm target)
    - Control loop latency (< 10 ms budget)
    - Safety violations (0 target)
    - Sim-to-real transfer gap
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KPI:
    """A single key performance indicator."""

    name: str
    target: float
    unit: str = ""
    higher_is_better: bool = True
    values: list[float] = field(default_factory=list)

    @property
    def current(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    @property
    def meets_target(self) -> bool:
        if self.higher_is_better:
            return self.current >= self.target
        return self.current <= self.target


# PHILOS KPIs from the proposal
PHILOS_KPIS = {
    "spill_rate": KPI("Spill Rate", target=0.05, unit="%", higher_is_better=False),
    "grasp_success": KPI("Grasp Success Rate", target=0.90, unit="%"),
    "object_id_accuracy": KPI("Object ID Accuracy", target=0.95, unit="%"),
    "placement_accuracy": KPI("Placement Accuracy", target=0.02, unit="m", higher_is_better=False),
    "control_latency": KPI("Control Latency", target=10.0, unit="ms", higher_is_better=False),
    "safety_violations": KPI("Safety Violations", target=0.0, unit="count", higher_is_better=False),
    "sim_to_real_gap": KPI("Sim-to-Real Gap", target=0.15, unit="%", higher_is_better=False),
}


class PhilosMetrics:
    """Collects and tracks all PHILOS evaluation metrics.

    Usage:
        metrics = PhilosMetrics()
        metrics.reset_episode()
        for step in episode:
            metrics.record_step(action, reward, info)
        episode_metrics = metrics.get_episode_metrics()
    """

    def __init__(self) -> None:
        self._step_rewards: list[float] = []
        self._step_actions: list[np.ndarray] = []
        self._safety_violations: int = 0
        self._spill_volume: float = 0.0
        self._total_volume: float = 1.0
        self._grasps_attempted: int = 0
        self._grasps_succeeded: int = 0
        self._placements: list[float] = []  # distance errors
        self._control_latencies: list[float] = []
        self._step_timer: float = 0.0

    def reset_episode(self) -> None:
        """Reset per-episode counters."""
        self._step_rewards = []
        self._step_actions = []
        self._safety_violations = 0
        self._spill_volume = 0.0
        self._grasps_attempted = 0
        self._grasps_succeeded = 0
        self._placements = []
        self._control_latencies = []

    def record_step(
        self,
        action: np.ndarray | None = None,
        reward: float = 0.0,
        info: dict | None = None,
    ) -> None:
        """Record metrics for one timestep."""
        self._step_rewards.append(reward)
        if action is not None:
            self._step_actions.append(np.asarray(action))

        if info is None:
            return

        # Extract specific metrics from info dict
        if "safety_violation" in info:
            self._safety_violations += int(info["safety_violation"])

        if "spill_volume" in info:
            self._spill_volume += info["spill_volume"]
        if "total_volume" in info:
            self._total_volume = info["total_volume"]

        if "grasp_attempted" in info:
            self._grasps_attempted += 1
            if info.get("grasp_success", False):
                self._grasps_succeeded += 1

        if "placement_error" in info:
            self._placements.append(info["placement_error"])

        if "control_latency_ms" in info:
            self._control_latencies.append(info["control_latency_ms"])

    def get_episode_metrics(self) -> dict[str, float]:
        """Compute episode-level metrics."""
        metrics: dict[str, float] = {}

        # Reward statistics
        if self._step_rewards:
            metrics["total_reward"] = sum(self._step_rewards)
            metrics["mean_reward"] = float(np.mean(self._step_rewards))

        # Action statistics (smoothness)
        if len(self._step_actions) > 1:
            actions = np.array(self._step_actions)
            diffs = np.diff(actions, axis=0)
            metrics["action_smoothness"] = float(np.mean(np.sum(diffs ** 2, axis=1)))

        # Safety
        metrics["safety_violations"] = float(self._safety_violations)

        # Spill rate
        if self._total_volume > 0:
            metrics["spill_rate"] = self._spill_volume / self._total_volume

        # Grasp success
        if self._grasps_attempted > 0:
            metrics["grasp_success_rate"] = self._grasps_succeeded / self._grasps_attempted

        # Placement accuracy
        if self._placements:
            metrics["placement_accuracy_mean"] = float(np.mean(self._placements))
            metrics["placement_accuracy_max"] = float(np.max(self._placements))

        # Latency
        if self._control_latencies:
            metrics["control_latency_mean_ms"] = float(np.mean(self._control_latencies))
            metrics["control_latency_p99_ms"] = float(np.percentile(self._control_latencies, 99))

        return metrics

    @staticmethod
    def check_kpis(metrics: dict[str, float]) -> dict[str, bool]:
        """Check which KPIs are met based on collected metrics."""
        results: dict[str, bool] = {}
        mapping = {
            "spill_rate": "spill_rate",
            "grasp_success": "grasp_success_rate",
            "placement_accuracy": "placement_accuracy_mean",
            "control_latency": "control_latency_mean_ms",
            "safety_violations": "safety_violations",
        }
        for kpi_name, metric_key in mapping.items():
            kpi = PHILOS_KPIS[kpi_name]
            if metric_key in metrics:
                met = (
                    metrics[metric_key] >= kpi.target
                    if kpi.higher_is_better
                    else metrics[metric_key] <= kpi.target
                )
                results[kpi_name] = met
        return results
