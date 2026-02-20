"""
Dynamic Reward Functions — the mathematical core of Semantic Reward Shaping.

Reward functions in PHILOS are NOT static. They are dynamically modified
by the Context Vector (z) at runtime. This module computes the scalar
reward signal from the environment state and the current reward weights.

Maps to:
    WP2 T2.2 — Language-to-Reward Interface
    D2.1 — Semantic Reward Shaping Core Codebase
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from philos.core.context_vector import ContextVector
from philos.core.state import RobotState

logger = logging.getLogger(__name__)

# Type alias for reward component functions
RewardComponentFn = Callable[[RobotState, ContextVector, dict[str, Any]], float]


class DynamicRewardFunction:
    """Dynamically weighted reward function conditioned on the Context Vector.

    Each reward component is a callable that computes a scalar value.
    The total reward is the weighted sum of all components, where
    weights are set by the Semantic Reward Shaping engine.
    """

    def __init__(self) -> None:
        self._components: dict[str, RewardComponentFn] = {}
        self._weights: dict[str, float] = {}
        self._register_default_components()

    def _register_default_components(self) -> None:
        """Register the default reward components."""
        self.register_component("position_error", _position_error_reward)
        self.register_component("orientation_error", _orientation_error_reward)
        self.register_component("velocity_penalty", _velocity_penalty)
        self.register_component("jerk_penalty", _jerk_penalty)
        self.register_component("collision_penalty", _collision_penalty)
        self.register_component("spill_penalty", _spill_penalty)
        self.register_component("force_penalty", _force_penalty)
        self.register_component("fluid_stability", _fluid_stability_reward)
        self.register_component("grip_stability", _grip_stability_reward)
        self.register_component("energy_efficiency", _energy_efficiency_reward)

    def register_component(self, name: str, fn: RewardComponentFn) -> None:
        """Register a custom reward component."""
        self._components[name] = fn

    def set_weights(self, weights: dict[str, float]) -> None:
        """Update reward weights (from Semantic Reward Shaping)."""
        self._weights.update(weights)

    def compute(
        self,
        state: RobotState,
        context: ContextVector,
        env_info: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute the total reward and individual component values.

        Args:
            state: Current robot state.
            context: Current context vector (contains weight modifiers).
            env_info: Additional environment info (e.g., target position, collisions).

        Returns:
            Tuple of (total_reward, component_breakdown).
        """
        if env_info is None:
            env_info = {}

        total = 0.0
        breakdown: dict[str, float] = {}

        for name, fn in self._components.items():
            weight = self._weights.get(name, 0.0)
            if abs(weight) < 1e-8:
                continue

            value = fn(state, context, env_info)
            weighted = weight * value
            breakdown[name] = weighted
            total += weighted

        return total, breakdown


# ─── Default Reward Component Functions ──────────────────────────────────────────


def _position_error_reward(
    state: RobotState, context: ContextVector, info: dict[str, Any]
) -> float:
    """Negative distance from end-effector to target position."""
    target = info.get("target_position", np.zeros(3))
    error = np.linalg.norm(state.end_effector.position - target)
    return float(-error)


def _orientation_error_reward(
    state: RobotState, context: ContextVector, info: dict[str, Any]
) -> float:
    """Penalty for deviation from the desired orientation (critical for fluids)."""
    # For fluid tasks, we want to keep containers upright
    desired_up = context.orientation_constraint
    # Simplified: compare z-axis of end-effector to desired up
    # In production, this uses quaternion distance
    ee_up = np.array([0, 0, 1], dtype=np.float32)  # Placeholder
    dot_product = float(np.clip(np.dot(ee_up, desired_up), -1, 1))
    return dot_product - 1.0  # 0 when aligned, -2 when opposite


def _velocity_penalty(
    state: RobotState, context: ContextVector, info: dict[str, Any]
) -> float:
    """Penalty for high velocities, scaled by context."""
    vel_norm = float(np.linalg.norm(state.amr.linear_velocity))
    limit = context.velocity_limit_scale * 1.5  # Max 1.5 m/s
    excess = max(0, vel_norm - limit)
    return -excess ** 2


def _jerk_penalty(
    state: RobotState, context: ContextVector, info: dict[str, Any]
) -> float:
    """Penalty for jerky motion (acceleration change)."""
    jerk = info.get("jerk_magnitude", 0.0)
    return float(-jerk * context.jerk_penalty)


def _collision_penalty(
    state: RobotState, context: ContextVector, info: dict[str, Any]
) -> float:
    """Large penalty for any collision."""
    collisions = info.get("collision_count", 0)
    return float(-collisions)


def _spill_penalty(
    state: RobotState, context: ContextVector, info: dict[str, Any]
) -> float:
    """Penalty for fluid spillage."""
    spill_amount = info.get("spill_amount", 0.0)
    return float(-spill_amount)


def _force_penalty(
    state: RobotState, context: ContextVector, info: dict[str, Any]
) -> float:
    """Penalty for excessive contact forces."""
    force_mag = float(np.linalg.norm(state.end_effector.force))
    max_force = (1.0 - context.impedance_scale) * 50.0 + 5.0  # Scale with impedance
    excess = max(0, force_mag - max_force)
    return -excess


def _fluid_stability_reward(
    state: RobotState, context: ContextVector, info: dict[str, Any]
) -> float:
    """Reward for minimizing fluid slosh in containers."""
    slosh_energy = info.get("fluid_slosh_energy", 0.0)
    return float(-slosh_energy)


def _grip_stability_reward(
    state: RobotState, context: ContextVector, info: dict[str, Any]
) -> float:
    """Reward for maintaining a stable grip (no slippage)."""
    slip_detected = info.get("slip_detected", False)
    return -1.0 if slip_detected else 0.0


def _energy_efficiency_reward(
    state: RobotState, context: ContextVector, info: dict[str, Any]
) -> float:
    """Reward for energy-efficient motion."""
    joint_efforts = [j.effort for j in state.joints]
    if not joint_efforts:
        return 0.0
    total_effort = sum(abs(e) for e in joint_efforts)
    return float(-total_effort * 0.01)
