"""
"The Sommelier" — Fluid Pouring Task (WP4 T4.2).

A mobile manipulator must:
    1. Navigate to a source container
    2. Grasp it (compliant grip, fluid inside)
    3. Transport it without spilling
    4. Pour a precise volume into a target container
    5. Return the source container

Success criteria (from PHILOS KPIs):
    - Spill rate < 5 % of total volume
    - Pour accuracy: volume error < 10 %
    - No collisions with environment
    - Platform tilt < 10° throughout

Requires PhysX 5 PBD fluid simulation (NVIDIA Flex / Flow).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from philos.simulation.isaac_env import IsaacSimEnv, EnvConfig
from philos.core.registry import register_component
from philos.learning.reward_functions import DynamicRewardFunction

logger = logging.getLogger(__name__)

# Isaac Sim imports (optional)
_ISAAC_AVAILABLE = False
try:
    from omni.isaac.core.utils.prims import create_prim  # type: ignore
    _ISAAC_AVAILABLE = True
except ImportError:
    pass


@dataclass
class PourTaskConfig(EnvConfig):
    """Configuration for the pouring task."""

    # Fluid
    target_volume_ml: float = 250.0
    source_volume_ml: float = 500.0
    max_spill_fraction: float = 0.05

    # Geometry
    source_position: tuple[float, float, float] = (0.5, 0.0, 0.8)
    target_position: tuple[float, float, float] = (0.5, 0.3, 0.6)

    # Reward
    pour_accuracy_weight: float = 10.0
    spill_penalty_weight: float = 20.0
    smoothness_weight: float = 1.0

    obs_dim: int = 56  # state(30) + context(18) + task(8)


@register_component("simulation", "pour_task")
class PourTaskEnv(IsaacSimEnv):
    """The Sommelier — pour a precise volume of fluid.

    Observation space (56-dim):
        [robot_state(30), context_vector(18), task_obs(8)]
        task_obs = [source_pos(3), target_pos(3), poured_frac, spill_frac]

    Action space (10-dim):
        [base_vx, base_vy, base_omega, j1..j6, gripper]
    """

    def __init__(self, config: PourTaskConfig | None = None) -> None:
        self._task_config = config or PourTaskConfig()
        super().__init__(config=self._task_config)

        # Task state
        self._poured_volume = 0.0
        self._spilled_volume = 0.0
        self._grasped = False
        self._source_pos = np.array(self._task_config.source_position)
        self._target_pos = np.array(self._task_config.target_position)

        # Reward function
        self._reward_fn = DynamicRewardFunction()

        # Stub state for testing without Isaac Sim
        self._stub_joint_pos = np.zeros(6)
        self._stub_ee_pos = np.array([0.0, 0.0, 0.5])
        self._stub_base_pos = np.zeros(3)

    def _setup_scene(self) -> None:
        """Load the lab environment, robot, containers, and fluid."""
        if _ISAAC_AVAILABLE and self._world is not None:
            # Ground plane
            self._world.scene.add_default_ground_plane()

            # TODO: Load robot USD (Jazzing AMR + 6-DoF arm)
            # self._robot = self._world.scene.add(Robot(...))

            # TODO: Add source/target containers as USD prims
            # create_prim("/World/source_container", ...)
            # create_prim("/World/target_container", ...)

            # TODO: Add PhysX 5 PBD fluid particles
            # create_prim("/World/fluid", "PhysxParticleSystem", ...)

            logger.info("Pour task scene loaded (Isaac Sim).")
        else:
            logger.info("Pour task scene loaded (stub mode).")

    def _on_reset(self, options: dict | None = None) -> None:
        """Reset fluid, container positions, and task state."""
        self._poured_volume = 0.0
        self._spilled_volume = 0.0
        self._grasped = False
        self._stub_joint_pos = np.zeros(6)
        self._stub_ee_pos = np.array([0.0, 0.0, 0.5])
        self._stub_base_pos = np.zeros(3)

        # Domain randomization could go here
        # self._randomizer.apply_to_env(self)

    def _compute_obs(self) -> np.ndarray:
        """Build the 56-dim observation."""
        obs = np.zeros(self._task_config.obs_dim, dtype=np.float32)

        # Robot state (simplified for stub)
        obs[:6] = self._stub_joint_pos
        obs[6:9] = self._stub_ee_pos
        obs[9:12] = self._stub_base_pos

        # Task-specific observations
        total_vol = self._task_config.source_volume_ml
        obs[48:51] = self._source_pos
        obs[51:54] = self._target_pos
        obs[54] = self._poured_volume / total_vol if total_vol > 0 else 0
        obs[55] = self._spilled_volume / total_vol if total_vol > 0 else 0

        return obs

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward: encourage accurate pouring, penalize spills."""
        reward = 0.0
        tc = self._task_config

        # Progress toward target volume
        target_frac = tc.target_volume_ml / tc.source_volume_ml
        poured_frac = self._poured_volume / tc.source_volume_ml
        volume_error = abs(poured_frac - target_frac)
        reward += tc.pour_accuracy_weight * max(0, 1.0 - volume_error)

        # Spill penalty
        spill_frac = self._spilled_volume / tc.source_volume_ml
        reward -= tc.spill_penalty_weight * spill_frac

        # Smoothness (penalize large actions)
        reward -= tc.smoothness_weight * float(np.sum(action ** 2)) * 0.01

        # Success bonus
        if poured_frac >= target_frac * 0.9 and spill_frac < tc.max_spill_fraction:
            reward += 50.0

        return float(reward)

    def _check_terminated(self) -> bool:
        """Episode ends on success, excessive spill, or collision."""
        tc = self._task_config
        total = tc.source_volume_ml

        # Success
        target_frac = tc.target_volume_ml / total
        poured_frac = self._poured_volume / total
        if poured_frac >= target_frac * 0.9:
            return True

        # Failure: too much spill
        if self._spilled_volume / total > tc.max_spill_fraction * 2:
            return True

        return False

    def _stub_step(self, action: np.ndarray) -> None:
        """Update stub state for testing."""
        # Simple kinematics update
        self._stub_base_pos[:2] += action[:2] * 0.02
        self._stub_joint_pos += action[3:9] * 0.02 if len(action) >= 9 else 0.0
        # Simulate some pouring progress
        if self._grasped and self._stub_ee_pos[2] > 0.7:
            self._poured_volume += 2.0  # ml per step
            self._spilled_volume += 0.05  # small spill per step
