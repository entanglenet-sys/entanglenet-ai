"""
"The Courier" — Fetch and Sort Task (WP4 T4.3).

A mobile manipulator must:
    1. Receive a natural-language command (e.g. "fetch the blue beaker
       from shelf B and place it in the hazardous zone")
    2. Navigate to the object location
    3. Identify the correct object via VLM
    4. Grasp and transport it safely
    5. Place it in the designated area, sorted by category

Success criteria (from PHILOS KPIs):
    - Correct object identification: >95%
    - Grasp success rate: >90%
    - Placement accuracy: <2 cm position error
    - No collisions during transport

This task validates the full Semantic-to-Control pipeline:
    Language → VLM → Context Vector → RL Policy → Safety Shield → Actuators
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from philos.simulation.isaac_env import IsaacSimEnv, EnvConfig
from philos.core.registry import register_component

logger = logging.getLogger(__name__)


@dataclass
class FetchSortConfig(EnvConfig):
    """Configuration for the fetch-and-sort task."""

    # Object library
    num_objects: int = 10
    num_categories: int = 4  # e.g. hazardous, fragile, standard, temperature-sensitive
    object_names: list[str] = field(default_factory=lambda: [
        "beaker_blue", "beaker_red", "flask_250ml", "flask_500ml",
        "bottle_reagent", "petri_dish", "test_tube_rack",
        "graduated_cylinder", "erlenmeyer_flask", "wash_bottle",
    ])
    shelf_positions: list[tuple[float, float, float]] = field(default_factory=lambda: [
        (1.0, -0.5, 0.8), (1.0, 0.0, 0.8), (1.0, 0.5, 0.8),
        (1.0, -0.5, 1.2), (1.0, 0.0, 1.2), (1.0, 0.5, 1.2),
    ])
    placement_zones: list[tuple[float, float, float]] = field(default_factory=lambda: [
        (-0.5, -0.5, 0.6),   # hazardous
        (-0.5, 0.0, 0.6),    # fragile
        (-0.5, 0.5, 0.6),    # standard
        (-0.5, 1.0, 0.6),    # temperature-sensitive
    ])

    # Reward
    grasp_reward: float = 5.0
    correct_placement_reward: float = 20.0
    wrong_placement_penalty: float = -10.0
    collision_penalty: float = -15.0
    time_penalty: float = -0.01

    # Thresholds
    grasp_distance_threshold: float = 0.05    # metres
    placement_accuracy: float = 0.02          # metres — KPI from proposal

    obs_dim: int = 64   # state(30) + context(18) + task(16)
    max_episode_steps: int = 1000


@register_component("simulation", "fetch_sort_task")
class FetchSortTaskEnv(IsaacSimEnv):
    """The Courier — fetch, identify, and sort objects.

    Observation (64-dim):
        [robot_state(30), context_vector(18), task_obs(16)]
        task_obs = [target_obj_pos(3), target_zone_pos(3),
                    ee_to_obj_vec(3), ee_to_zone_vec(3),
                    grasped_flag, correct_category, distance_to_obj, distance_to_zone]

    Action (10-dim):
        [base_vx, base_vy, base_omega, j1..j6, gripper]
    """

    def __init__(self, config: FetchSortConfig | None = None) -> None:
        self._task_config = config or FetchSortConfig()
        super().__init__(config=self._task_config)

        # Task state
        self._target_object_idx = 0
        self._target_category = 0
        self._target_obj_pos = np.zeros(3)
        self._target_zone_pos = np.zeros(3)
        self._grasped = False
        self._placed = False

        # Stub state
        self._stub_ee_pos = np.array([0.0, 0.0, 0.5])
        self._stub_base_pos = np.zeros(3)
        self._stub_joint_pos = np.zeros(6)
        self._rng = np.random.default_rng(42)

    def _setup_scene(self) -> None:
        """Set up the laboratory scene with shelves, objects, zones."""
        logger.info(
            f"Fetch-Sort task: {self._task_config.num_objects} objects, "
            f"{self._task_config.num_categories} categories"
        )
        # Isaac Sim USD loading would go here
        # For each object in object_names: create_prim(...)
        # For each zone: create visual marker

    def _on_reset(self, options: dict | None = None) -> None:
        """Randomize which object to fetch and its category."""
        tc = self._task_config

        # Pick a random target
        self._target_object_idx = int(self._rng.integers(0, tc.num_objects))
        self._target_category = int(self._rng.integers(0, tc.num_categories))

        # Set positions
        shelf_idx = self._target_object_idx % len(tc.shelf_positions)
        self._target_obj_pos = np.array(tc.shelf_positions[shelf_idx])
        self._target_zone_pos = np.array(tc.placement_zones[self._target_category])

        self._grasped = False
        self._placed = False

        # Reset stub
        self._stub_ee_pos = np.array([0.0, 0.0, 0.5])
        self._stub_base_pos = np.zeros(3)
        self._stub_joint_pos = np.zeros(6)

        logger.debug(
            f"Episode: fetch '{tc.object_names[self._target_object_idx]}' "
            f"cat={self._target_category} → zone {self._target_zone_pos}"
        )

    def _compute_obs(self) -> np.ndarray:
        obs = np.zeros(self._task_config.obs_dim, dtype=np.float32)

        # Robot state
        obs[:6] = self._stub_joint_pos
        obs[6:9] = self._stub_ee_pos
        obs[9:12] = self._stub_base_pos

        # Task observations
        ee = self._stub_ee_pos
        obj = self._target_obj_pos
        zone = self._target_zone_pos

        obs[48:51] = obj
        obs[51:54] = zone
        obs[54:57] = obj - ee
        obs[57:60] = zone - ee
        obs[60] = float(self._grasped)
        obs[61] = float(self._target_category)
        obs[62] = float(np.linalg.norm(obj - ee))
        obs[63] = float(np.linalg.norm(zone - ee))

        return obs

    def _compute_reward(self, action: np.ndarray) -> float:
        tc = self._task_config
        reward = tc.time_penalty  # Small time penalty each step

        ee = self._stub_ee_pos
        obj = self._target_obj_pos
        zone = self._target_zone_pos

        if not self._grasped:
            # Phase 1: Approach the object
            dist_to_obj = float(np.linalg.norm(ee - obj))
            reward += -dist_to_obj  # Distance shaping

            # Grasp detection
            if dist_to_obj < tc.grasp_distance_threshold and action[9] < 0.3:
                self._grasped = True
                reward += tc.grasp_reward
                logger.debug("Object grasped!")
        else:
            # Phase 2: Transport to correct zone
            dist_to_zone = float(np.linalg.norm(ee - zone))
            reward += -dist_to_zone  # Distance shaping

            # Placement detection
            if dist_to_zone < tc.placement_accuracy and action[9] > 0.7:
                self._placed = True
                reward += tc.correct_placement_reward
                logger.debug("Object placed correctly!")

        return float(reward)

    def _check_terminated(self) -> bool:
        return self._placed

    def _stub_step(self, action: np.ndarray) -> None:
        """Update stub state."""
        self._stub_base_pos[:2] += action[:2] * 0.02
        if len(action) >= 9:
            self._stub_joint_pos += action[3:9] * 0.02
        # Simple EE position update (follows joint motion)
        self._stub_ee_pos[0] += action[0] * 0.02
        self._stub_ee_pos[1] += action[1] * 0.02
        if len(action) > 5:
            self._stub_ee_pos[2] += action[5] * 0.01
