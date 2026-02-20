"""
Base Isaac Sim environment wrapper — Gymnasium-compatible interface.

Provides the abstract base for all PHILOS simulation environments.
Handles the lifecycle of the Omniverse Kit application, USD stage
management, and the step/reset loop that RL algorithms expect.

When Isaac Sim is not installed (e.g. unit testing, CI), the
environment falls back to a lightweight stub that mimics the
observation/action spaces.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Fallback for minimal installs
    gym = None  # type: ignore
    spaces = None  # type: ignore

from philos.core.config import SimulationConfig
from philos.core.state import RobotState, JointState, AMRState, EndEffectorState

logger = logging.getLogger(__name__)

# Check for Isaac Sim availability
_ISAAC_AVAILABLE = False
try:
    from omni.isaac.core import World  # type: ignore
    from omni.isaac.core.robots import Robot  # type: ignore
    _ISAAC_AVAILABLE = True
except ImportError:
    pass


@dataclass
class EnvConfig:
    """Environment configuration."""

    num_envs: int = 1
    env_spacing: float = 3.0      # metres between parallel envs
    max_episode_steps: int = 500
    dt: float = 0.02              # 50 Hz control frequency
    render: bool = False
    use_gpu_pipeline: bool = True
    physics_dt: float = 1 / 240   # PhysX substep

    # Observation dimensions
    obs_dim: int = 48             # state(30) + context(18)
    action_dim: int = 10          # base(3) + arm(6) + gripper(1)


class IsaacSimEnv(abc.ABC):
    """Gymnasium-compatible Isaac Sim environment base class.

    Subclasses implement:
        - _setup_scene()  : load USD assets, create robots
        - _compute_obs()  : build observation from sim state
        - _compute_reward(): task-specific reward function
        - _check_done()   : episode termination conditions
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: EnvConfig | None = None) -> None:
        self._config = config or EnvConfig()
        self._step_count = 0
        self._episode_count = 0

        # Observation / action spaces
        self.observation_space = (
            spaces.Box(-np.inf, np.inf, shape=(self._config.obs_dim,), dtype=np.float32)
            if spaces is not None
            else None
        )
        self.action_space = (
            spaces.Box(-1.0, 1.0, shape=(self._config.action_dim,), dtype=np.float32)
            if spaces is not None
            else None
        )

        # Isaac Sim world (lazy init)
        self._world: Any = None
        self._robot: Any = None
        self._is_initialized = False

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment and return initial observation."""
        if not self._is_initialized:
            self._initialize()

        self._step_count = 0
        self._episode_count += 1

        # Reset simulation
        if _ISAAC_AVAILABLE and self._world is not None:
            self._world.reset()

        self._on_reset(options)

        obs = self._compute_obs()
        info = {"episode": self._episode_count}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Returns:
            obs, reward, terminated, truncated, info
        """
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32)

        # Apply action to simulation
        self._apply_action(action)

        # Step physics
        if _ISAAC_AVAILABLE and self._world is not None:
            self._world.step(render=self._config.render)
        else:
            self._stub_step(action)

        # Compute outputs
        obs = self._compute_obs()
        reward = self._compute_reward(action)
        terminated = self._check_terminated()
        truncated = self._step_count >= self._config.max_episode_steps
        info = self._compute_info()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Clean up simulation resources."""
        if _ISAAC_AVAILABLE and self._world is not None:
            self._world.stop()
        self._is_initialized = False

    # ------------------------------------------------------------------
    # Methods to override in subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _setup_scene(self) -> None:
        """Load USD assets, create robot and environment."""
        ...

    @abc.abstractmethod
    def _compute_obs(self) -> np.ndarray:
        """Compute the observation vector."""
        ...

    @abc.abstractmethod
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute the scalar reward."""
        ...

    @abc.abstractmethod
    def _check_terminated(self) -> bool:
        """Check if the episode should terminate (success or failure)."""
        ...

    def _on_reset(self, options: dict[str, Any] | None = None) -> None:
        """Hook called during reset; override for domain randomization."""
        pass

    def _compute_info(self) -> dict[str, Any]:
        """Compute additional info for logging."""
        return {"step": self._step_count}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Initialize the Isaac Sim world (one-time setup)."""
        if _ISAAC_AVAILABLE:
            self._world = World(
                stage_units_in_meters=1.0,
                physics_dt=self._config.physics_dt,
                rendering_dt=self._config.dt,
            )
            self._setup_scene()
            self._world.reset()
            logger.info("Isaac Sim environment initialized.")
        else:
            logger.warning(
                "Isaac Sim not found — using stub environment. "
                "Install NVIDIA Isaac Sim for full simulation."
            )
            self._setup_scene()

        self._is_initialized = True

    def _apply_action(self, action: np.ndarray) -> None:
        """Send action to the robot in simulation."""
        if _ISAAC_AVAILABLE and self._robot is not None:
            base_vel = action[:3]
            joint_targets = action[3:9]
            gripper = action[9] if len(action) > 9 else 0.5
            # Isaac Sim robot API calls would go here
            self._robot.set_joint_positions(joint_targets)
        # In stub mode, _stub_step handles state updates

    def _stub_step(self, action: np.ndarray) -> None:
        """Lightweight physics step when Isaac Sim is unavailable."""
        # Override in subclasses for testing
        pass

    def get_robot_state(self) -> RobotState:
        """Get the current robot state from simulation."""
        if _ISAAC_AVAILABLE and self._robot is not None:
            # Extract from Isaac Sim
            joint_positions = self._robot.get_joint_positions()
            joint_velocities = self._robot.get_joint_velocities()
            joints = [
                JointState(name=f"joint_{i}", position=float(p), velocity=float(v))
                for i, (p, v) in enumerate(zip(joint_positions, joint_velocities))
            ]
            return RobotState(joints=joints)

        # Stub state
        return RobotState(
            joints=[JointState(name=f"joint_{i}", position=0.0) for i in range(6)],
            amr=AMRState(),
            end_effector=EndEffectorState(),
        )
