"""
Abstract base class for all controllers in the PHILOS pipeline.

A controller takes a desired action (from RL policy) and the current
robot state, then produces safe, feasible actuator commands.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from philos.core.state import RobotState


@dataclass
class ControlCommand:
    """Low-level actuator command produced by a controller.

    Fields:
        base_velocity: (vx, vy, omega) for the AMR base.
        joint_positions: Target joint positions (rad) for the arm.
        joint_velocities: Target joint velocities (rad/s).
        joint_torques: Feed-forward torques (Nm).
        gripper_position: Gripper opening [0=closed, 1=open].
        is_safe: Whether the command passed the safety shield.
        safety_overrides: Which fields were modified by the safety shield.
    """

    base_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_torques: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gripper_position: float = 0.5
    is_safe: bool = True
    safety_overrides: list[str] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        """Flatten to actuator array: [base(3), joints(6), gripper(1)]."""
        return np.concatenate([
            self.base_velocity,
            self.joint_positions,
            [self.gripper_position],
        ])


class BaseController(abc.ABC):
    """Abstract controller interface.

    All controllers must be deterministic (non-AI) to satisfy the
    PHILOS safety architecture.
    """

    @abc.abstractmethod
    def compute(
        self,
        action: np.ndarray,
        state: RobotState,
        dt: float = 0.02,
    ) -> ControlCommand:
        """Compute safe actuator commands.

        Args:
            action: Raw action from the RL policy (10-dim).
            state: Current robot state.
            dt: Control timestep (default 50 Hz = 0.02 s).

        Returns:
            ControlCommand with safe, feasible actuator targets.
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset controller state (call at episode start)."""
        ...
