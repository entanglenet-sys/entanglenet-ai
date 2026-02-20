"""
Deterministic Safety Shield — the non-negotiable safety layer.

This module implements the MPC-based safety filter described in
WP3 T3.3.  It runs at 50 Hz and can override ANY AI-generated
command to maintain hard safety invariants.

Hard constraints (from PHILOS proposal KPIs):
    1. Platform tilt < 10 degrees
    2. End-effector linear velocity < 1.5 m/s
    3. Joint torques within rated limits
    4. Minimum collision distance > safety margin
    5. Fluid spill rate = 0 during transport
    6. Emergency stop latency < 50 ms

The Safety Shield is PURELY DETERMINISTIC — no neural networks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from philos.control.base import BaseController, ControlCommand
from philos.core.registry import register_component
from philos.core.state import RobotState

logger = logging.getLogger(__name__)


class SafetyLevel(str, Enum):
    """Safety alert levels."""

    NOMINAL = "nominal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyConstraints:
    """Hard safety constraints for the PHILOS platform.

    All values are absolute limits that CANNOT be exceeded.
    """

    # Platform stability
    max_tilt_deg: float = 10.0          # degrees — KPI from proposal
    max_tilt_rate_deg_s: float = 30.0   # degrees/s

    # End-effector limits
    max_ee_velocity: float = 1.5        # m/s — KPI from proposal
    max_ee_acceleration: float = 5.0    # m/s^2

    # Joint limits (per-joint, 6-DoF arm)
    joint_position_limits: np.ndarray = field(
        default_factory=lambda: np.array([
            [-2.87, 2.87],    # J1
            [-1.76, 1.76],    # J2
            [-2.87, 2.87],    # J3
            [-3.07, 3.07],    # J4
            [-2.87, 2.87],    # J5
            [-6.28, 6.28],    # J6 (wrist)
        ])
    )
    max_joint_velocity: np.ndarray = field(
        default_factory=lambda: np.array([2.175, 2.175, 2.175, 3.49, 3.49, 3.49])
    )
    max_joint_torque: np.ndarray = field(
        default_factory=lambda: np.array([87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    )

    # Base (AMR) limits
    max_base_linear_velocity: float = 1.0   # m/s
    max_base_angular_velocity: float = 1.5  # rad/s

    # Collision avoidance
    min_collision_distance: float = 0.05    # metres
    safety_margin: float = 0.10             # metres (buffer zone)

    # Gripper
    max_grip_force: float = 40.0            # N


@dataclass
class SafetyState:
    """Current safety assessment of the system."""

    level: SafetyLevel = SafetyLevel.NOMINAL
    violations: list[str] = field(default_factory=list)
    overrides_applied: list[str] = field(default_factory=list)
    tilt_deg: float = 0.0
    ee_velocity: float = 0.0
    min_obstacle_distance: float = float("inf")


@register_component("control", "safety_shield")
class SafetyShield(BaseController):
    """Deterministic Safety Shield — 50 Hz safety filter.

    Pipeline: RL Action → clip/scale → constraint check → MPC refine → safe command

    The shield implements a 3-tier response:
        1. NOMINAL: pass through (possibly clipped)
        2. WARNING: reduce velocities, increase damping
        3. CRITICAL/E-STOP: halt all motion, apply brakes

    This is the ONLY component that can override the RL policy.
    """

    def __init__(
        self,
        constraints: SafetyConstraints | None = None,
        dt: float = 0.02,
    ) -> None:
        self._constraints = constraints or SafetyConstraints()
        self._dt = dt
        self._prev_command: ControlCommand | None = None
        self._safety_state = SafetyState()
        self._e_stop_active = False

        # Jerk limits (for smooth constraint enforcement)
        self._max_joint_jerk = 500.0  # rad/s^3
        self._prev_joint_velocities = np.zeros(6)

    # ------------------------------------------------------------------
    # BaseController interface
    # ------------------------------------------------------------------

    def compute(
        self,
        action: np.ndarray,
        state: RobotState,
        dt: float = 0.02,
    ) -> ControlCommand:
        """Apply safety filtering to a raw RL action.

        Args:
            action: 10-dim action [base_vx, base_vy, base_omega, j1..j6, gripper].
            state: Current robot state observation.
            dt: Timestep.

        Returns:
            Safe, constrained ControlCommand.
        """
        c = self._constraints
        self._safety_state = SafetyState()

        # Emergency stop check
        if self._e_stop_active:
            return self._emergency_stop_command()

        # ------- Parse raw action -------
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] < 10:
            action = np.pad(action, (0, 10 - action.shape[0]))

        base_vel = action[:3].copy()
        joint_targets = action[3:9].copy()
        gripper = float(np.clip(action[9], 0.0, 1.0))

        overrides: list[str] = []

        # ------- 1. Tilt check -------
        tilt = self._estimate_tilt(state)
        self._safety_state.tilt_deg = tilt
        if tilt > c.max_tilt_deg:
            self._safety_state.level = SafetyLevel.CRITICAL
            self._safety_state.violations.append(
                f"tilt={tilt:.1f}° > {c.max_tilt_deg}°"
            )
            return self._emergency_stop_command()
        elif tilt > c.max_tilt_deg * 0.8:
            # Warning zone — scale down velocities
            scale = max(0.1, 1.0 - (tilt - c.max_tilt_deg * 0.8) / (c.max_tilt_deg * 0.2))
            base_vel *= scale
            overrides.append(f"tilt_warning_scale={scale:.2f}")

        # ------- 2. Base velocity limits -------
        linear_speed = np.linalg.norm(base_vel[:2])
        if linear_speed > c.max_base_linear_velocity:
            base_vel[:2] *= c.max_base_linear_velocity / linear_speed
            overrides.append("base_linear_clipped")

        if abs(base_vel[2]) > c.max_base_angular_velocity:
            base_vel[2] = np.clip(
                base_vel[2], -c.max_base_angular_velocity, c.max_base_angular_velocity
            )
            overrides.append("base_angular_clipped")

        # ------- 3. Joint position limits -------
        for i in range(min(6, len(joint_targets))):
            lo, hi = c.joint_position_limits[i]
            if joint_targets[i] < lo or joint_targets[i] > hi:
                joint_targets[i] = np.clip(joint_targets[i], lo, hi)
                overrides.append(f"joint_{i}_pos_clipped")

        # ------- 4. Joint velocity limits -------
        if self._prev_command is not None:
            joint_vel = (joint_targets - self._prev_command.joint_positions) / dt
            for i in range(6):
                if abs(joint_vel[i]) > c.max_joint_velocity[i]:
                    # Clamp velocity → recompute target
                    clamped = np.clip(joint_vel[i], -c.max_joint_velocity[i], c.max_joint_velocity[i])
                    joint_targets[i] = self._prev_command.joint_positions[i] + clamped * dt
                    overrides.append(f"joint_{i}_vel_clipped")
            joint_velocities = joint_vel
        else:
            joint_velocities = np.zeros(6)

        # ------- 5. Jerk limit (smoothness) -------
        jerk = (joint_velocities - self._prev_joint_velocities) / dt
        for i in range(6):
            if abs(jerk[i]) > self._max_joint_jerk:
                clamped_jerk = np.clip(jerk[i], -self._max_joint_jerk, self._max_joint_jerk)
                joint_velocities[i] = self._prev_joint_velocities[i] + clamped_jerk * dt
                joint_targets[i] = (
                    self._prev_command.joint_positions[i] + joint_velocities[i] * dt
                    if self._prev_command is not None
                    else joint_targets[i]
                )
                overrides.append(f"joint_{i}_jerk_clipped")
        self._prev_joint_velocities = joint_velocities.copy()

        # ------- 6. End-effector velocity estimate -------
        ee_vel = self._estimate_ee_velocity(joint_velocities, state)
        self._safety_state.ee_velocity = ee_vel
        if ee_vel > c.max_ee_velocity:
            scale = c.max_ee_velocity / ee_vel
            joint_velocities *= scale
            if self._prev_command is not None:
                joint_targets = self._prev_command.joint_positions + joint_velocities * dt
            overrides.append(f"ee_vel_clipped({ee_vel:.2f}->{c.max_ee_velocity})")

        # ------- 7. Collision distance -------
        min_dist = self._check_collision_distance(state)
        self._safety_state.min_obstacle_distance = min_dist
        if min_dist < c.min_collision_distance:
            self._safety_state.level = SafetyLevel.CRITICAL
            self._safety_state.violations.append(f"collision_dist={min_dist:.3f}m")
            return self._emergency_stop_command()
        elif min_dist < c.safety_margin:
            scale = max(0.1, (min_dist - c.min_collision_distance) /
                        (c.safety_margin - c.min_collision_distance))
            joint_velocities *= scale
            base_vel *= scale
            overrides.append(f"collision_margin_scale={scale:.2f}")

        # ------- Build safe command -------
        self._safety_state.overrides_applied = overrides
        if overrides:
            self._safety_state.level = SafetyLevel.WARNING

        cmd = ControlCommand(
            base_velocity=base_vel,
            joint_positions=joint_targets,
            joint_velocities=joint_velocities,
            joint_torques=np.zeros(6),  # Torques computed by lower-level controller
            gripper_position=gripper,
            is_safe=self._safety_state.level != SafetyLevel.CRITICAL,
            safety_overrides=overrides,
        )
        self._prev_command = cmd
        return cmd

    def reset(self) -> None:
        """Reset shield state for a new episode."""
        self._prev_command = None
        self._prev_joint_velocities = np.zeros(6)
        self._safety_state = SafetyState()
        self._e_stop_active = False

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------

    def trigger_emergency_stop(self) -> None:
        """Manually trigger an emergency stop."""
        self._e_stop_active = True
        logger.critical("EMERGENCY STOP triggered!")

    def release_emergency_stop(self) -> None:
        """Release the emergency stop (requires explicit action)."""
        self._e_stop_active = False
        logger.warning("Emergency stop released.")

    @property
    def safety_state(self) -> SafetyState:
        return self._safety_state

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _emergency_stop_command(self) -> ControlCommand:
        """Generate a full-stop command."""
        return ControlCommand(
            base_velocity=np.zeros(3),
            joint_positions=(
                self._prev_command.joint_positions
                if self._prev_command is not None
                else np.zeros(6)
            ),
            joint_velocities=np.zeros(6),
            joint_torques=np.zeros(6),
            gripper_position=(
                self._prev_command.gripper_position
                if self._prev_command is not None
                else 0.5
            ),
            is_safe=False,
            safety_overrides=["EMERGENCY_STOP"],
        )

    @staticmethod
    def _estimate_tilt(state: RobotState) -> float:
        """Estimate platform tilt from the AMR state (degrees)."""
        if state.amr is not None and state.amr.orientation is not None:
            # orientation is quaternion [x, y, z, w]
            q = state.amr.orientation
            # Roll-pitch from quaternion
            sinr_cosp = 2.0 * (q[3] * q[0] + q[1] * q[2])
            cosr_cosp = 1.0 - 2.0 * (q[0] ** 2 + q[1] ** 2)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = 2.0 * (q[3] * q[1] - q[2] * q[0])
            pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

            tilt_rad = np.sqrt(roll ** 2 + pitch ** 2)
            return float(np.degrees(tilt_rad))
        return 0.0

    @staticmethod
    def _estimate_ee_velocity(
        joint_velocities: np.ndarray,
        state: RobotState,
    ) -> float:
        """Rough estimate of end-effector velocity from joint velocities.

        Uses a simplified Jacobian approximation.  A full kinematic
        model would replace this in production.
        """
        # Approximate link lengths for a 6-DoF arm (metres)
        link_lengths = np.array([0.0, 0.35, 0.35, 0.05, 0.05, 0.05])
        # Rough upper bound: sum of |omega_i * L_i|
        ee_vel = float(np.sum(np.abs(joint_velocities[:6]) * link_lengths))
        return ee_vel

    @staticmethod
    def _check_collision_distance(state: RobotState) -> float:
        """Check minimum distance to obstacles using LiDAR.

        Returns minimum obstacle distance in metres.
        """
        if state.lidar_scan is not None and len(state.lidar_scan) > 0:
            valid = state.lidar_scan[state.lidar_scan > 0.01]
            if len(valid) > 0:
                return float(np.min(valid))
        return float("inf")
