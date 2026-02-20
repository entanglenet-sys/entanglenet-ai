"""
Robot State definitions — the proprioceptive 's' in π(s, z).

High-frequency state vectors (50Hz) from the robot's sensors:
joint encoders, LiDAR, IMU, force/torque sensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class JointState:
    """State of a single robot joint."""

    name: str = ""
    position: float = 0.0  # rad
    velocity: float = 0.0  # rad/s
    effort: float = 0.0  # Nm (torque)

    def to_array(self) -> np.ndarray:
        return np.array([self.position, self.velocity, self.effort], dtype=np.float32)


@dataclass
class SensorReading:
    """A generic timestamped sensor reading."""

    sensor_id: str = ""
    data: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    timestamp: float = 0.0
    frame_id: str = "base_link"


@dataclass
class AMRState:
    """Autonomous Mobile Robot (base) state."""

    position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)  # x, y, z
    )
    orientation: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float32)  # quaternion xyzw
    )
    linear_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )


@dataclass
class EndEffectorState:
    """End-effector state (tip of the robotic arm)."""

    position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    orientation: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float32)
    )
    force: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    torque: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    gripper_state: float = 0.0  # 0=open, 1=closed


@dataclass
class RobotState:
    """Complete robot state — the high-frequency 's' vector for the RL policy.

    Combines AMR base state, arm joint states, end-effector, and sensor readings
    into a single observation space for the whole-body policy.
    """

    amr: AMRState = field(default_factory=AMRState)
    joints: list[JointState] = field(default_factory=list)
    end_effector: EndEffectorState = field(default_factory=EndEffectorState)
    lidar_scan: np.ndarray = field(
        default_factory=lambda: np.zeros(360, dtype=np.float32)  # 1-degree resolution
    )
    fluid_level: float = 0.0  # Estimated fluid fill level (0-1) if carrying container
    timestamp: float = 0.0

    def to_observation(self) -> np.ndarray:
        """Flatten to a fixed-size observation vector for the RL policy.

        Returns:
            1D numpy array combining all proprioceptive signals.
        """
        # AMR: 3 pos + 4 orient + 3 lin_vel + 3 ang_vel = 13
        amr_obs = np.concatenate([
            self.amr.position,
            self.amr.orientation,
            self.amr.linear_velocity,
            self.amr.angular_velocity,
        ])

        # Joints: N * 3 (pos, vel, effort)
        if self.joints:
            joint_obs = np.concatenate([j.to_array() for j in self.joints])
        else:
            joint_obs = np.zeros(18, dtype=np.float32)  # Default 6-DoF arm

        # End-effector: 3 pos + 4 orient + 3 force + 3 torque + 1 gripper = 14
        ee_obs = np.concatenate([
            self.end_effector.position,
            self.end_effector.orientation,
            self.end_effector.force,
            self.end_effector.torque,
            np.array([self.end_effector.gripper_state], dtype=np.float32),
        ])

        # Compressed LiDAR (downsample to 36 rays)
        lidar_compressed = self.lidar_scan[::10][:36] if len(self.lidar_scan) >= 360 else self.lidar_scan

        # Fluid level
        fluid = np.array([self.fluid_level], dtype=np.float32)

        return np.concatenate([amr_obs, joint_obs, ee_obs, lidar_compressed, fluid])

    @property
    def observation_dim(self) -> int:
        """Dimensionality of the observation vector."""
        return len(self.to_observation())
