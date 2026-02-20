"""
ROS2 topic definitions for the PHILOS system.

Centralizes all topic names, QoS profiles, and message types to
ensure consistency across nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QoSProfile(str, Enum):
    """Quality of Service profiles."""

    RELIABLE = "reliable"           # For commands/safety
    BEST_EFFORT = "best_effort"     # For sensor streams
    TRANSIENT_LOCAL = "transient_local"  # For latched topics


@dataclass
class TopicConfig:
    """Configuration for a single ROS2 topic."""

    name: str
    msg_type: str          # ROS2 message type (e.g. "sensor_msgs/Image")
    qos: QoSProfile = QoSProfile.RELIABLE
    frequency_hz: float = 50.0
    description: str = ""


# All PHILOS topics in one place
PHILOS_TOPICS = {
    # ── Perception ──────────────────────────────────────────
    "rgb_camera": TopicConfig(
        name="/philos/perception/rgb",
        msg_type="sensor_msgs/Image",
        qos=QoSProfile.BEST_EFFORT,
        frequency_hz=30.0,
        description="RGB camera feed (720p or 1080p)",
    ),
    "depth_camera": TopicConfig(
        name="/philos/perception/depth",
        msg_type="sensor_msgs/Image",
        qos=QoSProfile.BEST_EFFORT,
        frequency_hz=30.0,
        description="Depth camera (aligned to RGB)",
    ),
    "lidar_scan": TopicConfig(
        name="/philos/perception/lidar",
        msg_type="sensor_msgs/LaserScan",
        qos=QoSProfile.BEST_EFFORT,
        frequency_hz=10.0,
        description="2D LiDAR scan for navigation & collision",
    ),
    "point_cloud": TopicConfig(
        name="/philos/perception/pointcloud",
        msg_type="sensor_msgs/PointCloud2",
        qos=QoSProfile.BEST_EFFORT,
        frequency_hz=10.0,
        description="3D point cloud (fused RGB-D + LiDAR)",
    ),
    "detections": TopicConfig(
        name="/philos/perception/detections",
        msg_type="vision_msgs/Detection3DArray",
        qos=QoSProfile.RELIABLE,
        frequency_hz=50.0,
        description="YOLO-World 3D detections (System 1, 50 Hz)",
    ),

    # ── Cognitive ───────────────────────────────────────────
    "context_vector": TopicConfig(
        name="/philos/cognitive/context_vector",
        msg_type="std_msgs/Float32MultiArray",
        qos=QoSProfile.RELIABLE,
        frequency_hz=1.0,
        description="VLM context vector z (System 2, 1 Hz)",
    ),
    "language_command": TopicConfig(
        name="/philos/cognitive/command",
        msg_type="std_msgs/String",
        qos=QoSProfile.RELIABLE,
        frequency_hz=0.1,
        description="Natural-language command from operator",
    ),

    # ── Learning / Policy ──────────────────────────────────
    "policy_action": TopicConfig(
        name="/philos/learning/action",
        msg_type="std_msgs/Float32MultiArray",
        qos=QoSProfile.RELIABLE,
        frequency_hz=50.0,
        description="RL policy output (10-dim action)",
    ),

    # ── Control ─────────────────────────────────────────────
    "safe_command": TopicConfig(
        name="/philos/control/safe_command",
        msg_type="std_msgs/Float32MultiArray",
        qos=QoSProfile.RELIABLE,
        frequency_hz=50.0,
        description="Safety-filtered actuator command",
    ),
    "joint_command": TopicConfig(
        name="/philos/control/joint_command",
        msg_type="trajectory_msgs/JointTrajectoryPoint",
        qos=QoSProfile.RELIABLE,
        frequency_hz=50.0,
        description="Joint position/velocity targets for arm",
    ),
    "base_velocity": TopicConfig(
        name="/philos/control/cmd_vel",
        msg_type="geometry_msgs/Twist",
        qos=QoSProfile.RELIABLE,
        frequency_hz=50.0,
        description="AMR base velocity command",
    ),
    "safety_status": TopicConfig(
        name="/philos/control/safety_status",
        msg_type="std_msgs/String",
        qos=QoSProfile.RELIABLE,
        frequency_hz=50.0,
        description="Safety shield status (nominal/warning/critical/e-stop)",
    ),

    # ── Robot State ─────────────────────────────────────────
    "joint_states": TopicConfig(
        name="/philos/state/joints",
        msg_type="sensor_msgs/JointState",
        qos=QoSProfile.BEST_EFFORT,
        frequency_hz=100.0,
        description="Joint encoder readings",
    ),
    "odom": TopicConfig(
        name="/philos/state/odom",
        msg_type="nav_msgs/Odometry",
        qos=QoSProfile.BEST_EFFORT,
        frequency_hz=50.0,
        description="AMR odometry",
    ),
    "imu": TopicConfig(
        name="/philos/state/imu",
        msg_type="sensor_msgs/Imu",
        qos=QoSProfile.BEST_EFFORT,
        frequency_hz=200.0,
        description="IMU for tilt detection",
    ),

    # ── Diagnostics ─────────────────────────────────────────
    "diagnostics": TopicConfig(
        name="/philos/diagnostics",
        msg_type="diagnostic_msgs/DiagnosticArray",
        qos=QoSProfile.RELIABLE,
        frequency_hz=1.0,
        description="System health diagnostics",
    ),
}
