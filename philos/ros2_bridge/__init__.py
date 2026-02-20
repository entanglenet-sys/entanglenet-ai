"""
PHILOS ROS2 Bridge — Deterministic middleware for real-time control.

Provides the bridge between PHILOS modules and ROS2 Jazzy Jalisco
for deployment on physical hardware.

Maps to WP3 T3.1 — ROS2 Jazzy Deterministic Middleware.

Key requirements:
    - Motor command latency < 10 ms
    - 50 Hz control loop
    - Deterministic executor (SingleThreadedExecutor)
    - Safety shield runs as highest-priority callback
"""

from philos.ros2_bridge.bridge import PhilosROS2Bridge
from philos.ros2_bridge.topics import TopicConfig, PHILOS_TOPICS
from philos.ros2_bridge.transforms import TransformManager

__all__ = [
    "PhilosROS2Bridge",
    "TopicConfig",
    "PHILOS_TOPICS",
    "TransformManager",
]
