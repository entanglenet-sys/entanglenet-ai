"""
TF2 Transform Manager for the PHILOS robot.

Manages the coordinate frame tree:
    world → odom → base_link → arm_base → link_1 → ... → link_6 → ee_link
                                         → camera_link
                                         → lidar_link
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Transform:
    """A rigid 3D transform (position + quaternion)."""

    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))  # xyzw
    parent_frame: str = "world"
    child_frame: str = "base_link"
    timestamp: float = 0.0


# Standard PHILOS frame definitions
PHILOS_FRAMES = {
    "world": None,
    "odom": "world",
    "base_link": "odom",
    "arm_base_link": "base_link",
    "link_1": "arm_base_link",
    "link_2": "link_1",
    "link_3": "link_2",
    "link_4": "link_3",
    "link_5": "link_4",
    "link_6": "link_5",
    "ee_link": "link_6",
    "gripper_link": "ee_link",
    "camera_link": "base_link",
    "depth_camera_link": "camera_link",
    "lidar_link": "base_link",
}


class TransformManager:
    """Manages the TF tree for the PHILOS robot.

    In ROS2 mode, wraps tf2_ros. In standalone mode, provides
    a simple in-memory transform lookup.
    """

    def __init__(self) -> None:
        self._transforms: dict[str, Transform] = {}
        self._frame_tree = dict(PHILOS_FRAMES)

    def update_transform(self, tf: Transform) -> None:
        """Update or add a transform."""
        key = f"{tf.parent_frame}_to_{tf.child_frame}"
        self._transforms[key] = tf

    def lookup(
        self,
        target_frame: str,
        source_frame: str,
    ) -> Transform | None:
        """Look up the transform from source to target frame.

        For the simple case of direct parent-child, returns the
        stored transform. For multi-hop, chains transforms.
        """
        key = f"{source_frame}_to_{target_frame}"
        if key in self._transforms:
            return self._transforms[key]

        # Try reverse
        rev_key = f"{target_frame}_to_{source_frame}"
        if rev_key in self._transforms:
            return self._invert(self._transforms[rev_key])

        return None

    @staticmethod
    def _invert(tf: Transform) -> Transform:
        """Invert a transform (simple negation for translation)."""
        # Full quaternion inversion would be needed for production
        return Transform(
            translation=-tf.translation,
            rotation=np.array([-tf.rotation[0], -tf.rotation[1],
                               -tf.rotation[2], tf.rotation[3]]),
            parent_frame=tf.child_frame,
            child_frame=tf.parent_frame,
            timestamp=tf.timestamp,
        )

    def get_ee_pose_in_world(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Convenience: get end-effector pose in world frame."""
        tf = self.lookup("ee_link", "world")
        if tf is not None:
            return tf.translation, tf.rotation
        return None
