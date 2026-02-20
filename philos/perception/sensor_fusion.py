"""
Sensor Fusion — Combining RGB-D and LiDAR into a unified 3D representation.

Fuses data from multiple sensors to provide a complete spatial model
for both the reflex loop (System 1) and the reasoning loop (System 2).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from philos.core.registry import register_component
from philos.perception.base import BasePerception, PerceptionOutput

logger = logging.getLogger(__name__)


@register_component("perception", "sensor_fusion")
class SensorFusion(BasePerception):
    """Late fusion of RGB-D camera and LiDAR point clouds."""

    def __init__(
        self,
        fusion_method: str = "late_fusion",
        voxel_size: float = 0.01,  # 1cm voxels for point cloud
    ) -> None:
        self._method = fusion_method
        self._voxel_size = voxel_size

    @property
    def name(self) -> str:
        return "sensor_fusion"

    @property
    def hz(self) -> float:
        return 30.0  # Runs at camera framerate

    def initialize(self, config: Any = None) -> None:
        logger.info(f"Sensor fusion initialized (method={self._method})")

    def detect(self, rgb: np.ndarray, depth: np.ndarray | None = None) -> PerceptionOutput:
        """Fuse sensor data. This module doesn't do detection — it prepares
        the unified spatial representation consumed by YOLO-World and VLM."""
        point_cloud = None
        if depth is not None:
            point_cloud = self._depth_to_pointcloud(depth)

        return PerceptionOutput(
            depth_map=depth,
            point_cloud=point_cloud,
        )

    def fuse_lidar(
        self,
        depth_cloud: np.ndarray,
        lidar_cloud: np.ndarray,
        tf_lidar_to_camera: np.ndarray,
    ) -> np.ndarray:
        """Fuse depth camera point cloud with LiDAR point cloud.

        Args:
            depth_cloud: N x 3 points from RGB-D camera.
            lidar_cloud: M x 3 points from LiDAR.
            tf_lidar_to_camera: 4x4 transformation matrix from LiDAR to camera frame.

        Returns:
            Fused point cloud (N+M) x 3 in camera frame.
        """
        # Transform LiDAR to camera frame
        ones = np.ones((lidar_cloud.shape[0], 1), dtype=np.float32)
        lidar_homo = np.hstack([lidar_cloud, ones])  # M x 4
        lidar_transformed = (tf_lidar_to_camera @ lidar_homo.T).T[:, :3]

        # Concatenate
        fused = np.vstack([depth_cloud, lidar_transformed])

        # Voxel downsample if needed
        if self._voxel_size > 0:
            fused = self._voxel_downsample(fused, self._voxel_size)

        return fused

    def _depth_to_pointcloud(
        self,
        depth: np.ndarray,
        fx: float = 600.0,
        fy: float = 600.0,
    ) -> np.ndarray:
        """Convert a depth image to a 3D point cloud."""
        h, w = depth.shape[:2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32)
        x = (u - w / 2) * z / fx
        y = (v - h / 2) * z / fy

        mask = z > 0
        points = np.stack([x[mask], y[mask], z[mask]], axis=-1)
        return points

    @staticmethod
    def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        """Simple voxel grid downsampling."""
        quantized = np.floor(points / voxel_size).astype(np.int32)
        _, indices = np.unique(quantized, axis=0, return_index=True)
        return points[indices]

    def shutdown(self) -> None:
        logger.info("Sensor fusion shut down")
