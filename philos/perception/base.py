"""
Abstract Perception Interface.

All perception backends must implement this interface.
This ensures loose coupling — the cognitive/learning modules depend
on the interface, not on specific implementations (YOLO-World, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Detection:
    """A single object detection."""

    label: str
    confidence: float
    bbox_2d: np.ndarray  # [x1, y1, x2, y2] in pixels
    bbox_3d: np.ndarray | None = None  # [x, y, z, w, h, d] in meters
    mask: np.ndarray | None = None  # Segmentation mask
    position_world: np.ndarray | None = None  # 3D position in world frame
    semantic_properties: dict[str, Any] = field(default_factory=dict)
    # e.g., {"material": "glass", "state": "filled", "hazard_level": "high"}


@dataclass
class PerceptionOutput:
    """Output from the perception pipeline."""

    detections: list[Detection] = field(default_factory=list)
    depth_map: np.ndarray | None = None  # H x W depth image
    point_cloud: np.ndarray | None = None  # N x 3 points
    scene_embedding: np.ndarray | None = None  # Scene-level feature vector
    timestamp: float = 0.0
    processing_time_ms: float = 0.0


class BasePerception(ABC):
    """Abstract base class for perception backends.

    Implementations:
        - YoloWorldDetector: System 1 (50Hz) — fast open-vocabulary detection
        - VLMGrounding: System 2 (1Hz) — slow semantic scene understanding
        - SensorFusion: Fuses RGB-D + LiDAR into unified 3D representation
    """

    @abstractmethod
    def initialize(self, config: Any) -> None:
        """Initialize the perception model(s) and load weights."""
        ...

    @abstractmethod
    def detect(self, rgb: np.ndarray, depth: np.ndarray | None = None) -> PerceptionOutput:
        """Run detection on an input frame.

        Args:
            rgb: RGB image (H, W, 3).
            depth: Optional depth image (H, W) in meters.

        Returns:
            PerceptionOutput with detected objects.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this perception backend."""
        ...

    @property
    @abstractmethod
    def hz(self) -> float:
        """Target operating frequency."""
        ...
