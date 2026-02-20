"""
YOLO-World Open-Vocabulary Detector — System 1 (50Hz Reflex Loop).

This is the "fast eye" of the PHILOS architecture. It provides real-time,
open-vocabulary object detection at 50Hz, allowing the robot to track and
classify unknown objects without retraining.

Key capabilities:
    - Open-vocabulary: detects objects by text description, not fixed classes
    - Real-time: maintains 50Hz on edge GPU (NVIDIA Jetson / RTX)
    - Zero-shot: handles objects it has never seen in training

Maps to:
    WP2 T2.1 (VLM Fine-Tuning) — YOLO-World integration
    System 1 layer: high-frequency reflex detection
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from philos.core.registry import register_component
from philos.perception.base import BasePerception, Detection, PerceptionOutput

logger = logging.getLogger(__name__)


@register_component("perception", "yolo_world")
class YoloWorldDetector(BasePerception):
    """YOLO-World open-vocabulary object detector.

    Wraps the Ultralytics YOLO-World model with PHILOS-specific
    post-processing for industrial/chemical environments.
    """

    def __init__(
        self,
        model_name: str = "yolo-world-l",
        confidence_threshold: float = 0.3,
        device: str = "cuda:0",
        custom_classes: list[str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._model: Any = None
        self._custom_classes = custom_classes or [
            # Default classes for chemical/industrial environments
            "beaker",
            "flask",
            "bottle",
            "container",
            "liquid",
            "glass",
            "metal part",
            "plastic container",
            "hazardous material",
            "tool",
            "person",
            "obstacle",
        ]

    @property
    def name(self) -> str:
        return "yolo_world"

    @property
    def hz(self) -> float:
        return 50.0

    def initialize(self, config: Any = None) -> None:
        """Load YOLO-World model.

        Note: Requires `ultralytics` package.
        In production, the model runs on NVIDIA Jetson or RTX GPU.
        """
        try:
            from ultralytics import YOLO

            self._model = YOLO(f"{self._model_name}.pt")
            # Set the open-vocabulary classes
            self._model.set_classes(self._custom_classes)
            logger.info(
                f"YOLO-World '{self._model_name}' loaded on {self._device} "
                f"with {len(self._custom_classes)} custom classes"
            )
        except ImportError:
            logger.warning(
                "ultralytics not installed. YoloWorldDetector running in stub mode. "
                "Install with: pip install ultralytics"
            )
            self._model = None

    def detect(self, rgb: np.ndarray, depth: np.ndarray | None = None) -> PerceptionOutput:
        """Run open-vocabulary detection on an RGB frame.

        Args:
            rgb: RGB image (H, W, 3).
            depth: Optional depth image (H, W) in meters.

        Returns:
            PerceptionOutput with open-vocabulary detections.
        """
        start = time.monotonic()
        detections: list[Detection] = []

        if self._model is not None:
            results = self._model.predict(
                rgb,
                conf=self._confidence_threshold,
                device=self._device,
                verbose=False,
            )

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        label = (
                            self._custom_classes[cls_id]
                            if cls_id < len(self._custom_classes)
                            else f"object_{cls_id}"
                        )
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()

                        det = Detection(
                            label=label,
                            confidence=conf,
                            bbox_2d=xyxy,
                        )

                        # If depth is available, estimate 3D position
                        if depth is not None:
                            det.position_world = self._estimate_3d_position(xyxy, depth)

                        detections.append(det)
        else:
            # Stub mode — return empty detections for testing
            logger.debug("YOLO-World in stub mode, returning empty detections")

        processing_time = (time.monotonic() - start) * 1000

        return PerceptionOutput(
            detections=detections,
            depth_map=depth,
            timestamp=time.time(),
            processing_time_ms=processing_time,
        )

    def _estimate_3d_position(
        self, bbox_2d: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        """Estimate 3D position from 2D bbox center + depth map.

        Simple centroid-based depth lookup. In production, this would use
        proper camera intrinsics and point cloud projection.
        """
        cx = int((bbox_2d[0] + bbox_2d[2]) / 2)
        cy = int((bbox_2d[1] + bbox_2d[3]) / 2)

        h, w = depth.shape[:2]
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)

        z = float(depth[cy, cx])
        # Simplified pinhole camera model (replace with real intrinsics)
        fx, fy = 600.0, 600.0  # Focal lengths in pixels
        px, py = w / 2, h / 2  # Principal point

        x = (cx - px) * z / fx
        y = (cy - py) * z / fy

        return np.array([x, y, z], dtype=np.float32)

    def set_classes(self, classes: list[str]) -> None:
        """Dynamically update the open-vocabulary class set.

        This is a key PHILOS feature: the robot can be told to look for
        new objects via natural language without retraining.
        """
        self._custom_classes = classes
        if self._model is not None:
            self._model.set_classes(classes)
        logger.info(f"YOLO-World classes updated: {classes}")

    def shutdown(self) -> None:
        """Release model resources."""
        self._model = None
        logger.info("YOLO-World detector shut down")
