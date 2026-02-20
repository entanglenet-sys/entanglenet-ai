"""
Context Vector (z) — The bridge between semantic understanding and motor control.

In the PHILOS architecture, the VLM (System 2) produces a Latent Context Vector
that numerically encodes physical constraints from natural language and visual scene
understanding. This vector conditions the RL policy in real-time.

Example:
    "Handle the fragile beaker carefully" →
        z = [high_impedance=0.9, low_velocity=0.8, vertical_orientation=0.95, ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ManipulationMode(str, Enum):
    """High-level manipulation modes derived from semantic understanding."""

    STIFF = "stiff"  # Position control — rigid objects
    COMPLIANT = "compliant"  # Force control — fragile/deformable objects
    FLUID = "fluid"  # Fluid-adaptive — liquid containers
    PRECISION = "precision"  # High-precision — assembly tasks


@dataclass
class ContextVector:
    """Latent Context Vector (z) — conditions the RL policy π(s, z).

    This is the core data structure that bridges the probabilistic world
    of language/vision and the deterministic world of robotic control.

    Attributes:
        embedding: The raw latent vector from the VLM encoder (dim = context_vector_dim).
        manipulation_mode: High-level control mode derived from semantic analysis.
        impedance_scale: Scaling factor for impedance (0=free, 1=maximum stiffness).
        velocity_limit_scale: Scaling factor for max velocity (0=stop, 1=full speed).
        orientation_constraint: Target orientation constraint (e.g., keep vertical for fluids).
        jerk_penalty: Penalty weight for jerky motion (higher = smoother).
        semantic_labels: Object semantic labels detected in the scene.
        language_command: The raw natural language input that generated this context.
        confidence: Confidence score from the VLM (0-1).
        timestamp: Time of generation (for staleness checking).
    """

    embedding: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float32))
    manipulation_mode: ManipulationMode = ManipulationMode.STIFF
    impedance_scale: float = 0.5
    velocity_limit_scale: float = 1.0
    orientation_constraint: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Upright
    )
    jerk_penalty: float = 0.1
    semantic_labels: list[str] = field(default_factory=list)
    language_command: str = ""
    confidence: float = 1.0
    timestamp: float = 0.0

    def to_tensor(self) -> np.ndarray:
        """Flatten context to a fixed-size numpy array for policy conditioning.

        Returns:
            1D array concatenating: [embedding, mode_onehot, impedance, velocity, orientation, jerk]
        """
        mode_onehot = np.zeros(len(ManipulationMode), dtype=np.float32)
        mode_onehot[list(ManipulationMode).index(self.manipulation_mode)] = 1.0

        scalar_params = np.array(
            [self.impedance_scale, self.velocity_limit_scale, self.jerk_penalty, self.confidence],
            dtype=np.float32,
        )

        return np.concatenate([
            self.embedding,
            mode_onehot,
            scalar_params,
            self.orientation_constraint,
        ])

    @property
    def tensor_dim(self) -> int:
        """Total dimensionality of the flattened context tensor."""
        return len(self.embedding) + len(ManipulationMode) + 4 + 3

    def is_stale(self, current_time: float, max_age_s: float = 2.0) -> bool:
        """Check if this context vector is too old to be trusted."""
        return (current_time - self.timestamp) > max_age_s

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for API transport."""
        return {
            "embedding": self.embedding.tolist(),
            "manipulation_mode": self.manipulation_mode.value,
            "impedance_scale": self.impedance_scale,
            "velocity_limit_scale": self.velocity_limit_scale,
            "orientation_constraint": self.orientation_constraint.tolist(),
            "jerk_penalty": self.jerk_penalty,
            "semantic_labels": self.semantic_labels,
            "language_command": self.language_command,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextVector:
        """Deserialize from a dictionary."""
        return cls(
            embedding=np.array(data.get("embedding", []), dtype=np.float32),
            manipulation_mode=ManipulationMode(data.get("manipulation_mode", "stiff")),
            impedance_scale=data.get("impedance_scale", 0.5),
            velocity_limit_scale=data.get("velocity_limit_scale", 1.0),
            orientation_constraint=np.array(
                data.get("orientation_constraint", [0, 0, 1]), dtype=np.float32
            ),
            jerk_penalty=data.get("jerk_penalty", 0.1),
            semantic_labels=data.get("semantic_labels", []),
            language_command=data.get("language_command", ""),
            confidence=data.get("confidence", 1.0),
            timestamp=data.get("timestamp", 0.0),
        )
