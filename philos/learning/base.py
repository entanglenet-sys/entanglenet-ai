"""
Abstract RL Policy Interface.

All RL policies in PHILOS implement this interface. Policies are
conditioned on both the robot state (s) and the context vector (z),
following the PHILOS architecture: π(s, z).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from philos.core.context_vector import ContextVector
from philos.core.state import RobotState


class BasePolicy(ABC):
    """Abstract base class for RL policies.

    All PHILOS policies take the form π(a | s, z) where:
        s = robot state (proprioceptive, high-frequency 50Hz)
        z = context vector (semantic, low-frequency 1Hz)
        a = action (joint velocities, base velocity, gripper)
    """

    @abstractmethod
    def initialize(self, config: Any) -> None:
        """Initialize the policy network and load weights if available."""
        ...

    @abstractmethod
    def predict(
        self,
        state: RobotState,
        context: ContextVector,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Predict an action given state and context.

        Args:
            state: Current robot state observation.
            context: Current semantic context vector from the VLM.
            deterministic: If True, use mean action (no exploration noise).

        Returns:
            Action vector to be sent to the Safety Shield.
        """
        ...

    @abstractmethod
    def train_step(
        self,
        batch: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Dictionary containing: obs, actions, rewards, dones, context_vectors

        Returns:
            Dictionary of training metrics.
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save policy weights to disk."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load policy weights from disk."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name identifier."""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the action space."""
        ...

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """Dimensionality of the observation space (state + context)."""
        ...
