"""
Abstract Cognitive Engine Interface.

The cognitive engine is the "System 2" brain of PHILOS — it processes
natural language commands and visual context to produce reward-shaping
signals for the RL policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from philos.core.context_vector import ContextVector


class BaseCognitiveEngine(ABC):
    """Abstract base for cognitive engine implementations."""

    @abstractmethod
    def initialize(self, config: Any) -> None:
        """Initialize the cognitive engine."""
        ...

    @abstractmethod
    def process_command(
        self,
        command: str,
        scene_rgb: np.ndarray | None = None,
        scene_depth: np.ndarray | None = None,
    ) -> ContextVector:
        """Process a natural language command and generate a Context Vector.

        This is the core "Language-to-Reward" interface. The command is
        analyzed (potentially with visual context) to produce the vector z
        that conditions the RL policy.

        Args:
            command: Natural language instruction from operator.
            scene_rgb: Optional current camera frame.
            scene_depth: Optional depth image.

        Returns:
            ContextVector encoding the semantic constraints.
        """
        ...

    @abstractmethod
    def generate_reward_tensor(
        self,
        command: str,
        current_context: ContextVector | None = None,
    ) -> dict[str, float]:
        """Generate a reward tensor from a language command.

        Maps natural language constraints to mathematical reward weights.
        For example:
            "Pour slowly" → {"jerk_penalty": 0.9, "velocity_penalty": 0.8, ...}

        Args:
            command: Natural language constraint.
            current_context: Existing context to update.

        Returns:
            Dictionary of reward component weights.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""
        ...
