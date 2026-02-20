"""
Semantic Reward Shaping — The core PHILOS innovation.

Translates natural language constraints into mathematical reward tensors
for the RL policy. This is the "Language-to-Reward" interface described
in WP2 T2.2 of the PHILOS proposal.

The key insight: reward functions are NOT static. They are dynamically
modified by human language input, allowing real-time adaptation of
robot behavior without retraining.

Target: 50 distinct language constraints → reward tensors, <500ms latency.

Maps to:
    WP2 T2.2 (Language-to-Reward Interface)
    D2.1 (Semantic Reward Shaping Core Codebase) — due M6
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from philos.cognitive.base import BaseCognitiveEngine
from philos.core.context_vector import ContextVector, ManipulationMode
from philos.core.registry import register_component

logger = logging.getLogger(__name__)


# ─── Reward Components ──────────────────────────────────────────────────────────

# These are the individual reward terms that can be modulated by language.
# Each maps to a specific physical behavior of the robot.

REWARD_COMPONENTS = {
    "task_completion": "Reward for completing the primary task objective",
    "position_error": "Penalty for end-effector position error from target",
    "orientation_error": "Penalty for orientation deviation (critical for fluids)",
    "velocity_penalty": "Penalty for high velocities (safety-relevant)",
    "jerk_penalty": "Penalty for jerky motion (smoothness)",
    "collision_penalty": "Penalty for collisions with obstacles/environment",
    "spill_penalty": "Penalty for fluid spillage (specific to pouring tasks)",
    "force_penalty": "Penalty for excessive contact forces (fragile objects)",
    "energy_efficiency": "Reward for energy-efficient motion",
    "time_penalty": "Penalty for slow task completion",
    "safety_margin": "Reward for maintaining safety distance from constraints",
    "grip_stability": "Reward for stable grasping (no slippage)",
    "fluid_stability": "Reward for minimizing fluid slosh in containers",
    "path_efficiency": "Reward for efficient navigation paths",
}

# Default reward weights
DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "task_completion": 1.0,
    "position_error": -0.5,
    "orientation_error": -0.3,
    "velocity_penalty": -0.1,
    "jerk_penalty": -0.1,
    "collision_penalty": -10.0,
    "spill_penalty": -5.0,
    "force_penalty": -0.2,
    "energy_efficiency": 0.05,
    "time_penalty": -0.01,
    "safety_margin": 0.1,
    "grip_stability": 0.3,
    "fluid_stability": -0.5,
    "path_efficiency": 0.1,
}


# ─── Language-to-Reward Mapping Rules ────────────────────────────────────────────

# Each rule maps a set of language patterns to reward weight modifications.
# In production, these are learned by the fine-tuned VLM. Here they serve
# as the baseline heuristic implementation.

LANGUAGE_REWARD_RULES: list[dict[str, Any]] = [
    {
        "patterns": ["carefully", "gently", "soft", "delicate"],
        "modifications": {
            "velocity_penalty": -0.8,
            "jerk_penalty": -0.9,
            "force_penalty": -0.8,
            "time_penalty": -0.001,  # Relax time pressure
        },
        "mode": ManipulationMode.COMPLIANT,
    },
    {
        "patterns": ["pour", "fill", "transfer liquid", "decant"],
        "modifications": {
            "orientation_error": -0.9,
            "spill_penalty": -10.0,
            "fluid_stability": -2.0,
            "jerk_penalty": -0.8,
        },
        "mode": ManipulationMode.FLUID,
    },
    {
        "patterns": ["fast", "quickly", "hurry", "speed"],
        "modifications": {
            "velocity_penalty": -0.01,
            "time_penalty": -0.1,
            "jerk_penalty": -0.02,
        },
        "mode": None,  # Don't change mode
    },
    {
        "patterns": ["precise", "exact", "accurate", "insert", "assemble"],
        "modifications": {
            "position_error": -2.0,
            "velocity_penalty": -0.5,
            "jerk_penalty": -0.5,
        },
        "mode": ManipulationMode.PRECISION,
    },
    {
        "patterns": ["hazard", "danger", "toxic", "acid", "radiation"],
        "modifications": {
            "collision_penalty": -20.0,
            "spill_penalty": -20.0,
            "safety_margin": 0.5,
            "velocity_penalty": -0.6,
        },
        "mode": ManipulationMode.COMPLIANT,
    },
    {
        "patterns": ["avoid", "stay away", "don't touch", "keep distance"],
        "modifications": {
            "collision_penalty": -15.0,
            "safety_margin": 0.8,
        },
        "mode": None,
    },
    {
        "patterns": ["steady", "stable", "don't spill", "level"],
        "modifications": {
            "orientation_error": -1.5,
            "fluid_stability": -3.0,
            "jerk_penalty": -0.7,
        },
        "mode": ManipulationMode.FLUID,
    },
    {
        "patterns": ["grip", "hold tight", "secure", "clamp"],
        "modifications": {
            "grip_stability": 1.0,
            "force_penalty": -0.05,  # Allow more force for gripping
        },
        "mode": ManipulationMode.STIFF,
    },
]


@register_component("cognitive", "reward_shaping")
class SemanticRewardShaping(BaseCognitiveEngine):
    """Semantic Reward Shaping Engine.

    The core deliverable of WP2 — translates natural language into
    reward tensors that condition the RL policy in real-time.

    Architecture:
        1. Parse natural language command
        2. Match against reward rules (or use VLM for learned mapping)
        3. Generate reward weight modifications
        4. Produce Context Vector (z) with encoded constraints
        5. Publish to the RL policy via API
    """

    def __init__(
        self,
        context_vector_dim: int = 256,
        max_constraints: int = 50,
        latency_budget_ms: float = 500.0,
    ) -> None:
        self._context_dim = context_vector_dim
        self._max_constraints = max_constraints
        self._latency_budget = latency_budget_ms
        self._current_weights = dict(DEFAULT_REWARD_WEIGHTS)
        self._constraint_count = 0

    def initialize(self, config: Any = None) -> None:
        """Initialize the reward shaping engine."""
        self._current_weights = dict(DEFAULT_REWARD_WEIGHTS)
        self._constraint_count = 0
        logger.info(
            f"SemanticRewardShaping initialized "
            f"(dim={self._context_dim}, max_constraints={self._max_constraints})"
        )

    def process_command(
        self,
        command: str,
        scene_rgb: np.ndarray | None = None,
        scene_depth: np.ndarray | None = None,
    ) -> ContextVector:
        """Process a command and generate a Context Vector.

        This is the primary entry point for the "Language-to-Reward" pipeline.
        """
        start = time.monotonic()

        # Step 1: Generate reward tensor from command
        reward_mods = self.generate_reward_tensor(command)

        # Step 2: Apply modifications to current weights
        for key, value in reward_mods.items():
            if key in self._current_weights:
                self._current_weights[key] = value

        # Step 3: Determine manipulation mode
        mode = self._determine_mode(command)

        # Step 4: Encode into Context Vector
        context = self._encode_to_context(command, mode, reward_mods)

        latency = (time.monotonic() - start) * 1000
        if latency > self._latency_budget:
            logger.warning(
                f"Reward shaping latency {latency:.1f}ms exceeds budget "
                f"{self._latency_budget}ms"
            )

        self._constraint_count += 1
        logger.info(
            f"Constraint #{self._constraint_count}: '{command}' → "
            f"mode={mode.value}, latency={latency:.1f}ms"
        )

        return context

    def generate_reward_tensor(
        self,
        command: str,
        current_context: ContextVector | None = None,
    ) -> dict[str, float]:
        """Map a natural language command to reward weight modifications.

        Scans the command against the language-to-reward rules and
        produces a dictionary of reward component weights.
        """
        command_lower = command.lower()
        modifications: dict[str, float] = {}

        for rule in LANGUAGE_REWARD_RULES:
            if any(pattern in command_lower for pattern in rule["patterns"]):
                modifications.update(rule["modifications"])

        if not modifications:
            logger.debug(f"No reward rules matched for: '{command}'")

        return modifications

    def get_current_weights(self) -> dict[str, float]:
        """Get the current reward weight configuration."""
        return dict(self._current_weights)

    def reset_weights(self) -> None:
        """Reset reward weights to defaults."""
        self._current_weights = dict(DEFAULT_REWARD_WEIGHTS)
        self._constraint_count = 0
        logger.info("Reward weights reset to defaults")

    def _determine_mode(self, command: str) -> ManipulationMode:
        """Determine the manipulation mode from a command."""
        command_lower = command.lower()

        for rule in LANGUAGE_REWARD_RULES:
            if rule["mode"] is not None:
                if any(p in command_lower for p in rule["patterns"]):
                    return rule["mode"]

        return ManipulationMode.STIFF

    def _encode_to_context(
        self,
        command: str,
        mode: ManipulationMode,
        reward_mods: dict[str, float],
    ) -> ContextVector:
        """Encode the command and reward modifications into a Context Vector."""
        # Create a deterministic embedding from the reward modifications
        # In production, this comes from the fine-tuned VLM encoder
        embedding = np.zeros(self._context_dim, dtype=np.float32)

        # Encode reward modifications into the embedding
        for i, (key, value) in enumerate(sorted(reward_mods.items())):
            if i < self._context_dim:
                embedding[i] = value

        # Derive scalar parameters from reward weights
        velocity_scale = max(
            0.1,
            1.0 + self._current_weights.get("velocity_penalty", -0.1) * 0.5,
        )
        impedance_scale = max(
            0.1,
            abs(self._current_weights.get("force_penalty", -0.2)),
        )
        jerk_penalty = abs(self._current_weights.get("jerk_penalty", -0.1))

        return ContextVector(
            embedding=embedding,
            manipulation_mode=mode,
            impedance_scale=min(impedance_scale, 1.0),
            velocity_limit_scale=min(velocity_scale, 1.0),
            jerk_penalty=min(jerk_penalty, 1.0),
            language_command=command,
            confidence=0.9,
            timestamp=time.time(),
        )

    def shutdown(self) -> None:
        logger.info("SemanticRewardShaping shut down")
