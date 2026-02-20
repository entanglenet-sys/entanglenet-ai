"""
VLM Semantic Grounding — System 2 (1Hz Reasoning Loop).

This is the "slow brain" of the PHILOS architecture. It uses a fine-tuned
Vision-Language Model (LLaVA-Next) to analyze scenes at a deeper semantic level
and produce the Latent Context Vector (z) that conditions the RL policy.

Key capabilities:
    - Scene-level semantic understanding ("cracked beaker" vs "dirty beaker")
    - Hazard classification ("this contains acid")
    - Context Vector generation for the RL policy

Maps to:
    WP2 T2.1 (VLM Fine-Tuning)
    System 2 layer: low-frequency reasoning and context generation
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from philos.core.context_vector import ContextVector, ManipulationMode
from philos.core.registry import register_component
from philos.perception.base import BasePerception, Detection, PerceptionOutput

logger = logging.getLogger(__name__)


@register_component("perception", "vlm_grounding")
class VLMGrounding(BasePerception):
    """Vision-Language Model for semantic scene understanding.

    Fine-tuned on industrial/chemical environments. Produces a Latent Context
    Vector (z) rather than textual output — this is the key engineering innovation
    described in the PHILOS proposal (Section 1.3.1).
    """

    def __init__(
        self,
        model_name: str = "llava-next-7b",
        device: str = "cuda:0",
        context_vector_dim: int = 256,
        temperature: float = 0.1,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._context_dim = context_vector_dim
        self._temperature = temperature
        self._model: Any = None
        self._processor: Any = None

    @property
    def name(self) -> str:
        return "vlm_grounding"

    @property
    def hz(self) -> float:
        return 1.0  # System 2 operates at 1Hz

    def initialize(self, config: Any = None) -> None:
        """Load the VLM model.

        Note: Requires `transformers` package with LLaVA-Next support.
        """
        try:
            from transformers import AutoProcessor, LlavaNextForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(
                f"llava-hf/{self._model_name}-hf"
            )
            self._model = LlavaNextForConditionalGeneration.from_pretrained(
                f"llava-hf/{self._model_name}-hf",
                torch_dtype="auto",
                device_map=self._device,
            )
            logger.info(f"VLM '{self._model_name}' loaded on {self._device}")
        except (ImportError, Exception) as e:
            logger.warning(
                f"VLM model could not be loaded ({e}). Running in stub mode."
            )
            self._model = None

    def detect(self, rgb: np.ndarray, depth: np.ndarray | None = None) -> PerceptionOutput:
        """Run VLM scene analysis and produce semantic detections.

        Unlike YOLO-World (System 1), this doesn't produce bounding boxes.
        Instead, it provides scene-level semantic understanding.
        """
        start = time.monotonic()

        # In production, the VLM analyzes the scene and produces semantic output
        # For the base implementation, we return a placeholder
        scene_embedding = self._encode_scene(rgb)

        output = PerceptionOutput(
            detections=[],
            scene_embedding=scene_embedding,
            depth_map=depth,
            timestamp=time.time(),
            processing_time_ms=(time.monotonic() - start) * 1000,
        )

        return output

    def generate_context_vector(
        self,
        rgb: np.ndarray,
        language_command: str = "",
        depth: np.ndarray | None = None,
    ) -> ContextVector:
        """Generate a Context Vector (z) from visual scene + language command.

        This is the core "Language-to-Reward" interface. The VLM processes the
        scene and command to produce a Latent Context Vector that encodes:
        - Manipulation mode (stiff/compliant/fluid/precision)
        - Impedance parameters
        - Velocity constraints
        - Orientation constraints
        - Jerk penalties

        Args:
            rgb: Current camera frame (H, W, 3).
            language_command: Natural language instruction from operator.
            depth: Optional depth image.

        Returns:
            ContextVector conditioned on the scene and command.
        """
        start = time.monotonic()

        if self._model is not None:
            # Production path: use VLM to analyze scene + command
            context = self._vlm_inference(rgb, language_command)
        else:
            # Stub path: heuristic-based context generation
            context = self._heuristic_context(language_command)

        context.timestamp = time.time()
        latency = (time.monotonic() - start) * 1000
        logger.info(
            f"Context vector generated in {latency:.1f}ms "
            f"(mode={context.manipulation_mode.value}, cmd='{language_command}')"
        )

        return context

    def _vlm_inference(self, rgb: np.ndarray, command: str) -> ContextVector:
        """Run actual VLM inference to produce a context vector.

        In the fine-tuned model, we suppress textual output and instead
        extract the latent representation as the Context Vector.
        """
        # This would use the fine-tuned VLM to project multimodal context
        # into a latent vector. For now, return heuristic-based output.
        return self._heuristic_context(command)

    def _heuristic_context(self, command: str) -> ContextVector:
        """Generate a context vector using keyword heuristics.

        This is the fallback when the VLM is not loaded. Maps common
        language patterns to manipulation parameters.
        """
        command_lower = command.lower()

        # Default context
        mode = ManipulationMode.STIFF
        impedance = 0.5
        velocity = 1.0
        jerk_penalty = 0.1
        labels: list[str] = []

        # Keyword-based mapping (replaced by VLM in production)
        if any(w in command_lower for w in ["careful", "gently", "fragile", "delicate"]):
            mode = ManipulationMode.COMPLIANT
            impedance = 0.8
            velocity = 0.3
            jerk_penalty = 0.8

        if any(w in command_lower for w in ["pour", "liquid", "fluid", "water"]):
            mode = ManipulationMode.FLUID
            impedance = 0.7
            velocity = 0.4
            jerk_penalty = 0.9

        if any(w in command_lower for w in ["precise", "exact", "assembly", "insert"]):
            mode = ManipulationMode.PRECISION
            impedance = 0.9
            velocity = 0.2
            jerk_penalty = 0.5

        if any(w in command_lower for w in ["fast", "quick", "hurry"]):
            velocity = min(velocity + 0.4, 1.0)
            jerk_penalty = max(jerk_penalty - 0.3, 0.0)

        if any(w in command_lower for w in ["hazard", "acid", "danger", "toxic"]):
            labels.append("hazardous")
            velocity = min(velocity, 0.3)
            impedance = max(impedance, 0.8)

        return ContextVector(
            embedding=np.random.randn(self._context_dim).astype(np.float32) * 0.1,
            manipulation_mode=mode,
            impedance_scale=impedance,
            velocity_limit_scale=velocity,
            jerk_penalty=jerk_penalty,
            semantic_labels=labels,
            language_command=command,
            confidence=0.7 if command else 1.0,
        )

    def _encode_scene(self, rgb: np.ndarray) -> np.ndarray:
        """Encode the visual scene into a feature vector."""
        # Placeholder: return a random embedding
        # In production, this comes from the VLM's visual encoder
        return np.random.randn(self._context_dim).astype(np.float32) * 0.1

    def shutdown(self) -> None:
        self._model = None
        self._processor = None
        logger.info("VLM Grounding shut down")
