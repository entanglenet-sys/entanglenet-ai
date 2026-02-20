"""
Simulation-level domain randomizer that interfaces with Isaac Sim.

Extends the learning-level DomainRandomizer with Omniverse-specific
APIs for modifying physics materials, lighting, and sensor properties.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from philos.learning.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizer,
)

logger = logging.getLogger(__name__)

# Check for Omniverse/PhysX availability
_OMNI_AVAILABLE = False
try:
    import omni.isaac.core.utils.prims as prim_utils  # type: ignore
    _OMNI_AVAILABLE = True
except ImportError:
    pass


class SimDomainRandomizer(DomainRandomizer):
    """Domain randomizer with NVIDIA Isaac Sim integration.

    Extends the base randomizer to apply parameters directly to
    Omniverse USD prims and PhysX materials at the start of each
    episode.
    """

    def __init__(
        self,
        config: DomainRandomizationConfig | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(config=config, seed=seed)

    def apply_to_env(self, env: Any, params: dict[str, float] | None = None) -> None:
        """Apply randomized parameters to an Isaac Sim environment.

        If Omniverse is available, modifies USD prims directly.
        Otherwise, logs the parameters for debugging.
        """
        if params is None:
            params = self.sample()

        if _OMNI_AVAILABLE:
            self._apply_physics_params(env, params)
            self._apply_visual_params(env, params)
            self._apply_sensor_noise(env, params)
        else:
            logger.debug(f"Stub domain randomization: {params}")

        # Store on the environment for reward/observation access
        if hasattr(env, "_dr_params"):
            env._dr_params = params

    def _apply_physics_params(self, env: Any, params: dict[str, float]) -> None:
        """Apply physics parameters via USD / PhysX API."""
        # Fluid viscosity
        if "fluid_viscosity" in params and hasattr(env, "_fluid_prim_path"):
            try:
                prim_utils.set_prim_property(
                    env._fluid_prim_path,
                    "physxParticle:viscosity",
                    params["fluid_viscosity"],
                )
            except Exception as e:
                logger.warning(f"Failed to set fluid viscosity: {e}")

        # Ground friction
        if "ground_friction" in params and hasattr(env, "_ground_prim_path"):
            try:
                prim_utils.set_prim_property(
                    env._ground_prim_path,
                    "physics:dynamicFriction",
                    params["ground_friction"],
                )
                prim_utils.set_prim_property(
                    env._ground_prim_path,
                    "physics:staticFriction",
                    params["ground_friction"] * 1.2,
                )
            except Exception as e:
                logger.warning(f"Failed to set ground friction: {e}")

        # Object mass
        if "object_mass" in params and hasattr(env, "_object_prim_paths"):
            for prim_path in env._object_prim_paths:
                try:
                    prim_utils.set_prim_property(
                        prim_path, "physics:mass", params["object_mass"]
                    )
                except Exception:
                    pass

        # Joint friction/damping
        if hasattr(env, "_robot_prim_path"):
            for i in range(6):
                joint_path = f"{env._robot_prim_path}/joint_{i}"
                if "joint_friction" in params:
                    try:
                        prim_utils.set_prim_property(
                            joint_path,
                            "physxJoint:jointFriction",
                            params["joint_friction"],
                        )
                    except Exception:
                        pass
                if "joint_damping" in params:
                    try:
                        prim_utils.set_prim_property(
                            joint_path,
                            "drive:angular:physics:damping",
                            params["joint_damping"],
                        )
                    except Exception:
                        pass

    def _apply_visual_params(self, env: Any, params: dict[str, float]) -> None:
        """Randomize lighting and visual appearance."""
        if "light_intensity" in params and hasattr(env, "_light_prim_path"):
            try:
                prim_utils.set_prim_property(
                    env._light_prim_path,
                    "inputs:intensity",
                    params["light_intensity"] * 1000,  # Convert to lumens
                )
            except Exception:
                pass

    def _apply_sensor_noise(self, env: Any, params: dict[str, float]) -> None:
        """Configure sensor noise levels."""
        if hasattr(env, "_camera_noise_std"):
            env._camera_noise_std = params.get("camera_noise_std", 0.01)
        if hasattr(env, "_lidar_noise_std"):
            env._lidar_noise_std = params.get("lidar_noise_std", 0.005)
        if hasattr(env, "_encoder_noise_std"):
            env._encoder_noise_std = params.get("joint_encoder_noise", 0.001)
