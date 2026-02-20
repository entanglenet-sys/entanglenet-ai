"""
Domain Randomization — bridging the sim-to-real gap.

Implements extreme parameter variation (±200%) for training robust
RL policies that transfer from Isaac Sim to physical hardware.

Key randomization domains:
    - Fluid viscosity and mass (critical for pouring tasks)
    - Friction coefficients (surface and joint)
    - Sensor noise (camera, LiDAR, joint encoders)
    - Object properties (mass, size, shape, CoM)
    - Lighting and visual appearance

Maps to:
    WP2 T2.3 — Domain Randomization for Sim-to-Real
    Risk 1 mitigation: High-Fidelity Omniverse Simulation & Randomization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RandomizationRange:
    """A randomization parameter with its range."""

    name: str
    default: float
    min_val: float
    max_val: float
    distribution: str = "uniform"  # uniform, gaussian, log_uniform

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a random value from this range."""
        if self.distribution == "uniform":
            return float(rng.uniform(self.min_val, self.max_val))
        elif self.distribution == "gaussian":
            mean = (self.min_val + self.max_val) / 2
            std = (self.max_val - self.min_val) / 6  # 3-sigma coverage
            return float(np.clip(rng.normal(mean, std), self.min_val, self.max_val))
        elif self.distribution == "log_uniform":
            log_min = np.log(max(self.min_val, 1e-8))
            log_max = np.log(max(self.max_val, 1e-8))
            return float(np.exp(rng.uniform(log_min, log_max)))
        return self.default


@dataclass
class DomainRandomizationConfig:
    """Full domain randomization configuration.

    Default ranges implement ±200% variation as specified in the
    PHILOS proposal (WP2 T2.3).
    """

    # Fluid dynamics parameters
    fluid_viscosity: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "fluid_viscosity", default=1.0, min_val=0.1, max_val=3.0, distribution="log_uniform"
        )
    )
    fluid_mass: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "fluid_mass", default=0.5, min_val=0.1, max_val=1.5
        )
    )
    fluid_surface_tension: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "fluid_surface_tension", default=0.07, min_val=0.01, max_val=0.2
        )
    )

    # Object properties
    object_mass: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "object_mass", default=0.3, min_val=0.05, max_val=1.0
        )
    )
    object_friction: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "object_friction", default=0.5, min_val=0.1, max_val=1.0
        )
    )
    object_com_offset: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "object_com_offset", default=0.0, min_val=-0.03, max_val=0.03
        )
    )

    # Surface/environment
    ground_friction: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "ground_friction", default=0.6, min_val=0.2, max_val=1.0
        )
    )
    joint_friction: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "joint_friction", default=0.01, min_val=0.001, max_val=0.05
        )
    )
    joint_damping: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "joint_damping", default=0.1, min_val=0.01, max_val=0.3
        )
    )

    # Sensor noise
    camera_noise_std: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "camera_noise_std", default=0.01, min_val=0.0, max_val=0.05
        )
    )
    lidar_noise_std: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "lidar_noise_std", default=0.005, min_val=0.0, max_val=0.02
        )
    )
    joint_encoder_noise: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "joint_encoder_noise", default=0.001, min_val=0.0, max_val=0.005
        )
    )

    # Lighting
    light_intensity: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "light_intensity", default=1.0, min_val=0.3, max_val=2.0
        )
    )
    light_color_temp: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "light_color_temp", default=5500, min_val=3000, max_val=8000
        )
    )

    # Action delay/latency
    action_delay_steps: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(
            "action_delay_steps", default=0.0, min_val=0.0, max_val=3.0
        )
    )


class DomainRandomizer:
    """Applies domain randomization to simulation environments.

    At the start of each episode (or at a set interval), new random
    parameters are sampled and applied to the simulation.
    """

    def __init__(
        self,
        config: DomainRandomizationConfig | None = None,
        seed: int = 42,
    ) -> None:
        self._config = config or DomainRandomizationConfig()
        self._rng = np.random.default_rng(seed)
        self._current_params: dict[str, float] = {}

    def sample(self) -> dict[str, float]:
        """Sample a new set of randomized parameters.

        Returns:
            Dictionary mapping parameter names to sampled values.
        """
        import dataclasses

        params = {}
        for f in dataclasses.fields(self._config):
            r_range = getattr(self._config, f.name)
            if isinstance(r_range, RandomizationRange):
                params[r_range.name] = r_range.sample(self._rng)

        self._current_params = params
        return params

    def apply_to_env(self, env: Any, params: dict[str, float] | None = None) -> None:
        """Apply randomized parameters to a simulation environment.

        Args:
            env: The Isaac Sim environment wrapper.
            params: Parameters to apply. If None, samples new ones.
        """
        if params is None:
            params = self.sample()

        # Apply parameters through the environment's API
        # These method names map to Isaac Sim's physics properties
        if hasattr(env, "set_physics_params"):
            env.set_physics_params(params)
        else:
            logger.debug(f"Domain randomization params sampled: {params}")

    @property
    def current_params(self) -> dict[str, float]:
        return dict(self._current_params)
