"""
Configuration management for the PHILOS system.

Supports hierarchical YAML configs with OmegaConf and runtime overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SimulationConfig:
    """Isaac Sim simulation parameters."""

    backend: str = "isaac_sim"
    physics_dt: float = 1.0 / 120.0  # 120 Hz physics
    rendering_dt: float = 1.0 / 30.0  # 30 Hz rendering
    num_envs: int = 64  # Parallel environments for RL training
    gpu_id: int = 0
    headless: bool = False
    domain_randomization: bool = True
    fluid_simulation: bool = True  # PhysX 5 particle solver


@dataclass
class PerceptionConfig:
    """Perception stack configuration."""

    yolo_model: str = "yolo-world-l"
    yolo_confidence: float = 0.3
    vlm_model: str = "llava-next-7b"
    vlm_device: str = "cuda:0"
    rgbd_topic: str = "/camera/rgbd"
    lidar_topic: str = "/lidar/scan"
    fusion_method: str = "late_fusion"
    system1_hz: float = 50.0  # Fast reflex loop
    system2_hz: float = 1.0  # Slow reasoning loop


@dataclass
class CognitiveConfig:
    """Cognitive engine / Semantic Grounding configuration."""

    context_vector_dim: int = 256
    reward_shaping_latency_budget_ms: float = 500.0
    max_language_constraints: int = 50
    vlm_fine_tune_dataset: str = "philos-industrial-5k"
    temperature: float = 0.1  # Low temperature for deterministic outputs


@dataclass
class LearningConfig:
    """RL training configuration."""

    algorithm: str = "PPO"  # PPO, SAC, TD3
    policy_type: str = "whole_body"  # manipulation, navigation, whole_body
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    batch_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    domain_randomization_range: float = 2.0  # ±200% parameter variation
    context_conditioned: bool = True  # Policy π(s, z) conditioned on context vector
    checkpoint_dir: str = "checkpoints/"


@dataclass
class ControlConfig:
    """Safety Shield / MPC configuration."""

    mpc_horizon: int = 20
    mpc_dt: float = 0.02  # 50 Hz
    max_ee_velocity: float = 1.5  # m/s
    max_ee_acceleration: float = 5.0  # m/s²
    max_tilt_angle_deg: float = 10.0  # Max tilt for fluid containers
    max_joint_velocity: list[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]  # 6-DoF arm
    )
    collision_margin: float = 0.05  # 5 cm safety margin
    enable_safety_shield: bool = True
    deterministic_override: bool = True  # MPC overrides RL if unsafe


@dataclass
class ROS2Config:
    """ROS2 bridge configuration."""

    enable: bool = False  # Disabled in simulation-only mode
    middleware: str = "rmw_cyclonedds_cpp"
    namespace: str = "/philos"
    cmd_vel_topic: str = "/cmd_vel"
    joint_command_topic: str = "/joint_commands"
    safety_status_topic: str = "/safety_shield/status"
    bridge_latency_budget_ms: float = 10.0  # <10ms ROS2-to-motor requirement


@dataclass
class APIConfig:
    """REST API configuration for inter-module communication."""

    host: str = "0.0.0.0"
    port: int = 8000
    enable_cors: bool = True
    log_level: str = "info"
    websocket_enabled: bool = True  # For real-time telemetry streaming


@dataclass
class PhilosConfig:
    """Root configuration for the entire PHILOS system."""

    project_name: str = "PHILOS"
    version: str = "0.1.0"
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    ros2: ROS2Config = field(default_factory=ROS2Config)
    api: APIConfig = field(default_factory=APIConfig)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> PhilosConfig:
    """Load configuration from YAML file with optional runtime overrides.

    Args:
        path: Path to a YAML config file. If None, returns defaults.
        overrides: Dictionary of dot-separated key overrides, e.g.
                   {"learning.algorithm": "SAC", "simulation.num_envs": 128}

    Returns:
        A fully populated PhilosConfig instance.
    """
    config = PhilosConfig()

    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            config = _merge_dict_into_config(config, raw)

    if overrides:
        config = _apply_overrides(config, overrides)

    return config


def _merge_dict_into_config(config: PhilosConfig, raw: dict) -> PhilosConfig:
    """Merge a raw dictionary into a PhilosConfig, matching field names."""
    import dataclasses

    for section_name, section_dict in raw.items():
        if hasattr(config, section_name) and isinstance(section_dict, dict):
            section = getattr(config, section_name)
            if dataclasses.is_dataclass(section):
                for k, v in section_dict.items():
                    if hasattr(section, k):
                        setattr(section, k, v)
        elif hasattr(config, section_name):
            setattr(config, section_name, section_dict)
    return config


def _apply_overrides(config: PhilosConfig, overrides: dict[str, Any]) -> PhilosConfig:
    """Apply dot-separated overrides like 'learning.algorithm' = 'SAC'."""
    for key, value in overrides.items():
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return config
