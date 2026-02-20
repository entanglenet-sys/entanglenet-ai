"""Core module — shared data structures, configuration, and component registry."""

from philos.core.config import PhilosConfig, load_config
from philos.core.context_vector import ContextVector
from philos.core.registry import ComponentRegistry
from philos.core.state import RobotState, JointState, SensorReading

__all__ = [
    "PhilosConfig",
    "load_config",
    "ContextVector",
    "ComponentRegistry",
    "RobotState",
    "JointState",
    "SensorReading",
]
