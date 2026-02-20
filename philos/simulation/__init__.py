"""
PHILOS Simulation Module — NVIDIA Isaac Sim Environments.

Provides Gymnasium-compatible environments backed by NVIDIA Isaac Sim
/ Omniverse for high-fidelity physics simulation with PhysX 5.

Key environments (mapping to WP4 validation tasks):
    - PourTask      (T4.2 "The Sommelier")
    - FetchSortTask (T4.3 "The Courier")
    - NavigationTask (general AMR navigation)

Features:
    - Domain randomization (±200%) via DomainRandomizer
    - Parallel environments (64+) for massively parallel RL
    - PhysX 5 fluid simulation for pouring tasks
    - Camera / LiDAR sensor simulation
"""

from philos.simulation.isaac_env import IsaacSimEnv
from philos.simulation.domain_randomizer import SimDomainRandomizer
from philos.simulation.environments import PourTaskEnv, FetchSortTaskEnv

__all__ = [
    "IsaacSimEnv",
    "SimDomainRandomizer",
    "PourTaskEnv",
    "FetchSortTaskEnv",
]
