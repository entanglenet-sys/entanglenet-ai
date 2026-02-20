"""
PHILOS Control Module — Safety Shield & Whole-Body MPC.

Maps to WP3 T3.3 — Deterministic Safety Shield that can override any
AI-generated command to guarantee physical safety.

Architecture:
    RL Policy → Safety Shield (50 Hz) → MPC Solver → Actuator Commands

Safety invariants (hard constraints):
    - Platform tilt < 10 degrees
    - End-effector velocity < 1.5 m/s
    - Joint torques within rated limits
    - Collision distance > safety margin
"""

from philos.control.base import BaseController
from philos.control.safety_shield import SafetyShield
from philos.control.mpc_solver import WholeBodyMPC
from philos.control.trajectory import TrajectoryOptimizer

__all__ = [
    "BaseController",
    "SafetyShield",
    "WholeBodyMPC",
    "TrajectoryOptimizer",
]
