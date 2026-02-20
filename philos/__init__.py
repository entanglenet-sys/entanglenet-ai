"""
PHILOS — PHysical IntelLigence, Learning, Optimization, and Semantics

A Foundation Model for Adaptive Mobile Manipulation in Unstructured Industrial Environments.

Architecture Overview (Dual-Loop / System 1 + System 2):
    System 1 (50Hz Reflex Loop): YOLO-World + Whole-Body MPC → deterministic safety
    System 2 (1Hz Reasoning Loop): VLM → Latent Context Vector (z) → conditions RL policy

Pipeline:
    [Semantic Grounding] → [Context Vector z] → [RL Policy π(s,z)] → [Safety Shield / MPC] → [Actuators]

© 2025 Entanglenet GmbH — Vienna, Austria
"""

__version__ = "0.1.0"
__author__ = "Entanglenet GmbH"
