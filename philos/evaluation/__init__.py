"""
PHILOS Evaluation Module — benchmarks and metrics.

Implements the evaluation protocol from WP4:
    - Task success rates
    - Sim-to-real transfer gap (TRL progression)
    - Safety metrics (spill rate, collision rate, tilt violations)
    - Latency metrics (control loop, perception, VLM)

Produces structured results compatible with the PHILOS API.
"""

from philos.evaluation.benchmarks import BenchmarkRunner
from philos.evaluation.metrics import PhilosMetrics

__all__ = ["BenchmarkRunner", "PhilosMetrics"]
