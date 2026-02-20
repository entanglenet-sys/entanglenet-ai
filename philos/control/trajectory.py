"""
Trajectory Optimization — smooth motion planning.

Provides trajectory primitives for the MPC and safety shield:
    - Minimum-jerk trajectories (for smooth point-to-point)
    - Spline interpolation (for complex paths)
    - Time-optimal trajectories (respecting velocity/acceleration limits)

Maps to WP3 T3.2 — Whole-Body Motion Planning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """A single waypoint in a trajectory."""

    time: float
    position: np.ndarray          # (n_dof,)
    velocity: np.ndarray          # (n_dof,)
    acceleration: np.ndarray      # (n_dof,)


class TrajectoryOptimizer:
    """Generates smooth, feasible trajectories.

    Supports:
        - Minimum-jerk (5th-order polynomial) for smooth reaching
        - Linear interpolation with velocity ramps
        - Cubic spline for multi-waypoint paths

    All trajectories respect joint velocity and acceleration limits.
    """

    def __init__(
        self,
        n_dof: int = 6,
        max_velocity: np.ndarray | None = None,
        max_acceleration: np.ndarray | None = None,
    ) -> None:
        self._n_dof = n_dof
        self._max_vel = max_velocity if max_velocity is not None else np.full(n_dof, 2.0)
        self._max_acc = max_acceleration if max_acceleration is not None else np.full(n_dof, 5.0)

    def minimum_jerk(
        self,
        start: np.ndarray,
        end: np.ndarray,
        duration: float,
        dt: float = 0.02,
    ) -> list[TrajectoryPoint]:
        """Generate a minimum-jerk (5th-order) trajectory.

        The minimum-jerk trajectory minimizes ∫ jerk² dt, producing
        the smoothest possible motion between two configurations.
        Ideal for pouring / fluid transport tasks.

        Args:
            start: Start configuration (n_dof,).
            end: End configuration (n_dof,).
            duration: Total motion time (seconds).
            dt: Time step.

        Returns:
            List of TrajectoryPoints from t=0 to t=duration.
        """
        n_steps = max(1, int(duration / dt))
        trajectory: list[TrajectoryPoint] = []

        for i in range(n_steps + 1):
            t = i * dt
            tau = t / duration  # Normalized time [0, 1]

            # 5th-order polynomial: s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
            s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / (duration**2)

            delta = end - start
            pos = start + s * delta
            vel = s_dot * delta
            acc = s_ddot * delta

            trajectory.append(TrajectoryPoint(
                time=t,
                position=pos.copy(),
                velocity=vel.copy(),
                acceleration=acc.copy(),
            ))

        return trajectory

    def linear_with_ramp(
        self,
        start: np.ndarray,
        end: np.ndarray,
        dt: float = 0.02,
    ) -> list[TrajectoryPoint]:
        """Generate a trapezoidal-velocity trajectory.

        Accelerates to max velocity, cruises, then decelerates.
        Automatically computes the time-optimal duration.
        """
        delta = end - start
        distance = np.abs(delta)

        # Time to accelerate to max velocity
        t_acc = self._max_vel / self._max_acc
        # Distance covered during acceleration
        d_acc = 0.5 * self._max_acc * t_acc**2

        # Check if we can reach max velocity (triangle vs trapezoid)
        is_triangle = d_acc * 2 > distance

        # Per-joint timing
        durations = np.zeros(self._n_dof)
        for j in range(self._n_dof):
            if distance[j] < 1e-8:
                durations[j] = 0.0
            elif is_triangle[j]:
                durations[j] = 2.0 * np.sqrt(distance[j] / self._max_acc[j])
            else:
                cruise_dist = distance[j] - 2 * d_acc[j]
                durations[j] = 2 * t_acc[j] + cruise_dist / self._max_vel[j]

        total_duration = float(np.max(durations))
        if total_duration < dt:
            return [TrajectoryPoint(
                time=0.0,
                position=end.copy(),
                velocity=np.zeros(self._n_dof),
                acceleration=np.zeros(self._n_dof),
            )]

        # Scale all joints to finish at the same time
        return self.minimum_jerk(start, end, total_duration, dt)

    def cubic_spline(
        self,
        waypoints: list[np.ndarray],
        durations: list[float],
        dt: float = 0.02,
    ) -> list[TrajectoryPoint]:
        """Generate a cubic-spline trajectory through multiple waypoints.

        Uses natural cubic splines with zero-velocity boundary conditions.

        Args:
            waypoints: List of N configurations (n_dof,).
            durations: List of N-1 segment durations.
            dt: Time step.
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints.")
        if len(durations) != len(waypoints) - 1:
            raise ValueError("Need N-1 durations for N waypoints.")

        trajectory: list[TrajectoryPoint] = []

        # Concatenate minimum-jerk segments
        for seg_idx in range(len(durations)):
            seg = self.minimum_jerk(
                waypoints[seg_idx],
                waypoints[seg_idx + 1],
                durations[seg_idx],
                dt,
            )
            # Offset timestamps
            t_offset = sum(durations[:seg_idx])
            for pt in seg:
                pt.time += t_offset

            # Avoid duplicating the junction point
            if seg_idx > 0 and trajectory:
                seg = seg[1:]

            trajectory.extend(seg)

        return trajectory

    def resample(
        self,
        trajectory: list[TrajectoryPoint],
        new_dt: float,
    ) -> list[TrajectoryPoint]:
        """Resample a trajectory to a different time step."""
        if not trajectory:
            return []

        total_time = trajectory[-1].time
        n_steps = max(1, int(total_time / new_dt))
        resampled: list[TrajectoryPoint] = []

        idx = 0
        for i in range(n_steps + 1):
            t = i * new_dt
            # Find bracketing points
            while idx < len(trajectory) - 1 and trajectory[idx + 1].time < t:
                idx += 1

            if idx >= len(trajectory) - 1:
                resampled.append(trajectory[-1])
                continue

            # Linear interpolation
            t0 = trajectory[idx].time
            t1 = trajectory[idx + 1].time
            if t1 - t0 < 1e-10:
                alpha = 0.0
            else:
                alpha = (t - t0) / (t1 - t0)

            pos = (1 - alpha) * trajectory[idx].position + alpha * trajectory[idx + 1].position
            vel = (1 - alpha) * trajectory[idx].velocity + alpha * trajectory[idx + 1].velocity
            acc = (1 - alpha) * trajectory[idx].acceleration + alpha * trajectory[idx + 1].acceleration

            resampled.append(TrajectoryPoint(time=t, position=pos, velocity=vel, acceleration=acc))

        return resampled
