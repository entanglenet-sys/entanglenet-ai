"""
Whole-Body Model Predictive Control (MPC) solver.

Implements the coordinated AMR + 6-DoF arm trajectory optimization
described in WP3 T3.2.  The MPC plans over a short horizon to
produce smooth, dynamically-feasible trajectories.

This is a DETERMINISTIC controller (no neural networks) that can
serve as both:
    1. A standalone planner for slow / safety-critical motions
    2. A refinement layer after the RL policy's fast commands

Solver: Sequential Quadratic Programming (SQP) via scipy or
        CasADi (if installed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from philos.control.base import BaseController, ControlCommand
from philos.core.registry import register_component
from philos.core.state import RobotState

logger = logging.getLogger(__name__)


@dataclass
class MPCConfig:
    """MPC solver parameters."""

    horizon: int = 10                   # prediction steps
    dt: float = 0.02                    # 50 Hz
    max_iterations: int = 5             # SQP iterations per solve
    convergence_tol: float = 1e-4

    # Cost weights
    position_weight: float = 10.0       # tracking error
    velocity_weight: float = 1.0        # velocity smoothness
    acceleration_weight: float = 0.5    # jerk minimization
    torque_weight: float = 0.01         # energy efficiency
    terminal_weight: float = 20.0       # terminal cost


@register_component("control", "whole_body_mpc")
class WholeBodyMPC(BaseController):
    """Whole-Body MPC for coordinated AMR + arm control.

    State vector x = [base_x, base_y, base_theta, j1..j6]  (9-dim)
    Input vector u = [base_vx, base_vy, base_omega, jd1..jd6]  (9-dim)

    The MPC minimizes a quadratic cost:
        J = sum_{k=0}^{N-1} [ (x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k ]
            + (x_N - x_ref)^T P (x_N - x_ref)
    subject to dynamics, state, and input constraints.
    """

    STATE_DIM = 9   # base(3) + joints(6)
    INPUT_DIM = 9   # base_vel(3) + joint_vel(6)

    def __init__(self, config: MPCConfig | None = None) -> None:
        self._config = config or MPCConfig()
        self._Q = np.eye(self.STATE_DIM) * self._config.position_weight
        self._R = np.eye(self.INPUT_DIM) * self._config.velocity_weight
        self._P = np.eye(self.STATE_DIM) * self._config.terminal_weight

        # Warm-start buffer
        self._prev_solution: np.ndarray | None = None
        self._prev_state: np.ndarray | None = None

    def compute(
        self,
        action: np.ndarray,
        state: RobotState,
        dt: float = 0.02,
    ) -> ControlCommand:
        """Run MPC to produce a smooth trajectory toward the target.

        The RL policy action is treated as a *reference* that the MPC
        tracks while respecting dynamic constraints.
        """
        x_current = self._state_to_vector(state)
        x_ref = self._action_to_reference(action, x_current)

        # Solve the QP
        u_optimal = self._solve(x_current, x_ref)

        # Extract first control input
        base_vel = u_optimal[:3]
        joint_vel = u_optimal[3:9]

        # Integrate to get joint position targets
        joint_pos = x_current[3:9] + joint_vel * dt

        gripper = float(np.clip(action[9] if len(action) > 9 else 0.5, 0.0, 1.0))

        return ControlCommand(
            base_velocity=base_vel,
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            joint_torques=np.zeros(6),
            gripper_position=gripper,
            is_safe=True,
            safety_overrides=[],
        )

    def reset(self) -> None:
        self._prev_solution = None
        self._prev_state = None

    # ------------------------------------------------------------------
    # QP solver
    # ------------------------------------------------------------------

    def _solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
    ) -> np.ndarray:
        """Solve the finite-horizon QP.

        Uses a simple iterative approach (compatible without CasADi).
        For production, replace with CasADi / OSQP / HPIPM.
        """
        N = self._config.horizon
        dt = self._config.dt
        n_x = self.STATE_DIM
        n_u = self.INPUT_DIM

        # Warm start
        if self._prev_solution is not None:
            u_traj = np.roll(self._prev_solution.reshape(N, n_u), -1, axis=0)
            u_traj[-1] = u_traj[-2]  # repeat last
        else:
            u_traj = np.zeros((N, n_u))

        # Simple gradient descent on the QP
        for iteration in range(self._config.max_iterations):
            # Forward simulate
            x_traj = np.zeros((N + 1, n_x))
            x_traj[0] = x0

            for k in range(N):
                x_traj[k + 1] = self._dynamics(x_traj[k], u_traj[k], dt)

            # Compute gradient w.r.t. u
            grad = np.zeros((N, n_u))
            for k in range(N):
                # State cost gradient (dJ/du via chain rule with linear dynamics)
                state_error = x_traj[k + 1] - x_ref
                grad[k] = 2.0 * self._R @ u_traj[k] + 2.0 * dt * self._Q @ state_error

            # Terminal cost
            terminal_error = x_traj[N] - x_ref
            grad[-1] += 2.0 * dt * self._P @ terminal_error

            # Step size (simple backtracking)
            alpha = 0.1 / (iteration + 1)
            u_traj -= alpha * grad

            # Clamp inputs
            u_traj = self._clamp_inputs(u_traj)

            # Check convergence
            if np.max(np.abs(grad)) < self._config.convergence_tol:
                break

        self._prev_solution = u_traj.flatten()
        return u_traj[0]  # Return first input

    @staticmethod
    def _dynamics(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Simple integrator dynamics: x_{k+1} = x_k + u_k * dt."""
        return x + u * dt

    def _clamp_inputs(self, u_traj: np.ndarray) -> np.ndarray:
        """Clamp control inputs to feasible range."""
        # Base velocity limits
        u_traj[:, :2] = np.clip(u_traj[:, :2], -1.0, 1.0)
        u_traj[:, 2] = np.clip(u_traj[:, 2], -1.5, 1.5)
        # Joint velocity limits
        max_jv = np.array([2.175, 2.175, 2.175, 3.49, 3.49, 3.49])
        u_traj[:, 3:9] = np.clip(u_traj[:, 3:9], -max_jv, max_jv)
        return u_traj

    # ------------------------------------------------------------------
    # State conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _state_to_vector(state: RobotState) -> np.ndarray:
        """Convert RobotState to the 9-dim state vector."""
        x = np.zeros(9)
        if state.amr is not None:
            if state.amr.position is not None:
                x[0] = state.amr.position[0]
                x[1] = state.amr.position[1]
            if state.amr.orientation is not None:
                # Extract yaw from quaternion
                q = state.amr.orientation
                siny_cosp = 2.0 * (q[3] * q[2] + q[0] * q[1])
                cosy_cosp = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
                x[2] = np.arctan2(siny_cosp, cosy_cosp)
        if state.joint_states:
            for i, js in enumerate(state.joint_states[:6]):
                x[3 + i] = js.position
        return x

    @staticmethod
    def _action_to_reference(
        action: np.ndarray,
        current_state: np.ndarray,
    ) -> np.ndarray:
        """Convert RL action to a reference state for MPC tracking.

        The action's joint values are treated as target positions.
        Base velocity components are integrated one step to get a
        reference position.
        """
        x_ref = current_state.copy()
        if len(action) >= 3:
            x_ref[:3] += action[:3] * 0.02  # One-step integration
        if len(action) >= 9:
            x_ref[3:9] = action[3:9]  # Direct position targets
        return x_ref
