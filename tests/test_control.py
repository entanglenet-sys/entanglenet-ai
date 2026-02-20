"""Tests for the PHILOS control module — Safety Shield."""

import numpy as np
import pytest

from philos.control.safety_shield import SafetyShield, SafetyConstraints, SafetyLevel
from philos.control.mpc_solver import WholeBodyMPC
from philos.control.trajectory import TrajectoryOptimizer
from philos.core.state import RobotState, JointState, AMRState


def make_state(**kwargs) -> RobotState:
    """Helper to create a RobotState for testing."""
    return RobotState(
        joint_states=[JointState(name=f"j{i}", position=0.0) for i in range(6)],
        amr=AMRState(**kwargs.get("amr", {})),
    )


class TestSafetyShield:
    def test_nominal_pass_through(self):
        """Safe actions should pass through with minimal modification."""
        shield = SafetyShield()
        state = make_state()
        action = np.zeros(10)
        action[3] = 0.1  # Small joint movement

        cmd = shield.compute(action, state)
        assert cmd.is_safe is True
        assert len(cmd.safety_overrides) == 0 or shield.safety_state.level == SafetyLevel.WARNING

    def test_base_velocity_clipping(self):
        """Excessive base velocity should be clipped."""
        shield = SafetyShield()
        state = make_state()
        action = np.zeros(10)
        action[0] = 5.0  # Way too fast
        action[1] = 5.0

        cmd = shield.compute(action, state)
        speed = np.linalg.norm(cmd.base_velocity[:2])
        assert speed <= 1.0 + 1e-6  # max_base_linear_velocity

    def test_joint_position_clipping(self):
        """Joint positions beyond limits should be clamped."""
        shield = SafetyShield()
        state = make_state()
        action = np.zeros(10)
        action[3] = 10.0  # Beyond J1 limit of 2.87

        cmd = shield.compute(action, state)
        assert cmd.joint_positions[0] <= 2.87 + 1e-6

    def test_emergency_stop(self):
        """Emergency stop should produce zero velocities."""
        shield = SafetyShield()
        shield.trigger_emergency_stop()
        state = make_state()
        action = np.ones(10)

        cmd = shield.compute(action, state)
        assert cmd.is_safe is False
        assert "EMERGENCY_STOP" in cmd.safety_overrides
        np.testing.assert_array_equal(cmd.base_velocity, np.zeros(3))
        np.testing.assert_array_equal(cmd.joint_velocities, np.zeros(6))

    def test_reset(self):
        """Reset should clear all state."""
        shield = SafetyShield()
        shield.trigger_emergency_stop()
        shield.reset()
        assert shield._e_stop_active is False


class TestMPC:
    def test_basic_solve(self):
        """MPC should produce a feasible control command."""
        mpc = WholeBodyMPC()
        state = make_state()
        action = np.zeros(10)
        action[3:9] = [0.5, 0.3, -0.2, 0.1, 0.0, 0.0]

        cmd = mpc.compute(action, state)
        assert cmd.joint_positions.shape == (6,)
        assert cmd.base_velocity.shape == (3,)


class TestTrajectory:
    def test_minimum_jerk(self):
        """Minimum jerk should produce smooth trajectory."""
        traj_opt = TrajectoryOptimizer(n_dof=6)
        start = np.zeros(6)
        end = np.ones(6) * 0.5
        traj = traj_opt.minimum_jerk(start, end, duration=1.0, dt=0.02)

        assert len(traj) > 0
        # Start and end positions
        np.testing.assert_array_almost_equal(traj[0].position, start, decimal=3)
        np.testing.assert_array_almost_equal(traj[-1].position, end, decimal=3)
        # Zero velocity at boundaries
        np.testing.assert_array_almost_equal(traj[0].velocity, np.zeros(6), decimal=2)
        np.testing.assert_array_almost_equal(traj[-1].velocity, np.zeros(6), decimal=2)

    def test_cubic_spline(self):
        """Cubic spline through waypoints."""
        traj_opt = TrajectoryOptimizer(n_dof=3)
        waypoints = [np.zeros(3), np.array([1, 0, 0]), np.array([1, 1, 0])]
        durations = [0.5, 0.5]
        traj = traj_opt.cubic_spline(waypoints, durations)
        assert len(traj) > 0
        assert traj[-1].time == pytest.approx(1.0, abs=0.02)
