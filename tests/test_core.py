"""Tests for the PHILOS core module."""

import numpy as np
import pytest

from philos.core.config import PhilosConfig, SimulationConfig, load_config
from philos.core.context_vector import ContextVector, ManipulationMode
from philos.core.state import RobotState, JointState, AMRState, EndEffectorState
from philos.core.registry import ComponentRegistry


class TestPhilosConfig:
    def test_default_config(self):
        config = PhilosConfig()
        assert config.simulation.backend == "isaac_sim"
        assert config.control.safety_enabled is True

    def test_simulation_config(self):
        cfg = SimulationConfig(num_envs=32, render=True)
        assert cfg.num_envs == 32
        assert cfg.render is True


class TestContextVector:
    def test_creation(self):
        cv = ContextVector(
            embedding=np.random.randn(18),
            mode=ManipulationMode.FLUID,
            impedance_scale=0.5,
        )
        assert cv.mode == ManipulationMode.FLUID
        assert cv.embedding.shape == (18,)

    def test_to_tensor(self):
        cv = ContextVector(embedding=np.ones(18))
        tensor = cv.to_tensor()
        assert tensor.shape == (18,)

    def test_staleness(self):
        import time
        cv = ContextVector(embedding=np.zeros(18), timestamp=time.time() - 10)
        assert cv.is_stale(max_age=5.0) is True
        assert cv.is_stale(max_age=20.0) is False

    def test_serialization(self):
        cv = ContextVector(
            embedding=np.array([1.0, 2.0, 3.0]),
            mode=ManipulationMode.PRECISION,
        )
        d = cv.to_dict()
        cv2 = ContextVector.from_dict(d)
        assert cv2.mode == ManipulationMode.PRECISION
        np.testing.assert_array_almost_equal(cv2.embedding, cv.embedding)


class TestRobotState:
    def test_default_state(self):
        state = RobotState(
            joint_states=[JointState(name=f"j{i}", position=0.0) for i in range(6)]
        )
        assert len(state.joint_states) == 6

    def test_to_observation(self):
        state = RobotState(
            joint_states=[
                JointState(name=f"j{i}", position=float(i) * 0.1)
                for i in range(6)
            ],
            amr=AMRState(),
            end_effector=EndEffectorState(),
        )
        obs = state.to_observation()
        assert isinstance(obs, np.ndarray)
        assert len(obs) > 0


class TestRegistry:
    def test_register_and_create(self):
        registry = ComponentRegistry()

        class DummyComponent:
            def __init__(self, value=42):
                self.value = value

        registry.register("test", "dummy", DummyComponent)
        assert registry.has("test", "dummy")

        instance = registry.create("test", "dummy", value=99)
        assert instance.value == 99

    def test_list_components(self):
        registry = ComponentRegistry()
        components = registry.list_components()
        assert isinstance(components, dict)
