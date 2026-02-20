"""Tests for PHILOS simulation environments (stub mode)."""

import numpy as np
import pytest

from philos.simulation.environments.pour_task import PourTaskEnv, PourTaskConfig
from philos.simulation.environments.fetch_sort_task import FetchSortTaskEnv, FetchSortConfig
from philos.learning.domain_randomization import DomainRandomizer, DomainRandomizationConfig


class TestPourTask:
    def test_reset(self):
        env = PourTaskEnv()
        obs, info = env.reset()
        assert obs.shape == (56,)
        assert "episode" in info

    def test_step(self):
        env = PourTaskEnv()
        env.reset()
        action = np.zeros(10)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (56,)
        assert isinstance(reward, float)

    def test_multiple_episodes(self):
        env = PourTaskEnv()
        for _ in range(3):
            obs, _ = env.reset()
            for _ in range(10):
                action = np.random.uniform(-1, 1, size=10).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        env.close()


class TestFetchSortTask:
    def test_reset(self):
        env = FetchSortTaskEnv()
        obs, info = env.reset()
        assert obs.shape == (64,)

    def test_step(self):
        env = FetchSortTaskEnv()
        env.reset()
        action = np.zeros(10)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (64,)
        assert isinstance(reward, float)


class TestDomainRandomization:
    def test_sample(self):
        dr = DomainRandomizer(seed=42)
        params = dr.sample()
        assert "fluid_viscosity" in params
        assert "ground_friction" in params
        assert len(params) > 10

    def test_reproducibility(self):
        dr1 = DomainRandomizer(seed=42)
        dr2 = DomainRandomizer(seed=42)
        assert dr1.sample() == dr2.sample()

    def test_range_coverage(self):
        dr = DomainRandomizer(seed=0)
        viscosities = [dr.sample()["fluid_viscosity"] for _ in range(1000)]
        assert min(viscosities) < 0.5  # Should explore low range
        assert max(viscosities) > 1.5  # Should explore high range
