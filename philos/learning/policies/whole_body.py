"""
Whole-Body Policy — Context-conditioned RL policy for AMR + Arm.

This is the primary PHILOS policy: a single neural network that controls
both the mobile base (AMR) and the robotic arm simultaneously, conditioned
on the Context Vector (z) from the VLM.

The policy architecture: π(a | s, z)
    - s: Robot state (AMR pose + joint positions + LiDAR + force/torque)
    - z: Context Vector (semantic embedding + manipulation parameters)
    - a: Whole-body action (base velocity + joint velocities + gripper)

Training:
    - Algorithm: PPO (default) or SAC
    - Environment: NVIDIA Isaac Sim with domain randomization
    - Sim episodes: 10,000+ with ±200% parameter variation

Maps to:
    WP2 T2.3 (Sim-to-Real RL Training)
    KR2.1 (Isaac Sim to Jazzing robot transfer)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from philos.core.context_vector import ContextVector
from philos.core.registry import register_component
from philos.core.state import RobotState
from philos.learning.base import BasePolicy

logger = logging.getLogger(__name__)


@register_component("learning", "whole_body_ppo")
class WholeBodyPolicy(BasePolicy):
    """Whole-body (AMR + 6-DoF Arm) policy using PPO.

    Network architecture:
        Observation encoder (MLP) → shared features
        ├─ Policy head → action mean + log_std
        └─ Value head → state value V(s,z)

    The context vector z is concatenated with the state observation s
    before being fed to the encoder.
    """

    def __init__(
        self,
        state_dim: int = 82,  # From RobotState.to_observation()
        context_dim: int = 267,  # From ContextVector.to_tensor()
        action_dim: int = 10,  # 3 base + 6 joints + 1 gripper
        hidden_dims: list[int] | None = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        device: str = "cuda:0",
    ) -> None:
        self._state_dim = state_dim
        self._context_dim = context_dim
        self._action_dim = action_dim
        self._hidden_dims = hidden_dims or [512, 256, 128]
        self._lr = learning_rate
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._clip_range = clip_range
        self._entropy_coef = entropy_coef
        self._device = device

        # PyTorch network (initialized in initialize())
        self._policy_net: Any = None
        self._value_net: Any = None
        self._optimizer: Any = None
        self._step_count: int = 0

    @property
    def name(self) -> str:
        return "whole_body_ppo"

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def observation_dim(self) -> int:
        return self._state_dim + self._context_dim

    def initialize(self, config: Any = None) -> None:
        """Initialize the policy and value networks."""
        try:
            import torch
            import torch.nn as nn

            obs_dim = self._state_dim + self._context_dim

            # Shared encoder
            encoder_layers = []
            in_dim = obs_dim
            for h_dim in self._hidden_dims:
                encoder_layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.ELU(),
                ])
                in_dim = h_dim

            # Policy head (actor)
            self._policy_net = nn.Sequential(
                *encoder_layers,
                nn.Linear(self._hidden_dims[-1], self._action_dim * 2),  # mean + log_std
            ).to(self._device)

            # Value head (critic)
            value_layers = []
            in_dim = obs_dim
            for h_dim in self._hidden_dims:
                value_layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.ELU(),
                ])
                in_dim = h_dim
            value_layers.append(nn.Linear(self._hidden_dims[-1], 1))

            self._value_net = nn.Sequential(*value_layers).to(self._device)

            # Optimizer
            params = list(self._policy_net.parameters()) + list(self._value_net.parameters())
            self._optimizer = torch.optim.Adam(params, lr=self._lr)

            total_params = sum(p.numel() for p in params)
            logger.info(
                f"WholeBodyPolicy initialized: obs_dim={obs_dim}, "
                f"action_dim={self._action_dim}, params={total_params:,}"
            )

        except ImportError:
            logger.warning("PyTorch not available. Policy running in stub mode.")

    def predict(
        self,
        state: RobotState,
        context: ContextVector,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Predict a whole-body action given state and context.

        Returns:
            Action vector: [vx, vy, omega, j1, j2, j3, j4, j5, j6, gripper]
        """
        if self._policy_net is None:
            # Stub mode: return zero action
            return np.zeros(self._action_dim, dtype=np.float32)

        import torch

        # Concatenate state and context into observation
        obs = np.concatenate([
            state.to_observation(),
            context.to_tensor(),
        ])
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._policy_net(obs_tensor)
            mean = output[:, :self._action_dim]
            log_std = output[:, self._action_dim:]
            log_std = torch.clamp(log_std, -5, 2)

            if deterministic:
                action = mean
            else:
                std = torch.exp(log_std)
                action = mean + std * torch.randn_like(std)

            action = torch.tanh(action)  # Bound to [-1, 1]

        return action.cpu().numpy().flatten()

    def get_value(self, state: RobotState, context: ContextVector) -> float:
        """Estimate the value of a state-context pair."""
        if self._value_net is None:
            return 0.0

        import torch

        obs = np.concatenate([state.to_observation(), context.to_tensor()])
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self._device)

        with torch.no_grad():
            value = self._value_net(obs_tensor)

        return float(value.item())

    def predict_with_value(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """Predict action, log-probability, and state value from a raw observation.

        This is a convenience method for training loops that work with
        flat numpy observations rather than structured RobotState/ContextVector.

        Returns:
            Tuple of (action, log_prob, value).
        """
        if self._policy_net is None or self._value_net is None:
            action = np.zeros(self._action_dim, dtype=np.float32)
            return action, 0.0, 0.0

        import torch

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self._device)

        with torch.no_grad():
            # Policy
            output = self._policy_net(obs_tensor)
            mean = output[:, :self._action_dim]
            log_std = torch.clamp(output[:, self._action_dim:], -5, 2)
            std = torch.exp(log_std)

            if deterministic:
                action = mean
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            action = torch.tanh(action)

            # Value
            value = self._value_net(obs_tensor).squeeze(-1)

        return (
            action.cpu().numpy().flatten(),
            float(log_prob.item()),
            float(value.item()),
        )

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Perform a PPO training step.

        Args:
            batch: Dictionary with keys:
                - observations: (B, obs_dim) — concatenated [state, context]
                - actions: (B, action_dim)
                - rewards: (B,)
                - dones: (B,)
                - old_log_probs: (B,)
                - advantages: (B,)
                - returns: (B,)

        Returns:
            Training metrics.
        """
        if self._policy_net is None or self._optimizer is None:
            return {"loss": 0.0}

        import torch
        import torch.nn.functional as F

        obs = torch.FloatTensor(batch["observations"]).to(self._device)
        actions = torch.FloatTensor(batch["actions"]).to(self._device)
        old_log_probs = torch.FloatTensor(batch["old_log_probs"]).to(self._device)
        advantages = torch.FloatTensor(batch["advantages"]).to(self._device)
        returns = torch.FloatTensor(batch["returns"]).to(self._device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy forward
        output = self._policy_net(obs)
        mean = output[:, :self._action_dim]
        log_std = torch.clamp(output[:, self._action_dim:], -5, 2)
        std = torch.exp(log_std)

        # Log probability of taken actions
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        # PPO clipped objective
        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1 - self._clip_range, 1 + self._clip_range)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        # Value loss
        values = self._value_net(obs).squeeze(-1)
        value_loss = F.mse_loss(values, returns)

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - self._entropy_coef * entropy

        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self._policy_net.parameters()) + list(self._value_net.parameters()),
            max_norm=0.5,
        )
        self._optimizer.step()

        self._step_count += 1

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "total_loss": float(total_loss.item()),
            "step": self._step_count,
        }

    def save(self, path: str | Path) -> None:
        """Save policy and value network weights."""
        if self._policy_net is None:
            logger.warning("Cannot save — networks not initialized")
            return

        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._policy_net.state_dict(), path / "policy.pt")
        torch.save(self._value_net.state_dict(), path / "value.pt")
        if self._optimizer:
            torch.save(self._optimizer.state_dict(), path / "optimizer.pt")
        logger.info(f"Policy saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load policy and value network weights."""
        if self._policy_net is None:
            logger.warning("Cannot load — networks not initialized. Call initialize() first.")
            return

        import torch

        path = Path(path)
        self._policy_net.load_state_dict(torch.load(path / "policy.pt", weights_only=True))
        self._value_net.load_state_dict(torch.load(path / "value.pt", weights_only=True))
        if self._optimizer and (path / "optimizer.pt").exists():
            self._optimizer.load_state_dict(torch.load(path / "optimizer.pt", weights_only=True))
        logger.info(f"Policy loaded from {path}")
