"""
"The Sommelier" — Chemical-Liquid Pouring Task (WP4 T4.2).

A UR5e 6-DoF arm on a FIXED PEDESTAL (no AMR in initial setup)
holds a beaker of chemical liquid and must:
    1. Position the arm to bring the beaker over the target glass
    2. Tilt the wrist to pour liquid precisely into the glass
    3. Pour the correct amount without spilling

The robot model is loaded from the project URDF file, which is the
single source of truth for link lengths, joint limits, masses, and
visual geometry.  Both this simulation and the 3D digital twin
read from the same URDF, guaranteeing a 1:1 match.

Physical constraints (from URDF):
    - 6 revolute joints with hard position/velocity/torque limits
    - Real UR5e DH parameters
    - Gravity: 9.81 m/s²  (affects liquid, joint torques)
    - Payload: beaker + liquid ≈ 0.55 kg

Fluid model (simplified physics):
    - Pour rate = f(tilt_angle): zero below 15°, ramps to max at 90°
    - Spill if beaker tilt > 8° when not over glass
    - Surface tension → small delay before flow starts
    - Gravity-driven parabolic stream

Success criteria (PHILOS KPIs):
    - Spill rate   < 5 % of original volume
    - Pour accuracy: volume error < 10 %
    - No collisions
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from philos.simulation.isaac_env import IsaacSimEnv, EnvConfig
from philos.core.registry import register_component
from philos.learning.reward_functions import DynamicRewardFunction
from philos.utils.urdf_parser import (
    load_default_robot,
    forward_kinematics,
    get_end_effector_pose,
    URDFRobot,
)

logger = logging.getLogger(__name__)

# Isaac Sim imports (optional)
_ISAAC_AVAILABLE = False
try:
    from omni.isaac.core.utils.prims import create_prim  # type: ignore
    _ISAAC_AVAILABLE = True
except ImportError:
    pass


# ─── Task phases ──────────────────────────────────────────────────────────────

class TaskPhase(str, Enum):
    REACH = "reach"        # Move EE to pouring position above glass
    POUR  = "pour"         # Tilt wrist to pour liquid
    DONE  = "done"         # Target volume reached


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class PourTaskConfig(EnvConfig):
    """Configuration for the chemical-liquid pouring task.

    Robot geometry is loaded from URDF at runtime — no hardcoded link lengths.
    """

    # ── Fluid ──────────────────────────────────────────────
    source_volume_ml: float = 250.0
    target_volume_ml: float = 250.0
    glass_capacity_ml: float = 350.0
    max_spill_fraction: float = 0.05

    # ── Lab geometry (metres, URDF world frame) ────────────
    # Robot pedestal is at origin (0,0,0).  Bench is in FRONT
    # of the robot, NOT overlapping the pedestal.
    glass_position: tuple[float, float, float] = (0.45, 0.25, 0.78)
    bench_height: float = 0.75
    pour_arrive_distance: float = 0.10  # EE-to-glass XY for POUR phase

    # Bench collision AABB (URDF frame: x=forward, y=left, z=up)
    bench_x_min: float = 0.15
    bench_x_max: float = 0.85
    bench_y_min: float = -0.05
    bench_y_max: float = 0.55
    bench_z_top: float = 0.77  # slightly above bench surface (col margin)

    # ── Joint velocity scaling ─────────────────────────────
    # Policy outputs in [-1, 1], scaled by these per-joint max velocities
    max_joint_velocity: tuple[float, ...] = (2.094, 2.094, 3.142, 3.491, 3.491, 3.491)

    # ── Pouring physics ────────────────────────────────────
    pour_tilt_threshold_deg: float = 45.0   # flow starts at 45° tilt
    pour_rate_max_ml_per_step: float = 1.5  # ~75 ml/s at 50 Hz
    spill_tilt_threshold_deg: float = 20.0  # drips start at 20° (if not over glass)
    pour_xy_tolerance_m: float = 0.08

    # ── Reward weights (shaped for visible learning) ───────
    reach_weight: float = 5.0
    height_bonus_weight: float = 2.0
    pour_progress_weight: float = 15.0
    spill_penalty_weight: float = 30.0
    smoothness_weight: float = 0.3
    success_bonus: float = 100.0
    time_penalty: float = 0.05
    collision_penalty_weight: float = 50.0   # big penalty for hitting bench
    orientation_penalty_weight: float = 5.0  # keep beaker upright during REACH

    obs_dim: int = 56
    action_dim: int = 7  # 6 joint velocities + 1 gripper (NO AMR base)


# ─── Environment ──────────────────────────────────────────────────────────────

@register_component("simulation", "pour_task")
class PourTaskEnv(IsaacSimEnv):
    """Chemical-liquid pouring with a URDF-driven UR5e on a fixed pedestal.

    Observation space (56-dim):
        [joint_pos_norm(6), ee_pos(3), glass_pos(3),
         wrist_tilt_sin, wrist_tilt_cos, grasped,
         phase_onehot(2), fluid_in_beaker_frac, poured_frac,
         spill_frac, ee_to_glass_dist, ee_to_glass_z_diff,
         joint_velocities(6), prev_reward, ...context(18)]

    Action space (7-dim):
        [j1_vel, j2_vel, j3_vel, j4_vel, j5_vel, j6_vel, gripper]
    """

    def __init__(self, config: PourTaskConfig | None = None) -> None:
        self._task_config = config or PourTaskConfig()
        super().__init__(config=self._task_config)

        tc = self._task_config

        # ── Load robot model from URDF (single source of truth) ──
        self._urdf: URDFRobot = load_default_robot()
        self._num_joints = len(self._urdf.actuated_joints)
        assert self._num_joints == 6, f"Expected 6 actuated joints, got {self._num_joints}"

        # Extract joint limits from URDF
        self._joint_lo = np.array([
            self._urdf.joints[jn].lower for jn in self._urdf.actuated_joints
        ], dtype=np.float64)
        self._joint_hi = np.array([
            self._urdf.joints[jn].upper for jn in self._urdf.actuated_joints
        ], dtype=np.float64)
        self._max_jvel = np.array(tc.max_joint_velocity[:6], dtype=np.float64)

        # ── Task state ──
        self._poured_volume: float = 0.0
        self._spilled_volume: float = 0.0
        self._grasped: bool = True
        self._phase: TaskPhase = TaskPhase.REACH
        self._glass_pos = np.array(tc.glass_position, dtype=np.float64)
        self._fluid_in_beaker: float = tc.source_volume_ml
        self._wrist_tilt_deg: float = 0.0

        # ── Kinematic state ──
        self._joint_pos = np.zeros(6, dtype=np.float64)
        self._joint_vel = np.zeros(6, dtype=np.float64)
        self._ee_pos = np.zeros(3, dtype=np.float64)
        self._ee_rot = np.eye(3)
        self._beaker_pos = np.zeros(3, dtype=np.float64)
        self._gripper_openness: float = 0.0

        # ── Collision state ──
        self._collision: bool = False
        self._max_penetration: float = 0.0
        self._min_bench_clearance: float = 1.0  # min z above bench for any link
        self._link_positions: dict[str, np.ndarray] = {}  # link → xyz

        # ── Reward tracking ──
        self._prev_reward: float = 0.0
        self._prev_ee_to_glass: float = 1.0
        self._episode_rewards: list = []

        # Backward compat: aliases used by server.py telemetry
        self._stub_joint_pos = self._joint_pos
        self._stub_ee_pos = self._ee_pos
        self._stub_base_pos = np.zeros(3, dtype=np.float64)  # fixed pedestal

        # Reward function
        self._reward_fn = DynamicRewardFunction()

    # ──────────────────────────────────────────────────────────────────────
    # Scene
    # ──────────────────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        if _ISAAC_AVAILABLE and self._world is not None:
            self._world.scene.add_default_ground_plane()
            logger.info("Pour task scene loaded (Isaac Sim).")
        else:
            logger.info(f"Pour task scene loaded (stub, URDF={self._urdf.name}, "
                        f"{self._num_joints} joints).")

    # ──────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────

    def _on_reset(self, options: dict | None = None) -> None:
        tc = self._task_config

        # Fluid
        self._poured_volume = 0.0
        self._spilled_volume = 0.0
        self._fluid_in_beaker = tc.source_volume_ml
        self._wrist_tilt_deg = 0.0
        self._prev_reward = 0.0

        # Robot starts holding the beaker, gripper closed
        self._grasped = True
        self._gripper_openness = 0.0
        self._phase = TaskPhase.REACH

        # Collision state
        self._collision = False
        self._max_penetration = 0.0
        self._min_bench_clearance = 1.0

        # Glass on bench (with small domain randomization)
        self._glass_pos = np.array(tc.glass_position, dtype=np.float64)
        self._glass_pos[:2] += np.random.uniform(-0.03, 0.03, size=2)

        # ── Initial joint config: beaker held UPRIGHT above bench ──
        # This pose gives: tool_z ≈ (0,0,1), tilt ≈ 0°, EE above glass
        self._joint_pos = np.array([0.4, -0.8, 1.2, -1.97, -1.5708, 0.0],
                                    dtype=np.float64)
        self._joint_vel = np.zeros(6, dtype=np.float64)

        # Update aliases
        self._stub_joint_pos = self._joint_pos
        self._stub_base_pos = np.zeros(3, dtype=np.float64)

        # FK from URDF
        self._update_kinematics()
        self._beaker_pos = self._ee_pos.copy()
        self._prev_ee_to_glass = float(np.linalg.norm(
            self._ee_pos[:2] - self._glass_pos[:2]))
        self._episode_rewards = []

    # ──────────────────────────────────────────────────────────────────────
    # Forward kinematics (from URDF — proper 3D)
    # ──────────────────────────────────────────────────────────────────────

    def _update_kinematics(self) -> None:
        """Recompute EE pose from current joints using URDF FK."""
        transforms = forward_kinematics(self._urdf, self._joint_pos)
        tool_T = transforms.get("tool0", np.eye(4))
        self._ee_pos[:] = tool_T[:3, 3]
        self._ee_rot = tool_T[:3, :3].copy()

        # Wrist tilt: angle between tool Z-axis and world up (0,0,1)
        tool_z = self._ee_rot[:, 2]
        cos_angle = np.clip(np.dot(tool_z, np.array([0, 0, 1])), -1, 1)
        self._wrist_tilt_deg = float(np.degrees(np.arccos(cos_angle)))

        # Store all link positions for collision checks
        self._link_positions = {}
        for name, T in transforms.items():
            self._link_positions[name] = T[:3, 3].copy()

        # Check bench collision
        self._check_bench_collision()

        # Keep alias in sync
        self._stub_ee_pos = self._ee_pos

    def _check_bench_collision(self) -> None:
        """Check if any moving link intersects the bench volume."""
        tc = self._task_config
        self._collision = False
        self._max_penetration = 0.0
        self._min_bench_clearance = 10.0  # large default

        # Skip fixed base links — only check moving parts
        skip = {'world', 'pedestal', 'base_link'}
        for name, pos in self._link_positions.items():
            if name in skip:
                continue
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

            # Is this link within the bench footprint (XY)?
            in_x = tc.bench_x_min <= x <= tc.bench_x_max
            in_y = tc.bench_y_min <= y <= tc.bench_y_max

            if in_x and in_y:
                clearance = z - tc.bench_z_top
                self._min_bench_clearance = min(self._min_bench_clearance, clearance)
                if clearance < 0:
                    self._collision = True
                    self._max_penetration = max(self._max_penetration, -clearance)

    # ──────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_obs(self) -> np.ndarray:
        """Build the 56-dim observation."""
        obs = np.zeros(self._task_config.obs_dim, dtype=np.float32)
        tc = self._task_config
        total = tc.source_volume_ml if tc.source_volume_ml > 0 else 1.0

        # Joint positions (normalized to [-1, 1] by URDF limits)
        for i in range(6):
            rng = self._joint_hi[i] - self._joint_lo[i]
            if rng > 0:
                obs[i] = 2.0 * (self._joint_pos[i] - self._joint_lo[i]) / rng - 1.0

        # EE position (world frame)
        obs[6:9] = self._ee_pos

        # Glass position
        obs[9:12] = self._glass_pos

        # Wrist tilt (sin/cos encoding)
        tilt_rad = np.radians(self._wrist_tilt_deg)
        obs[12] = np.sin(tilt_rad)
        obs[13] = np.cos(tilt_rad)

        # Grasped flag
        obs[14] = 1.0 if self._grasped else 0.0

        # Phase one-hot (REACH / POUR)
        if self._phase == TaskPhase.REACH:
            obs[15] = 1.0
        elif self._phase == TaskPhase.POUR:
            obs[16] = 1.0

        # Fluid fractions
        obs[17] = self._fluid_in_beaker / total
        obs[18] = self._poured_volume / total
        obs[19] = self._spilled_volume / total

        # Distance features
        ee_to_glass_xy = float(np.linalg.norm(self._ee_pos[:2] - self._glass_pos[:2]))
        ee_to_glass_z = self._ee_pos[2] - self._glass_pos[2]
        obs[20] = ee_to_glass_xy
        obs[21] = ee_to_glass_z

        # Joint velocities
        obs[22:28] = self._joint_vel / (self._max_jvel + 1e-8)

        # Previous reward (helps correlate actions → outcomes)
        obs[28] = np.clip(self._prev_reward / 20.0, -1.0, 1.0)

        # Collision / environment awareness (indices 29-34)
        obs[29] = 1.0 if self._collision else 0.0
        obs[30] = np.clip(self._min_bench_clearance / 0.5, -1.0, 1.0)
        obs[31] = self._wrist_tilt_deg / 180.0  # 0=upright, 0.5=horizontal, 1=inverted
        obs[32] = self._max_penetration * 10.0  # scale up for visibility
        obs[33] = 1.0 if (self._phase == TaskPhase.REACH and self._wrist_tilt_deg > 15.0) else 0.0
        obs[34] = np.clip(float(np.linalg.norm(self._ee_pos[:2] - self._glass_pos[:2])) / 0.5, 0, 1)

        # Context (indices 38-55)
        obs[38:41] = self._ee_pos
        obs[41:44] = self._glass_pos
        obs[44] = self._poured_volume / total
        obs[45] = self._spilled_volume / total
        obs[46] = self._wrist_tilt_deg / 180.0
        obs[47] = float(self._phase == TaskPhase.POUR)

        return obs

    # ──────────────────────────────────────────────────────────────────────
    # Reward — shaped for VISIBLE learning progress
    # ──────────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> float:
        tc = self._task_config
        total = tc.source_volume_ml if tc.source_volume_ml > 0 else 1.0
        reward = 0.0

        ee = self._ee_pos
        d_glass_xy = float(np.linalg.norm(ee[:2] - self._glass_pos[:2]))
        d_glass_z = abs(ee[2] - self._glass_pos[2] - 0.10)
        poured_frac = self._poured_volume / total
        spill_frac = self._spilled_volume / total
        target_frac = tc.target_volume_ml / total

        # ── 0) COLLISION PENALTY (critical — must learn to avoid bench) ──
        if self._collision:
            reward -= tc.collision_penalty_weight * (1.0 + self._max_penetration * 10.0)
        elif self._min_bench_clearance < 0.05:
            # Soft penalty: getting dangerously close to bench
            reward -= tc.collision_penalty_weight * 0.3 * (0.05 - self._min_bench_clearance) / 0.05

        # ── 1) ORIENTATION PENALTY during REACH ──
        # Beaker MUST stay upright while reaching (tilt < 10°)
        if self._phase == TaskPhase.REACH:
            tilt_excess = max(0.0, self._wrist_tilt_deg - 10.0)
            reward -= tc.orientation_penalty_weight * (tilt_excess / 80.0)
            # Bonus for keeping beaker upright
            if self._wrist_tilt_deg < 5.0:
                reward += 0.5

        # ── 2) Dense reach reward: closing distance to glass ──
        approach_improvement = self._prev_ee_to_glass - d_glass_xy
        reward += tc.reach_weight * approach_improvement * 10.0
        if d_glass_xy < 0.5:
            reward += tc.reach_weight * (0.5 - d_glass_xy)

        # ── 3) Height bonus: right altitude for pouring ──
        # Want EE 5-15cm above glass, NOT below bench level
        if d_glass_z < 0.15:
            reward += tc.height_bonus_weight * (0.15 - d_glass_z) / 0.15
        if ee[2] < tc.bench_z_top:
            reward -= 5.0  # below bench surface

        # ── 4) Pour progress reward (only in POUR phase) ──
        if self._phase == TaskPhase.POUR:
            reward += tc.pour_progress_weight * poured_frac
            # Bonus for being positioned correctly over glass while pouring
            if d_glass_xy < tc.pour_xy_tolerance_m:
                reward += 3.0
                # Additional bonus for controlled tilt (45-100°)
                if 45 <= self._wrist_tilt_deg <= 120:
                    reward += 2.0
            else:
                reward -= 3.0 * (d_glass_xy - tc.pour_xy_tolerance_m)

        # ── 5) Spill penalty (always, but proportional) ──
        reward -= tc.spill_penalty_weight * spill_frac

        # ── 6) Smoothness ──
        action_mag = float(np.sum(action[:6] ** 2))
        reward -= tc.smoothness_weight * action_mag * 0.01

        # ── 7) Time penalty ──
        reward -= tc.time_penalty

        # ── 8) Success bonus ──
        if poured_frac >= target_frac * 0.90 and spill_frac < tc.max_spill_fraction:
            reward += tc.success_bonus

        # Track for next step
        self._prev_ee_to_glass = d_glass_xy
        self._prev_reward = reward
        self._episode_rewards.append(reward)

        return float(reward)

    # ──────────────────────────────────────────────────────────────────────
    # Termination
    # ──────────────────────────────────────────────────────────────────────

    def _check_terminated(self) -> bool:
        tc = self._task_config
        total = tc.source_volume_ml if tc.source_volume_ml > 0 else 1.0

        # Success
        if self._poured_volume / total >= (tc.target_volume_ml / total) * 0.90:
            self._phase = TaskPhase.DONE
            return True

        # Failure: bench collision (severe penetration)
        if self._collision and self._max_penetration > 0.02:
            return True

        # Failure: too much spillage
        if self._spilled_volume / total > tc.max_spill_fraction * 2:
            return True

        # Failure: beaker empty, didn't pour enough
        if self._fluid_in_beaker <= 0 and self._poured_volume / total < 0.5:
            return True

        # Failure: EE too low (floor crash)
        if self._ee_pos[2] < 0.2:
            return True

        return False

    # ──────────────────────────────────────────────────────────────────────
    # Stub physics step — URDF kinematics, no AMR
    # ──────────────────────────────────────────────────────────────────────

    def _stub_step(self, action: np.ndarray) -> None:
        """Physics step using URDF FK.  No AMR base.

        Action: [j1_vel..j6_vel, gripper] — 7 dims.
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] < 7:
            action = np.pad(action, (0, 7 - action.shape[0]))

        tc = self._task_config
        dt = tc.dt  # 0.02 s

        # 1. Joint velocities (scaled, clamped, with URDF damping)
        raw_vel = action[:6] * self._max_jvel
        clamped_vel = np.clip(raw_vel, -self._max_jvel, self._max_jvel)
        for i, jname in enumerate(self._urdf.actuated_joints):
            damp = self._urdf.joints[jname].damping
            clamped_vel[i] -= damp * self._joint_vel[i] * dt

        new_joints = self._joint_pos + clamped_vel * dt
        self._joint_pos = np.clip(new_joints, self._joint_lo, self._joint_hi)
        self._joint_vel = clamped_vel.copy()

        # Update alias
        self._stub_joint_pos = self._joint_pos

        # 2. Forward kinematics from URDF (also checks bench collision)
        self._update_kinematics()

        # 2b. Bench repulsion: if a link penetrates the bench, push joints back
        if self._collision:
            # Revert to pre-step joints (simple collision response)
            self._joint_pos = np.clip(
                self._joint_pos - clamped_vel * dt * 0.5,  # half-revert
                self._joint_lo, self._joint_hi
            )
            self._update_kinematics()

        # 3. Gripper
        self._gripper_openness = float(np.clip(action[6], 0, 1))
        self._grasped = self._gripper_openness < 0.5

        # 4. Beaker follows EE
        if self._grasped:
            self._beaker_pos = self._ee_pos.copy()

        # 5. Phase transitions
        # REACH → POUR: based on position only (close + above glass)
        # The policy must then decide to tilt for pouring
        ee_to_glass_xy = float(np.linalg.norm(
            self._ee_pos[:2] - self._glass_pos[:2]))

        if self._phase == TaskPhase.REACH:
            if (ee_to_glass_xy < tc.pour_arrive_distance
                    and self._ee_pos[2] > self._glass_pos[2]):
                self._phase = TaskPhase.POUR

        # 6. Fluid dynamics with gravity
        if self._fluid_in_beaker > 0:
            over_glass = ee_to_glass_xy < tc.pour_xy_tolerance_m
            tilt = self._wrist_tilt_deg

            if tilt > tc.pour_tilt_threshold_deg:
                tilt_frac = min(1.0,
                    (tilt - tc.pour_tilt_threshold_deg)
                    / (90.0 - tc.pour_tilt_threshold_deg))
                gravity_factor = 0.5 + 0.5 * (tilt_frac ** 0.5)
                flow_ml = tc.pour_rate_max_ml_per_step * tilt_frac * gravity_factor
                flow_ml = min(flow_ml, self._fluid_in_beaker)
                self._fluid_in_beaker -= flow_ml

                if over_glass and self._ee_pos[2] > self._glass_pos[2]:
                    accuracy = max(0.5, 1.0 - ee_to_glass_xy / tc.pour_xy_tolerance_m)
                    self._poured_volume += flow_ml * accuracy
                    self._spilled_volume += flow_ml * (1.0 - accuracy)
                else:
                    self._spilled_volume += flow_ml

            elif tilt > tc.spill_tilt_threshold_deg:
                drip_rate = 0.03 * (tilt - tc.spill_tilt_threshold_deg) / 10.0
                drip = min(drip_rate, self._fluid_in_beaker)
                self._fluid_in_beaker -= drip
                self._spilled_volume += drip

        # 7. Cap volumes
        total = tc.source_volume_ml
        self._poured_volume = float(np.clip(self._poured_volume, 0, total))
        self._spilled_volume = float(np.clip(self._spilled_volume, 0, total))
        self._fluid_in_beaker = float(np.clip(self._fluid_in_beaker, 0, total))
