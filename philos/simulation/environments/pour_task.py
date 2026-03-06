"""
"The Sommelier" — Chemical-Liquid Pick-and-Pour Task (WP4 T4.2).

A UR5e 6-DoF arm with a parallel-jaw gripper on a FIXED PEDESTAL
must perform a complete pick-and-pour sequence:
    1. APPROACH — Move gripper (open) above the source beaker on the bench
    2. GRASP   — Close gripper fingers around the beaker
    3. LIFT    — Lift the beaker off the bench surface
    4. POUR    — Transport beaker over the target glass, tilt to pour
    5. DONE    — Target volume reached

The robot model (including the Robotiq-style gripper) is loaded from
the project URDF file, which is the single source of truth for link
lengths, joint limits, masses, and visual geometry.  Both this
simulation and the 3D digital twin read from the same URDF.

Physical constraints (from URDF):
    - 6 revolute arm joints with hard position/velocity/torque limits
    - 2 prismatic finger joints (parallel-jaw gripper)
    - Real UR5e DH parameters
    - Gravity: 9.81 m/s² (affects liquid, falling objects, joint torques)
    - Payload: beaker + liquid ≈ 0.55 kg

Gripper notes:
    - Beaker diameter ≈ 70 mm → grasp at finger_q ≈ 0.027 per finger
    - Finger gap: 16 mm (fully closed) → 96 mm (fully open)
    - Grasped when: proximity OK + fingers closed on beaker

Fluid model (simplified physics):
    - Pour rate = f(beaker_tilt): zero below 45°, ramps to max at 90°
    - Beaker tilt computed from gripper orientation: tilt = arccos(-tool_z · up)
    - When beaker is on table, tilt = 0° (upright)
    - Spill if beaker tilt > 20° when not over glass

Success criteria (PHILOS KPIs):
    - Spill rate   < 5 % of original volume
    - Pour accuracy: volume error < 10 %
    - No collisions with bench/objects
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
    APPROACH = "approach"  # Move to above source beaker, open gripper
    GRASP    = "grasp"     # Close gripper around beaker
    LIFT     = "lift"      # Lift beaker off the bench
    POUR     = "pour"      # Move over glass + tilt to pour
    DONE     = "done"      # Success


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class PourTaskConfig(EnvConfig):
    """Configuration for the pick-and-pour task.

    Robot geometry (including Robotiq-style gripper) is loaded from URDF.
    """

    # ── Fluid ──────────────────────────────────────────────
    source_volume_ml: float = 250.0
    target_volume_ml: float = 250.0
    glass_capacity_ml: float = 350.0
    max_spill_fraction: float = 0.05

    # ── Lab geometry (metres, URDF world frame) ────────────
    # Robot pedestal at origin.  Bench is in FRONT of robot.
    source_beaker_position: tuple[float, float, float] = (0.35, 0.10, 0.815)
    target_glass_position: tuple[float, float, float] = (0.60, 0.35, 0.78)
    bench_height: float = 0.75
    beaker_radius: float = 0.035   # 70 mm diameter
    beaker_height: float = 0.09    # 90 mm tall
    beaker_mass: float = 0.55      # kg, beaker+liquid

    # Bench collision AABB (URDF frame: x=forward, y=left, z=up)
    bench_x_min: float = 0.15
    bench_x_max: float = 0.85
    bench_y_min: float = -0.05
    bench_y_max: float = 0.55
    bench_z_top: float = 0.77  # slightly above surface (collision margin)

    # ── Gripper geometry ───────────────────────────────────
    finger_closed_offset: float = 0.008   # metres each side at q=0
    finger_max_travel: float = 0.040      # metres per finger
    grasp_gap_tolerance: float = 0.010    # gap tolerance for grasping
    grasp_proximity_xy: float = 0.04      # XY distance threshold
    grasp_proximity_z:  float = 0.05      # Z distance threshold

    # ── Joint velocity scaling ─────────────────────────────
    max_joint_velocity: tuple[float, ...] = (
        2.094, 2.094, 3.142, 3.491, 3.491, 3.491)

    # ── Pouring physics ────────────────────────────────────
    pour_tilt_threshold_deg: float = 45.0
    pour_rate_max_ml_per_step: float = 1.5
    spill_tilt_threshold_deg: float = 20.0
    pour_xy_tolerance_m: float = 0.08
    pour_arrive_distance: float = 0.10
    lift_clearance: float = 0.08

    # ── Reward weights ─────────────────────────────────────
    approach_weight: float = 8.0
    grasp_bonus: float = 20.0
    lift_weight: float = 6.0
    pour_progress_weight: float = 15.0
    spill_penalty_weight: float = 30.0
    collision_penalty_weight: float = 50.0
    orientation_weight: float = 4.0
    smoothness_weight: float = 0.3
    success_bonus: float = 100.0
    time_penalty: float = 0.05

    obs_dim: int = 56
    action_dim: int = 7  # 6 arm joint velocities + 1 gripper

    # Legacy aliases used by old code
    glass_position: tuple[float, float, float] = (0.60, 0.35, 0.78)


# ─── Environment ──────────────────────────────────────────────────────────────

@register_component("simulation", "pour_task")
class PourTaskEnv(IsaacSimEnv):
    """Pick-and-pour with a URDF-driven UR5e + parallel-jaw gripper.

    Observation space (56-dim):
        [arm_joint_pos_norm(6), gripper_tip_pos(3),
         source_beaker_pos(3), target_glass_pos(3),
         beaker_tilt_sin, beaker_tilt_cos,
         grasped, gripper_openness,
         phase_onehot(4),
         fluid_in_beaker_frac, poured_frac, spill_frac,
         ee_to_beaker_xy, ee_to_beaker_z,
         ee_to_glass_xy, ee_to_glass_z,
         arm_joint_vel(6), prev_reward,
         collision, ...context(18)]

    Action space (7-dim):
        [j1_vel, j2_vel, j3_vel, j4_vel, j5_vel, j6_vel, gripper_cmd]
        gripper_cmd: -1 = close, +1 = full open
    """

    def __init__(self, config: PourTaskConfig | None = None) -> None:
        self._task_config = config or PourTaskConfig()
        super().__init__(config=self._task_config)

        tc = self._task_config

        # ── Load robot model from URDF ──
        self._urdf: URDFRobot = load_default_robot()

        # Separate arm joints from gripper (finger) joints
        all_act = self._urdf.actuated_joints
        self._arm_joint_names = [j for j in all_act if 'finger' not in j]
        self._grip_joint_names = [j for j in all_act if 'finger' in j]
        self._num_arm_joints = len(self._arm_joint_names)
        self._num_grip_joints = len(self._grip_joint_names)
        assert self._num_arm_joints == 6, (
            f"Expected 6 arm joints, got {self._num_arm_joints}")

        # Arm joint limits from URDF
        self._joint_lo = np.array([
            self._urdf.joints[jn].lower for jn in self._arm_joint_names
        ], dtype=np.float64)
        self._joint_hi = np.array([
            self._urdf.joints[jn].upper for jn in self._arm_joint_names
        ], dtype=np.float64)
        self._max_jvel = np.array(
            tc.max_joint_velocity[:6], dtype=np.float64)

        # Finger joint limits
        self._finger_lo = np.array([
            self._urdf.joints[jn].lower for jn in self._grip_joint_names
        ], dtype=np.float64)
        self._finger_hi = np.array([
            self._urdf.joints[jn].upper for jn in self._grip_joint_names
        ], dtype=np.float64)

        # ── Kinematic state ──
        self._joint_pos = np.zeros(6, dtype=np.float64)
        self._joint_vel = np.zeros(6, dtype=np.float64)
        self._finger_pos = np.zeros(self._num_grip_joints, dtype=np.float64)
        self._ee_pos = np.zeros(3, dtype=np.float64)
        self._ee_rot = np.eye(3)
        self._gripper_tip_pos = np.zeros(3, dtype=np.float64)
        self._gripper_openness: float = 0.0

        # ── Task state ──
        self._phase: TaskPhase = TaskPhase.APPROACH
        self._grasped: bool = False
        self._poured_volume: float = 0.0
        self._spilled_volume: float = 0.0
        self._fluid_in_beaker: float = tc.source_volume_ml
        self._beaker_tilt_deg: float = 0.0
        self._wrist_tilt_deg: float = 0.0  # legacy alias

        # Source beaker (free object on bench)
        self._beaker_pos = np.array(
            tc.source_beaker_position, dtype=np.float64)
        self._beaker_vel = np.zeros(3, dtype=np.float64)
        self._beaker_on_bench: bool = True

        # Target glass
        self._glass_pos = np.array(
            tc.target_glass_position, dtype=np.float64)

        # ── Collision state ──
        self._collision: bool = False
        self._max_penetration: float = 0.0
        self._min_bench_clearance: float = 1.0
        self._link_positions: dict[str, np.ndarray] = {}

        # ── Reward tracking ──
        self._prev_reward: float = 0.0
        self._prev_ee_to_beaker: float = 1.0
        self._prev_ee_to_glass: float = 1.0
        self._episode_rewards: list = []
        self._grasp_step: int = 0

        # ── Backward compat aliases for server.py ──
        self._stub_joint_pos = self._joint_pos
        self._stub_ee_pos = self._ee_pos
        self._stub_base_pos = np.zeros(3, dtype=np.float64)

        # Reward function
        self._reward_fn = DynamicRewardFunction()

    # ──────────────────────────────────────────────────────────────────
    # Scene
    # ──────────────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        if _ISAAC_AVAILABLE and self._world is not None:
            self._world.scene.add_default_ground_plane()
            logger.info("Pour task scene loaded (Isaac Sim).")
        else:
            logger.info(
                f"Pour task scene loaded (stub, URDF={self._urdf.name}, "
                f"{self._num_arm_joints} arm + "
                f"{self._num_grip_joints} finger joints).")

    # ──────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────

    def _on_reset(self, options: dict | None = None) -> None:
        tc = self._task_config

        # Fluid
        self._poured_volume = 0.0
        self._spilled_volume = 0.0
        self._fluid_in_beaker = tc.source_volume_ml
        self._beaker_tilt_deg = 0.0
        self._wrist_tilt_deg = 0.0
        self._prev_reward = 0.0

        # Phase
        self._phase = TaskPhase.APPROACH
        self._grasped = False
        self._grasp_step = 0

        # Collision
        self._collision = False
        self._max_penetration = 0.0
        self._min_bench_clearance = 1.0

        # Source beaker on bench (with domain randomization)
        self._beaker_pos = np.array(
            tc.source_beaker_position, dtype=np.float64)
        self._beaker_pos[:2] += np.random.uniform(-0.03, 0.03, size=2)
        self._beaker_vel = np.zeros(3, dtype=np.float64)
        self._beaker_on_bench = True

        # Target glass on bench
        self._glass_pos = np.array(
            tc.target_glass_position, dtype=np.float64)
        self._glass_pos[:2] += np.random.uniform(-0.03, 0.03, size=2)

        # Gripper starts OPEN (ready to approach)
        self._gripper_openness = 1.0
        self._finger_pos = self._finger_hi.copy()

        # Home pose: tool pointing DOWN, above bench
        # EE ≈ (0.728, 0.308, 1.065), gripper_tip z ≈ 0.942
        self._joint_pos = np.array(
            [0.4, -0.8, 1.2, -1.97, -1.5708, 3.14159],
            dtype=np.float64)
        self._joint_vel = np.zeros(6, dtype=np.float64)

        # Aliases
        self._stub_joint_pos = self._joint_pos
        self._stub_base_pos = np.zeros(3, dtype=np.float64)

        # FK
        self._update_kinematics()
        self._prev_ee_to_beaker = float(np.linalg.norm(
            self._gripper_tip_pos[:2] - self._beaker_pos[:2]))
        self._prev_ee_to_glass = float(np.linalg.norm(
            self._gripper_tip_pos[:2] - self._glass_pos[:2]))
        self._episode_rewards = []

    # ──────────────────────────────────────────────────────────────────
    # Forward kinematics
    # ──────────────────────────────────────────────────────────────────

    def _update_kinematics(self) -> None:
        """Recompute EE/gripper poses from URDF FK."""
        full_angles = np.concatenate([self._joint_pos, self._finger_pos])
        transforms = forward_kinematics(self._urdf, full_angles)

        # Tool0 (EE flange)
        tool_T = transforms.get("tool0", np.eye(4))
        self._ee_pos[:] = tool_T[:3, 3]
        self._ee_rot = tool_T[:3, :3].copy()

        # Gripper tip
        tip_T = transforms.get("gripper_tip", tool_T)
        self._gripper_tip_pos[:] = tip_T[:3, 3]

        # Beaker tilt: when grasped, open end = -tool_z
        tool_z = self._ee_rot[:, 2]
        if self._grasped:
            beaker_up = -tool_z
        else:
            beaker_up = np.array([0.0, 0.0, 1.0])
        cos_a = np.clip(np.dot(beaker_up, [0, 0, 1]), -1.0, 1.0)
        self._beaker_tilt_deg = float(np.degrees(np.arccos(cos_a)))
        self._wrist_tilt_deg = self._beaker_tilt_deg

        # Link positions for collision
        self._link_positions = {}
        for name, T in transforms.items():
            self._link_positions[name] = T[:3, 3].copy()

        self._check_bench_collision()
        self._stub_ee_pos = self._ee_pos
        self._stub_joint_pos = self._joint_pos

    def _check_bench_collision(self) -> None:
        """Check if any moving link intersects the bench AABB."""
        tc = self._task_config
        self._collision = False
        self._max_penetration = 0.0
        self._min_bench_clearance = 10.0

        skip = {'world', 'pedestal', 'base_link'}
        for name, pos in self._link_positions.items():
            if name in skip:
                continue
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            in_x = tc.bench_x_min <= x <= tc.bench_x_max
            in_y = tc.bench_y_min <= y <= tc.bench_y_max
            if in_x and in_y:
                clearance = z - tc.bench_z_top
                self._min_bench_clearance = min(
                    self._min_bench_clearance, clearance)
                if clearance < 0:
                    self._collision = True
                    self._max_penetration = max(
                        self._max_penetration, -clearance)

    # ──────────────────────────────────────────────────────────────────
    # Gripper helpers
    # ──────────────────────────────────────────────────────────────────

    def _finger_gap(self) -> float:
        """Current gap between finger pads (metres)."""
        tc = self._task_config
        return 2.0 * (tc.finger_closed_offset + self._finger_pos[0])

    def _check_grasp(self) -> bool:
        """Check if gripper has grasped the source beaker."""
        tc = self._task_config

        if self._grasped:
            # Release check
            if self._gripper_openness > 0.85:
                return False
            return True

        # Proximity
        tip = self._gripper_tip_pos
        bk = self._beaker_pos
        dxy = float(np.linalg.norm(tip[:2] - bk[:2]))
        dz = abs(tip[2] - bk[2])
        if dxy > tc.grasp_proximity_xy or dz > tc.grasp_proximity_z:
            return False

        # Finger closure
        gap = self._finger_gap()
        beaker_diam = 2.0 * tc.beaker_radius
        if gap > beaker_diam + tc.grasp_gap_tolerance:
            return False
        if gap < beaker_diam * 0.4:
            return False

        # Must be on bench still
        if not self._beaker_on_bench:
            return False

        return True

    # ──────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────

    def _compute_obs(self) -> np.ndarray:
        """Build 56-dim observation."""
        obs = np.zeros(self._task_config.obs_dim, dtype=np.float32)
        tc = self._task_config
        total = tc.source_volume_ml if tc.source_volume_ml > 0 else 1.0

        # [0:6] Arm joints normalised
        for i in range(6):
            rng = self._joint_hi[i] - self._joint_lo[i]
            if rng > 0:
                obs[i] = (2.0 * (self._joint_pos[i] - self._joint_lo[i])
                          / rng - 1.0)

        # [6:9] Gripper tip position
        obs[6:9] = self._gripper_tip_pos
        # [9:12] Source beaker
        obs[9:12] = self._beaker_pos
        # [12:15] Target glass
        obs[12:15] = self._glass_pos

        # [15:17] Beaker tilt sin/cos
        tilt_rad = np.radians(self._beaker_tilt_deg)
        obs[15] = np.sin(tilt_rad)
        obs[16] = np.cos(tilt_rad)

        # [17] Grasped
        obs[17] = 1.0 if self._grasped else 0.0
        # [18] Gripper openness
        obs[18] = self._gripper_openness

        # [19:23] Phase one-hot
        phase_map = {
            TaskPhase.APPROACH: 19, TaskPhase.GRASP: 20,
            TaskPhase.LIFT: 21, TaskPhase.POUR: 22}
        if self._phase in phase_map:
            obs[phase_map[self._phase]] = 1.0

        # [23:26] Fluid fractions
        obs[23] = self._fluid_in_beaker / total
        obs[24] = self._poured_volume / total
        obs[25] = self._spilled_volume / total

        # [26:30] Distances
        tip = self._gripper_tip_pos
        obs[26] = float(np.linalg.norm(tip[:2] - self._beaker_pos[:2]))
        obs[27] = tip[2] - self._beaker_pos[2]
        obs[28] = float(np.linalg.norm(tip[:2] - self._glass_pos[:2]))
        obs[29] = tip[2] - self._glass_pos[2]

        # [30:36] Joint velocities
        obs[30:36] = self._joint_vel / (self._max_jvel + 1e-8)
        # [36] Previous reward
        obs[36] = np.clip(self._prev_reward / 20.0, -1.0, 1.0)
        # [37] Collision
        obs[37] = 1.0 if self._collision else 0.0

        # [38:56] Context
        obs[38:41] = self._gripper_tip_pos
        obs[41:44] = self._beaker_pos
        obs[44:47] = self._glass_pos
        obs[47] = self._poured_volume / total
        obs[48] = self._spilled_volume / total
        obs[49] = self._beaker_tilt_deg / 180.0
        obs[50] = float(self._phase == TaskPhase.POUR)
        obs[51] = float(self._grasped)
        obs[52] = self._gripper_openness
        obs[53] = float(self._beaker_on_bench)
        obs[54] = np.clip(self._min_bench_clearance / 0.5, -1.0, 1.0)
        obs[55] = self._max_penetration * 10.0

        return obs

    # ──────────────────────────────────────────────────────────────────
    # Reward — shaped for VISIBLE learning on pick-and-pour
    # ──────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> float:
        tc = self._task_config
        total = tc.source_volume_ml if tc.source_volume_ml > 0 else 1.0
        reward = 0.0

        tip = self._gripper_tip_pos
        bk = self._beaker_pos
        gl = self._glass_pos

        d_bk_xy = float(np.linalg.norm(tip[:2] - bk[:2]))
        d_bk_z = abs(tip[2] - bk[2])
        d_gl_xy = float(np.linalg.norm(tip[:2] - gl[:2]))
        poured_frac = self._poured_volume / total
        spill_frac = self._spilled_volume / total
        target_frac = tc.target_volume_ml / total

        # ── 0) Collision penalty ──
        if self._collision:
            reward -= tc.collision_penalty_weight * (
                1.0 + self._max_penetration * 10.0)
        elif self._min_bench_clearance < 0.03:
            reward -= tc.collision_penalty_weight * 0.2 * (
                0.03 - self._min_bench_clearance) / 0.03

        # ── Phase-specific rewards ──

        if self._phase == TaskPhase.APPROACH:
            # Dense approach reward
            delta = self._prev_ee_to_beaker - d_bk_xy
            reward += tc.approach_weight * delta * 10.0
            if d_bk_xy < 0.3:
                reward += tc.approach_weight * (0.3 - d_bk_xy) * 2.0
            if d_bk_z < 0.1:
                reward += 2.0 * (0.1 - d_bk_z) / 0.1
            # Keep gripper open
            reward += 0.5 * self._gripper_openness
            # Tool should point down
            if self._beaker_tilt_deg < 30.0:
                reward += 1.0
            else:
                reward -= tc.orientation_weight * (
                    self._beaker_tilt_deg - 30.0) / 150.0

        elif self._phase == TaskPhase.GRASP:
            if self._grasped:
                reward += tc.grasp_bonus
            else:
                reward += 2.0 * (1.0 - self._gripper_openness)
                if d_bk_xy < tc.grasp_proximity_xy:
                    reward += 3.0

        elif self._phase == TaskPhase.LIFT:
            lift_h = self._beaker_pos[2] - tc.bench_z_top
            if lift_h > 0:
                reward += tc.lift_weight * min(
                    lift_h / tc.lift_clearance, 1.0)
            if self._beaker_tilt_deg < 10.0:
                reward += 1.0
            elif self._beaker_tilt_deg > 20.0:
                reward -= tc.orientation_weight * (
                    self._beaker_tilt_deg - 20.0) / 160.0

        elif self._phase == TaskPhase.POUR:
            glass_delta = self._prev_ee_to_glass - d_gl_xy
            reward += tc.approach_weight * glass_delta * 5.0
            reward += tc.pour_progress_weight * poured_frac
            if d_gl_xy < tc.pour_xy_tolerance_m:
                reward += 3.0
                if 45 <= self._beaker_tilt_deg <= 120:
                    reward += 2.0
            else:
                reward -= 2.0 * (d_gl_xy - tc.pour_xy_tolerance_m)

        # ── Always-on ──
        reward -= tc.spill_penalty_weight * spill_frac
        action_mag = float(np.sum(action[:6] ** 2))
        reward -= tc.smoothness_weight * action_mag * 0.01
        reward -= tc.time_penalty

        if (poured_frac >= target_frac * 0.90
                and spill_frac < tc.max_spill_fraction):
            reward += tc.success_bonus

        self._prev_ee_to_beaker = d_bk_xy
        self._prev_ee_to_glass = d_gl_xy
        self._prev_reward = reward
        self._episode_rewards.append(reward)
        return float(reward)

    # ──────────────────────────────────────────────────────────────────
    # Termination
    # ──────────────────────────────────────────────────────────────────

    def _check_terminated(self) -> bool:
        tc = self._task_config
        total = tc.source_volume_ml if tc.source_volume_ml > 0 else 1.0

        # Success
        if self._poured_volume / total >= (
                tc.target_volume_ml / total) * 0.90:
            self._phase = TaskPhase.DONE
            return True
        # Collision
        if self._collision and self._max_penetration > 0.02:
            return True
        # Too much spill
        if self._spilled_volume / total > tc.max_spill_fraction * 2:
            return True
        # Beaker empty without enough pour
        if (self._fluid_in_beaker <= 0
                and self._poured_volume / total < 0.5):
            return True
        # EE too low
        if self._ee_pos[2] < 0.2:
            return True
        # Beaker fell to floor
        if (not self._grasped and not self._beaker_on_bench
                and self._beaker_pos[2] < 0.1):
            return True
        return False

    # ──────────────────────────────────────────────────────────────────
    # Stub physics step — gripper + free beaker + collision
    # ──────────────────────────────────────────────────────────────────

    def _stub_step(self, action: np.ndarray) -> None:
        """Physics step with gripper + free beaker + collision.

        Action: [j1..j6 velocity, gripper_cmd] — 7 dims.
        gripper_cmd: -1 = close, +1 = full open
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] < 7:
            action = np.pad(action, (0, 7 - action.shape[0]))

        tc = self._task_config
        dt = tc.dt

        # ── 1. Arm joints ──
        raw_vel = action[:6] * self._max_jvel
        clamped_vel = np.clip(raw_vel, -self._max_jvel, self._max_jvel)
        for i, jname in enumerate(self._arm_joint_names):
            damp = self._urdf.joints[jname].damping
            clamped_vel[i] -= damp * self._joint_vel[i] * dt

        new_joints = self._joint_pos + clamped_vel * dt
        self._joint_pos = np.clip(
            new_joints, self._joint_lo, self._joint_hi)
        self._joint_vel = clamped_vel.copy()

        # ── 2. Gripper finger control ──
        grip_cmd = float(np.clip(action[6], -1.0, 1.0))
        target = (grip_cmd + 1.0) / 2.0  # map to [0,1]
        grip_speed = 3.0
        delta = (target - self._gripper_openness) * min(
            1.0, grip_speed * dt)
        self._gripper_openness = float(np.clip(
            self._gripper_openness + delta, 0.0, 1.0))
        self._finger_pos = (self._finger_lo
            + (self._finger_hi - self._finger_lo) * self._gripper_openness)

        # ── 3. FK + collision ──
        self._update_kinematics()
        if self._collision:
            self._joint_pos = np.clip(
                self._joint_pos - clamped_vel * dt * 0.5,
                self._joint_lo, self._joint_hi)
            self._update_kinematics()

        # ── 4. Grasp detection ──
        was_grasped = self._grasped
        self._grasped = self._check_grasp()
        if self._grasped and not was_grasped:
            self._grasp_step = self.num_steps

        # ── 5. Source beaker physics ──
        if self._grasped:
            self._beaker_pos[:] = self._gripper_tip_pos
            self._beaker_vel[:] = 0.0
            self._beaker_on_bench = False
        else:
            # Gravity
            self._beaker_vel[2] += -9.81 * dt
            self._beaker_pos += self._beaker_vel * dt

            # Bench support
            bx, by = self._beaker_pos[0], self._beaker_pos[1]
            on_xy = (tc.bench_x_min < bx < tc.bench_x_max
                     and tc.bench_y_min < by < tc.bench_y_max)
            beaker_bottom = self._beaker_pos[2] - tc.beaker_height / 2.0
            bench_surface = tc.bench_height + 0.02

            if on_xy and beaker_bottom <= bench_surface:
                self._beaker_pos[2] = bench_surface + tc.beaker_height / 2.0
                self._beaker_vel[2] = max(0.0, self._beaker_vel[2])
                self._beaker_vel[:2] *= 0.9  # friction
                self._beaker_on_bench = True
            elif self._beaker_pos[2] < 0.01:
                self._beaker_pos[2] = 0.01
                self._beaker_vel[:] = 0.0
                self._beaker_on_bench = False

        # ── 6. Phase transitions ──
        tip = self._gripper_tip_pos
        d_bk_xy = float(np.linalg.norm(tip[:2] - self._beaker_pos[:2]))
        d_bk_z = abs(tip[2] - self._beaker_pos[2])

        if self._phase == TaskPhase.APPROACH:
            if (d_bk_xy < tc.grasp_proximity_xy * 1.5
                    and d_bk_z < tc.grasp_proximity_z * 1.5):
                self._phase = TaskPhase.GRASP

        elif self._phase == TaskPhase.GRASP:
            if self._grasped:
                self._phase = TaskPhase.LIFT
            elif d_bk_xy > tc.grasp_proximity_xy * 3.0:
                self._phase = TaskPhase.APPROACH

        elif self._phase == TaskPhase.LIFT:
            if not self._grasped:
                self._phase = TaskPhase.APPROACH
            elif self._beaker_pos[2] > tc.bench_z_top + tc.lift_clearance:
                self._phase = TaskPhase.POUR

        elif self._phase == TaskPhase.POUR:
            if not self._grasped:
                self._phase = TaskPhase.APPROACH

        # ── 7. Fluid dynamics ──
        if self._fluid_in_beaker > 0 and self._grasped:
            d_gl_xy = float(np.linalg.norm(
                self._beaker_pos[:2] - self._glass_pos[:2]))
            over_glass = d_gl_xy < tc.pour_xy_tolerance_m
            tilt = self._beaker_tilt_deg

            if tilt > tc.pour_tilt_threshold_deg:
                tilt_frac = min(1.0,
                    (tilt - tc.pour_tilt_threshold_deg)
                    / (90.0 - tc.pour_tilt_threshold_deg))
                grav_f = 0.5 + 0.5 * (tilt_frac ** 0.5)
                flow = tc.pour_rate_max_ml_per_step * tilt_frac * grav_f
                flow = min(flow, self._fluid_in_beaker)
                self._fluid_in_beaker -= flow

                if over_glass and self._beaker_pos[2] > self._glass_pos[2]:
                    acc = max(0.5,
                        1.0 - d_gl_xy / tc.pour_xy_tolerance_m)
                    self._poured_volume += flow * acc
                    self._spilled_volume += flow * (1.0 - acc)
                else:
                    self._spilled_volume += flow

            elif tilt > tc.spill_tilt_threshold_deg:
                drip = 0.03 * (tilt - tc.spill_tilt_threshold_deg) / 10.0
                drip = min(drip, self._fluid_in_beaker)
                self._fluid_in_beaker -= drip
                self._spilled_volume += drip

        elif self._fluid_in_beaker > 0 and not self._grasped:
            if not self._beaker_on_bench and self._beaker_pos[2] < 0.5:
                spill = min(5.0, self._fluid_in_beaker)
                self._fluid_in_beaker -= spill
                self._spilled_volume += spill

        # Cap
        vol = tc.source_volume_ml
        self._poured_volume = float(np.clip(self._poured_volume, 0, vol))
        self._spilled_volume = float(np.clip(self._spilled_volume, 0, vol))
        self._fluid_in_beaker = float(np.clip(
            self._fluid_in_beaker, 0, vol))
