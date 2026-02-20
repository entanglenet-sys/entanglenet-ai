"""
"The Sommelier" — Chemical-Liquid Pouring Task (WP4 T4.2).

A UR5e-class 6-DoF arm on a differential-drive AMR must:
    1. Navigate to a lab bench where a beaker of chemical liquid sits
    2. Reach the beaker and grasp it (compliant grip)
    3. Transport the beaker to a target glass without spilling
    4. Tilt the wrist to pour the liquid precisely into the glass

Physical robot model (UR5e-class):
    - 6 revolute joints with hard position / velocity / torque limits
    - Link lengths: [0.152, 0.425, 0.392, 0.109, 0.095, 0.082] m
    - Payload: 5 kg  (beaker + liquid ≈ 0.55 kg → well within limits)
    - Base: max 1.0 m/s linear, 1.5 rad/s angular

Fluid model (simplified stub):
    - Pour rate = f(tilt_angle): zero below 15°, ramps linearly to max at 90°
    - Spill during transport if beaker tilt > 8° and NOT over target glass
    - Surface tension → small delay before flow starts

Success criteria (PHILOS KPIs):
    - Spill rate   < 5 % of original volume
    - Pour accuracy: volume error < 10 %
    - No collisions
    - Platform tilt < 10° throughout
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from philos.simulation.isaac_env import IsaacSimEnv, EnvConfig
from philos.core.registry import register_component
from philos.learning.reward_functions import DynamicRewardFunction

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
    APPROACH = "approach"       # navigate base & arm toward beaker
    GRASP = "grasp"             # close gripper on beaker
    TRANSPORT = "transport"     # carry beaker to glass
    POUR = "pour"               # tilt wrist → liquid flows
    DONE = "done"               # target volume reached


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class PourTaskConfig(EnvConfig):
    """Configuration for the chemical-liquid pouring task."""

    # ── Fluid ──────────────────────────────────────────────
    source_volume_ml: float = 250.0           # liquid in beaker at start
    target_volume_ml: float = 250.0           # how much to pour (= all of it)
    glass_capacity_ml: float = 350.0          # glass can hold
    max_spill_fraction: float = 0.05          # KPI < 5 %

    # ── Lab geometry (metres, world frame) ─────────────────
    beaker_start: tuple[float, float, float] = (0.50, 0.0, 0.85)
    glass_position: tuple[float, float, float] = (0.50, 0.38, 0.78)
    bench_height: float = 0.75

    # ── Robot arm — UR5e-class dimensions ──────────────────
    arm_base_height: float = 0.30              # AMR deck to shoulder
    link_lengths: tuple[float, ...] = (0.152, 0.425, 0.392, 0.109, 0.095, 0.082)
    joint_limits_lo: tuple[float, ...] = (-2.87, -1.76, -2.87, -3.07, -2.87, -6.28)
    joint_limits_hi: tuple[float, ...] = ( 2.87,  1.76,  2.87,  3.07,  2.87,  6.28)
    max_joint_velocity: tuple[float, ...] = (2.175, 2.175, 2.175, 3.49, 3.49, 3.49)
    base_max_linear: float = 1.0               # m / s
    base_max_angular: float = 1.5              # rad / s

    # ── Pouring physics ────────────────────────────────────
    pour_tilt_threshold_deg: float = 15.0      # min tilt for flow
    pour_rate_max_ml_per_step: float = 1.2     # ≈ 60 ml/s at 50 Hz
    spill_tilt_threshold_deg: float = 8.0      # spills during transport
    grasp_reach_m: float = 0.08                # EE must be within 8 cm to grasp
    pour_xy_tolerance_m: float = 0.12          # beaker must be within 12 cm over glass

    # ── Reward weights ─────────────────────────────────────
    approach_weight: float = 2.0
    grasp_reward: float = 15.0
    transport_weight: float = 3.0
    pour_accuracy_weight: float = 12.0
    spill_penalty_weight: float = 25.0
    smoothness_weight: float = 0.5
    success_bonus: float = 80.0

    obs_dim: int = 56


# ─── Environment ──────────────────────────────────────────────────────────────

@register_component("simulation", "pour_task")
class PourTaskEnv(IsaacSimEnv):
    """Chemical-liquid pouring with a physically constrained robot arm.

    Observation space (56-dim):
        [joint_pos(6), ee_pos(3), base_pos(3), beaker_pos(3),
         glass_pos(3), wrist_tilt_sin, wrist_tilt_cos, grasped,
         phase_onehot(5), fluid_in_beaker_frac, poured_frac,
         spill_frac, ee_to_beaker_dist, ee_to_glass_dist,
         ... padding/context(18)]

    Action space (10-dim):
        [base_vx, base_vy, base_omega, j1..j6, gripper]
    """

    def __init__(self, config: PourTaskConfig | None = None) -> None:
        self._task_config = config or PourTaskConfig()
        super().__init__(config=self._task_config)

        tc = self._task_config

        # ── Physical arm parameters ──
        self._link_lengths = np.array(tc.link_lengths, dtype=np.float64)
        self._joint_lo = np.array(tc.joint_limits_lo, dtype=np.float64)
        self._joint_hi = np.array(tc.joint_limits_hi, dtype=np.float64)
        self._max_jvel = np.array(tc.max_joint_velocity, dtype=np.float64)

        # ── Task state ──
        self._poured_volume: float = 0.0
        self._spilled_volume: float = 0.0
        self._grasped: bool = False
        self._phase: TaskPhase = TaskPhase.APPROACH
        self._beaker_pos = np.array(tc.beaker_start, dtype=np.float64)
        self._glass_pos = np.array(tc.glass_position, dtype=np.float64)
        self._fluid_in_beaker: float = tc.source_volume_ml  # ml
        self._wrist_tilt_deg: float = 0.0

        # ── Stub kinematic state ──
        self._stub_joint_pos = np.zeros(6, dtype=np.float64)
        self._stub_ee_pos = np.array([0.0, 0.0, 0.5], dtype=np.float64)
        self._stub_base_pos = np.zeros(3, dtype=np.float64)
        self._gripper_openness: float = 1.0  # 1 = open, 0 = closed

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
            logger.info("Pour task scene loaded (stub mode).")

    # ──────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────

    def _on_reset(self, options: dict | None = None) -> None:
        tc = self._task_config

        # Fluid
        self._poured_volume = 0.0
        self._spilled_volume = 0.0
        self._fluid_in_beaker = tc.source_volume_ml
        self._grasped = False
        self._phase = TaskPhase.APPROACH
        self._wrist_tilt_deg = 0.0

        # Scene objects
        self._beaker_pos = np.array(tc.beaker_start, dtype=np.float64)
        self._glass_pos = np.array(tc.glass_position, dtype=np.float64)

        # Robot starts at origin, arm in home configuration
        self._stub_base_pos = np.zeros(3, dtype=np.float64)
        self._stub_joint_pos = np.array([0.0, -0.5, 0.8, 0.0, -0.3, 0.0],
                                         dtype=np.float64)
        self._gripper_openness = 1.0
        self._stub_ee_pos = self._forward_kinematics()

    # ──────────────────────────────────────────────────────────────────────
    # Forward kinematics  (UR5e simplified — planar arm rotated by J1)
    # ──────────────────────────────────────────────────────────────────────

    def _forward_kinematics(self) -> np.ndarray:
        """Compute EE world position from joint angles + base position.

        Models a UR5e-like arm:
            J1 = azimuth rotation (vertical Z axis)
            J2 = shoulder lift   (vertical plane)
            J3 = elbow           (vertical plane)
            J4 = wrist-1 axial   (minimal positional effect in stub)
            J5 = wrist-2 tilt    (vertical plane)
            J6 = tool rotation   (no positional effect)
        """
        j = self._stub_joint_pos
        L = self._link_lengths  # [L0, L1, L2, L3, L4, L5]

        azimuth = j[0]
        shoulder = j[1]
        elbow = shoulder + j[2]
        wrist = elbow + j[4]  # J4 is axial → skip for position

        # Radial reach in horizontal plane & vertical height
        r = (L[1] * np.cos(shoulder)
             + L[2] * np.cos(elbow)
             + (L[3] + L[4] + L[5]) * np.cos(wrist))
        z = (self._task_config.arm_base_height + L[0]
             + L[1] * np.sin(shoulder)
             + L[2] * np.sin(elbow)
             + (L[3] + L[4] + L[5]) * np.sin(wrist))

        ee = np.array([
            self._stub_base_pos[0] + r * np.cos(azimuth),
            self._stub_base_pos[1] + r * np.sin(azimuth),
            z,
        ], dtype=np.float64)
        return ee

    def _compute_wrist_tilt(self) -> float:
        """Beaker tilt angle in degrees (from vertical) based on wrist joints."""
        j = self._stub_joint_pos
        # Cumulative arm angle in vertical plane → EE angle from horizontal
        angle_from_horiz = j[1] + j[2] + j[4]
        # Tilt from vertical = 90° − angle_from_horizontal
        tilt_deg = 90.0 - np.degrees(angle_from_horiz)
        return float(np.clip(tilt_deg, 0.0, 180.0))

    # ──────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_obs(self) -> np.ndarray:
        """Build the 56-dim observation."""
        obs = np.zeros(self._task_config.obs_dim, dtype=np.float32)
        tc = self._task_config
        total = tc.source_volume_ml if tc.source_volume_ml > 0 else 1.0

        # Robot state
        obs[0:6] = self._stub_joint_pos
        obs[6:9] = self._stub_ee_pos
        obs[9:12] = self._stub_base_pos

        # Beaker & glass
        obs[12:15] = self._beaker_pos
        obs[15:18] = self._glass_pos

        # Wrist tilt (sin/cos encoding)
        tilt_rad = np.radians(self._wrist_tilt_deg)
        obs[18] = np.sin(tilt_rad)
        obs[19] = np.cos(tilt_rad)

        # Grasped flag
        obs[20] = 1.0 if self._grasped else 0.0

        # Phase one-hot (5 phases)
        phase_idx = list(TaskPhase).index(self._phase)
        obs[21 + phase_idx] = 1.0

        # Fluid fractions
        obs[26] = self._fluid_in_beaker / total          # fluid remaining
        obs[27] = self._poured_volume / total             # poured
        obs[28] = self._spilled_volume / total            # spilled

        # Distance features
        ee_to_beaker = np.linalg.norm(self._stub_ee_pos - self._beaker_pos)
        ee_to_glass = np.linalg.norm(self._stub_ee_pos[:2] - self._glass_pos[:2])
        obs[29] = float(ee_to_beaker)
        obs[30] = float(ee_to_glass)

        # Context-vector placeholder (indices 38-55)
        obs[48:51] = self._beaker_pos
        obs[51:54] = self._glass_pos
        obs[54] = self._poured_volume / total
        obs[55] = self._spilled_volume / total

        return obs

    # ──────────────────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> float:
        tc = self._task_config
        total = tc.source_volume_ml if tc.source_volume_ml > 0 else 1.0
        reward = 0.0

        ee = self._stub_ee_pos
        d_beaker = float(np.linalg.norm(ee - self._beaker_pos))
        d_glass_xy = float(np.linalg.norm(ee[:2] - self._glass_pos[:2]))
        poured_frac = self._poured_volume / total
        spill_frac = self._spilled_volume / total
        target_frac = tc.target_volume_ml / total

        if self._phase == TaskPhase.APPROACH:
            # Reward approach to beaker
            reward += tc.approach_weight * max(0, 1.0 - d_beaker / 1.5)

        elif self._phase == TaskPhase.GRASP:
            # Bonus for grasping
            reward += tc.grasp_reward

        elif self._phase == TaskPhase.TRANSPORT:
            # Reward proximity to glass
            reward += tc.transport_weight * max(0, 1.0 - d_glass_xy / 1.0)
            # Small penalty for tilt during transport
            if self._wrist_tilt_deg > tc.spill_tilt_threshold_deg:
                reward -= 0.5

        elif self._phase == TaskPhase.POUR:
            # Reward pour progress
            reward += tc.pour_accuracy_weight * poured_frac

        # Always: spill penalty
        reward -= tc.spill_penalty_weight * spill_frac

        # Smoothness
        reward -= tc.smoothness_weight * float(np.sum(action ** 2)) * 0.01

        # Success bonus
        if poured_frac >= target_frac * 0.9 and spill_frac < tc.max_spill_fraction:
            reward += tc.success_bonus

        return float(reward)

    # ──────────────────────────────────────────────────────────────────────
    # Termination
    # ──────────────────────────────────────────────────────────────────────

    def _check_terminated(self) -> bool:
        tc = self._task_config
        total = tc.source_volume_ml if tc.source_volume_ml > 0 else 1.0

        # Success: poured enough
        if self._poured_volume / total >= (tc.target_volume_ml / total) * 0.90:
            self._phase = TaskPhase.DONE
            return True

        # Failure: too much spill
        if self._spilled_volume / total > tc.max_spill_fraction * 2:
            return True

        # Failure: beaker is empty but didn't pour enough (all spilled)
        if self._fluid_in_beaker <= 0 and self._poured_volume / total < 0.5:
            return True

        return False

    # ──────────────────────────────────────────────────────────────────────
    # Stub physics step — physically constrained
    # ──────────────────────────────────────────────────────────────────────

    def _stub_step(self, action: np.ndarray) -> None:
        """Physically realistic stub step (no Isaac Sim).

        Applies UR5e joint/velocity limits, proper FK, beaker-follows-EE
        when grasped, tilt-based pouring, spill during transport.
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] < 10:
            action = np.pad(action, (0, 10 - action.shape[0]))

        tc = self._task_config
        dt = tc.dt  # 0.02 s

        # ── 1. Base velocity (clamped) ──────────────────
        base_lin = action[:2] * tc.base_max_linear
        base_ang = np.clip(action[2], -1, 1) * tc.base_max_angular
        self._stub_base_pos[0] += float(base_lin[0]) * dt
        self._stub_base_pos[1] += float(base_lin[1]) * dt
        # (heading stored as base_pos[2] for simplicity)
        self._stub_base_pos[2] += float(base_ang) * dt

        # ── 2. Joint velocities (clamped to max) then positions (clamped) ──
        raw_vel = action[3:9]
        clamped_vel = np.clip(raw_vel, -self._max_jvel, self._max_jvel)
        new_joints = self._stub_joint_pos + clamped_vel * dt
        self._stub_joint_pos = np.clip(new_joints, self._joint_lo, self._joint_hi)

        # ── 3. Forward kinematics ──────────────────────
        self._stub_ee_pos = self._forward_kinematics()
        self._wrist_tilt_deg = self._compute_wrist_tilt()

        # ── 4. Gripper ──────────────────────────────────
        gripper_cmd = float(action[9]) if len(action) > 9 else 0.0
        self._gripper_openness = float(np.clip(1.0 - gripper_cmd, 0, 1))

        # ── 5. Grasping logic ───────────────────────────
        if not self._grasped:
            d = float(np.linalg.norm(self._stub_ee_pos - self._beaker_pos))
            if d < tc.grasp_reach_m and self._gripper_openness < 0.4:
                self._grasped = True
                self._phase = TaskPhase.GRASP
                logger.debug("Beaker grasped!")
        elif self._gripper_openness > 0.7:
            # Released
            self._grasped = False
            self._beaker_pos = self._stub_ee_pos.copy()

        # ── 6. Beaker follows EE when grasped ──────────
        if self._grasped:
            self._beaker_pos = self._stub_ee_pos.copy()

            # Update phase
            d_glass_xy = float(np.linalg.norm(
                self._stub_ee_pos[:2] - self._glass_pos[:2]))
            if d_glass_xy < tc.pour_xy_tolerance_m:
                if self._wrist_tilt_deg > tc.pour_tilt_threshold_deg:
                    self._phase = TaskPhase.POUR
                else:
                    self._phase = TaskPhase.TRANSPORT
            elif self._phase not in (TaskPhase.POUR,):
                self._phase = TaskPhase.TRANSPORT

        else:
            if self._phase not in (TaskPhase.DONE,):
                self._phase = TaskPhase.APPROACH

        # ── 7. Fluid dynamics ──────────────────────────
        if self._grasped and self._fluid_in_beaker > 0:
            d_glass_xy = float(np.linalg.norm(
                self._beaker_pos[:2] - self._glass_pos[:2]))
            over_glass = d_glass_xy < tc.pour_xy_tolerance_m

            tilt = self._wrist_tilt_deg

            if tilt > tc.pour_tilt_threshold_deg:
                # Pouring — rate proportional to tilt
                tilt_frac = min(1.0,
                    (tilt - tc.pour_tilt_threshold_deg)
                    / (90.0 - tc.pour_tilt_threshold_deg))
                flow_ml = tc.pour_rate_max_ml_per_step * tilt_frac

                flow_ml = min(flow_ml, self._fluid_in_beaker)
                self._fluid_in_beaker -= flow_ml

                if over_glass:
                    self._poured_volume += flow_ml * 0.97  # 3 % splash loss
                    self._spilled_volume += flow_ml * 0.03
                else:
                    # Pouring but not over glass → all spills
                    self._spilled_volume += flow_ml

            elif tilt > tc.spill_tilt_threshold_deg:
                # Slight tilt during transport — small drip
                drip = 0.02 * (tilt - tc.spill_tilt_threshold_deg) / 10.0
                drip = min(drip, self._fluid_in_beaker)
                self._fluid_in_beaker -= drip
                self._spilled_volume += drip

        # ── 8. Cap volumes ─────────────────────────────
        total = tc.source_volume_ml
        self._poured_volume = float(np.clip(self._poured_volume, 0, total))
        self._spilled_volume = float(np.clip(self._spilled_volume, 0, total))
        self._fluid_in_beaker = float(np.clip(self._fluid_in_beaker, 0, total))
