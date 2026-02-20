"""
Pydantic schemas — API contracts for the PHILOS system.

These define the payloads exchanged between loosely-coupled modules via REST/WebSocket.
Every module communicates exclusively through these schemas, ensuring decoupling.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─── Enums ──────────────────────────────────────────────────────────────────────


class ManipulationModeEnum(str, Enum):
    STIFF = "stiff"
    COMPLIANT = "compliant"
    FLUID = "fluid"
    PRECISION = "precision"


class SafetyLevel(str, Enum):
    NOMINAL = "nominal"
    WARNING = "warning"
    OVERRIDE = "override"  # MPC overriding AI command
    EMERGENCY_STOP = "emergency_stop"


class ModuleStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    OFFLINE = "offline"


# ─── Language Command ────────────────────────────────────────────────────────────


class LanguageCommandRequest(BaseModel):
    """Natural language command from the operator.

    Example: {"command": "Pour the blue liquid carefully", "scene_image_b64": "..."}
    """

    command: str = Field(..., description="Natural language instruction for the robot")
    scene_image_b64: str | None = Field(
        None, description="Optional base64-encoded RGB image of the current scene"
    )
    priority: int = Field(1, ge=1, le=5, description="Command priority (1=normal, 5=critical)")


class LanguageCommandResponse(BaseModel):
    """Response after processing a language command."""

    success: bool
    context_vector_id: str
    interpretation: str = Field(..., description="Human-readable interpretation of the command")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")
    constraints_generated: int = Field(
        ..., description="Number of reward constraints generated"
    )


# ─── Context Vector ─────────────────────────────────────────────────────────────


class ContextVectorPayload(BaseModel):
    """Serialized Context Vector (z) for API transport.

    This is the central data contract — produced by the Cognitive module,
    consumed by the Learning module to condition the RL policy.
    """

    embedding: list[float] = Field(..., description="Latent embedding vector")
    manipulation_mode: ManipulationModeEnum = ManipulationModeEnum.STIFF
    impedance_scale: float = Field(0.5, ge=0.0, le=1.0)
    velocity_limit_scale: float = Field(1.0, ge=0.0, le=1.0)
    orientation_constraint: list[float] = Field(
        default=[0.0, 0.0, 1.0], description="Target orientation (unit vector)"
    )
    jerk_penalty: float = Field(0.1, ge=0.0)
    semantic_labels: list[str] = []
    language_command: str = ""
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    timestamp: float = 0.0


# ─── Robot State ─────────────────────────────────────────────────────────────────


class JointStatePayload(BaseModel):
    name: str
    position: float
    velocity: float
    effort: float


class RobotStatePayload(BaseModel):
    """Complete robot state observation — the 's' in π(s, z)."""

    amr_position: list[float] = Field(default=[0, 0, 0], min_length=3, max_length=3)
    amr_orientation: list[float] = Field(default=[0, 0, 0, 1], min_length=4, max_length=4)
    amr_linear_velocity: list[float] = Field(default=[0, 0, 0])
    amr_angular_velocity: list[float] = Field(default=[0, 0, 0])
    joints: list[JointStatePayload] = []
    ee_position: list[float] = Field(default=[0, 0, 0])
    ee_orientation: list[float] = Field(default=[0, 0, 0, 1])
    ee_force: list[float] = Field(default=[0, 0, 0])
    ee_torque: list[float] = Field(default=[0, 0, 0])
    gripper_state: float = 0.0
    lidar_scan: list[float] = []
    fluid_level: float = 0.0
    timestamp: float = 0.0


# ─── Policy Action ───────────────────────────────────────────────────────────────


class PolicyActionResponse(BaseModel):
    """Action output from the RL policy — before safety filtering.

    Sent to the Safety Shield (MPC) for validation before execution.
    """

    base_velocity: list[float] = Field(
        default=[0, 0, 0], description="AMR base velocity [vx, vy, omega]"
    )
    joint_velocities: list[float] = Field(
        default=[], description="Target joint velocities for the arm"
    )
    gripper_command: float = Field(0.0, ge=0.0, le=1.0, description="Gripper open/close")
    policy_confidence: float = Field(1.0, ge=0.0, le=1.0)
    timestamp: float = 0.0


# ─── Safety ──────────────────────────────────────────────────────────────────────


class SafetyStatus(BaseModel):
    """Status report from the Safety Shield (MPC)."""

    level: SafetyLevel = SafetyLevel.NOMINAL
    override_active: bool = False
    original_action: PolicyActionResponse | None = None
    filtered_action: PolicyActionResponse | None = None
    violations: list[str] = []
    mpc_solve_time_ms: float = 0.0
    timestamp: float = 0.0


# ─── System Health ───────────────────────────────────────────────────────────────


class ModuleHealth(BaseModel):
    module: str
    status: ModuleStatus
    latency_ms: float = 0.0
    last_heartbeat: float = 0.0
    details: dict[str, Any] = {}


class SystemHealthResponse(BaseModel):
    """Overall system health including all modules."""

    system_status: ModuleStatus = ModuleStatus.READY
    modules: list[ModuleHealth] = []
    uptime_seconds: float = 0.0


# ─── Training ────────────────────────────────────────────────────────────────────


class TrainingRequest(BaseModel):
    """Request to start or configure a training run."""

    algorithm: str = "PPO"
    task: str = "pour_task"
    num_envs: int = 64
    total_timesteps: int = 10_000_000
    resume_checkpoint: str | None = None
    config_overrides: dict[str, Any] = {}


class TrainingStatusResponse(BaseModel):
    """Current status of a training run."""

    run_id: str
    algorithm: str
    task: str
    current_step: int = 0
    total_steps: int = 0
    mean_reward: float = 0.0
    success_rate: float = 0.0
    elapsed_seconds: float = 0.0
    status: str = "running"


# ─── Evaluation ──────────────────────────────────────────────────────────────────


class BenchmarkResult(BaseModel):
    """Result from a benchmark evaluation run."""

    benchmark_name: str
    success_rate: float
    mean_reward: float
    spill_rate: float = 0.0
    collision_count: int = 0
    mean_latency_ms: float = 0.0
    episodes: int = 0
    details: dict[str, Any] = {}
