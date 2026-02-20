# PHILOS Architecture

## Dual-Loop Cognitive Architecture

PHILOS implements a biologically-inspired dual-process architecture:

### System 1 — Fast Reflexes (50 Hz)

The reactive loop handles real-time perception and control:

```
Sensors → YOLO-World → State s → Policy π(s,z) → Safety Shield → Actuators
            (50 Hz)                   (50 Hz)        (50 Hz)
```

**Components:**
- `perception/yolo_world.py` — Open-vocabulary object detection
- `learning/policies/whole_body.py` — PPO actor-critic policy
- `control/safety_shield.py` — Deterministic safety filter
- `control/mpc_solver.py` — Whole-body MPC trajectory optimization

### System 2 — Slow Reasoning (1 Hz)

The deliberative loop provides semantic understanding:

```
Language Command → VLM (LLaVA-Next) → Context Vector z → conditions System 1
                      (1 Hz)
```

**Components:**
- `perception/vlm_grounding.py` — VLM → latent context vector
- `cognitive/reward_shaping.py` — Language → reward weight modulation

## Context Vector z

The **Context Vector** is the key innovation bridging language and control.
Instead of passing text to the RL policy, the VLM produces a structured
latent vector:

```python
@dataclass
class ContextVector:
    embedding: np.ndarray        # 18-dim latent vector
    mode: ManipulationMode       # STIFF | COMPLIANT | FLUID | PRECISION
    impedance_scale: float       # How soft/stiff the arm should be
    velocity_limit_scale: float  # Speed modifier
    orientation_constraint: float # How strictly to maintain orientation
    jerk_penalty: float          # Smoothness requirement
    semantic_labels: list[str]   # Detected relevant objects
    confidence: float            # VLM confidence
```

This vector conditions the RL policy: **π(s, z)** instead of π(s).

## Safety Architecture

The Safety Shield is the ONLY component that can override the RL policy.
It is **purely deterministic** (no neural networks):

```
RL Action ──► [Tilt Check] ──► [Velocity Clamp] ──► [Joint Limits]
                                                          │
                                                          ▼
                                                    [Collision Check]
                                                          │
                                                          ▼
                                                    [Jerk Limit]
                                                          │
                                                          ▼
                                                  Safe Actuator Command
```

Hard constraints (cannot be overridden):
- Platform tilt < 10°
- End-effector velocity < 1.5 m/s
- Joint positions within mechanical limits
- Minimum obstacle distance > 5 cm

## Module Communication

Modules communicate via two paths:

1. **In-Process** (training/simulation): Direct Python calls via ComponentRegistry
2. **API** (deployment/distributed): FastAPI REST + WebSocket endpoints

The ComponentRegistry enables loose coupling:

```python
from philos.core.registry import ComponentRegistry

registry = ComponentRegistry()
detector = registry.create("perception", "yolo_world", confidence=0.3)
policy = registry.create("learning", "whole_body_ppo", state_dim=30)
shield = registry.create("control", "safety_shield")
```

## Domain Randomization

Training with ±200% physics variation:

| Parameter | Default | Range |
|-----------|---------|-------|
| Fluid viscosity | 1.0 | 0.1 – 3.0 |
| Object mass | 0.3 kg | 0.05 – 1.0 kg |
| Ground friction | 0.6 | 0.2 – 1.0 |
| Camera noise | 0.01 | 0.0 – 0.05 |
| Light intensity | 1.0x | 0.3x – 2.0x |
| Joint friction | 0.01 | 0.001 – 0.05 |
