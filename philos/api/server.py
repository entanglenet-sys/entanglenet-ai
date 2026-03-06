"""
PHILOS API Server — FastAPI-based inter-module communication hub.

All PHILOS modules communicate through this REST + WebSocket API,
ensuring loose coupling and scalability. Each module can run as an
independent microservice or be composed in-process.

Endpoints:
    POST /api/v1/command         — Submit a natural language command
    GET  /api/v1/context         — Get the current context vector
    POST /api/v1/state           — Submit robot state observation
    GET  /api/v1/action          — Get the latest policy action
    GET  /api/v1/safety          — Get safety shield status
    GET  /api/v1/health          — System health check
    POST /api/v1/training/start  — Start a training run
    GET  /api/v1/training/status — Get training status
    WS   /ws/telemetry           — Real-time telemetry stream
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from philos.api.schemas import (
    BenchmarkResult,
    ContextVectorPayload,
    LanguageCommandRequest,
    LanguageCommandResponse,
    ModuleHealth,
    ModuleStatus,
    PolicyActionResponse,
    RobotStatePayload,
    SafetyStatus,
    SystemHealthResponse,
    TrainingRequest,
    TrainingStatusResponse,
)

logger = logging.getLogger(__name__)

# ─── In-Memory State Store (would be Redis/shared memory in production) ──────────

_state_store: dict[str, Any] = {
    "context_vector": None,
    "robot_state": None,
    "policy_action": None,
    "safety_status": None,
    "training_status": None,
    "start_time": time.time(),
}

_ws_clients: list[WebSocket] = []
_sim_task: asyncio.Task | None = None
_sim_running: bool = False


# ─── Simulation Loop ─────────────────────────────────────────────────────────────

async def _run_simulation_loop() -> None:
    """Background coroutine: steps the real PourTaskEnv + WholeBodyPolicy,
    broadcasts state to all WebSocket clients at ~20Hz.

    Uses URDF-driven UR5e on a fixed pedestal (no AMR).
    Action dim = 7: 6 joint velocities + 1 gripper.
    """
    global _sim_running

    try:
        from philos.simulation.environments.pour_task import PourTaskEnv, PourTaskConfig
        from philos.learning.policies.whole_body import WholeBodyPolicy
        from philos.control.safety_shield import SafetyShield
        from philos.utils.urdf_parser import load_default_robot, urdf_to_json

        cfg = PourTaskConfig(max_episode_steps=500)
        env = PourTaskEnv(config=cfg)

        # Store URDF JSON for the dashboard endpoint
        urdf_model = load_default_robot()
        _state_store["urdf_json"] = urdf_to_json(urdf_model)

        # Build a policy and initialise on CPU
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        policy = WholeBodyPolicy(
            state_dim=cfg.obs_dim - 18,
            context_dim=18,
            action_dim=cfg.action_dim,  # 7: 6 joints + 1 gripper
            hidden_dims=[256, 128],
            device=device,
        )
        policy.initialize()
        safety = SafetyShield()
        logger.info(f"Simulation loop ready (device={device}, action_dim={cfg.action_dim})")

        # ── Online PPO rollout buffer ──
        rollout_obs: list = []
        rollout_actions: list = []
        rollout_log_probs: list = []
        rollout_rewards: list = []
        rollout_values: list = []
        rollout_dones: list = []
        _gamma, _lam = 0.99, 0.95

        # ── Learning progress tracking ──
        episode_rewards_history: list = []
        best_episode_reward = float("-inf")
        total_train_steps = 0

        episode = 0
        _sim_running = True
        while _sim_running:
            obs, _ = env.reset()
            safety.reset()
            episode += 1
            step = 0
            ep_reward = 0.0
            done = False

            await _broadcast_telemetry({
                "type": "sim_episode_start",
                "episode": episode,
                "beaker_pos": env._beaker_pos.tolist(),
                "glass_pos": env._glass_pos.tolist(),
                "joints": (env._joint_pos.tolist() + env._finger_pos.tolist()),
                "grasped": False,
                "phase": "approach",
            })

            while not done and _sim_running:
                # Policy inference
                action, log_prob, value = policy.predict_with_value(obs)

                # ── Collect transition for PPO ──
                rollout_obs.append(obs.copy())
                rollout_actions.append(action.copy())
                rollout_log_probs.append(float(log_prob))
                rollout_values.append(float(value))

                # Safety filter — no AMR, fixed base
                from philos.core.state import RobotState, JointState, AMRState, EndEffectorState
                robot_state = RobotState(
                    amr=AMRState(position=np.zeros(3, dtype=np.float32)),
                    joints=[JointState(name=f"j{i}", position=float(env._joint_pos[i]))
                            for i in range(6)],
                    end_effector=EndEffectorState(
                        position=np.array(env._ee_pos, dtype=np.float32),
                        gripper_state=float(np.clip(action[6], 0, 1)) if len(action) > 6 else 0.0,
                    ),
                )
                safe_cmd = safety.compute(action, robot_state)
                # Build safe action: 6 joint vels + 1 gripper
                safe_action = np.concatenate([
                    safe_cmd.joint_positions,
                    np.array([safe_cmd.gripper_position]),
                ])

                # Step env
                next_obs, reward, terminated, truncated, info = env.step(safe_action)
                done = terminated or truncated
                ep_reward += reward
                step += 1

                # ── Collect reward/done for PPO ──
                rollout_rewards.append(float(reward))
                rollout_dones.append(1.0 if done else 0.0)

                # Build telemetry payload
                joint_positions = env._joint_pos.tolist() + env._finger_pos.tolist()
                ee_pos = env._ee_pos.tolist()
                total_vol = cfg.source_volume_ml if cfg.source_volume_ml > 0 else 1.0
                poured_frac = env._poured_volume / total_vol
                spill_frac = env._spilled_volume / total_vol

                telem = {
                    "type": "sim_step",
                    "episode": episode,
                    "step": step,
                    "joints": joint_positions,
                    "ee_pos": ee_pos,
                    "base_pos": [0, 0, 0],  # fixed pedestal
                    "gripper": env._gripper_openness,
                    "action": safe_action[:7].tolist(),
                    "reward": round(float(reward), 3),
                    "ep_reward": round(ep_reward, 2),
                    "poured_frac": round(poured_frac, 4),
                    "spill_frac": round(spill_frac, 4),
                    "value": round(float(value), 3),
                    "done": done,
                    "safety_ok": safe_cmd.is_safe,
                    # Physical state
                    "phase": env._phase.value,
                    "beaker_pos": env._beaker_pos.tolist(),
                    "glass_pos": env._glass_pos.tolist(),
                    "beaker_tilt_deg": round(env._wrist_tilt_deg, 1),
                    "fluid_in_beaker": round(env._fluid_in_beaker, 1),
                    "grasped": env._grasped,
                    "collision": env._collision,
                    "bench_clearance": round(env._min_bench_clearance, 3),
                    "finger_q": env._finger_pos.tolist(),
                    "gripper_openness": env._gripper_openness,
                    "beaker_on_bench": env._beaker_on_bench,
                }
                await _broadcast_telemetry(telem)

                obs = next_obs

                # ~20 Hz — yield to event loop
                await asyncio.sleep(0.05)

            # Episode end — track learning progress
            total_vol = cfg.source_volume_ml if cfg.source_volume_ml > 0 else 1.0
            episode_rewards_history.append(ep_reward)
            if ep_reward > best_episode_reward:
                best_episode_reward = ep_reward

            # Rolling average for trend
            window = min(20, len(episode_rewards_history))
            avg_recent = sum(episode_rewards_history[-window:]) / window

            await _broadcast_telemetry({
                "type": "sim_episode_end",
                "episode": episode,
                "total_reward": round(ep_reward, 2),
                "steps": step,
                "poured_frac": round(env._poured_volume / total_vol, 4),
                "spill_frac": round(env._spilled_volume / total_vol, 4),
                "phase": env._phase.value,
                # Learning progress
                "best_reward": round(best_episode_reward, 2),
                "avg_reward_20": round(avg_recent, 2),
                "total_episodes": episode,
            })

            # ── PPO training update ──
            if len(rollout_obs) >= 64:
                obs_arr = np.array(rollout_obs, dtype=np.float32)
                act_arr = np.array(rollout_actions, dtype=np.float32)
                lp_arr = np.array(rollout_log_probs, dtype=np.float32)
                rew_arr = np.array(rollout_rewards, dtype=np.float32)
                val_arr = np.array(rollout_values, dtype=np.float32)
                done_arr = np.array(rollout_dones, dtype=np.float32)

                # GAE advantages
                adv = np.zeros_like(rew_arr)
                last_gae = 0.0
                for t in reversed(range(len(rew_arr))):
                    nv = val_arr[t + 1] if t < len(rew_arr) - 1 else 0.0
                    delta = rew_arr[t] + _gamma * nv * (1 - done_arr[t]) - val_arr[t]
                    adv[t] = last_gae = delta + _gamma * _lam * (1 - done_arr[t]) * last_gae
                returns = adv + val_arr

                batch = {
                    "observations": obs_arr,
                    "actions": act_arr,
                    "old_log_probs": lp_arr,
                    "advantages": adv,
                    "returns": returns,
                }
                try:
                    metrics = policy.train_step(batch)
                    total_train_steps += 1
                    await _broadcast_telemetry({
                        "type": "train_step",
                        "episode": episode,
                        "metrics": {k: round(float(v), 5) for k, v in metrics.items()},
                        "avg_reward": round(float(rew_arr.mean()), 3),
                        "ep_reward": round(ep_reward, 2),
                        "total_train_steps": total_train_steps,
                    })
                    logger.info(
                        f"[PPO] ep={episode} "
                        f"loss={metrics.get('total_loss', 0):.4f} "
                        f"reward={ep_reward:.1f} "
                        f"avg20={avg_recent:.1f}"
                    )
                except Exception:
                    logger.exception("PPO train_step failed")

                rollout_obs.clear()
                rollout_actions.clear()
                rollout_log_probs.clear()
                rollout_rewards.clear()
                rollout_values.clear()
                rollout_dones.clear()

            # Brief pause between episodes
            await asyncio.sleep(1.0)

    except asyncio.CancelledError:
        logger.info("Simulation loop cancelled.")
    except Exception:
        logger.exception("Simulation loop crashed!")
    finally:
        env.close()
        _sim_running = False
        logger.info("Simulation loop stopped.")


# ─── Lifespan ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup/shutdown — auto-starts simulation loop."""
    global _sim_task
    logger.info("PHILOS API Server starting...")
    _state_store["start_time"] = time.time()
    # Auto-start simulation in background
    _sim_task = asyncio.create_task(_run_simulation_loop())
    yield
    # Shutdown
    global _sim_running
    _sim_running = False
    if _sim_task:
        _sim_task.cancel()
        try:
            await _sim_task
        except asyncio.CancelledError:
            pass
    logger.info("PHILOS API Server shutting down.")


# ─── App ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PHILOS API",
    description=(
        "Physical Intelligence, Learning, Optimization, and Semantics — "
        "Inter-module communication API for adaptive mobile manipulation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Root / Dashboard ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Interactive PHILOS dashboard with live architecture, robot arm simulation, and telemetry."""
    from pathlib import Path

    html_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/v1/urdf")
async def get_urdf_json() -> dict:
    """Return the URDF robot model as JSON for the 3D digital twin.

    The dashboard fetches this on load to build the Three.js scene
    from the exact same URDF that drives the simulation — guaranteeing
    a 1:1 match between physics and visualization.
    """
    urdf = _state_store.get("urdf_json")
    if urdf is None:
        # Load on demand if simulation hasn't started yet
        from philos.utils.urdf_parser import load_default_robot, urdf_to_json
        robot = load_default_robot()
        urdf = urdf_to_json(robot)
        _state_store["urdf_json"] = urdf
    return urdf


# ─── Language Command Endpoint ───────────────────────────────────────────────────

@app.post("/api/v1/command", response_model=LanguageCommandResponse)
async def submit_command(request: LanguageCommandRequest) -> LanguageCommandResponse:
    """Submit a natural language command to the Cognitive Engine.

    The command is processed by the VLM to produce a Context Vector (z)
    that will condition the RL policy. This is the "Language-to-Reward" interface.

    Example:
        POST /api/v1/command
        {"command": "Pour the blue liquid carefully"}
    """
    start = time.monotonic()

    # TODO: Forward to the Cognitive Engine module for real processing
    # For now, return a placeholder response
    latency = (time.monotonic() - start) * 1000

    response = LanguageCommandResponse(
        success=True,
        context_vector_id="ctx_placeholder",
        interpretation=f"Understood: '{request.command}' — generating reward constraints",
        latency_ms=latency,
        constraints_generated=0,
    )

    logger.info(f"Command processed: '{request.command}' in {latency:.1f}ms")
    return response


# ─── Context Vector Endpoints ────────────────────────────────────────────────────

@app.get("/api/v1/context", response_model=ContextVectorPayload | None)
async def get_context() -> ContextVectorPayload | None:
    """Get the current Context Vector (z) — consumed by the RL policy."""
    return _state_store.get("context_vector")


@app.put("/api/v1/context")
async def update_context(payload: ContextVectorPayload) -> dict[str, str]:
    """Update the Context Vector (z) — called by the Cognitive Engine."""
    _state_store["context_vector"] = payload
    await _broadcast_telemetry({"type": "context_update", "data": payload.model_dump()})
    return {"status": "ok"}


# ─── Robot State Endpoints ───────────────────────────────────────────────────────

@app.post("/api/v1/state")
async def submit_state(payload: RobotStatePayload) -> dict[str, str]:
    """Submit robot state observation — from sensors or simulation."""
    _state_store["robot_state"] = payload
    return {"status": "ok"}


@app.get("/api/v1/state", response_model=RobotStatePayload | None)
async def get_state() -> RobotStatePayload | None:
    """Get the latest robot state."""
    return _state_store.get("robot_state")


# ─── Policy Action Endpoints ─────────────────────────────────────────────────────

@app.get("/api/v1/action", response_model=PolicyActionResponse | None)
async def get_action() -> PolicyActionResponse | None:
    """Get the latest action from the RL policy (pre-safety-filter)."""
    return _state_store.get("policy_action")


@app.put("/api/v1/action")
async def update_action(payload: PolicyActionResponse) -> dict[str, str]:
    """Update the policy action — called by the Learning module."""
    _state_store["policy_action"] = payload
    await _broadcast_telemetry({"type": "action_update", "data": payload.model_dump()})
    return {"status": "ok"}


# ─── Safety Shield Endpoints ─────────────────────────────────────────────────────

@app.get("/api/v1/safety", response_model=SafetyStatus | None)
async def get_safety_status() -> SafetyStatus | None:
    """Get the current Safety Shield status."""
    return _state_store.get("safety_status")


@app.put("/api/v1/safety")
async def update_safety_status(payload: SafetyStatus) -> dict[str, str]:
    """Update safety shield status — called by the Control module."""
    _state_store["safety_status"] = payload
    if payload.override_active:
        await _broadcast_telemetry({"type": "safety_override", "data": payload.model_dump()})
    return {"status": "ok"}


# ─── Training Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/v1/training/start", response_model=TrainingStatusResponse)
async def start_training(request: TrainingRequest) -> TrainingStatusResponse:
    """Start a new training run.

    This triggers the Learning module to begin RL training in Isaac Sim.
    """
    # TODO: Forward to Learning module
    run_id = f"run_{int(time.time())}"
    status = TrainingStatusResponse(
        run_id=run_id,
        algorithm=request.algorithm,
        task=request.task,
        total_steps=request.total_timesteps,
        status="queued",
    )
    _state_store["training_status"] = status
    logger.info(f"Training run {run_id} queued: {request.algorithm} on {request.task}")
    return status


@app.get("/api/v1/training/status", response_model=TrainingStatusResponse | None)
async def get_training_status() -> TrainingStatusResponse | None:
    """Get the current training run status."""
    return _state_store.get("training_status")


# ─── Health ──────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=SystemHealthResponse)
async def health_check() -> SystemHealthResponse:
    """System health check — reports status of all modules."""
    uptime = time.time() - _state_store.get("start_time", time.time())

    modules = [
        ModuleHealth(module="api", status=ModuleStatus.RUNNING, latency_ms=0),
        ModuleHealth(
            module="perception",
            status=ModuleStatus.READY if _state_store.get("robot_state") else ModuleStatus.OFFLINE,
        ),
        ModuleHealth(
            module="cognitive",
            status=ModuleStatus.READY if _state_store.get("context_vector") else ModuleStatus.OFFLINE,
        ),
        ModuleHealth(
            module="learning",
            status=ModuleStatus.RUNNING
            if _state_store.get("training_status")
            else ModuleStatus.OFFLINE,
        ),
        ModuleHealth(
            module="control",
            status=ModuleStatus.READY if _state_store.get("safety_status") else ModuleStatus.OFFLINE,
        ),
    ]

    return SystemHealthResponse(
        system_status=ModuleStatus.RUNNING,
        modules=modules,
        uptime_seconds=uptime,
    )


# ─── Evaluation ──────────────────────────────────────────────────────────────────

@app.post("/api/v1/evaluate", response_model=BenchmarkResult)
async def run_benchmark(task: str = "pour_task", episodes: int = 50) -> BenchmarkResult:
    """Run a benchmark evaluation.

    Supported tasks:
        - pour_task: "The Sommelier" — fluid pouring (WP4, T4.2)
        - fetch_sort_task: "The Courier" — fetch and sort (WP4, T4.3)
        - navigation_task: Dynamic obstacle avoidance
    """
    # TODO: Forward to Evaluation module
    return BenchmarkResult(
        benchmark_name=task,
        success_rate=0.0,
        mean_reward=0.0,
        episodes=episodes,
    )


# ─── WebSocket Telemetry ─────────────────────────────────────────────────────────

@app.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket) -> None:
    """Real-time telemetry stream for monitoring dashboards."""
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"WS received: {data}")
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)


# ─── Simulation Control ─────────────────────────────────────────────────────────

@app.post("/api/v1/simulation/start")
async def start_simulation() -> dict:
    """Start the background simulation loop (PourTaskEnv + WholeBodyPolicy)."""
    global _sim_task, _sim_running
    if _sim_running:
        return {"status": "already_running"}
    _sim_task = asyncio.create_task(_run_simulation_loop())
    return {"status": "started"}


@app.post("/api/v1/simulation/stop")
async def stop_simulation() -> dict:
    """Stop the background simulation loop."""
    global _sim_running, _sim_task
    if not _sim_running:
        return {"status": "not_running"}
    _sim_running = False
    if _sim_task:
        _sim_task.cancel()
        try:
            await _sim_task
        except asyncio.CancelledError:
            pass
        _sim_task = None
    return {"status": "stopped"}


@app.get("/api/v1/simulation/status")
async def simulation_status() -> dict:
    """Check if the background simulation loop is running."""
    return {"running": _sim_running}


async def _broadcast_telemetry(message: dict) -> None:
    """Broadcast a telemetry event to all connected WebSocket clients."""
    disconnected = []
    for ws in _ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _ws_clients.remove(ws)


# ─── Entry Point ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Launch the PHILOS API server."""
    import uvicorn

    uvicorn.run(
        "philos.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
