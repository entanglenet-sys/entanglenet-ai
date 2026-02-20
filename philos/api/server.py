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
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

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


# ─── Lifespan ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup/shutdown."""
    logger.info("PHILOS API Server starting...")
    _state_store["start_time"] = time.time()
    yield
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
            # Keep alive — clients can also send commands through WS
            data = await websocket.receive_text()
            logger.debug(f"WS received: {data}")
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)


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
