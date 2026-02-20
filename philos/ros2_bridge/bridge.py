"""
PHILOS ROS2 Bridge — connects PHILOS modules to ROS2 Jazzy.

This node orchestrates the full pipeline on real hardware:
    Sensors → Perception → Cognitive → Learning → Control → Actuators

Design:
    - SingleThreadedExecutor for deterministic timing
    - Timer callbacks at 50 Hz for the control loop
    - Safety shield runs as highest-priority callback
    - All inter-module communication via shared memory (zero-copy)
    - ROS2 topics for external interfaces (sensors, actuators, diagnostics)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from philos.core.config import PhilosConfig, load_config
from philos.core.context_vector import ContextVector
from philos.core.state import RobotState, JointState, AMRState
from philos.control.safety_shield import SafetyShield
from philos.ros2_bridge.topics import PHILOS_TOPICS

logger = logging.getLogger(__name__)

# ROS2 imports (optional — for use without ROS2 installed)
_ROS2_AVAILABLE = False
try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from rclpy.executors import SingleThreadedExecutor  # type: ignore
    from rclpy.qos import QoSProfile as ROS2QoS, ReliabilityPolicy, DurabilityPolicy  # type: ignore
    _ROS2_AVAILABLE = True
except ImportError:
    # Stub for development without ROS2
    Node = object  # type: ignore


class PhilosROS2Bridge(Node):  # type: ignore[misc]
    """Main ROS2 node bridging PHILOS modules to hardware.

    Pipeline (50 Hz loop):
        1. Read sensors (joint_states, odom, imu, cameras)
        2. Run perception (YOLO-World at 50 Hz)
        3. Run VLM grounding (1 Hz, asynchronous)
        4. Query RL policy for action
        5. Apply safety shield
        6. Publish actuator commands

    Usage (with ROS2):
        rclpy.init()
        bridge = PhilosROS2Bridge()
        executor = SingleThreadedExecutor()
        executor.add_node(bridge)
        executor.spin()

    Usage (without ROS2 — for testing):
        bridge = PhilosROS2Bridge()
        bridge.spin_once()  # Manual step
    """

    def __init__(
        self,
        config_path: str | None = None,
        node_name: str = "philos_bridge",
    ) -> None:
        if _ROS2_AVAILABLE:
            super().__init__(node_name)
        else:
            logger.warning("ROS2 not available — running in standalone mode.")

        # Load configuration
        self._config = load_config(config_path) if config_path else PhilosConfig()

        # Core components (lazy-loaded from registry)
        self._safety_shield = SafetyShield()
        self._context_vector: ContextVector | None = None
        self._robot_state = RobotState(
            joints=[JointState(name=f"joint_{i}", position=0.0) for i in range(6)],
            amr=AMRState(),
        )

        # Timing
        self._dt = 1.0 / 50.0  # 50 Hz
        self._last_vlm_time = 0.0
        self._vlm_interval = 1.0  # 1 Hz
        self._step_count = 0

        # ROS2 publishers / subscribers
        self._publishers: dict[str, Any] = {}
        self._subscribers: dict[str, Any] = {}

        if _ROS2_AVAILABLE:
            self._setup_ros2_interfaces()
            self._control_timer = self.create_timer(self._dt, self._control_loop)
            logger.info(f"PHILOS ROS2 bridge initialized at {1/self._dt:.0f} Hz.")

    def _setup_ros2_interfaces(self) -> None:
        """Create ROS2 publishers and subscribers."""
        # Publishers for actuator commands
        for key in ["safe_command", "joint_command", "base_velocity", "safety_status"]:
            topic = PHILOS_TOPICS[key]
            # In production: self.create_publisher(msg_type, topic.name, qos)
            self._publishers[key] = None  # Placeholder

        # Subscribers for sensor data
        for key in ["joint_states", "odom", "imu", "rgb_camera", "depth_camera", "lidar_scan"]:
            topic = PHILOS_TOPICS[key]
            # In production: self.create_subscription(msg_type, topic.name, callback, qos)
            self._subscribers[key] = None  # Placeholder

        logger.info(f"Set up {len(self._publishers)} publishers, {len(self._subscribers)} subscribers.")

    def _control_loop(self) -> None:
        """Main 50 Hz control loop callback.

        This is the beating heart of the PHILOS system.
        """
        t_start = time.perf_counter()
        self._step_count += 1

        # 1. Update robot state from latest sensor readings
        state = self._robot_state

        # 2. Run VLM grounding at 1 Hz (System 2)
        now = time.monotonic()
        if now - self._last_vlm_time >= self._vlm_interval:
            self._last_vlm_time = now
            # In production: call VLM grounding asynchronously
            # self._context_vector = self._vlm.generate_context_vector(...)

        # 3. Query RL policy for action
        # In production:
        #   obs = state.to_observation()
        #   if self._context_vector:
        #       obs = np.concatenate([obs, self._context_vector.to_tensor()])
        #   action = self._policy.predict(obs, self._context_vector)
        action = np.zeros(10)  # Placeholder

        # 4. Apply safety shield (ALWAYS runs, non-negotiable)
        safe_cmd = self._safety_shield.compute(action, state, self._dt)

        # 5. Publish actuator commands
        self._publish_commands(safe_cmd)

        # 6. Timing check
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        if elapsed_ms > self._dt * 1000:
            logger.warning(
                f"Control loop overrun: {elapsed_ms:.1f} ms "
                f"(budget: {self._dt * 1000:.1f} ms)"
            )

    def _publish_commands(self, cmd: Any) -> None:
        """Publish safe commands to actuator topics."""
        # In production: serialize and publish via ROS2
        pass

    # ------------------------------------------------------------------
    # Standalone (non-ROS2) interface
    # ------------------------------------------------------------------

    def spin_once(self, action: np.ndarray | None = None) -> dict[str, Any]:
        """Execute one control cycle without ROS2.

        Useful for testing and integration with Isaac Sim.
        """
        if action is None:
            action = np.zeros(10)

        safe_cmd = self._safety_shield.compute(action, self._robot_state, self._dt)
        self._step_count += 1

        return {
            "command": safe_cmd,
            "safety_level": self._safety_shield.safety_state.level.value,
            "overrides": safe_cmd.safety_overrides,
            "step": self._step_count,
        }

    def update_state(self, state: RobotState) -> None:
        """Update the robot state (for standalone mode)."""
        self._robot_state = state

    def update_context(self, context: ContextVector) -> None:
        """Update the VLM context vector (for standalone mode)."""
        self._context_vector = context

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._safety_shield.trigger_emergency_stop()
        logger.info("PHILOS ROS2 bridge shut down.")
        if _ROS2_AVAILABLE:
            self.destroy_node()
