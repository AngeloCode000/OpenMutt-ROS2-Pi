#!/usr/bin/env python3
"""
Bridge a 12-dim policy action vector into sensor_msgs/JointState for the OpenMutt ODrive bridge.

Usage (example, headless):
  /home/eppl/workspaces/IsaacLab/isaaclab.sh -p OpenMutt-ROS2-Pi/openmutt_startup/openmutt_startup/policy_to_jointstate.py

By default this loads a TorchScript policy from POLICY_PATH (override with
OPENMUTT_POLICY_PATH) plus optional obs RMS from OPENMUTT_OBS_RMS_PATH and
publishes its action as JointState.

Publishes at 100 Hz to "joint_states" in the CHAMP order expected by joint2odrive_pub.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import os
from typing import Optional
import numpy as np
import torch

# Policy index -> CHAMP slot
POLICY_TO_CHAMP = {
    "lf_hip": 2,
    "lf_upper_leg": 6,
    "lf_lower_leg": 10,
    "rf_hip": 3,
    "rf_upper_leg": 7,
    "rf_lower_leg": 11,
    "lh_hip": 1,
    "lh_upper_leg": 5,
    "lh_lower_leg": 9,
    "rh_hip": 0,
    "rh_upper_leg": 4,
    "rh_lower_leg": 8,
}

# Ordered names expected by joint2odrive_pub (CHAMP order)
CHAMP_ORDER = [
    "lf_hip_joint",
    "lf_upper_leg_joint",
    "lf_lower_leg_joint",
    "rf_hip_joint",
    "rf_upper_leg_joint",
    "rf_lower_leg_joint",
    "lh_hip_joint",
    "lh_upper_leg_joint",
    "lh_lower_leg_joint",
    "rh_hip_joint",
    "rh_upper_leg_joint",
    "rh_lower_leg_joint",
]

# Paths to TorchScript policy and optional obs normalization stats.
POLICY_PATH = os.getenv(
    "OPENMUTT_POLICY_PATH",
    "/home/eppl/workspaces/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/openmutt_flat/2025-12-03_21-02-13_Run-0/policy_jit.pt",
)
OBS_RMS_PATH = os.getenv("OPENMUTT_OBS_RMS_PATH", "")
POLICY_DEVICE = os.getenv("OPENMUTT_POLICY_DEVICE", "cpu")
OBS_CLAMP = 5.0
DEFAULT_OBS_DIM = int(os.getenv("OPENMUTT_OBS_DIM", "48"))
ACTION_SCALE = float(os.getenv("OPENMUTT_ACTION_SCALE", "0.3"))
HEIGHT_SCAN_DIM = int(os.getenv("OPENMUTT_HEIGHT_SCAN_DIM", "0"))
USE_FAKE_OBS = bool(int(os.getenv("OPENMUTT_FAKE_OBS", "0")))
FAKE_FREQ = float(os.getenv("OPENMUTT_FAKE_FREQ", "0.8"))  # Hz, controls swing speed
FAKE_AMP = float(os.getenv("OPENMUTT_FAKE_AMP", "0.5"))    # rad amplitude for joints
FAKE_CMD_X = float(os.getenv("OPENMUTT_FAKE_CMD_X", "0.0"))  # m/s forward command


last_action = np.zeros(12, dtype=np.float32)
fake_obs_gen = None


class FakeObservationGenerator:
    """
    Simple synthetic observation generator to let the policy produce changing actions
    when real sensors are unavailable.
    """

    def __init__(self):
        self.phase = 0.0
        self.dt = 0.01  # matches 100 Hz timer

    def next(self) -> np.ndarray:
        # Offsets per joint to de-phase legs
        offsets = np.arange(12, dtype=np.float32) * 0.3
        joint_pos = FAKE_AMP * np.sin(self.phase + offsets)
        joint_vel = FAKE_AMP * (FAKE_FREQ * 2 * np.pi) * np.cos(self.phase + offsets)

        base_lin_vel = np.array([FAKE_CMD_X, 0.0, 0.0], dtype=np.float32)
        base_ang_vel = np.zeros(3, dtype=np.float32)
        projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # approx gravity in body frame
        velocity_commands = np.array([FAKE_CMD_X, 0.0, 0.0], dtype=np.float32)
        actions_prev = last_action.astype(np.float32)
        height_scan = np.zeros(HEIGHT_SCAN_DIM, dtype=np.float32) if HEIGHT_SCAN_DIM > 0 else np.array([], dtype=np.float32)

        obs = np.concatenate(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                velocity_commands,
                joint_pos,
                joint_vel,
                actions_prev,
                height_scan,
            ]
        ).astype(np.float32)

        self.phase += FAKE_FREQ * 2 * np.pi * self.dt
        return obs


def build_observation_from_sensors() -> np.ndarray:
    """
    Build observation in the same order as the Isaac-Velocity-Flat-OpenMutt-v0 task:
      [base_lin_vel(3), base_ang_vel(3), projected_gravity(3), velocity_commands(3),
       joint_pos(12), joint_vel(12), actions(12), height_scan(N)]

    Replace the placeholder zeros below with real sensor values in that order.
    """
    global fake_obs_gen

    if USE_FAKE_OBS:
        if fake_obs_gen is None:
            fake_obs_gen = FakeObservationGenerator()
        obs = fake_obs_gen.next()
        if obs.shape[0] != DEFAULT_OBS_DIM:
            raise ValueError(f"Observation length {obs.shape[0]} != OPENMUTT_OBS_DIM ({DEFAULT_OBS_DIM})")
        return obs

    # TODO: wire these to real sensors/estimators. Placeholders keep the node running.
    base_lin_vel = np.zeros(3, dtype=np.float32)   # m/s
    base_ang_vel = np.zeros(3, dtype=np.float32)   # rad/s
    projected_gravity = np.zeros(3, dtype=np.float32)  # IMU gravity in body frame
    velocity_commands = np.zeros(3, dtype=np.float32)  # commanded vx, vy, yaw_rate
    joint_pos = np.zeros(12, dtype=np.float32)     # joint angles in training order
    joint_vel = np.zeros(12, dtype=np.float32)     # joint velocities
    actions_prev = last_action.astype(np.float32)  # previous action sent
    height_scan = np.zeros(HEIGHT_SCAN_DIM, dtype=np.float32) if HEIGHT_SCAN_DIM > 0 else np.array([], dtype=np.float32)

    obs = np.concatenate(
        [
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            velocity_commands,
            joint_pos,
            joint_vel,
            actions_prev,
            height_scan,
        ]
    ).astype(np.float32)

    if obs.shape[0] != DEFAULT_OBS_DIM:
        raise ValueError(f"Observation length {obs.shape[0]} != OPENMUTT_OBS_DIM ({DEFAULT_OBS_DIM})")

    return obs


class PolicyRunner:
    def __init__(self):
        self.device = torch.device(POLICY_DEVICE)
        self.policy = self._load_policy()
        self.obs_rms = self._load_obs_rms()

    def _load_policy(self):
        if not POLICY_PATH or not os.path.exists(POLICY_PATH):
            raise FileNotFoundError(f"Policy file not found: {POLICY_PATH}")
        policy = torch.jit.load(POLICY_PATH, map_location=self.device)
        policy.eval()
        return policy

    def _load_obs_rms(self):
        if OBS_RMS_PATH and os.path.exists(OBS_RMS_PATH):
            return torch.load(OBS_RMS_PATH, map_location=self.device)
        return None

    def act(self, obs_np: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.obs_rms is not None:
            obs = torch.clamp(
                (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8),
                -OBS_CLAMP,
                OBS_CLAMP,
            )
        with torch.no_grad():
            action = self.policy(obs)
        return action.squeeze(0).cpu().numpy()


policy_runner: Optional[PolicyRunner] = None
_policy_error_logged = False


def get_policy_action() -> np.ndarray:
    global policy_runner, _policy_error_logged

    try:
        if policy_runner is None:
            policy_runner = PolicyRunner()
        obs = build_observation_from_sensors()
        return policy_runner.act(obs)
    except Exception as exc:  # keep publishing zeros if policy fails
        if not _policy_error_logged:
            print(f"[policy_to_jointstate] policy inference failed: {exc}")
            _policy_error_logged = True
        return np.zeros(12, dtype=np.float32)


class PolicyPublisher(Node):
    def __init__(self):
        super().__init__("openmutt_policy_pub")
        self.pub = self.create_publisher(JointState, "joint_states", 10)
        self.timer = self.create_timer(0.01, self._tick)  # 100 Hz

    def _tick(self):
        action = get_policy_action()
        # Safety clamp; scale to radians if your task uses an action scale.
        action = np.clip(action, -1.0, 1.0) * ACTION_SCALE
        if action.shape[0] < 12:
            self.get_logger().warn("Policy action vector has fewer than 12 elements")
            return

        champ_positions = []
        for name in CHAMP_ORDER:
            key = name.replace("_joint", "")
            idx = POLICY_TO_CHAMP[key]
            champ_positions.append(float(action[idx]))

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.name = CHAMP_ORDER
        msg.position = champ_positions
        # velocity/effort left empty by design
        self.pub.publish(msg)
        # Track last action for next observation build
        global last_action
        last_action = np.array(champ_positions, dtype=np.float32)


def main():
    rclpy.init()
    node = PolicyPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
