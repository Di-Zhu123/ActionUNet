"""Lightweight WebSocket adapter for RoboTwin's legacy evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class RoboTwinActionUNetClient:
    """Translate RoboTwin observations to the OpenPI WebSocket protocol."""

    def __init__(self, args: dict[str, Any]):
        self.action_chunk = int(args.get("action_chunk", args.get("pi05_step", 50)))
        self.instruction: str | None = None
        self.observation: dict[str, Any] | None = None
        self.policy = WebsocketClientPolicy(
            host=str(args.get("host", "127.0.0.1")),
            port=int(args.get("port", 8000)),
        )

    def update(self, observation: dict[str, Any]) -> None:
        head = np.asarray(observation["observation"]["head_camera"]["rgb"])
        left = np.asarray(observation["observation"]["left_camera"]["rgb"])
        right = np.asarray(observation["observation"]["right_camera"]["rgb"])
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        self.observation = {
            "state": state,
            "images": {
                "cam_high": np.transpose(head, (2, 0, 1)),
                "cam_left_wrist": np.transpose(left, (2, 0, 1)),
                "cam_right_wrist": np.transpose(right, (2, 0, 1)),
            },
            "prompt": self.instruction,
        }

    def reset(self) -> None:
        self.instruction = None
        self.observation = None


def get_model(usr_args: dict[str, Any]) -> RoboTwinActionUNetClient:
    """Entry point loaded by RoboTwin's ``script/eval_policy.py``."""
    return RoboTwinActionUNetClient(usr_args)


def eval(task_env, model: RoboTwinActionUNetClient, observation: dict[str, Any]) -> None:
    """Infer one action chunk remotely and execute it in RoboTwin."""
    if model.instruction is None:
        model.instruction = str(task_env.get_instruction())
    model.update(observation)
    if model.observation is None:
        raise RuntimeError("RoboTwin observation was not initialized")
    actions = np.asarray(model.policy.infer(model.observation)["actions"])
    for action in actions[: model.action_chunk]:
        if getattr(task_env, "eval_success", False):
            break
        if getattr(task_env, "take_action_cnt", 0) >= getattr(task_env, "step_lim", float("inf")):
            break
        task_env.take_action(action)
        observation = task_env.get_obs()
    model.update(observation)


def reset_model(model: RoboTwinActionUNetClient) -> None:
    """Reset language and observation state between episodes."""
    model.reset()
