#!/usr/bin/env python3
"""LIBERO evaluation worker for standard LIBERO and LIBERO-Plus."""

# This worker runs on the pinned Python 3.8 LIBERO environment. Robosuite's
# public wrapper doesn't expose camera-observable toggles, so the optimized
# rendering path intentionally uses its stable underlying observable API.
# ruff: noqa: FBT001, FBT003, SLF001, UP006, UP035, UP045

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import pathlib
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import eval_queue
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
DEFAULT_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data/libero/videos"
    seed: int = 7

    # Queue-worker extensions used by run_parallel_eval.sh.
    job_db: Optional[str] = None
    worker_id: str = "worker-0"
    no_save_videos: bool = False
    max_steps_override: int = -1
    skip_camera_rendering_during_action_chunk: bool = True
    minimal_policy_observables: bool = False
    soft_reset_env: bool = False


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / denominator


def _get_libero_env(task: Any, resolution: int, seed: int, soft_reset: bool) -> Tuple[Any, str]:
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        # LIBERO-Plus parses view/init-state suffixes with string operations,
        # while upstream LIBERO also accepts the resulting string path.
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
        hard_reset=not soft_reset,
    )
    env.seed(seed)
    return env, task.language


def _camera_observable_names(env: Any) -> Tuple[str, ...]:
    return tuple(
        name
        for name, observable in env.env._observables.items()
        if name.endswith("_image") or getattr(observable, "modality", None) == "image"
    )


def _set_camera_active(env: Any, active: bool) -> None:
    for name in _camera_observable_names(env):
        env.env._observables[name].set_active(active)


def _configure_minimal_observables(env: Any) -> None:
    required = {
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    }
    for name, observable in env.env._observables.items():
        if name not in required:
            observable.set_active(False)


def _fresh_observation(env: Any) -> Dict[str, np.ndarray]:
    _set_camera_active(env, True)
    env._update_observables(force=True)
    return env.env._get_observations()


def _policy_observation(obs: Dict[str, np.ndarray], description: str, resize_size: int) -> Dict[str, Any]:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, resize_size, resize_size)
    )
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    if state.shape != (8,):
        raise RuntimeError(f"Expected eight-dimensional LIBERO state, got {state.shape}")
    return {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": str(description),
    }


def _episode_horizon(args: Args, suite_name: str) -> int:
    if args.max_steps_override >= 0:
        return args.max_steps_override
    try:
        return DEFAULT_HORIZONS[suite_name]
    except KeyError as exc:
        raise ValueError(f"Unknown task suite: {suite_name}") from exc


def _run_episode(
    args: Args,
    client: _websocket_client_policy.WebsocketClientPolicy,
    env: Any,
    description: str,
    initial_state: np.ndarray,
) -> Tuple[bool, list]:
    env.reset()
    obs = env.set_init_state(initial_state)
    action_plan = collections.deque()
    replay_images = []
    done = False
    t = 0
    final_t = args.num_steps_wait + _episode_horizon(args, args.task_suite_name)

    if args.skip_camera_rendering_during_action_chunk:
        _set_camera_active(env, False)
    while t < final_t:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if not action_plan:
            obs = _fresh_observation(env)
            element = _policy_observation(obs, description, args.resize_size)
            if not args.no_save_videos:
                replay_images.append(element["observation/image"])
            action_chunk = np.asarray(client.infer(element)["actions"])
            if len(action_chunk) < args.replan_steps:
                raise RuntimeError(
                    f"Policy returned {len(action_chunk)} actions, fewer than replan_steps={args.replan_steps}"
                )
            action_plan.extend(action_chunk[: args.replan_steps])
            if args.skip_camera_rendering_during_action_chunk:
                _set_camera_active(env, False)

        action = np.asarray(action_plan.popleft())
        if action.shape[-1] != 7:
            raise RuntimeError(f"Expected seven-dimensional LIBERO action, got {action.shape}")
        obs, _, done, _ = env.step(action.tolist())
        t += 1
        if done:
            break
    return bool(done), replay_images


def _write_video(args: Args, description: str, success: bool, images: list, job_id: str) -> None:
    if args.no_save_videos or not images:
        return
    output_dir = pathlib.Path(args.video_out_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "success" if success else "failure"
    task_segment = description.replace(" ", "_")
    imageio.mimwrite(output_dir / f"rollout_{job_id}_{task_segment}_{suffix}.mp4", images, fps=10)


def eval_queue_worker(args: Args) -> None:
    if args.job_db is None:
        raise ValueError("job_db is required")
    database = pathlib.Path(args.job_db).resolve()
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    benchmark_dict = benchmark.get_benchmark_dict()
    suites: Dict[str, Any] = {}
    env = None
    env_key = None
    completed = 0
    np.random.seed(args.seed)
    logging.info(
        "queue_worker_start worker=%s db=%s host=%s port=%s replan=%s max_steps_override=%s",
        args.worker_id,
        database,
        args.host,
        args.port,
        args.replan_steps,
        args.max_steps_override,
    )
    try:
        while True:
            job = eval_queue.claim(database, args.worker_id)
            if job is None:
                break
            started_at = time.time()
            job_id = int(job["id"])
            suite_name = str(job["suite"])
            task_id = int(job["task_id"])
            trial_id = int(job["trial_id"])
            try:
                if suite_name not in suites:
                    suites[suite_name] = benchmark_dict[suite_name]()
                suite = suites[suite_name]
                task = suite.get_task(task_id)
                initial_states = suite.get_task_init_states(task_id)
                if trial_id >= len(initial_states):
                    raise IndexError(
                        f"trial_id={trial_id} exceeds {len(initial_states)} states for {suite_name}/{task_id}"
                    )
                requested_env_key = (suite_name, task_id)
                if env is None or requested_env_key != env_key:
                    if env is not None:
                        env.close()
                    env, description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.soft_reset_env)
                    env_key = requested_env_key
                    if args.minimal_policy_observables:
                        _configure_minimal_observables(env)
                else:
                    description = task.language
                args.task_suite_name = suite_name
                success, images = _run_episode(
                    args,
                    client,
                    env,
                    description,
                    initial_states[trial_id],
                )
                _write_video(args, description, success, images, str(job_id))
                eval_queue.finish(database, job_id, args.worker_id, success, started_at)
                completed += 1
                logging.info(
                    "job_done worker=%s id=%s suite=%s task=%s trial=%s success=%s duration_s=%.2f",
                    args.worker_id,
                    job_id,
                    suite_name,
                    task_id,
                    trial_id,
                    success,
                    time.time() - started_at,
                )
            except Exception:
                detail = traceback.format_exc()
                logging.error("job_failed worker=%s id=%s\n%s", args.worker_id, job_id, detail)
                eval_queue.fail(database, job_id, args.worker_id, detail, started_at)
                if env is not None:
                    env.close()
                    env = None
                    env_key = None
    finally:
        if env is not None:
            env.close()
    logging.info("queue_worker_complete worker=%s jobs=%s", args.worker_id, completed)


def eval_libero(args: Args) -> None:
    if args.job_db:
        eval_queue_worker(args)
        return

    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    total_episodes = 0
    total_successes = 0
    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.soft_reset_env)
        if args.minimal_policy_observables:
            _configure_minimal_observables(env)
        try:
            for trial_id in tqdm.tqdm(range(args.num_trials_per_task)):
                success, images = _run_episode(args, client, env, description, initial_states[trial_id])
                _write_video(args, description, success, images, f"{task_id}_{trial_id}")
                total_episodes += 1
                total_successes += int(success)
                logging.info(
                    "episodes=%s successes=%s rate=%.4f",
                    total_episodes,
                    total_successes,
                    total_successes / total_episodes,
                )
        finally:
            env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    try:
        tyro.cli(eval_libero)
    except KeyboardInterrupt:
        logging.info("LIBERO worker interrupted")
        sys.exit(130)
