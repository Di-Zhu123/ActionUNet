"""Serve an ActionUNet checkpoint through OpenPI's WebSocket policy API."""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from actionunet.model_actionunet import ActionUNetPytorch
from actionunet.release import data_config_for_repo
from actionunet.release import model_config_from_checkpoint
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as openpi_config


def build_policy(
    base_config: str,
    checkpoint_dir: str | Path,
    *,
    device: str,
    num_steps: int,
    data_repo_id: str | None = None,
    asset_id: str | None = None,
):
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not (checkpoint_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing model.safetensors in {checkpoint_dir}")
    base = openpi_config.get_config(base_config)
    model = model_config_from_checkpoint(checkpoint_dir)
    if base.model.action_dim != model.action_dim or base.model.action_horizon != model.action_horizon:
        raise ValueError("Base OpenPI data config does not match the checkpoint action shape")
    data = data_config_for_repo(base.data, data_repo_id, asset_id=asset_id)
    train_config = dataclasses.replace(base, model=model, data=data)
    pi0_pytorch.PI0Pytorch = ActionUNetPytorch
    return policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        sample_kwargs={"num_steps": num_steps},
        pytorch_device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--data-repo-id", help="LeRobot dataset id; preserves the base LIBERO or ALOHA data pipeline")
    parser.add_argument("--asset-id", help="Normalization asset id; defaults to data repo id")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    policy = build_policy(
        args.base_config,
        args.checkpoint,
        device=args.device,
        num_steps=args.num_steps,
        data_repo_id=args.data_repo_id,
        asset_id=args.asset_id,
    )
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
