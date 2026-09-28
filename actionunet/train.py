"""Train the published ActionUNet variant with the unmodified OpenPI data pipeline."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import tqdm

from actionunet import openpi_train_pytorch as train_pytorch
from actionunet.model_actionunet import ActionUNetPytorch
from actionunet.release import build_train_config
from openpi import transforms
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
from openpi.shared import normalize
from openpi.training import data_loader


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {key: value for key, value in x.items() if not np.issubdtype(np.asarray(value).dtype, np.str_)}


def compute_norm_stats(config) -> Path:
    """Compute the OpenPI state/action statistics for this resolved training config."""
    data = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
    dataset = data_loader.TransformedDataset(
        dataset,
        [*data.repack_transforms.inputs, *data.data_transforms.inputs, RemoveStrings()],
    )
    if len(dataset) < 2:
        raise ValueError("At least two dataset frames are required for normalization statistics")
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=min(config.batch_size, len(dataset)),
        num_workers=config.num_workers,
        shuffle=False,
        num_batches=len(dataset) // min(config.batch_size, len(dataset)),
        framework="pytorch",
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, desc="Computing OpenPI normalization stats"):
        for key, running_stats in stats.items():
            running_stats.update(np.asarray(batch[key]))
    output_root = Path(config.data.assets.assets_dir or config.assets_dirs)
    output_path = output_root / data.asset_id
    normalize.save(output_path, {key: stat.get_statistics() for key, stat in stats.items()})
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_config", help="An existing OpenPI pi0.5 config, e.g. pi05_libero")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--base-weights", help="Directory containing base model.safetensors (required for training)")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--data-repo-id", help="LeRobot dataset id; preserves the base LIBERO or ALOHA data pipeline")
    parser.add_argument("--asset-id", help="Normalization asset id; defaults to data repo id")
    parser.add_argument("--assets-dir", help="Directory containing training normalization assets")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-train-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--peak-lr", type=float)
    parser.add_argument("--decay-steps", type=int)
    parser.add_argument("--decay-lr", type=float)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compute-norm-stats", action="store_true", help="Compute stats on one process before training")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")
    overrides = {
        "batch_size": args.batch_size,
        "num_train_steps": args.num_train_steps,
        "save_interval": args.save_interval,
        "num_workers": args.num_workers,
        "wandb_enabled": args.wandb,
        "resume": args.resume,
        "overwrite": args.overwrite,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    config = build_train_config(
        args.base_config,
        exp_name=args.exp_name,
        checkpoint_base_dir=args.checkpoint_root,
        pytorch_weight_path=args.base_weights,
        data_repo_id=args.data_repo_id,
        asset_id=args.asset_id,
        assets_dir=args.assets_dir,
        **overrides,
    )
    schedule_updates = {
        "warmup_steps": args.warmup_steps,
        "peak_lr": args.peak_lr,
        "decay_steps": args.decay_steps,
        "decay_lr": args.decay_lr,
    }
    schedule_updates = {key: value for key, value in schedule_updates.items() if value is not None}
    if schedule_updates:
        config = dataclasses.replace(config, lr_schedule=dataclasses.replace(config.lr_schedule, **schedule_updates))
    if args.dry_run:
        print(config)
        return
    if args.compute_norm_stats:
        print(compute_norm_stats(config))
        return
    if args.base_weights is None:
        raise ValueError("--base-weights is required for training")
    pi0_pytorch.PI0Pytorch = ActionUNetPytorch
    train_pytorch.init_logging()
    train_pytorch.train_loop(config)


if __name__ == "__main__":
    main()
