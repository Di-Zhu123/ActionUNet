"""The single published ActionUNet architecture and its OpenPI training config."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from actionunet.config import ActionUNetConfig
from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as openpi_config

ARCHITECTURE = "unet_nearest_fc_matern25"

# Metadata written by the pre-extraction implementation. These values are
# checked only when loading an old checkpoint; they are not runtime options.
_LEGACY_ARCHITECTURE = {
    "ms_enabled": True,
    "ms_code_mode": "unet",
    "ms_decode_mode": "direct_siren",
    "ms_tau_mode": "per_layer",
    "ms_agg_mode": "matern_index",
    "ms_unet_upsample_mode": "nearest",
    "ms_unet_bottleneck_type": "fc",
    "matern_gamma": 2.5,
    "siren_matern_rou": 0.5,
    "noise_mode": "gp",
    "noise_matern_rou": 1.0,
    "ms_siren_w0": 1.0,
    "ms_force_c2_gp": False,
    "ms_unet_num_levels": None,
    "ms_unet_downsample_factor": 2,
    "ms_unet_fusion_mode": "replace",
    "ms_output_residual": False,
    "ms_siren_num_layers": 2,
    "ms_modulation_type": "bias_only",
    "ms_learnable_centers": False,
    "ms_learnable_widths": False,
}


def config_name(base_name: str) -> str:
    return f"{base_name}__actionunet"


def build_model_config(base: Pi0Config) -> ActionUNetConfig:
    if not base.pi05:
        raise ValueError("ActionUNet requires an OpenPI pi0.5 base config")
    return ActionUNetConfig.from_pi0_config(base)


def validate_model_config(model: ActionUNetConfig) -> None:
    if model.architecture != ARCHITECTURE:
        raise ValueError(f"Unsupported ActionUNet architecture: {model.architecture!r}")


def aloha_data_config(
    repo_id: str,
    *,
    asset_id: str | None = None,
    assets_dir: str | Path | None = None,
) -> openpi_config.LeRobotAlohaDataConfig:
    """Use the RoboTwin three-camera ALOHA mapping."""
    repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                }
            )
        ]
    )
    assets = openpi_config.AssetsConfig(
        assets_dir=str(Path(assets_dir).expanduser().resolve()) if assets_dir is not None else None,
        asset_id=asset_id or repo_id,
    )
    return openpi_config.LeRobotAlohaDataConfig(
        repo_id=repo_id,
        repack_transforms=repack,
        adapt_to_pi=False,
        base_config=openpi_config.DataConfig(prompt_from_task=True),
        assets=assets,
    )


def data_config_for_repo(
    base: openpi_config.DataConfigFactory,
    repo_id: str | None,
    *,
    asset_id: str | None = None,
    assets_dir: str | Path | None = None,
) -> openpi_config.DataConfigFactory:
    """Keep the upstream LIBERO pipeline or select the three-camera ALOHA mapping."""
    if isinstance(base, openpi_config.LeRobotLiberoDataConfig):
        assets = dataclasses.replace(
            base.assets,
            asset_id=asset_id or repo_id or base.assets.asset_id,
            assets_dir=str(Path(assets_dir).expanduser().resolve()) if assets_dir is not None else base.assets.assets_dir,
        )
        return dataclasses.replace(base, repo_id=repo_id or base.repo_id, assets=assets)
    if repo_id is not None:
        return aloha_data_config(repo_id, asset_id=asset_id, assets_dir=assets_dir)
    if asset_id is not None or assets_dir is not None:
        raise ValueError("--asset-id and --assets-dir require --data-repo-id for ALOHA configs")
    return base


def build_train_config(
    base_name: str,
    *,
    exp_name: str,
    checkpoint_base_dir: str | Path | None = None,
    pytorch_weight_path: str | Path | None = None,
    data_repo_id: str | None = None,
    asset_id: str | None = None,
    assets_dir: str | Path | None = None,
    **overrides,
) -> openpi_config.TrainConfig:
    base = openpi_config.get_config(base_name)
    if not isinstance(base.model, Pi0Config):
        raise TypeError(f"{base_name} is not an OpenPI pi0/0.5 config")
    fields = {field.name for field in dataclasses.fields(base)}
    invalid = set(overrides) - fields
    if invalid:
        raise TypeError(f"Unknown OpenPI training config fields: {sorted(invalid)}")
    updates = {
        "name": config_name(base_name),
        "model": build_model_config(base.model),
        "exp_name": exp_name,
        **overrides,
    }
    updates["data"] = data_config_for_repo(base.data, data_repo_id, asset_id=asset_id, assets_dir=assets_dir)
    if checkpoint_base_dir is not None:
        updates["checkpoint_base_dir"] = str(Path(checkpoint_base_dir).expanduser().resolve())
    if pytorch_weight_path is not None:
        weight = Path(pytorch_weight_path).expanduser().resolve()
        if weight.is_file() and weight.name == "model.safetensors":
            weight = weight.parent
        if not (weight / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing base model.safetensors in {weight}")
        updates["pytorch_weight_path"] = str(weight)
    return dataclasses.replace(base, **updates)


def model_config_from_checkpoint(checkpoint_dir: str | Path) -> ActionUNetConfig:
    """Reconstruct the exact model config recorded by the OpenPI trainer."""
    metadata_path = Path(checkpoint_dir) / "metadata.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata is required: {metadata_path}")
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    model_dict = metadata["config"]["model"]
    if "architecture" in model_dict:
        if model_dict["architecture"] != ARCHITECTURE:
            raise ValueError(f"Unsupported checkpoint architecture: {model_dict['architecture']!r}")
    else:
        missing = set(_LEGACY_ARCHITECTURE) - set(model_dict)
        invalid = {
            key: model_dict[key]
            for key, expected in _LEGACY_ARCHITECTURE.items()
            if key in model_dict and model_dict[key] != expected
        }
        if missing or invalid:
            raise ValueError(
                "Checkpoint metadata is not the fixed ActionUNet variant: "
                f"missing={sorted(missing)}, invalid={invalid}"
            )
    fields = {field.name for field in dataclasses.fields(ActionUNetConfig)}
    model = ActionUNetConfig(**{key: value for key, value in model_dict.items() if key in fields})
    validate_model_config(model)
    return model
