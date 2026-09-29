"""Minimal ActionUNet model: π0.5 + UNet/nearest/FC/SIREN/Matérn-2.5."""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from typing_extensions import override

from actionunet.config import ActionUNetConfig
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class _DownBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.conv(x.transpose(1, 2)).transpose(1, 2)))


class ActionUNetPytorch(PI0Pytorch):
    """The fixed ActionUNet variant; generic π0.5 logic is inherited from OpenPI."""

    _IGNORED_OLD_CHECKPOINT_PREFIXES = (
        "ms_local_proj.",
        "ms_pyramid_blocks.",
        "ms_fuse.",
        "ms_unet_bottleneck_mlp.",
        "ms_aux_",
        "ms_siren_",
        "ms_bias_heads.",
    )
    _ACTIONUNET_PARAMETER_PREFIXES = (
        "action_out_",
        "ms_unet_down_blocks.",
        "ms_unet_decoder_fuse.",
        "ms_unet_bottleneck_fc.",
    )

    def __init__(self, config: ActionUNetConfig):
        if not isinstance(config, ActionUNetConfig):
            raise TypeError("ActionUNetPytorch requires ActionUNetConfig")

        # This is the π0.5 constructor with its linear output layer replaced by
        # the fixed ActionUNet decoder. All generic forward/inference code stays
        # inherited from the unmodified OpenPI PI0Pytorch class.
        nn.Module.__init__(self)
        self.config = config
        self.pi05 = True
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        expert_config = _gemma.get_config(config.action_expert_variant)
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )
        width = expert_config.width
        self.action_in_proj = nn.Linear(config.action_dim, width)
        self.time_mlp_in = nn.Linear(width, width)
        self.time_mlp_out = nn.Linear(width, width)

        self.action_out_norm = nn.LayerNorm(width)
        self.action_out_proj_e1 = nn.Linear(width, width)
        self.action_out_proj_tau1 = nn.Linear(1, width, bias=False)
        self.action_out_proj_e2 = nn.Linear(width, width)
        self.action_out_proj_tau2 = nn.Linear(1, width, bias=False)
        self.action_out_final = nn.Linear(width, config.action_dim)
        self._init_siren(width)

        levels = math.ceil(math.log2(config.action_horizon))
        self.ms_unet_down_blocks = nn.ModuleList(_DownBlock(width) for _ in range(levels))
        self.ms_unet_decoder_fuse = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(levels))
        self.ms_unet_bottleneck_fc = nn.Linear(width, width)

        torch.set_float32_matmul_precision("high")
        self.gradient_checkpointing_enabled = False
        self._check_transformers_patch()

    def _init_siren(self, width: int) -> None:
        with torch.no_grad():
            radius = 1.0 / width
            self.action_out_proj_e1.weight.uniform_(-radius, radius)
            self.action_out_proj_tau1.weight.uniform_(-radius, radius)
            self.action_out_proj_e1.bias.uniform_(-radius, radius)
            radius = math.sqrt(6.0 / width)
            self.action_out_proj_e2.weight.uniform_(-radius, radius)
            self.action_out_proj_tau2.weight.uniform_(-radius, radius)
            self.action_out_proj_e2.bias.uniform_(-radius, radius)
            nn.init.kaiming_normal_(self.action_out_final.weight, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(self.action_out_final.bias)

    @staticmethod
    def _check_transformers_patch() -> None:
        message = (
            "OpenPI transformers_replace is not installed. Copy "
            "src/openpi/models_pytorch/transformers_replace/* into the installed transformers package."
        )
        try:
            check = importlib.import_module("transformers.models.siglip.check")
        except ImportError:
            raise ValueError(message) from None
        if not check.check_whether_transformers_replace_is_installed_correctly():
            raise ValueError(message)

    def sample_noise(self, shape, device):
        """Draw Matérn-2.5 GP noise along the action horizon."""
        white = torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)
        horizon = shape[1]
        index = torch.arange(horizon, dtype=torch.float32, device=device)
        distance = (index[:, None] - index[None, :]).abs() / self.config.noise_matern_rou
        scaled = math.sqrt(5.0) * distance
        kernel = (1.0 + scaled + scaled * scaled / 3.0) * torch.exp(-scaled)
        kernel = kernel + 1e-6 * torch.eye(horizon, dtype=torch.float32, device=device)
        return torch.einsum("ij,bjd->bid", torch.linalg.cholesky(kernel), white)

    def action_out_proj(self, suffix_out: torch.Tensor) -> torch.Tensor:
        """Decode transformer features with UNet, nearest upsampling and Matérn aggregation."""
        encoder = [self.action_out_norm(suffix_out)]
        for block in self.ms_unet_down_blocks:
            encoder.append(block(encoder[-1]))

        decoded = self.ms_unet_bottleneck_fc(encoder[-1])
        for level in range(len(encoder) - 2, -1, -1):
            upsampled = F.interpolate(
                decoded.transpose(1, 2), size=encoder[level].shape[1], mode="nearest"
            ).transpose(1, 2)
            decoded = self.ms_unet_decoder_fuse[level](torch.cat([upsampled, encoder[level]], dim=-1))

        batch, horizon, _ = decoded.shape
        tau = torch.linspace(-1.0, 1.0, horizon, dtype=decoded.dtype, device=decoded.device)
        tau = tau.view(1, 1, horizon, 1).expand(batch, horizon, -1, -1)
        code = decoded[:, :, None, :].expand(batch, horizon, horizon, -1)
        hidden = torch.sin(self.action_out_proj_e1(code) + self.action_out_proj_tau1(tau))
        hidden = torch.sin(self.action_out_proj_e2(hidden) + self.action_out_proj_tau2(tau))
        local_actions = self.action_out_final(hidden)

        index = torch.arange(horizon, dtype=decoded.dtype, device=decoded.device)
        distance = (index[:, None] - index[None, :]).abs() / self.config.siren_matern_rou
        scaled = math.sqrt(5.0) * distance
        weights = (1.0 + scaled + scaled * scaled / 3.0) * torch.exp(-scaled)
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
        return torch.einsum("st,bstd->btd", weights, local_actions)

    @override
    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False):
        """Load current checkpoints, old fixed-variant checkpoints, or π0.5 base weights."""
        is_base_checkpoint = "action_out_proj.weight" in state_dict
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key not in {"action_out_proj.weight", "action_out_proj.bias"}
            and not key.startswith(self._IGNORED_OLD_CHECKPOINT_PREFIXES)
        }
        result = super().load_state_dict(filtered, strict=False, assign=assign)
        missing = list(result.missing_keys)
        if is_base_checkpoint:
            missing = [key for key in missing if not key.startswith(self._ACTIONUNET_PARAMETER_PREFIXES)]
        unexpected = list(result.unexpected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(f"State-dict mismatch: missing={missing}, unexpected={unexpected}")
        return type(result)(missing, unexpected)
