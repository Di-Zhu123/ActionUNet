"""Configuration for the single published ActionUNet architecture."""

from __future__ import annotations

import dataclasses
from typing import Literal

from openpi.models.pi0_config import Pi0Config


@dataclasses.dataclass(frozen=True)
class ActionUNetConfig(Pi0Config):
    """π0.5 with the fixed ActionUNet decoder used in this repository."""

    architecture: Literal["unet_nearest_fc_matern25"] = "unet_nearest_fc_matern25"
    noise_matern_rou: float = 1.0
    siren_matern_rou: float = 0.5
    matern_gamma: Literal[2.5] = 2.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.pi05:
            raise ValueError("ActionUNet requires pi05=True")
        expected = {
            "architecture": "unet_nearest_fc_matern25",
            "noise_matern_rou": 1.0,
            "siren_matern_rou": 0.5,
            "matern_gamma": 2.5,
        }
        invalid = {name: getattr(self, name) for name, value in expected.items() if getattr(self, name) != value}
        if invalid:
            raise ValueError(f"Only UNet + nearest + FC + Matérn 2.5 is supported; invalid values: {invalid}")

    @classmethod
    def from_pi0_config(cls, base: Pi0Config) -> ActionUNetConfig:
        fields = {field.name for field in dataclasses.fields(Pi0Config)}
        return cls(**{name: getattr(base, name) for name in fields})
