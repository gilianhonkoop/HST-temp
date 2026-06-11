from dataclasses import dataclass, field
from typing import Literal

@dataclass
class PearlConfig:
    num_phi_layers: int = 4
    num_samples: int = 32
    sample_pool: Literal["mean", "sum"] = "mean"
    random_std: float = 1.0

@dataclass
class FraudGTConfig:
    heads: int = 4

@dataclass
class PNAConfig:
    num_layers: int = 3
    aggregators: tuple[str, ...] = ("mean", "min", "max", "std")
    scalers: tuple[str, ...] = ("identity", "amplification", "attenuation")
    towers: int = 2
    pre_layers: int = 1
    post_layers: int = 1
    norm: Literal["batch", "layer", "none"] = "batch"

@dataclass
class ModelConfig:
    hidden_dim: int
    pearl: PearlConfig
    fraudgt: FraudGTConfig
    pna: PNAConfig = field(default_factory=PNAConfig)
    backbone: Literal["fraudgt", "pna"] = "fraudgt"
    dropout: float = 0.0
