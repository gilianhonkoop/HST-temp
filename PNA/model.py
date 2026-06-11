from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool, global_max_pool, global_mean_pool


class PNAGraphClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        deg: torch.Tensor,
        edge_dim: int = 0,
        dropout: float = 0.0,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        aggregators: List[str] | None = None,
        scalers: List[str] | None = None,
        readout: str = "mean_sum_max",
    ):
        super().__init__()
        if aggregators is None:
            aggregators = ["mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["identity", "amplification", "attenuation"]
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.dropout = float(dropout)
        self.readout = str(readout)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                PNAConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg.cpu(),
                    edge_dim=edge_dim if edge_dim > 0 else None,
                    towers=int(towers),
                    pre_layers=int(pre_layers),
                    post_layers=int(post_layers),
                    divide_input=False,
                )
            )
            self.norms.append(BatchNorm(hidden_channels))

        readout_dim = hidden_channels * self._readout_multiplier()
        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels, 1),
        )

    def _readout_multiplier(self) -> int:
        if self.readout in {"mean", "sum", "max"}:
            return 1
        if self.readout in {"mean_sum", "mean_max", "sum_max"}:
            return 2
        if self.readout in {"mean_sum_max", "all"}:
            return 3
        raise ValueError(f"Unsupported readout: {self.readout}")

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean":
            return global_mean_pool(x, batch)
        if self.readout == "sum":
            return global_add_pool(x, batch)
        if self.readout == "max":
            return global_max_pool(x, batch)
        if self.readout == "mean_sum":
            return torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=-1)
        if self.readout == "mean_max":
            return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        if self.readout == "sum_max":
            return torch.cat([global_add_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return torch.cat(
            [
                global_mean_pool(x, batch),
                global_add_pool(x, batch),
                global_max_pool(x, batch),
            ],
            dim=-1,
        )

    def forward(self, data) -> torch.Tensor:
        x = self.input_proj(data.x.float())
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None and edge_attr.numel() == 0:
            edge_attr = None

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        pooled = self._pool(x, data.batch)
        return self.head(pooled).view(-1)
