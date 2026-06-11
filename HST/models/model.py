import torch
import time
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import PNAConv
import torch.nn.functional as F

from .pearl import RandomPEARLEncoder
from .fraudgt import FraudGTConv
from configs.config import ModelConfig


class PNANodeBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        deg: torch.Tensor,
        num_layers: int = 3,
        aggregators: list[str] | tuple[str, ...] | None = None,
        scalers: list[str] | tuple[str, ...] | None = None,
        towers: int = 2,
        pre_layers: int = 1,
        post_layers: int = 1,
        dropout: float = 0.0,
        norm: str = "batch",
    ):
        super().__init__()
        if hidden_dim % towers != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by PNA towers ({towers})")
        aggregators = list(aggregators or ["mean", "min", "max", "std"])
        scalers = list(scalers or ["identity", "amplification", "attenuation"])
        norm = str(norm).lower()
        if norm not in {"batch", "layer", "none"}:
            raise ValueError(f"Unsupported PNA norm={norm!r}; expected 'batch', 'layer', or 'none'")
        self.edge_proj = nn.LazyLinear(hidden_dim)
        self.pearl_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.convs.append(
                PNAConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg.detach().cpu(),
                    edge_dim=hidden_dim,
                    towers=int(towers),
                    pre_layers=int(pre_layers),
                    post_layers=int(post_layers),
                    divide_input=False,
                )
            )
            if norm == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif norm == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.norms.append(nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        pearl_encodings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pearl_encodings is not None:
            x = x + self.pearl_proj(pearl_encodings)

        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), x.size(-1)))
        else:
            edge_attr = self.edge_proj(edge_attr.float())

        h = x
        for conv, norm in zip(self.convs, self.norms):
            updated = conv(h, edge_index, edge_attr)
            updated = norm(updated)
            updated = F.relu(updated)
            updated = self.dropout(updated)
            h = h + updated
        return h


class HierarchicalSubgraphTransformer(nn.Module):
    def __init__(self, config: ModelConfig, seed: int | None = None, pna_deg: torch.Tensor | None = None):
        super().__init__()
        self.config = config
        self.encoder = RandomPEARLEncoder(config.hidden_dim, config.pearl.num_phi_layers, 
                                          config.pearl.num_samples, config.pearl.sample_pool, 
                                          config.pearl.random_std, seed)
        # Add a projection layer for the input features
        self.input_proj = nn.LazyLinear(config.hidden_dim)
        self.backbone_name = str(getattr(config, "backbone", "fraudgt")).lower()
        if self.backbone_name == "pna":
            if pna_deg is None:
                pna_deg = torch.ones(1, dtype=torch.long)
            self.graph_backbone = PNANodeBackbone(
                hidden_dim=config.hidden_dim,
                deg=pna_deg,
                num_layers=config.pna.num_layers,
                aggregators=config.pna.aggregators,
                scalers=config.pna.scalers,
                towers=config.pna.towers,
                pre_layers=config.pna.pre_layers,
                post_layers=config.pna.post_layers,
                dropout=config.dropout,
                norm=config.pna.norm,
            )
        elif self.backbone_name == "fraudgt":
            self.graph_backbone = FraudGTConv(config.hidden_dim, config.fraudgt.heads)
        else:
            raise ValueError(f"Unsupported model.backbone={config.backbone!r}; expected 'fraudgt' or 'pna'")
        self.dropout = nn.Dropout(config.dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
        start_time = time.time()
        pe_encodings = self.encoder(data)
        encoding_time = time.time()

        # Project input features to hidden_dim
        x = self.input_proj(x)
        
        out = self.graph_backbone(x, edge_index, edge_attr, pearl_encodings=pe_encodings)
        out = self.dropout(out)
        out = self.classifier(out).squeeze(-1)
        fraudgt_time = time.time()

        # print(f"PEARL encoding time: {encoding_time - start_time:.4f}s, FraudGT time: {fraudgt_time - encoding_time:.4f}s")

        return out
