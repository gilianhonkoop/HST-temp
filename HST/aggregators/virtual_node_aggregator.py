import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn import PNAConv, TransformerConv
from torch_geometric.utils import softmax
import logging
from models import FraudGTConv
from .node_aggregator import NodeAggregator
from .utils import reconstruct_local_edges
import time


def _degree_histogram(edge_index: torch.Tensor | None, num_nodes: int, device: torch.device) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0 or num_nodes <= 0:
        return torch.ones(1, dtype=torch.long, device=device)
    dst = edge_index[1].detach().to(device=device, dtype=torch.long)
    node_degrees = torch.bincount(dst, minlength=int(num_nodes))
    hist = torch.bincount(node_degrees.cpu(), minlength=int(node_degrees.max().item()) + 1)
    return hist.clamp_min(1).to(device)


class PNAVirtualNodeConv(nn.Module):
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
    ) -> torch.Tensor:
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), x.size(-1)))
        h = x
        for conv, norm in zip(self.convs, self.norms):
            updated = conv(h, edge_index, edge_attr)
            updated = norm(updated)
            updated = F.relu(updated)
            updated = self.dropout(updated)
            h = h + updated
        return h

class VirtualNodeAggregator(NodeAggregator):
    """
    Base class for aggregators that use a Global Readout methodology via a Virtual Node.
    """
    def __init__(self, hidden_dim: int = 128, heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.node_proj = nn.LazyLinear(hidden_dim)
        self.edge_proj = nn.LazyLinear(hidden_dim)
        
        self.virtual_node = Parameter(torch.Tensor(1, hidden_dim))
        self.virtual_edge = Parameter(torch.Tensor(1, hidden_dim))
        nn.init.xavier_uniform_(self.virtual_node)
        nn.init.xavier_uniform_(self.virtual_edge)
        
        assert hidden_dim % heads == 0, "hidden_dim must be strictly divisible by heads!"
        self.conv: nn.Module  # To be initialized by subclasses
        
        # Prepared/batched structural cache (built on first forward per process)
        self._prepared: bool = False
        self._concat_node_ids: torch.Tensor | None = None
        self._sizes: list[int] | None = None
        self._batched_edge_index: torch.Tensor | None = None
        self._edge_keep_idx_list: list[torch.Tensor | None] | None = None
        self._virtual_indices: torch.Tensor | None = None
        self._total_nodes_with_v: int | None = None
        self._total_local_edges: int | None = None
        self._total_virtual_edges: int | None = None
        self._local_edge_counts: list[int] | None = None
        self._edge_attr_offsets: list[int] | None = None
        self._cache_key: tuple | None = None
        self._node_positions: torch.Tensor | None = None
        self._virtual_positions: torch.Tensor | None = None
        self._virtual_edge_index_cache: dict[int, torch.Tensor] = {}
        self._virtual_edge_attr_cache: dict[int, torch.Tensor] = {}
        self._local_edge_positions: torch.Tensor | None = None
        self._raw_local_edge_positions: torch.Tensor | None = None
        self._missing_local_edge_positions: torch.Tensor | None = None
        self._virtual_edge_positions: torch.Tensor | None = None
        self._flat_local_edge_attr: torch.Tensor | None = None
        self._prepared_cache: dict[tuple, dict[str, object]] = {}
        self.profile: bool = False
        self._last_profile: dict[str, float] | None = None

    def prepare(self, global_data: Data, subgraph_data_list: list[Data]) -> None:
        x = getattr(global_data, 'x', None)
        device = x.device if x is not None else torch.device('cpu')
        cache_key = tuple(
            (id(sg), sg.x_idx.data_ptr(), int(sg.x_idx.numel()))
            for sg in subgraph_data_list
        )
        if cache_key not in self._prepared_cache:
            self._build_cache(subgraph_data_list, device, cache_key)

    def _store_cache(self, cache_key: tuple) -> None:
        self._prepared_cache[cache_key] = {
            "concat_node_ids": self._concat_node_ids,
            "sizes": self._sizes,
            "batched_edge_index": self._batched_edge_index,
            "edge_keep_idx_list": self._edge_keep_idx_list,
            "virtual_indices": self._virtual_indices,
            "total_nodes_with_v": self._total_nodes_with_v,
            "total_local_edges": self._total_local_edges,
            "total_virtual_edges": self._total_virtual_edges,
            "local_edge_counts": self._local_edge_counts,
            "edge_attr_offsets": self._edge_attr_offsets,
            "node_positions": self._node_positions,
            "virtual_positions": self._virtual_positions,
            "local_edge_positions": self._local_edge_positions,
            "raw_local_edge_positions": self._raw_local_edge_positions,
            "missing_local_edge_positions": self._missing_local_edge_positions,
            "virtual_edge_positions": self._virtual_edge_positions,
            "flat_local_edge_attr": self._flat_local_edge_attr,
        }

    def _load_cache(self, cache_key: tuple) -> bool:
        cached = self._prepared_cache.get(cache_key)
        if cached is None:
            return False
        self._concat_node_ids = cached["concat_node_ids"]  # type: ignore[assignment]
        self._sizes = cached["sizes"]  # type: ignore[assignment]
        self._batched_edge_index = cached["batched_edge_index"]  # type: ignore[assignment]
        self._edge_keep_idx_list = cached["edge_keep_idx_list"]  # type: ignore[assignment]
        self._virtual_indices = cached["virtual_indices"]  # type: ignore[assignment]
        self._total_nodes_with_v = cached["total_nodes_with_v"]  # type: ignore[assignment]
        self._total_local_edges = cached["total_local_edges"]  # type: ignore[assignment]
        self._total_virtual_edges = cached["total_virtual_edges"]  # type: ignore[assignment]
        self._local_edge_counts = cached["local_edge_counts"]  # type: ignore[assignment]
        self._edge_attr_offsets = cached["edge_attr_offsets"]  # type: ignore[assignment]
        self._node_positions = cached["node_positions"]  # type: ignore[assignment]
        self._virtual_positions = cached["virtual_positions"]  # type: ignore[assignment]
        self._local_edge_positions = cached["local_edge_positions"]  # type: ignore[assignment]
        self._raw_local_edge_positions = cached["raw_local_edge_positions"]  # type: ignore[assignment]
        self._missing_local_edge_positions = cached["missing_local_edge_positions"]  # type: ignore[assignment]
        self._virtual_edge_positions = cached["virtual_edge_positions"]  # type: ignore[assignment]
        self._flat_local_edge_attr = cached["flat_local_edge_attr"]  # type: ignore[assignment]
        self._cache_key = cache_key
        self._prepared = True
        return True

    def _build_cache(self, subgraph_data_list: list[Data], device: torch.device, cache_key: tuple) -> None:
        sizes: list[int] = []
        concat_node_ids: list[torch.Tensor] = []
        edge_keep_idx_list: list[torch.Tensor | None] = []
        virtual_indices: list[int] = []
        local_edge_counts: list[int] = []
        edge_attr_offsets: list[int] = []
        node_positions: list[torch.Tensor] = []
        virtual_positions: list[int] = []
        node_offsets: list[int] = []
        edge_index_offsets: list[int] = []
        local_edge_positions: list[torch.Tensor] = []
        raw_local_edge_positions: list[torch.Tensor] = []
        missing_local_edge_positions: list[torch.Tensor] = []
        virtual_edge_positions: list[torch.Tensor] = []
        flat_local_edge_attr_parts: list[torch.Tensor] = []

        offset = 0
        total_local_edges = 0
        total_virtual_edges = 0
        total_edge_attr = 0
        total_edge_index = 0

        for sub_data in subgraph_data_list:
            node_ids = sub_data.x_idx.flatten().to(device)
            concat_node_ids.append(node_ids)
            n = int(node_ids.numel())
            sizes.append(n)

            if not hasattr(sub_data, '_local_edge_index'):
                if sub_data.edge_index is not None and sub_data.edge_index.size(1) > 0 and node_ids.numel() > 0:
                    edge_index = sub_data.edge_index.to(device)
                    u = edge_index[0]
                    v = edge_index[1]

                    min_id = int(node_ids.min().item())
                    max_id = int(node_ids.max().item())
                    id_range = max_id - min_id + 1

                    if min_id >= 0 and id_range <= 5 * node_ids.numel():
                        lookup = torch.full((id_range,), -1, device=device, dtype=torch.long)
                        lookup[node_ids - min_id] = torch.arange(node_ids.numel(), device=device, dtype=torch.long)
                        local_u = lookup[(u - min_id).clamp(min=0, max=id_range - 1)]
                        local_v = lookup[(v - min_id).clamp(min=0, max=id_range - 1)]
                        mask = (local_u >= 0) & (local_v >= 0)
                    else:
                        ids_sorted, perm = torch.sort(node_ids)
                        n_ids = int(node_ids.numel())
                        pos_u = torch.searchsorted(ids_sorted, u)
                        pos_v = torch.searchsorted(ids_sorted, v)
                        pos_u_clamped = pos_u.clamp(max=n_ids - 1)
                        pos_v_clamped = pos_v.clamp(max=n_ids - 1)
                        found_u = (pos_u < n_ids) & (ids_sorted[pos_u_clamped] == u)
                        found_v = (pos_v < n_ids) & (ids_sorted[pos_v_clamped] == v)
                        mask = found_u & found_v
                        local_u = perm[pos_u_clamped]
                        local_v = perm[pos_v_clamped]

                    keep_idx = mask.nonzero(as_tuple=False).flatten()
                    local_edge_index = torch.stack([local_u[mask], local_v[mask]], dim=0)
                else:
                    local_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    keep_idx = None

                sub_data._local_edge_index = local_edge_index
                sub_data._edge_keep_idx = keep_idx

            local_edge_index = sub_data._local_edge_index  # type: ignore[attr-defined]
            keep_idx = getattr(sub_data, '_edge_keep_idx', None)
            edge_keep_idx_list.append(keep_idx)
            local_e_count = int(local_edge_index.size(1))
            local_edge_counts.append(local_e_count)

            node_offsets.append(offset)
            edge_index_offsets.append(total_edge_index)
            if local_e_count > 0:
                total_local_edges += local_e_count

            v_local = n
            v_global = offset + v_local
            virtual_indices.append(v_global)

            if n > 0:
                total_virtual_edges += 2 * n

            if n > 0:
                node_positions.append(torch.arange(n, device=device, dtype=torch.long) + offset)
            virtual_positions.append(v_global)

            edge_attr_offsets.append(total_edge_attr)
            total_edge_attr += local_e_count + (2 * n)
            total_edge_index += local_e_count + (2 * n)

            if local_e_count > 0:
                local_pos = torch.arange(local_e_count, device=device, dtype=torch.long) + edge_attr_offsets[-1]
                local_edge_positions.append(local_pos)
                sub_edge_attr = getattr(sub_data, 'edge_attr', None)
                if sub_edge_attr is not None and keep_idx is not None and keep_idx.numel() > 0:
                    raw_local_edge_positions.append(local_pos)
                    flat_local_edge_attr_parts.append(sub_edge_attr[keep_idx].to(device))
                else:
                    missing_local_edge_positions.append(local_pos)

            if n > 0:
                v_pos = torch.arange(2 * n, device=device, dtype=torch.long) + edge_attr_offsets[-1] + local_e_count
                virtual_edge_positions.append(v_pos)

            offset += n + 1

        self._concat_node_ids = torch.cat(concat_node_ids, dim=0) if len(concat_node_ids) > 0 else torch.empty(0, dtype=torch.long, device=device)
        self._sizes = sizes
        if total_edge_index > 0:
            batched_edge_index = torch.empty((2, total_edge_index), dtype=torch.long, device=device)
            for sub_data, n, local_e_count, edge_offset, node_offset in zip(
                subgraph_data_list, sizes, local_edge_counts, edge_index_offsets, node_offsets
            ):
                if local_e_count > 0:
                    local_edge_index = sub_data._local_edge_index  # type: ignore[attr-defined]
                    batched_edge_index[:, edge_offset:edge_offset + local_e_count] = local_edge_index + node_offset

                if n > 0:
                    v_cache = self._virtual_edge_index_cache.get(n)
                    if v_cache is None or v_cache.device != device:
                        base_src = torch.cat(
                            [torch.arange(n, device=device, dtype=torch.long), torch.full((n,), n, device=device, dtype=torch.long)]
                        )
                        base_dst = torch.cat(
                            [torch.full((n,), n, device=device, dtype=torch.long), torch.arange(n, device=device, dtype=torch.long)]
                        )
                        v_cache = torch.stack([base_src, base_dst], dim=0)
                        self._virtual_edge_index_cache[n] = v_cache

                    v_offset = edge_offset + local_e_count
                    batched_edge_index[:, v_offset:v_offset + (2 * n)] = v_cache + node_offset

            self._batched_edge_index = batched_edge_index
        else:
            self._batched_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        self._edge_keep_idx_list = edge_keep_idx_list
        self._virtual_indices = torch.tensor(virtual_indices, dtype=torch.long, device=device)
        self._total_nodes_with_v = int(offset)
        self._total_local_edges = int(total_local_edges)
        self._total_virtual_edges = int(total_virtual_edges)
        self._local_edge_counts = local_edge_counts
        self._edge_attr_offsets = edge_attr_offsets
        self._node_positions = torch.cat(node_positions, dim=0) if len(node_positions) > 0 else torch.empty(0, dtype=torch.long, device=device)
        self._virtual_positions = torch.tensor(virtual_positions, dtype=torch.long, device=device)
        self._local_edge_positions = torch.cat(local_edge_positions, dim=0) if len(local_edge_positions) > 0 else torch.empty(0, dtype=torch.long, device=device)
        self._raw_local_edge_positions = torch.cat(raw_local_edge_positions, dim=0) if len(raw_local_edge_positions) > 0 else torch.empty(0, dtype=torch.long, device=device)
        self._missing_local_edge_positions = torch.cat(missing_local_edge_positions, dim=0) if len(missing_local_edge_positions) > 0 else torch.empty(0, dtype=torch.long, device=device)
        self._virtual_edge_positions = torch.cat(virtual_edge_positions, dim=0) if len(virtual_edge_positions) > 0 else torch.empty(0, dtype=torch.long, device=device)
        self._flat_local_edge_attr = torch.cat(flat_local_edge_attr_parts, dim=0) if len(flat_local_edge_attr_parts) > 0 else None
        self._cache_key = cache_key
        self._prepared = True
        self._store_cache(cache_key)
        
    def forward(self, global_data: Data, subgraph_data_list: list[Data]) -> torch.Tensor:
        global_x = global_data.x
        device = global_x.device if global_x is not None else torch.device('cpu')
        num_subgraphs = len(subgraph_data_list)

        profile = self.profile
        if profile:
            def _sync():
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            t_total_start = time.perf_counter()
            t_rebuild = 0.0
            t_node_proj = 0.0
            t_x_batched = 0.0
            t_edge_attr = 0.0
            t_conv = 0.0
            t_gather = 0.0
        
        if global_x is None:
            return torch.ones((num_subgraphs, self.hidden_dim), dtype=torch.float, device=device)

        cache_key = tuple(
            (id(sg), sg.x_idx.data_ptr(), int(sg.x_idx.numel()))
            for sg in subgraph_data_list
        )

        if cache_key != self._cache_key and not self._load_cache(cache_key):
            self._prepared = False

        if not self._prepared:
            if profile:
                _sync()
                t0 = time.perf_counter()
            self._build_cache(subgraph_data_list, device, cache_key)
            if profile:
                _sync()
                t_rebuild = time.perf_counter() - t0

        # 1) Project all nodes in one batched pass
        if profile:
            _sync()
            t0 = time.perf_counter()
        sub_x_all = global_x[self._concat_node_ids]  # [sum(N_s), F]
        hidden_all = self.node_proj(sub_x_all)       # [sum(N_s), H]
        if profile:
            _sync()
            t_node_proj = time.perf_counter() - t0

        # 2) Rebuild x_batched by inserting one virtual node after each segment (vectorized)
        if profile:
            _sync()
            t0 = time.perf_counter()
        if self._total_nodes_with_v is not None and self._total_nodes_with_v > 0:
            x_batched = torch.empty((self._total_nodes_with_v, self.hidden_dim), device=device)
            assert self._node_positions is not None and self._virtual_positions is not None
            if self._node_positions.numel() > 0:
                x_batched[self._node_positions] = hidden_all
            x_batched[self._virtual_positions] = self.virtual_node
        else:
            x_batched = torch.empty((0, self.hidden_dim), device=device)
        if profile:
            _sync()
            t_x_batched = time.perf_counter() - t0

        # 3) Build edge_attr per-subgraph to match edge_index block order [L_s, V_s] for each subgraph (preallocated)
        if profile:
            _sync()
            t0 = time.perf_counter()
        total_edge_attr = (self._total_local_edges or 0) + (self._total_virtual_edges or 0)
        edge_attr = None
        if total_edge_attr > 0:
            edge_attr = torch.empty((total_edge_attr, self.hidden_dim), device=device)

        if edge_attr is not None:
            assert self._local_edge_positions is not None
            assert self._raw_local_edge_positions is not None
            assert self._missing_local_edge_positions is not None
            assert self._virtual_edge_positions is not None

            if self._raw_local_edge_positions.numel() > 0 and self._flat_local_edge_attr is not None:
                projected = self.edge_proj(self._flat_local_edge_attr)
                edge_attr[self._raw_local_edge_positions] = projected

            if self._missing_local_edge_positions.numel() > 0:
                edge_attr[self._missing_local_edge_positions] = 0.0

            if self._virtual_edge_positions.numel() > 0:
                edge_attr[self._virtual_edge_positions] = self.virtual_edge.expand(self._virtual_edge_positions.numel(), -1)
        if profile:
            _sync()
            t_edge_attr = time.perf_counter() - t0

        # 4) Single batched conv
        if profile:
            _sync()
            t0 = time.perf_counter()
        out_all = F.relu(self.conv(x_batched, self._batched_edge_index, edge_attr))
        if profile:
            _sync()
            t_conv = time.perf_counter() - t0

        # 5) Gather virtual node embeddings in original order
        if profile:
            _sync()
            t0 = time.perf_counter()
        supernodes = out_all[self._virtual_indices]
        if profile:
            _sync()
            t_gather = time.perf_counter() - t0
            total = time.perf_counter() - t_total_start
            self._last_profile = {
                "rebuild": t_rebuild,
                "node_proj": t_node_proj,
                "x_batched": t_x_batched,
                "edge_attr": t_edge_attr,
                "conv": t_conv,
                "gather": t_gather,
                "total": total,
            }
            logging.info(f"VirtualNodeAggregator profile: {self._last_profile}")
        return supernodes

class TransformerconvNodeAggregator(VirtualNodeAggregator):
    """Standard PyG TransformerConv Virtual Node implementation"""
    def __init__(self, hidden_dim: int = 128, heads: int = 4):
        super().__init__(hidden_dim, heads)
        self.conv = TransformerConv(
            in_channels=hidden_dim, 
            out_channels=hidden_dim // heads, 
            heads=heads, 
            edge_dim=hidden_dim
        )

class FraudgtNodeAggregator(VirtualNodeAggregator):
    """Strict Mathematical implementation of the FraudGT mechanism via Virtual Node"""
    def __init__(self, hidden_dim: int = 128, heads: int = 4):
        super().__init__(hidden_dim, heads)
        self.conv = FraudGTConv(hidden_dim=hidden_dim, heads=heads)


class PnaNodeAggregator(VirtualNodeAggregator):
    """PyG PNAConv virtual-node implementation for component node summarization."""
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        aggregators: list[str] | tuple[str, ...] | None = None,
        scalers: list[str] | tuple[str, ...] | None = None,
        towers: int = 2,
        pre_layers: int = 1,
        post_layers: int = 1,
        dropout: float = 0.0,
        norm: str = "batch",
    ):
        super().__init__(hidden_dim, heads=int(towers))
        self.pna_num_layers = int(num_layers)
        self.pna_aggregators = list(aggregators or ["mean", "min", "max", "std"])
        self.pna_scalers = list(scalers or ["identity", "amplification", "attenuation"])
        self.pna_towers = int(towers)
        self.pna_pre_layers = int(pre_layers)
        self.pna_post_layers = int(post_layers)
        self.pna_dropout = float(dropout)
        self.pna_norm = str(norm).lower()
        self.conv: PNAVirtualNodeConv | None = None

    def _ensure_conv(self, device: torch.device) -> None:
        if self.conv is not None:
            return
        if self._batched_edge_index is None or self._total_nodes_with_v is None:
            raise RuntimeError("PnaNodeAggregator must build its virtual-node graph before initializing PNAConv")
        deg = _degree_histogram(self._batched_edge_index, int(self._total_nodes_with_v), device)
        self.conv = PNAVirtualNodeConv(
            hidden_dim=self.hidden_dim,
            deg=deg,
            num_layers=self.pna_num_layers,
            aggregators=self.pna_aggregators,
            scalers=self.pna_scalers,
            towers=self.pna_towers,
            pre_layers=self.pna_pre_layers,
            post_layers=self.pna_post_layers,
            dropout=self.pna_dropout,
            norm=self.pna_norm,
        ).to(device)

    def prepare(self, global_data: Data, subgraph_data_list: list[Data]) -> None:
        super().prepare(global_data, subgraph_data_list)
        x = getattr(global_data, 'x', None)
        device = x.device if x is not None else torch.device('cpu')
        self._ensure_conv(device)

    def forward(self, global_data: Data, subgraph_data_list: list[Data]) -> torch.Tensor:
        global_x = global_data.x
        device = global_x.device if global_x is not None else torch.device('cpu')
        cache_key = tuple(
            (id(sg), sg.x_idx.data_ptr(), int(sg.x_idx.numel()))
            for sg in subgraph_data_list
        )
        if cache_key != self._cache_key and not self._load_cache(cache_key):
            self._prepared = False
        if not self._prepared:
            self._build_cache(subgraph_data_list, device, cache_key)
        self._ensure_conv(device)
        return super().forward(global_data, subgraph_data_list)


PNANodeAggregator = PnaNodeAggregator
