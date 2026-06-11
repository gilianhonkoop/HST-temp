import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import softmax
from models import FraudGTConv
from .node_aggregator import NodeAggregator
from .utils import reconstruct_local_edges

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
        
    def forward(self, global_data: Data, subgraph_data_list: list[Data]) -> torch.Tensor:
        global_x = global_data.x
        device = global_x.device if global_x is not None else torch.device('cpu')
        num_subgraphs = len(subgraph_data_list)
        
        if global_x is None:
            return torch.ones((num_subgraphs, self.hidden_dim), dtype=torch.float, device=device)

        # Prepare static batched structure on the first call
        if not self._prepared:
            sizes: list[int] = []
            concat_node_ids: list[torch.Tensor] = []
            batched_src: list[torch.Tensor] = []
            batched_dst: list[torch.Tensor] = []
            edge_keep_idx_list: list[torch.Tensor | None] = []
            virtual_indices: list[int] = []

            offset = 0
            total_local_edges = 0
            total_virtual_edges = 0

            for sub_data in subgraph_data_list:
                node_ids = sub_data.x_idx.flatten().to(device)
                concat_node_ids.append(node_ids)
                n = int(node_ids.numel())
                sizes.append(n)

                # Build/cached local edge index
                if not hasattr(sub_data, '_local_edge_index'):
                    mapping = {global_id.item(): local_id for local_id, global_id in enumerate(node_ids)}
                    src_list, dst_list, keep_idx = [], [], []
                    if sub_data.edge_index is not None and sub_data.edge_index.size(1) > 0:
                        for i in range(sub_data.edge_index.size(1)):
                            u = sub_data.edge_index[0, i].item()
                            v = sub_data.edge_index[1, i].item()
                            if u in mapping and v in mapping:
                                src_list.append(mapping[u])
                                dst_list.append(mapping[v])
                                keep_idx.append(i)
                    local_edge_index = (
                        torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
                        if len(src_list) > 0 else torch.empty((2, 0), dtype=torch.long, device=device)
                    )
                    sub_data._local_edge_index = local_edge_index
                    sub_data._edge_keep_idx = (
                        torch.tensor(keep_idx, dtype=torch.long, device=device) if len(keep_idx) > 0 else None
                    )

                local_edge_index = sub_data._local_edge_index  # type: ignore[attr-defined]
                keep_idx = getattr(sub_data, '_edge_keep_idx', None)
                edge_keep_idx_list.append(keep_idx)

                # Offset and accumulate structural edges
                if local_edge_index.size(1) > 0:
                    batched_src.append(local_edge_index[0] + offset)
                    batched_dst.append(local_edge_index[1] + offset)
                    total_local_edges += int(local_edge_index.size(1))

                # Virtual node placement and virtual edges (offset applied)
                v_local = n
                v_global = offset + v_local
                virtual_indices.append(v_global)

                if n > 0:
                    ar = torch.arange(n, device=device, dtype=torch.long)
                    v_src = torch.cat([ar + offset, torch.full((n,), v_global, device=device, dtype=torch.long)])
                    v_dst = torch.cat([torch.full((n,), v_global, device=device, dtype=torch.long), ar + offset])
                    batched_src.append(v_src)
                    batched_dst.append(v_dst)
                    total_virtual_edges += 2 * n

                offset += n + 1

            # Finalize cached tensors
            self._concat_node_ids = torch.cat(concat_node_ids, dim=0) if len(concat_node_ids) > 0 else torch.empty(0, dtype=torch.long, device=device)
            self._sizes = sizes
            if len(batched_src) > 0:
                src = torch.cat(batched_src, dim=0)
                dst = torch.cat(batched_dst, dim=0)
                self._batched_edge_index = torch.stack([src, dst], dim=0)
            else:
                self._batched_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            self._edge_keep_idx_list = edge_keep_idx_list
            self._virtual_indices = torch.tensor(virtual_indices, dtype=torch.long, device=device)
            self._total_nodes_with_v = int(offset)
            self._total_local_edges = int(total_local_edges)
            self._total_virtual_edges = int(total_virtual_edges)
            self._prepared = True

        assert self._prepared
        assert self._concat_node_ids is not None and self._sizes is not None
        assert self._batched_edge_index is not None and self._virtual_indices is not None
        assert self._edge_keep_idx_list is not None

        # 1) Project all nodes in one batched pass
        sub_x_all = global_x[self._concat_node_ids]  # [sum(N_s), F]
        hidden_all = self.node_proj(sub_x_all)       # [sum(N_s), H]

        # 2) Rebuild x_batched by inserting one virtual node after each segment
        x_batched_list: list[torch.Tensor] = []
        start = 0
        for n in self._sizes:
            end = start + n
            seg = hidden_all[start:end]
            x_batched_list.append(torch.cat([seg, self.virtual_node], dim=0))
            start = end
        x_batched = torch.cat(x_batched_list, dim=0) if len(x_batched_list) > 0 else torch.empty((0, self.hidden_dim), device=device)

        # 3) Build edge_attr in a single batched projection
        local_attr_list: list[torch.Tensor] = []
        for keep_idx, sub_data in zip(self._edge_keep_idx_list, subgraph_data_list):
            sub_edge_attr = getattr(sub_data, 'edge_attr', None)
            if sub_edge_attr is not None and keep_idx is not None and keep_idx.numel() > 0:
                local_attr_list.append(sub_edge_attr[keep_idx])
            else:
                # No local edges => empty slice
                pass
        if len(local_attr_list) > 0:
            local_attr_all = torch.cat(local_attr_list, dim=0)
            local_attr_proj = self.edge_proj(local_attr_all)
        else:
            local_attr_proj = torch.empty((0, self.hidden_dim), device=device)

        # Virtual edges all share the same learnable embedding
        total_v = int(self._total_virtual_edges or 0)
        v_attr = self.virtual_edge.expand(total_v, -1) if total_v > 0 else torch.empty((0, self.hidden_dim), device=device)

        edge_attr = torch.cat([local_attr_proj, v_attr], dim=0) if (local_attr_proj.numel() + v_attr.numel()) > 0 else None

        # 4) Single batched conv
        out_all = F.relu(self.conv(x_batched, self._batched_edge_index, edge_attr))

        # 5) Gather virtual node embeddings in original order
        supernodes = out_all[self._virtual_indices]
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