import torch.nn as nn
import torch
from torch_geometric.data import Data
import torch_sparse
from typing import cast

class EdgeAggregator(nn.Module):
    """Base class for structural and edge feature aggregation strategies."""
    def forward(self, global_data: Data, m_index: torch.Tensor, m_value: torch.Tensor, num_subgraphs: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

class MeanEdgeAggregator(EdgeAggregator):
    """Calculates subgraph connectivity and mean-pools edge features using sparse math."""
    def forward(self, global_data: Data, m_index: torch.Tensor, m_value: torch.Tensor, num_subgraphs: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_global_nodes = int(global_data.num_nodes) if global_data.num_nodes is not None else 0
        a_index = global_data.edge_index
        if a_index is None:
            return torch.empty((2, 0), dtype=torch.long), None
        
        # Ensure value vectors are on the same device as indices for torch_sparse
        a_value = torch.ones(a_index.size(1), dtype=torch.float, device=a_index.device)
        
        # M_t is the transpose of M (shape: N * S)
        m_t_index = torch.stack([m_index[1], m_index[0]])
        m_t_value = m_value.clone()

        # Multiply M * A_global --> M_A (shape S * N)
        ma_index, ma_value = torch_sparse.spspmm(
            indexA=m_index, valueA=m_value,
            indexB=a_index, valueB=a_value,
            m=num_subgraphs, k=num_global_nodes, n=num_global_nodes
        )
        
        # Multiply M_A * M^T --> A_sub (shape S * S)
        # This gives us the TOPOLOGICAL structural connections (asub_index)
        # and the SUM of how many raw edges overlap (asub_value)
        asub_index, asub_value = torch_sparse.spspmm(
            indexA=ma_index, valueA=ma_value,
            indexB=m_t_index, valueB=m_t_value,
            m=num_subgraphs, k=num_global_nodes, n=num_subgraphs
        )

        # If there are no connections between subgraphs, stop here.
        if asub_index is None or asub_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=a_index.device), None

        # If multi-dimensional edge features are present, handle them
        if global_data.edge_attr is not None and len(global_data.edge_attr.shape) > 1:
            edge_attr = global_data.edge_attr
            num_features = edge_attr.shape[1]
            pooled_features = []
            eps = 1e-8

            # Use torch_sparse to multiply the adjacency index with EACH feature column independently
            for f_idx in range(num_features):
                a_feat_value = edge_attr[:, f_idx]
                
                ma_idx, ma_feat_val = torch_sparse.spspmm(
                    m_index, m_value,
                    a_index, a_feat_value,
                    num_subgraphs, num_global_nodes, num_global_nodes
                )
                
                _, asub_feat_val = torch_sparse.spspmm(
                    ma_idx, ma_feat_val,
                    m_t_index, m_t_value,
                    num_subgraphs, num_global_nodes, num_subgraphs
                )

                # Mean Pool: Sum / Count (plus epsilon safety)
                asub_value_cast = cast(torch.Tensor, asub_value)
                pooled_mean_feat = asub_feat_val / (asub_value_cast + eps)
                pooled_features.append(pooled_mean_feat)
                
            # Stack all columns back into an [E, F] vector and replace the basic scalar tracker
            if len(pooled_features) > 0:
                asub_value = torch.stack(pooled_features, dim=1)
        elif asub_value is not None:
            # Convert to shape [num_edges, 1] for consistency
            asub_value = asub_value.unsqueeze(1)  

        # shapes [2, num_edges], [num_edges, 1] or [num_edges, F]
        return asub_index, asub_value 


class StatsEdgeAggregator(EdgeAggregator):
    """Simple non-attention edge summarizer: concatenate mean, max, std per super-edge."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def _edge_features(self, global_data: Data, edge_index: torch.Tensor) -> torch.Tensor:
        num_edges = edge_index.size(1)
        edge_attr = global_data.edge_attr
        device = edge_index.device

        if edge_attr is None:
            return torch.ones((num_edges, 1), dtype=torch.float, device=device)
        if edge_attr.dim() == 1:
            return edge_attr.unsqueeze(1)
        return edge_attr

    def forward(self, global_data: Data, m_index: torch.Tensor, m_value: torch.Tensor, num_subgraphs: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        del m_value, num_subgraphs
        a_index = global_data.edge_index
        if a_index is None or a_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long), None

        device = a_index.device
        num_global_nodes = int(global_data.num_nodes) if global_data.num_nodes is not None else 0
        edge_feat = self._edge_features(global_data, a_index)

        # Build node -> subgraph membership list from sparse M[s, n] = 1
        node_to_subgraphs: list[list[int]] = [[] for _ in range(num_global_nodes)]
        sub_ids = m_index[0].tolist()
        node_ids = m_index[1].tolist()
        for s_id, n_id in zip(sub_ids, node_ids):
            node_to_subgraphs[n_id].append(s_id)

        stats: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        src_nodes = a_index[0].tolist()
        dst_nodes = a_index[1].tolist()

        for e_idx, (u, v) in enumerate(zip(src_nodes, dst_nodes)):
            su_list = node_to_subgraphs[u]
            sv_list = node_to_subgraphs[v]
            if len(su_list) == 0 or len(sv_list) == 0:
                continue

            x = edge_feat[e_idx]
            for su in su_list:
                for sv in sv_list:
                    key = (su, sv)
                    if key not in stats:
                        zeros = torch.zeros_like(x)
                        stats[key] = {
                            "sum_x": zeros.clone(),
                            "sum_x2": zeros.clone(),
                            "max_x": torch.full_like(x, float("-inf")),
                            "count": torch.zeros((), device=device),
                        }

                    entry = stats[key]
                    entry["sum_x"] = entry["sum_x"] + x
                    entry["sum_x2"] = entry["sum_x2"] + (x * x)
                    entry["max_x"] = torch.maximum(entry["max_x"], x)
                    entry["count"] = entry["count"] + 1.0

        if len(stats) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device), None

        pairs = sorted(stats.keys())
        s_edge_index = torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()

        out_features = []
        for key in pairs:
            entry = stats[key]
            mean = entry["sum_x"] / (entry["count"] + self.eps)
            second = entry["sum_x2"] / (entry["count"] + self.eps)
            var = torch.clamp(second - mean * mean, min=0.0)
            std = torch.sqrt(var + self.eps)
            max_feat = entry["max_x"]
            out_features.append(torch.cat([mean, max_feat, std], dim=0))

        s_edge_attr = torch.stack(out_features, dim=0)
        return s_edge_index, s_edge_attr
    

class AttentionStatsEdgeAggregator(EdgeAggregator):
    """
    Stronger edge summarizer:
    1) Learns edge importance weights (attention) from endpoint node features + edge features.
    2) Aggregates edge features per (subgraph_i, subgraph_j) with weighted mean, max, and std.
    3) Concatenates [mean, max, std, structural_count] as the final super-edge feature.
    """
    def __init__(self, hidden_dim: int = 64, eps: float = 1e-8, temperature: float = 2.0, clamp_max: float = 20.0):
        super().__init__()
        self.eps = eps
        self.temperature = temperature
        self.clamp_max = clamp_max
        self.attn_mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _edge_features(self, global_data: Data, edge_index: torch.Tensor) -> torch.Tensor:
        """Build raw edge features [E, F]. If no edge_attr exists, fallback to ones."""
        num_edges = edge_index.size(1)
        device = edge_index.device
        edge_attr = global_data.edge_attr

        if edge_attr is None:
            return torch.ones((num_edges, 1), dtype=torch.float, device=device)

        if edge_attr.dim() == 1:
            return edge_attr.unsqueeze(1)

        return edge_attr

    def _edge_attention(self, global_data: Data, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute scalar edge importance in [0,1] for each global edge."""
        src, dst = edge_index[0], edge_index[1]
        parts: list[torch.Tensor] = []

        if global_data.x is not None:
            parts.append(global_data.x[src])
            parts.append(global_data.x[dst])

        edge_attr = global_data.edge_attr
        if edge_attr is not None:
            parts.append(edge_attr.unsqueeze(1) if edge_attr.dim() == 1 else edge_attr)

        if len(parts) == 0:
            return torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)

        attn_in = torch.cat(parts, dim=1)
        logits = self.attn_mlp(attn_in).squeeze(1)
        return torch.sigmoid(logits)

    def forward(self, global_data: Data, m_index: torch.Tensor, m_value: torch.Tensor, num_subgraphs: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Vectorized implementation using sparse-sparse matrix multiplications.
        Computes per super-edge statistics with attention weights w:
          - mean: sum(w*x) / sum(w)
          - std: sqrt(max(sum(w*x^2)/sum(w) - mean^2, 0) + eps)
          - smooth-max: 1/t * log(sum(exp(t*x))) (log-sum-exp approx.)
          - count: structural edge count via M * A * M^T
        Returns concatenation [mean, smooth_max, std, count].
        """
        a_index = global_data.edge_index
        if a_index is None or a_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long), None

        device = a_index.device
        num_global_nodes = int(global_data.num_nodes) if global_data.num_nodes is not None else 0
        edge_feat = self._edge_features(global_data, a_index)  # [E, F]
        edge_feat = torch.nan_to_num(edge_feat, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attn = self._edge_attention(global_data, a_index)  # [E]
        edge_attn = torch.clamp(torch.nan_to_num(edge_attn, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        # Build M^T index
        m_t_index = torch.stack([m_index[1], m_index[0]])
        m_t_value = m_value.clone()

        # 1) Structural counts per super-edge (S x S)
        a_ones = torch.ones(a_index.size(1), dtype=torch.float, device=device)
        ma_idx, ma_val = torch_sparse.spspmm(
            m_index, m_value,
            a_index, a_ones,
            num_subgraphs, num_global_nodes, num_global_nodes
        )
        as_idx, as_count = torch_sparse.spspmm(
            ma_idx, ma_val,
            m_t_index, m_t_value,
            num_subgraphs, num_global_nodes, num_subgraphs
        )

        if as_idx is None or as_idx.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device), None

        # 2) Sum of attention weights per super-edge: sum_w
        ma_idx_w, ma_val_w = torch_sparse.spspmm(
            m_index, m_value,
            a_index, edge_attn,
            num_subgraphs, num_global_nodes, num_global_nodes
        )
        _, sum_w = torch_sparse.spspmm(
            ma_idx_w, ma_val_w,
            m_t_index, m_t_value,
            num_subgraphs, num_global_nodes, num_subgraphs
        )
        sum_w = cast(torch.Tensor, sum_w)

        # Handle feature dimensionality
        if edge_feat.dim() == 1:
            edge_feat = edge_feat.unsqueeze(1)

        num_features = edge_feat.size(1)
        eps = self.eps
        t = float(self.temperature)  # temperature for smooth max

        means = []
        stds = []
        smooth_maxes = []

        for f in range(num_features):
            x_f = edge_feat[:, f]

            # sum(w * x)
            ma_idx_wx, ma_val_wx = torch_sparse.spspmm(
                m_index, m_value,
                a_index, edge_attn * x_f,
                num_subgraphs, num_global_nodes, num_global_nodes
            )
            _, sum_wx = torch_sparse.spspmm(
                ma_idx_wx, ma_val_wx,
                m_t_index, m_t_value,
                num_subgraphs, num_global_nodes, num_subgraphs
            )
            sum_wx = cast(torch.Tensor, sum_wx)

            # sum(w * x^2)
            ma_idx_wx2, ma_val_wx2 = torch_sparse.spspmm(
                m_index, m_value,
                a_index, edge_attn * (x_f * x_f),
                num_subgraphs, num_global_nodes, num_global_nodes
            )
            _, sum_wx2 = torch_sparse.spspmm(
                ma_idx_wx2, ma_val_wx2,
                m_t_index, m_t_value,
                num_subgraphs, num_global_nodes, num_subgraphs
            )
            sum_wx2 = cast(torch.Tensor, sum_wx2)

            # Smooth max via stabilized log-sum-exp: log(sum(exp(t*x)))
            tx = torch.clamp(t * x_f, min=-self.clamp_max, max=self.clamp_max)
            exp_tx = torch.exp(tx)
            ma_idx_exptx, ma_val_exptx = torch_sparse.spspmm(
                m_index, m_value,
                a_index, exp_tx,
                num_subgraphs, num_global_nodes, num_global_nodes
            )
            _, sum_exptx = torch_sparse.spspmm(
                ma_idx_exptx, ma_val_exptx,
                m_t_index, m_t_value,
                num_subgraphs, num_global_nodes, num_subgraphs
            )
            sum_exptx = cast(torch.Tensor, sum_exptx)

            mean_f = sum_wx / (sum_w + eps)
            second_moment = sum_wx2 / (sum_w + eps)
            var_f = torch.clamp(second_moment - mean_f * mean_f, min=0.0)
            std_f = torch.sqrt(var_f + eps)
            max_f = (1.0 / t) * torch.log(torch.clamp(sum_exptx, min=eps))

            means.append(mean_f)
            stds.append(std_f)
            smooth_maxes.append(max_f)

        # Stack features into [E_s, F]
        mean_mat = torch.stack(means, dim=1)
        max_mat = torch.stack(smooth_maxes, dim=1)
        std_mat = torch.stack(stds, dim=1)
        as_count = cast(torch.Tensor, as_count)
        count_feat = as_count.unsqueeze(1)

        # Final numeric sanitation
        mean_mat = torch.nan_to_num(mean_mat, nan=0.0, posinf=0.0, neginf=0.0)
        max_mat = torch.nan_to_num(max_mat, nan=0.0, posinf=0.0, neginf=0.0)
        std_mat = torch.nan_to_num(std_mat, nan=0.0, posinf=0.0, neginf=0.0)
        count_feat = torch.nan_to_num(count_feat, nan=0.0, posinf=0.0, neginf=0.0)

        s_edge_attr = torch.cat([mean_mat, max_mat, std_mat, count_feat], dim=1)
        return as_idx, s_edge_attr
