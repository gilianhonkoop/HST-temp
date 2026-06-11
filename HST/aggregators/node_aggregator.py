import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import softmax
from models import FraudGTConv

class NodeAggregator(nn.Module):
    """Base class for node aggregation strategies."""
    def forward(self, global_data: Data, subgraph_data_list: list[Data]) -> torch.Tensor:
        raise NotImplementedError

class MeanNodeAggregator(NodeAggregator):
    """Standard mean pooling for subgraph node features, followed by a projection to hidden_dim."""
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.LazyLinear(hidden_dim)

    def forward(self, global_data: Data, subgraph_data_list: list[Data]) -> torch.Tensor:
        global_x = global_data.x
        num_subgraphs = len(subgraph_data_list)
        device = global_x.device if global_x is not None else torch.device('cpu')
        
        if global_x is None:
            # Fallback if no embeddings are available
            return torch.ones((num_subgraphs, self.hidden_dim), dtype=torch.float, device=device)
            
        sub_x_list = []
        for sub_data in subgraph_data_list:
            n_ids = sub_data.x_idx.flatten()
            sub_feats = global_x[n_ids].mean(dim=0)
            sub_x_list.append(sub_feats)
        
        # Stack to [S, F_in] then project to [S, hidden_dim]
        sub_x = torch.stack(sub_x_list)
        return self.proj(sub_x)


class MaxNodeAggregator(NodeAggregator):
    """Standard max pooling for subgraph node features, followed by a projection to hidden_dim."""
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.LazyLinear(hidden_dim)

    def forward(self, global_data: Data, subgraph_data_list: list[Data]) -> torch.Tensor:
        global_x = global_data.x
        num_subgraphs = len(subgraph_data_list)
        device = global_x.device if global_x is not None else torch.device('cpu')
        
        if global_x is None:
            # Fallback if no embeddings are available
            return torch.ones((num_subgraphs, self.hidden_dim), dtype=torch.float, device=device)
            
        sub_x_list = []
        for sub_data in subgraph_data_list:
            n_ids = sub_data.x_idx.flatten()
            # torch.max returns a (values, indices) namedtuple; select only values
            # Handle potential empty subgraphs by falling back to zeros
            if n_ids.numel() == 0:
                sub_feats = torch.zeros(global_x.size(-1), device=global_x.device, dtype=global_x.dtype)
            else:
                sub_feats = global_x[n_ids].max(dim=0).values
            sub_x_list.append(sub_feats)
        
        # Stack to [S, F_in] then project to [S, hidden_dim]
        sub_x = torch.stack(sub_x_list)
        return self.proj(sub_x)


class StatsNodeAggregator(NodeAggregator):
    """Simple non-attention node summarizer: concatenate mean, max, std per subgraph."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, global_data: Data, subgraph_data_list: list[Data]) -> torch.Tensor:
        global_x = global_data.x
        num_subgraphs = len(subgraph_data_list)
        device = global_x.device if global_x is not None else torch.device('cpu')

        if global_x is None:
            return torch.ones((num_subgraphs, 3), dtype=torch.float, device=device)

        if global_x.dim() == 1:
            global_x = global_x.unsqueeze(1)

        sub_x_list = []
        feature_dim = global_x.size(-1)
        empty_stats = torch.zeros(feature_dim * 3, dtype=global_x.dtype, device=device)

        for sub_data in subgraph_data_list:
            n_ids = sub_data.x_idx.flatten().to(device)
            if n_ids.numel() == 0:
                sub_x_list.append(empty_stats.clone())
                continue

            sub_x = global_x[n_ids]
            mean = sub_x.mean(dim=0)
            max_feat = sub_x.max(dim=0).values
            std = torch.sqrt(torch.clamp(sub_x.var(dim=0, unbiased=False), min=0.0) + self.eps)
            sub_x_list.append(torch.cat([mean, max_feat, std], dim=0))

        return torch.stack(sub_x_list)


class AttentionNodeAggregator(NodeAggregator):
    """
    Attention-based pooling inspired by FraudGT.
    It calculates an attention score for each node within a subgraph by looking 
    at both its node features AND the sum of features from its incident edges.
    """
    def __init__(self, hidden_dim: int = 128, **kwargs):
        super().__init__()
        # We use Lazy modules so we don't need to hardcode input dimensions from your datasets
        self.node_proj = nn.LazyLinear(hidden_dim)
        
        # We only project edge features if the dataset provides them
        self.edge_proj = nn.LazyLinear(hidden_dim) 
        
        # The attention mechanism
        self.attn_net = nn.Sequential(
            nn.LazyLinear(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Edge-based Message Passing Gate (FraudGT #2)
        # We learn a gating mechanism directly from edge geometries to modulate 
        # how much an edge influences its neighboring nodes
        self.edge_gate_net = nn.Sequential(
            nn.LazyLinear(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim) # Outputs a gating vector per edge
        )

    def forward(self, global_data: Data, subgraph_data_list: list[Data]) -> torch.Tensor:
        global_x = global_data.x

        num_subgraphs = len(subgraph_data_list)
        device = global_data.edge_index.device if global_data.edge_index is not None else torch.device('cpu')

        if global_x is None:
            return torch.ones((num_subgraphs, 1), dtype=torch.float, device=device)

        supernodes = []
        
        for sub_data in subgraph_data_list:
            n_ids = sub_data.x_idx.flatten()  # Global IDs of nodes in this subgraph
            sub_x = global_x[n_ids]           # [N, node_feature_dim]
            
            # Project nodes to uniform hidden dimension
            h_nodes = self.node_proj(sub_x)   # [N, hidden_dim]
            
            # Incorporate edge features (similar to FraudGT's edge attention bias)
            sub_edge_attr = getattr(sub_data, 'edge_attr', None)
            if sub_edge_attr is not None and sub_data.edge_index is not None and sub_data.edge_index.size(1) > 0:
                h_edges_for_nodes = torch.zeros_like(h_nodes)
                
                # Sift through subgraph edges (which retain global IDs currently from dataset.py)
                for i in range(sub_data.edge_index.size(1)):
                    u = sub_data.edge_index[0, i].item()
                    v = sub_data.edge_index[1, i].item()

                    e_feat = sub_edge_attr[i]
                    e_proj = self.edge_proj(e_feat)
                    
                    # Generate Edge Gate (FraudGT step #2)
                    edge_gate = F.sigmoid(self.edge_gate_net(e_feat))
                    
                    # Modulate the projected edge features with the gate
                    # This turns off edges that the network thinks are irrelevant/noise
                    e_proj_gated = e_proj * edge_gate
                    
                    # Add gated edge bias to the local nodes receiving the edge
                    # (Finding the local index of the global node ID)
                    u_local = (n_ids == u).nonzero(as_tuple=True)[0]
                    v_local = (n_ids == v).nonzero(as_tuple=True)[0]
                    
                    if len(u_local) > 0:
                        h_edges_for_nodes[u_local[0]] += e_proj_gated
                    if len(v_local) > 0:
                        h_edges_for_nodes[v_local[0]] += e_proj_gated
                
                # Combine node features and their incident edge features softly
                context = h_nodes + h_edges_for_nodes
            else:
                context = h_nodes
                
            # Compute attention score for each node: [N, 1]
            attn_scores = self.attn_net(context)
            # Softmax to get a valid probability distribution over the bounding subgraph
            alpha = F.softmax(attn_scores, dim=0) 
            
            # Weighted sum for the supernode
            subgraph_representation = (h_nodes * alpha).sum(dim=0)
            supernodes.append(subgraph_representation)

        return torch.stack(supernodes)
