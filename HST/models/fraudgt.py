import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from typing import Optional
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class FraudGTConv(MessagePassing):
    """
    A custom PyG MessagePassing layer that mimics the FraudGT SparseNodeTransformer block 
    (including Edge Bias and Edge Gate). Provides all structural FraudGT capabilities 
    without the overhead of the GraphGym global configuration dictionaries.
    """
    def __init__(self, hidden_dim: int, heads: int=4, **kwargs):
        super().__init__(node_dim=0, **kwargs)
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.d_k = hidden_dim // heads
        
        # MHA projections for node queries, keys, and values
        self.q_lin = nn.Linear(hidden_dim, hidden_dim)
        self.k_lin = nn.Linear(hidden_dim, hidden_dim)
        self.v_lin = nn.Linear(hidden_dim, hidden_dim)
        
        # FraudGT Edge enhancements
        # Input dimensions are different due to using mean max std edge features 
        # instead of raw edge attributes
        self.e_lin = nn.LazyLinear(hidden_dim) # (Yellow Box) Edge attention bias
        self.g_lin = nn.LazyLinear(hidden_dim) # (Blue Box) Edge gate
        
        # Optional PEARL fusion: x_fused = x + proj(pearl_encodings)
        self.pearl_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        pearl_encodings: Optional[torch.Tensor] = None,
    ):
        if pearl_encodings is not None:
            x = x + self.pearl_proj(pearl_encodings)

        # Project inputs and split into H heads
        q = self.q_lin(x).view(-1, self.heads, self.d_k)
        k = self.k_lin(x).view(-1, self.heads, self.d_k)
        v = self.v_lin(x).view(-1, self.heads, self.d_k)
        
        # Project edge attributes if they exist
        e_feat = self.e_lin(edge_attr).view(-1, self.heads, self.d_k) if edge_attr is not None else None
        e_gate = self.g_lin(edge_attr).view(-1, self.heads, self.d_k) if edge_attr is not None else None

        # Start PyG's blazing fast optimized message-passing scatter engine
        out = self.propagate(edge_index, q=q, k=k, v=v, e_feat=e_feat, e_gate=e_gate)
        
        # Re-concatenate the heads
        return out.view(-1, self.hidden_dim) 
        
    def message(self, q_i: torch.Tensor, k_j: torch.Tensor, v_j: torch.Tensor, 
                e_feat: Optional[torch.Tensor], e_gate: Optional[torch.Tensor], 
                index: torch.Tensor, ptr: Optional[torch.Tensor], size_i: Optional[int]):
        
        # 1. Graph Attention with Edge Bias (FraudGT Yellow Box)
        # Element-wise product of Query and Key
        edge_scores = q_i * k_j
        if e_feat is not None:
            # FraudGT Eq: directly add edge representation to structural attention
            edge_scores = edge_scores + e_feat  
            
        # Sum across latent feature dimension and softly scale by sqrt(d)
        alpha = edge_scores.sum(dim=-1) / math.sqrt(self.d_k)  # Shape: [TotalEdges, heads]
        
        # Determine neighborhood attention weights
        alpha = softmax(alpha, index, ptr, size_i)
        
        # 2. Edge-based Message Passing Gate (FraudGT Blue Box)
        if e_gate is not None:
            # FraudGT Eq: aggressively throttle the Value vector of noisy/fake edges
            v_j = v_j * torch.sigmoid(e_gate)
            
        # Final Attention message weighted sum
        return v_j * alpha.unsqueeze(-1)