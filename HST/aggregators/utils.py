import torch.nn as nn
import torch


def reconstruct_local_edges(mapping: dict, sub_edge_index: torch.Tensor | None, sub_edge_attr: torch.Tensor | None, 
                            hidden_dim: int, device: torch.device, edge_proj: nn.Module):
    """Helper function to map global edges to local subgraph IDs and project their features."""
    src_list, dst_list, e_attr_list = [], [], []
    
    if sub_edge_index is not None and sub_edge_index.size(1) > 0:
        for i in range(sub_edge_index.size(1)):
            u = sub_edge_index[0, i].item()
            v = sub_edge_index[1, i].item()
            
            if u in mapping and v in mapping:
                src_list.append(mapping[u])
                dst_list.append(mapping[v])
                
                if sub_edge_attr is not None:
                    # Pull feature physically aligned with this specific edge
                    e_attr_list.append(edge_proj(sub_edge_attr[i]))
                else:
                    e_attr_list.append(torch.zeros(hidden_dim, device=device))
                    
    return src_list, dst_list, e_attr_list