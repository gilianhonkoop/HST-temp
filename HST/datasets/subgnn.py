import os
import torch
import networkx as nx
from torch_geometric.data import Data

from typing import List, Literal, Tuple, Union, Any

from .base import BaseSubgraphDataset, Embeddingtype
from .utils import read_subgraphs, get_data_list_from_subgraphs

class SubGNNDataset(BaseSubgraphDataset):
    def __init__(self, root: str, name: str, embedding_type: Embeddingtype, verbose: bool = False):
        super().__init__(root, name, verbose)
        self.embedding_type = embedding_type
        valid_datasets = {"hpo_metab", "hpo_neuro", "ppi_bp", "em_user"}
        if name not in valid_datasets:
            raise ValueError(f"Dataset must be one of {valid_datasets}")

    def load(self) -> Tuple[Data, List[Data]]:
        # Global graph
        edge_list_path = os.path.join(self.base_path, "edge_list.txt")
        if self.verbose: print(f"Loading global graph from {edge_list_path}...")
        global_nxg = nx.read_edgelist(edge_list_path, nodetype=int)
        
        # Convert properly (assumes nodes 0..N-1 are contiguous)
        # Get max node id to ensure tensor sizing is correct
        max_node = max(global_nxg.nodes())
        # NOTE : all edges should be undirected, but are dircted in edge_list.txt (for non elliptic datasets), 
        # so we convert to undirected and back to directed to ensure both directions are present
        edge_index = torch.tensor(list(global_nxg.to_directed().edges)).t().contiguous()
        global_data = Data(edge_index=edge_index, num_nodes=max_node + 1)
        
        # Embeddings
        emb_path = os.path.join(self.base_path, f"{self.embedding_type}_embeddings.pth")
        if os.path.exists(emb_path):
            global_data.x = torch.load(emb_path, map_location="cpu", weights_only=True)
            if self.verbose: print(f"Loaded {self.embedding_type} embeddings. Shape: {global_data.x.shape}")
        else:
            if self.verbose: print(f"Warning: Embeddings not found at {emb_path}")
            global_data.x = None

        # Subgraphs
        subgraph_path = os.path.join(self.base_path, "subgraphs.pth")
        if self.verbose: print(f"Loading subgraphs from {subgraph_path}...")
        
        # NOTE : nothing is actually done with multilabels yet
        all_nodes, all_ys, all_splits, is_multi = read_subgraphs(subgraph_path)
        
        if self.verbose: print("Building PyG Data objects for subgraphs...")
        all_data = get_data_list_from_subgraphs(global_data, all_nodes, all_ys, sub_splits=all_splits)
        
        # Cumulative Edge Splitting (Inductive mask simulation based on node sets)
        train_n, val_n = set(), set()
        for nodes, split in zip(all_nodes, all_splits):
            if split == 'train': train_n.update(nodes)
            elif split == 'val': val_n.update(nodes)
            
        num_nodes = global_data.num_nodes if global_data.num_nodes is not None else 0
        is_train_node = torch.zeros(num_nodes, dtype=torch.bool)
        if len(train_n) > 0:
            is_train_node[torch.tensor(list(train_n), dtype=torch.long)] = True
            
        is_val_node = torch.zeros(num_nodes, dtype=torch.bool)
        if len(train_n | val_n) > 0:
            is_val_node[torch.tensor(list(train_n | val_n), dtype=torch.long)] = True

        if getattr(global_data, 'edge_index', None) is not None:
            ei = global_data.edge_index
            if ei is not None:
                src, dst = ei[0], ei[1]
                global_data.edge_train_mask = is_train_node[src] & is_train_node[dst]
                global_data.edge_val_mask = is_val_node[src] & is_val_node[dst]
                global_data.edge_test_mask = torch.ones(ei.size(1), dtype=torch.bool)

        if self.verbose: print(f"Done. Total Subgraphs: {len(all_data)}")
        return global_data, all_data

    # Uses predefined splits from the dataset (if available) to split into train/val/test
    def split(self, subgraphs: List[Data]) -> Tuple[List[Data], List[Data], List[Data]]:
        if len(subgraphs) == 0:
            return [], [], []
            
        train_data = [d for d in subgraphs if getattr(d, 'preset_split', '') == 'train']
        val_data = [d for d in subgraphs if getattr(d, 'preset_split', '') == 'val']
        test_data = [d for d in subgraphs if getattr(d, 'preset_split', '') == 'test']
        
        return train_data, val_data, test_data
