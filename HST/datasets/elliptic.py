import os
import torch
import networkx as nx
import pandas as pd
import numpy as np
from torch_geometric.data import Data

from typing import List, Literal, Tuple, Union, Any

from .base import BaseSubgraphDataset
from .utils import read_subgraphs, get_data_list_from_subgraphs

class EllipticDataset(BaseSubgraphDataset):
    def load(self):
        if self.verbose: print(f"Loading Elliptic dataset from {self.base_path}...")
        
        # Node features
        nodes_csv = os.path.join(self.base_path, "background_nodes.csv")
        bg_nodes_df = pd.read_csv(nodes_csv)
        # IMPORTANT: keep IDs as strings (background files may contain mixed IDs like "U_123")
        bg_nodes_df['clId'] = bg_nodes_df['clId'].astype(str)
        node_mapping = {clId: idx for idx, clId in enumerate(bg_nodes_df['clId'])}
        
        feat_cols = [c for c in bg_nodes_df.columns if c.startswith('feat')]
        if len(feat_cols) > 0:
            global_x_tensor = torch.tensor(bg_nodes_df[feat_cols].values, dtype=torch.float)
        else:
            raise ValueError("No node features found in background_nodes.csv.")
            global_x_tensor = torch.ones((len(node_mapping), 1), dtype=torch.float)
            
        if self.verbose: print(f"Loaded {len(node_mapping)} background nodes with {len(feat_cols)} features.")
            
        # Edge features
        edges_csv = os.path.join(self.base_path, "background_edges.csv")
        bg_edges_df = pd.read_csv(edges_csv)
        bg_edges_df['clId1'] = bg_edges_df['clId1'].astype(str)
        bg_edges_df['clId2'] = bg_edges_df['clId2'].astype(str)
        valid_edges = bg_edges_df[bg_edges_df['clId1'].isin(node_mapping) & bg_edges_df['clId2'].isin(node_mapping)]
        
        src = valid_edges['clId1'].map(node_mapping).to_numpy()
        dst = valid_edges['clId2'].map(node_mapping).to_numpy()
        global_edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
        
        edge_feat_cols = [c for c in bg_edges_df.columns if c.startswith('feat')]
        if len(edge_feat_cols) > 0:
            global_edge_attr = torch.tensor(valid_edges[edge_feat_cols].values, dtype=torch.float)
        else:
            raise ValueError("No edge features found in background_edges.csv.")
            
        if self.verbose: print(f"Loaded {global_edge_index.size(1)} background edges with {len(edge_feat_cols)} features.")

        global_data = Data(x=global_x_tensor, edge_index=global_edge_index, edge_attr=global_edge_attr, num_nodes=len(node_mapping))
        
        # Connected Components (subgraphs & labels)
        cc_csv = os.path.join(self.base_path, "connected_components.csv")
        if os.path.exists(cc_csv):
            cc_df = pd.read_csv(cc_csv)
            cc_df['ccId'] = cc_df['ccId'].astype(str)
            unique_labels = sorted(cc_df['ccLabel'].unique())
            label_mapping = {lbl: i for i, lbl in enumerate(unique_labels)}
            if self.verbose: print(f"Discovered labels: {label_mapping}")
            cc_dict = {row['ccId']: label_mapping[row['ccLabel']] for _, row in cc_df.iterrows()}
        else:
            raise ValueError(f"Connected components file not found at {cc_csv}")    

        # Nodes
        sub_csv = os.path.join(self.base_path, "nodes.csv")
        sub_df = pd.read_csv(sub_csv)
        sub_df['clId'] = sub_df['clId'].astype(str)
        sub_df['ccId'] = sub_df['ccId'].astype(str)
        
        sub_nodes_list = []
        sub_ys_list = []
        
        for ccId, group in sub_df.groupby('ccId'):
            mapped_nodes = [node_mapping[c] for c in group['clId'] if c in node_mapping]
            if len(mapped_nodes) > 0:
                sub_nodes_list.append(mapped_nodes)
                # Assign mapped label if present, else default to 0
                lbl = cc_dict.get(ccId, 0)
                sub_ys_list.append(torch.tensor(lbl, dtype=torch.long))

        if len(sub_nodes_list) == 0:
            raise ValueError(
                "No labeled subgraphs could be mapped to background node IDs. "
                "Check ID type consistency between background_nodes.csv and nodes.csv."
            )

        if self.verbose: print("Building PyG Data objects for subgraphs...")
        all_data = get_data_list_from_subgraphs(global_data, sub_nodes_list, sub_ys_list)
        
        if self.verbose: print(f"Done. Total Subgraphs: {len(all_data)}")
        return global_data, all_data

    # Does a random split
    # TODO : could also implement a time-based split based on the original elliptic dataset's timestamps if
    def split(self, subgraphs: List[Data], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Data], List[Data], List[Data]]:
        if len(subgraphs) == 0:
            return [], [], []
            
        import random
        random.seed(seed)
        shuffled_data = list(subgraphs)
        random.shuffle(shuffled_data)
        
        n = len(shuffled_data)
        t_end = int(train_ratio * n)
        v_end = t_end + int(val_ratio * n)
        
        train_data = shuffled_data[:t_end]
        val_data = shuffled_data[t_end:v_end]
        test_data = shuffled_data[v_end:]
        
        return train_data, val_data, test_data