import os
import networkx as nx
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Literal, Tuple, Union, Any

DatasetType = Literal["elliptic", "subgnn", "ibm_aml"]
Embeddingtype = Literal["glass", "gin", "graphsaint_gcn"]

def read_subgraphs(subgraph_path: str) -> Tuple[List[List[int]], torch.Tensor, List[str], bool]:
    label_idx = 0
    labels = {}
    
    all_nodes = []
    all_ys = []
    all_splits = []
    
    multilabel = False
    
    with open(subgraph_path) as fin:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            
            nodes = [int(n) for n in parts[0].split("-") if n != ""]
            if len(nodes) == 0: continue
            
            label_cell = parts[1]
            l = label_cell.split("-")
            if len(l) > 1:
                multilabel = True
                
            cur_labels = []
            for lab in l:
                if lab not in labels.keys():
                    labels[lab] = label_idx
                    label_idx += 1
                cur_labels.append(labels[lab])
                
            split = parts[2].strip()
            
            all_nodes.append(nodes)
            all_ys.append(cur_labels)
            all_splits.append(split)

    if multilabel:
        mlb = MultiLabelBinarizer()
        mlb.fit(all_ys)
        all_ys = torch.Tensor(mlb.transform(all_ys))
    else:
        all_ys = torch.tensor([y[0] for y in all_ys], dtype=torch.long)
        
    return all_nodes, all_ys, all_splits, multilabel

def get_data_list_from_subgraphs(global_data: Data, sub_nodes: List[List[int]], sub_ys: Any, sub_splits: Union[List[str], None] = None, return_subgraph_edges: bool = True) -> List[Data]:
    data_list = []
    global_edge_index = global_data.edge_index if global_data.edge_index is not None else torch.empty((2, 0), dtype=torch.long)
    global_x = global_data.x
    
    for i, (x_index, y) in enumerate(zip(sub_nodes, sub_ys)):
        x_index_tensor = torch.tensor(x_index, dtype=torch.long).view(-1, 1)
        if len(y.size()) == 0:  # single-label
            y = y.unsqueeze(0)
        else:  # multi-label
            y = y.view(1, -1)
            
        if return_subgraph_edges:
            edge_index, _ = subgraph(x_index_tensor.flatten(), global_edge_index, relabel_nodes=False)
            # NOTE : old pyg version for directed / undirected edges
            # if edge_index.size(1) >= 2:
            #     edge_index = to_directed(edge_index)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # Get Node Embeddings for nodes inside this subgraph
        sub_x = None
        if global_x is not None:
            sub_x = global_x[x_index_tensor.flatten()]
            
        data = Data(x=sub_x, x_idx=x_index_tensor, edge_index=edge_index, y=y)
        if sub_splits is not None:
            data.preset_split = sub_splits[i]
        data_list.append(data)
    return data_list


class BaseSubgraphDataset:
    def __init__(self, root: str, name: str, verbose: bool = False):
        self.root = root
        self.name = name
        self.verbose = verbose
        self.base_path = os.path.join(root, name)

    def load(self) -> Tuple[Data, List[Data]]:
        raise NotImplementedError

    def split(self, subgraphs: List[Data]) -> Tuple[List[Data], List[Data], List[Data]]:
        raise NotImplementedError

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
    # TODO : could also implement a time-based split based on the original elliptic dataset's timestamps if needed
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

class AMLDataset(BaseSubgraphDataset):
    def load(self) -> Tuple[Data, List[Data]]:
        if self.verbose: print(f"Loading IBM AML dataset from {self.base_path}...")

        # Node features
        nodes_csv = os.path.join(self.base_path, "background_nodes.csv")
        bg_nodes_df = pd.read_csv(nodes_csv)
        bg_nodes_df['clId'] = bg_nodes_df['clId'].astype(str)
        node_mapping = {clId: idx for idx, clId in enumerate(bg_nodes_df['clId'])}

        feat_cols = [c for c in bg_nodes_df.columns if c.startswith('feat')]
        if len(feat_cols) > 0:
            global_x_tensor = torch.tensor(bg_nodes_df[feat_cols].values, dtype=torch.float)
        else:
            raise ValueError("No node features found in background_nodes.csv.")

        if self.verbose: print(f"Loaded {len(node_mapping)} AML nodes with {len(feat_cols)} features.")

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

        if self.verbose: print(f"Loaded {global_edge_index.size(1)} AML edges with {len(edge_feat_cols)} features.")

        global_data = Data(x=global_x_tensor, edge_index=global_edge_index, edge_attr=global_edge_attr, num_nodes=len(node_mapping))

        # Connected components + labels (+ optional temporal metadata)
        cc_csv = os.path.join(self.base_path, "connected_components.csv")
        if os.path.exists(cc_csv):
            cc_df = pd.read_csv(cc_csv)
            cc_df['ccId'] = cc_df['ccId'].astype(str)

            unique_labels = sorted(cc_df['ccLabel'].astype(str).unique())
            label_mapping = {lbl: i for i, lbl in enumerate(unique_labels)}
            if self.verbose: print(f"Discovered AML labels: {label_mapping}")
            cc_label_dict = {row['ccId']: label_mapping[str(row['ccLabel'])] for _, row in cc_df.iterrows()}

            cc_time_dict = {}
            if 'ccStartTime' in cc_df.columns:
                cc_df['ccStartTime'] = pd.to_datetime(cc_df['ccStartTime'], errors='coerce')
                cc_time_dict = {
                    row['ccId']: row['ccStartTime']
                    for _, row in cc_df.iterrows()
                    if pd.notna(row['ccStartTime'])
                }
        else:
            raise ValueError(f"Connected components file not found at {cc_csv}")

        # Nodes per component
        sub_csv = os.path.join(self.base_path, "nodes.csv")
        sub_df = pd.read_csv(sub_csv)
        sub_df['clId'] = sub_df['clId'].astype(str)
        sub_df['ccId'] = sub_df['ccId'].astype(str)

        sub_nodes_list = []
        sub_ys_list = []
        sub_times_list = []

        for ccId, group in sub_df.groupby('ccId'):
            mapped_nodes = [node_mapping[c] for c in group['clId'] if c in node_mapping]
            if len(mapped_nodes) > 0:
                sub_nodes_list.append(mapped_nodes)
                lbl = cc_label_dict.get(ccId, 0)
                sub_ys_list.append(torch.tensor(lbl, dtype=torch.long))
                sub_times_list.append(cc_time_dict.get(ccId, pd.NaT))

        if len(sub_nodes_list) == 0:
            raise ValueError(
                "No AML subgraphs could be mapped to background node IDs. "
                "Check ID type consistency between background_nodes.csv and nodes.csv."
            )

        if self.verbose: print("Building PyG Data objects for AML subgraphs...")
        all_data = get_data_list_from_subgraphs(global_data, sub_nodes_list, sub_ys_list)
        for d, t in zip(all_data, sub_times_list):
            if pd.notna(t):
                d.preset_time = pd.Timestamp(t)

        if self.verbose: print(f"Done. Total AML Subgraphs: {len(all_data)}")
        return global_data, all_data

    # Temporal split by component start time (earliest -> latest)
    def split(self, subgraphs: List[Data], train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[List[Data], List[Data], List[Data]]:
        if len(subgraphs) == 0:
            return [], [], []

        if not all(hasattr(d, 'preset_time') for d in subgraphs):
            raise ValueError(
                "Temporal split requested, but some subgraphs are missing `ccStartTime` metadata. "
                "Rebuild AML dataset so connected_components.csv includes ccStartTime."
            )

        ordered = sorted(subgraphs, key=lambda d: d.preset_time)

        n = len(ordered)
        t_end = int(train_ratio * n)
        v_end = t_end + int(val_ratio * n)

        train_data = ordered[:t_end]
        val_data = ordered[t_end:v_end]
        test_data = ordered[v_end:]
        return train_data, val_data, test_data

def init_dataset(root: str, name: str, dataset_type: DatasetType, 
                 embedding_type: Union[Embeddingtype, None] = None, verbose: bool = False) -> BaseSubgraphDataset:
    if dataset_type == "elliptic":
        root = os.path.join(root, "elliptic")
        ds = EllipticDataset(root, name, verbose=verbose)
    elif dataset_type == "subgnn":
        root = os.path.join(root, "subgnn")
        if embedding_type is None:
            raise ValueError("embedding_type must be specified for subgnn dataset.")
        ds = SubGNNDataset(root, name, embedding_type, verbose=verbose)
    elif dataset_type == "ibm_aml":
        # Supports either:
        #   root=/home/.../data, name=aml
        # or
        #   root=/home/.../data/aml, name=""
        ds = AMLDataset(root, name, verbose=verbose)
    else:
        raise ValueError(f"Unknown dataset_type {dataset_type}")
    return ds

if __name__ == "__main__":
    PATH = "/home/ghonkoop/data"
    
    # Using SubGNN Data via class instance directly
    subgnn_ds = init_dataset(PATH, "ppi_bp", "subgnn", "glass", verbose=False)
    global_data, subgraphs_list = subgnn_ds.load()
    train_sub, val_sub, test_sub = subgnn_ds.split(subgraphs_list)
    print(f"SubGNN Splitted -> Train: {len(train_sub)}, Val: {len(val_sub)}, Test: {len(test_sub)}")
    
    # Using Elliptic Data via class instance directly
    elliptic_ds = init_dataset(PATH, "elliptic2_bfs_100", "elliptic", verbose=False)
    global_data, subgraphs_list = elliptic_ds.load()
    train_sub, val_sub, test_sub = elliptic_ds.split(subgraphs_list)
    print(f"Elliptic Splitted -> Train: {len(train_sub)}, Val: {len(val_sub)}, Test: {len(test_sub)}")
