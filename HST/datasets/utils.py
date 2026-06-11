import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from sklearn.preprocessing import MultiLabelBinarizer

from typing import List, Literal, Tuple, Union, Any

from .base import BaseSubgraphDataset

DatasetType = Literal["elliptic", "subgnn", "aml", "saml_d"]
Embeddingtype = Literal["glass", "gin", "graphsaint_gcn"]

def init_dataset(root: str, name: str, dataset_type: DatasetType, embedding_type: Union[Embeddingtype, None] = None, 
                 verbose: bool = False) -> BaseSubgraphDataset:
    if dataset_type == "elliptic":
        from .elliptic import EllipticDataset
        root = os.path.join(root, "elliptic")
        ds = EllipticDataset(root, name, verbose=verbose)
    elif dataset_type == "subgnn":
        from .subgnn import SubGNNDataset
        root = os.path.join(root, "subgnn")
        if embedding_type is None:
            raise ValueError("embedding_type must be specified for subgnn dataset.")
        ds = SubGNNDataset(root, name, embedding_type, verbose=verbose)
    elif dataset_type == "aml":
        from .aml import AMLDataset
        root = os.path.join(root, "aml")
        ds = AMLDataset(root, name, verbose=verbose)
    elif dataset_type == "saml_d":
        from .aml import AMLDataset
        root = os.path.join(root, "saml-d")
        ds = AMLDataset(root, name, verbose=verbose)
    else:
        raise ValueError(f"Unknown dataset_type {dataset_type}")
    return ds

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

