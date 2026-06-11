import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
import torch
from torch_geometric.utils import is_undirected, to_undirected, negative_sampling, to_networkx
from torch_geometric.data import Data
import networkx as nx
import os


def _scalar_edge_weight(edge_attr):
    if edge_attr.dim() == 1:
        return edge_attr
    if edge_attr.size(-1) == 1:
        return edge_attr.reshape(-1)
    return torch.ones(edge_attr.size(0), device=edge_attr.device)


def _normalize_edge_features(edge_features: pd.DataFrame):
    feature_values = edge_features.apply(pd.to_numeric, errors="coerce")
    feature_values = feature_values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_values = feature_values.to_numpy(dtype=np.float32)
    mean = feature_values.mean(axis=0, keepdims=True)
    std = feature_values.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return torch.from_numpy((feature_values - mean) / std)


def _normalize_edge_feature_tensor(edge_attr: torch.Tensor, clip=None):
    edge_attr = edge_attr.to(torch.float)
    mean = edge_attr.mean(dim=0, keepdim=True)
    std = edge_attr.std(dim=0, keepdim=True, unbiased=False)
    std[std < 1e-12] = 1.0
    edge_attr = (edge_attr - mean) / std
    if clip is not None:
        edge_attr = torch.clamp(edge_attr, min=-clip, max=clip)
    return edge_attr


def _select_edge_feature_cols(edges_df: pd.DataFrame, edge_feature_cols):
    if edge_feature_cols:
        selected = [col for col in edge_feature_cols if col in edges_df.columns]
        missing = sorted(set(edge_feature_cols) - set(selected))
        if missing:
            raise ValueError(f"Missing edge feature columns: {missing}")
        return selected

    feature_cols = [col for col in edges_df.columns if col.startswith("feat")]
    return [
        col for col in feature_cols
        if pd.to_numeric(edges_df[col], errors="coerce").notna().any()
    ]


def _set_temporal_edge_masks(
    graph_data,
    edge_time: pd.Series,
    component_edges: pd.DataFrame,
    train_cc_ids,
    val_cc_ids,
    train_ratio: float,
    val_ratio: float,
):
    if edge_time is None or graph_data.edge_index is None:
        return
    if graph_data.edge_index.size(1) != len(edge_time):
        return

    if component_edges is not None and len(component_edges) > 0:
        train_ratio = float(component_edges["ccId"].isin(train_cc_ids).mean())
        val_ratio = float(component_edges["ccId"].isin(val_cc_ids).mean())

    edge_time_np = edge_time.to_numpy()
    total_edges = edge_time_np.shape[0]
    order = np.argsort(edge_time_np, kind="mergesort")
    train_cut = max(0, min(int(train_ratio * total_edges), total_edges))
    val_cut = max(train_cut, min(int(val_ratio * total_edges), total_edges))

    train_np = np.zeros(total_edges, dtype=bool)
    val_np = np.zeros(total_edges, dtype=bool)
    test_np = np.ones(total_edges, dtype=bool)
    if train_cut > 0:
        train_np[order[:train_cut]] = True
    if val_cut > 0:
        val_np[order[:val_cut]] = True

    graph_data.edge_train_mask = torch.from_numpy(train_np)
    graph_data.edge_val_mask = torch.from_numpy(val_np)
    graph_data.edge_test_mask = torch.from_numpy(test_np)


def _build_aml_split_mask(labels, train_ratio: float, val_ratio: float,
                          split_mode: str):
    labels = np.asarray(labels)
    n = labels.shape[0]
    mask = np.full(n, 2, dtype=np.int64)
    split_mode = str(split_mode).strip().lower().replace("-", "_")

    if split_mode in {"temporal", "global_temporal"}:
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)
        mask[:train_end] = 0
        mask[train_end:val_end] = 1
        return torch.from_numpy(mask)

    if split_mode in {"class_temporal", "classwise_temporal", "stratified_temporal"}:
        for label_value in sorted(np.unique(labels)):
            idx = np.flatnonzero(labels == label_value)
            train_end = int(train_ratio * len(idx))
            val_end = train_end + int(val_ratio * len(idx))
            mask[idx[:train_end]] = 0
            mask[idx[train_end:val_end]] = 1
        return torch.from_numpy(mask)

    raise ValueError(
        "AML_SPLIT_MODE must be one of temporal, class_temporal, or stratified_temporal; "
        f"got {split_mode!r}."
    )


class BaseGraph(Data):
    def __init__(self, x, edge_index, edge_weight, subG_node, subG_label,
                 mask):
        '''
        A general format for datasets.
        Args:
            x: node feature. For our used datasets, x is empty vector.
            subG_node: a matrix like [[0,2,3],[1,4,5],[6,7,-1]], whose i-th row contains the nodes in the i-th subgraph. -1 is for padding.
            subG_label: the target of subgraphs.
            mask: of shape (number of subgraphs), type torch.long. mask[i]=0,1,2 if i-th subgraph is in the training set, validation set and test set respectively. 
        '''
        super(BaseGraph, self).__init__(x=x,
                                        edge_index=edge_index,
                                        edge_attr=edge_weight,
                                        pos=subG_node,
                                        y=subG_label)
        self.mask = mask
        self.to_undirected()

    def addDegreeFeature(self):
        # For GNN-seg only, use one-hot node degree as node features.
        adj = torch.sparse_coo_tensor(self.edge_index,
                                      _scalar_edge_weight(self.edge_attr),
                                      (self.x.shape[0], self.x.shape[0]))
        degree = torch.sparse.sum(adj, dim=1).to_dense().to(torch.int64)
        self.x = torch.cat((self.x, one_hot(degree).to(torch.float).reshape(
            self.x.shape[0], 1, -1)),
            dim=-1)
    
    def addOneFeature(self):
        # For GNN-seg only, use one as node features.
        self.x = torch.cat(
            (self.x, torch.ones(self.x.shape[0], self.x.shape[1], 1)),
            dim=-1)

    def setDegreeFeature(self, mod=1):
        # use node degree as node features.
        adj = torch.sparse_coo_tensor(self.edge_index,
                                      _scalar_edge_weight(self.edge_attr),
                                      (self.x.shape[0], self.x.shape[0]))
        degree = torch.sparse.sum(adj, dim=1).to_dense().to(torch.int64)
        degree = torch.div(degree, mod, rounding_mode='floor')
        degree = torch.unique(degree, return_inverse=True)[1]
        self.x = degree.reshape(self.x.shape[0], 1, -1)

    def setOneFeature(self):
        # use homogeneous node features.
        self.x = torch.ones((self.x.shape[0], 1, 1), dtype=torch.int64)

    def setNodeIdFeature(self):
        # use nodeid as node features.
        self.x = torch.arange(self.x.shape[0], dtype=torch.int64).reshape(
            self.x.shape[0], 1, -1)

    def get_split(self, split: str):
        tar_mask = {"train": 0, "valid": 1, "test": 2}[split]
        return self.x, self.edge_index, self.edge_attr, self.pos[
            self.mask == tar_mask], self.y[self.mask == tar_mask]

    def to_undirected(self):
        if not is_undirected(self.edge_index):
            reduce = "mean" if self.edge_attr.dim() > 1 else "add"
            self.edge_index, self.edge_attr = to_undirected(
                self.edge_index, self.edge_attr, reduce=reduce)

    def get_LPdataset(self, use_loop=False):
        # generate link prediction dataset for pretraining GNNs
        neg_edge = negative_sampling(self.edge_index)
        x = self.x
        ei = self.edge_index
        ea = self.edge_attr
        pos = torch.cat((self.edge_index, neg_edge), dim=1).t()
        y = torch.cat((torch.ones(ei.shape[1]),
                       torch.zeros(neg_edge.shape[1]))).to(ei.device)
        if use_loop:
            mask = (ei[0] == ei[1])
            pos_loops = ei[0][mask]
            all_loops = torch.arange(x.shape[0],
                                     device=x.device).reshape(-1, 1)[:, [0, 0]]
            y_loop = torch.zeros(x.shape[0], device=y.device)
            y_loop[pos_loops] = 1
            pos = torch.cat((pos, all_loops), dim=0)
            y = torch.cat((y, y_loop), dim=0)
        return x, ei, ea, pos, y

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        self.mask = self.mask.to(device)
        if hasattr(self, "edge_train_mask"):
            self.edge_train_mask = self.edge_train_mask.to(device)
        if hasattr(self, "edge_val_mask"):
            self.edge_val_mask = self.edge_val_mask.to(device)
        if hasattr(self, "edge_test_mask"):
            self.edge_test_mask = self.edge_test_mask.to(device)
        return self


def load_dataset(name: str,
                 use_edge_features=False,
                 edge_feature_cols=None,
                 edge_feature_clip=None):
    # To use your own dataset, add a branch returning a BaseGraph Object here.
    if name in ["coreness", "cut_ratio", "density", "component"]:
        obj = np.load(f"./dataset_/{name}/tmp.npy", allow_pickle=True).item()
        # copied from https://github.com/mims-harvard/SubGNN/blob/main/SubGNN/subgraph_utils.py
        edge = np.array([[i[0] for i in obj['G'].edges],
                         [i[1] for i in obj['G'].edges]])
        degree = obj['G'].degree
        node = [n for n in obj['G'].nodes]
        subG = obj["subG"]
        subG_pad = pad_sequence([torch.tensor(i) for i in subG],
                                batch_first=True,
                                padding_value=-1)
        subGLabel = torch.tensor([ord(i) - ord('A') for i in obj["subGLabel"]])
        #mask = torch.tensor(obj['mask'])
        cnt = subG_pad.shape[0]
        mask = torch.cat(
            (torch.zeros(cnt - cnt // 2, dtype=torch.int64),
             torch.ones(cnt // 4, dtype=torch.int64),
             2 * torch.ones(cnt // 2 - cnt // 4, dtype=torch.int64)))
        mask = mask[torch.randperm(mask.shape[0])]
        return BaseGraph(torch.empty(
            (len(node), 1, 0)), torch.from_numpy(edge),
                         torch.ones(edge.shape[1]), subG_pad, subGLabel, mask)
    elif name in ["ppi_bp", "hpo_metab", "hpo_neuro", "em_user"]:
        multilabel = False

        # copied from https://github.com/mims-harvard/SubGNN/blob/main/SubGNN/subgraph_utils.py
        def read_subgraphs(sub_f, split=True):
            label_idx = 0
            labels = {}
            train_sub_G, val_sub_G, test_sub_G = [], [], []
            train_sub_G_label, val_sub_G_label, test_sub_G_label = [], [], []
            train_mask, val_mask, test_mask = [], [], []
            nonlocal multilabel
            # Parse data
            with open(sub_f) as fin:
                subgraph_idx = 0
                for line in fin:
                    nodes = [
                        int(n) for n in line.split("\t")[0].split("-")
                        if n != ""
                    ]
                    if len(nodes) != 0:
                        if len(nodes) == 1:
                            print(nodes)
                        l = line.split("\t")[1].split("-")
                        if len(l) > 1:
                            multilabel = True
                        for lab in l:
                            if lab not in labels.keys():
                                labels[lab] = label_idx
                                label_idx += 1
                        if line.split("\t")[2].strip() == "train":
                            train_sub_G.append(nodes)
                            train_sub_G_label.append(
                                [labels[lab] for lab in l])
                            train_mask.append(subgraph_idx)
                        elif line.split("\t")[2].strip() == "val":
                            val_sub_G.append(nodes)
                            val_sub_G_label.append([labels[lab] for lab in l])
                            val_mask.append(subgraph_idx)
                        elif line.split("\t")[2].strip() == "test":
                            test_sub_G.append(nodes)
                            test_sub_G_label.append([labels[lab] for lab in l])
                            test_mask.append(subgraph_idx)
                        subgraph_idx += 1
            if not multilabel:
                train_sub_G_label = torch.tensor(train_sub_G_label).squeeze()
                val_sub_G_label = torch.tensor(val_sub_G_label).squeeze()
                test_sub_G_label = torch.tensor(test_sub_G_label).squeeze()

            if len(val_mask) < len(test_mask):
                return train_sub_G, train_sub_G_label, test_sub_G, test_sub_G_label, val_sub_G, val_sub_G_label

            return train_sub_G, train_sub_G_label, val_sub_G, val_sub_G_label, test_sub_G, test_sub_G_label

        if os.path.exists(
                f"./dataset/{name}/train_sub_G.pt") and name != "hpo_neuro":
            train_sub_G = torch.load(f"./dataset/{name}/train_sub_G.pt")
            train_sub_G_label = torch.load(
                f"./dataset/{name}/train_sub_G_label.pt")
            val_sub_G = torch.load(f"./dataset/{name}/val_sub_G.pt")
            val_sub_G_label = torch.load(
                f"./dataset/{name}/val_sub_G_label.pt")
            test_sub_G = torch.load(f"./dataset/{name}/test_sub_G.pt")
            test_sub_G_label = torch.load(
                f"./dataset/{name}/test_sub_G_label.pt")
        else:
            train_sub_G, train_sub_G_label, val_sub_G, val_sub_G_label, test_sub_G, test_sub_G_label = read_subgraphs(
                f"./dataset/{name}/subgraphs.pth")
            torch.save(train_sub_G, f"./dataset/{name}/train_sub_G.pt")
            torch.save(train_sub_G_label,
                       f"./dataset/{name}/train_sub_G_label.pt")
            torch.save(val_sub_G, f"./dataset/{name}/val_sub_G.pt")
            torch.save(val_sub_G_label, f"./dataset/{name}/val_sub_G_label.pt")
            torch.save(test_sub_G, f"./dataset/{name}/test_sub_G.pt")
            torch.save(test_sub_G_label,
                       f"./dataset/{name}/test_sub_G_label.pt")
        mask = torch.cat(
            (torch.zeros(len(train_sub_G_label), dtype=torch.int64),
             torch.ones(len(val_sub_G_label), dtype=torch.int64),
             2 * torch.ones(len(test_sub_G_label))),
            dim=0)
        if multilabel:
            tlist = train_sub_G_label + val_sub_G_label + test_sub_G_label
            max_label = max([max(i) for i in tlist])
            label = torch.zeros(len(tlist), max_label + 1)
            for idx, ll in enumerate(tlist):
                label[idx][torch.LongTensor(ll)] = 1
        else:
            label = torch.cat(
                (train_sub_G_label, val_sub_G_label, test_sub_G_label))
        pos = pad_sequence(
            [torch.tensor(i) for i in train_sub_G + val_sub_G + test_sub_G],
            batch_first=True,
            padding_value=-1)
        rawedge = nx.read_edgelist(f"./dataset/{name}/edge_list.txt").edges
        edge_index = torch.tensor([[int(i[0]), int(i[1])]
                                   for i in rawedge]).t()
        num_node = max([torch.max(pos), torch.max(edge_index)]) + 1
        x = torch.empty((num_node, 1, 0))

        return BaseGraph(x, edge_index, torch.ones(edge_index.shape[1]), pos,
                         label.to(torch.float), mask)
    elif name in ["aml", "aml_hi_small"] or name.startswith("aml_hi_small_"):
        base_path = os.environ.get("AML_BASE_PATH",
                                   "/home/ghonkoop/data/aml/HI-Small")
        if not os.path.exists(base_path):
            raise FileNotFoundError(
                f"AML dataset path not found: {base_path}. Set AML_BASE_PATH to your dataset directory."
            )

        nodes_df = pd.read_csv(os.path.join(base_path, "background_nodes.csv"))
        nodes_df["clId"] = nodes_df["clId"].astype(str)
        node_map = {clId: i for i, clId in enumerate(nodes_df["clId"])}

        edges_df = pd.read_csv(os.path.join(base_path, "background_edges.csv"))
        valid = edges_df["clId1"].astype(str).isin(node_map) & edges_df[
            "clId2"].astype(str).isin(node_map)
        edges_df = edges_df[valid]
        edge_time = None
        if "feat0" in edges_df.columns:
            edge_time = edges_df["feat0"].astype(float).reset_index(drop=True)

        src = edges_df["clId1"].astype(str).map(node_map).to_numpy(
            dtype=np.int64)
        dst = edges_df["clId2"].astype(str).map(node_map).to_numpy(
            dtype=np.int64)
        edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
        edge_attr = torch.ones(edge_index.shape[1])
        selected_edge_feature_cols = []
        if use_edge_features:
            selected_edge_feature_cols = _select_edge_feature_cols(
                edges_df, edge_feature_cols)
            if not selected_edge_feature_cols:
                raise ValueError(
                    "use_edge_features=True but no numeric edge feature columns were found"
                )
            edge_attr = _normalize_edge_features(
                edges_df[selected_edge_feature_cols])

        cc_df = pd.read_csv(
            os.path.join(base_path, "connected_components.csv"))
        cc_df["ccId"] = cc_df["ccId"].astype(str)

        if "ccStartTime" in cc_df.columns:
            cc_df["ccStartTime"] = pd.to_datetime(cc_df["ccStartTime"],
                                                   errors="coerce")
            cc_df = cc_df.sort_values(["ccStartTime", "ccId"])\
                       .reset_index(drop=True)

        lbl_map = {l: i for i, l in enumerate(sorted(cc_df["ccLabel"].unique()))}

        sub_df = pd.read_csv(os.path.join(base_path, "nodes.csv"))
        sub_df["ccId"] = sub_df["ccId"].astype(str)
        sub_df["clId"] = sub_df["clId"].astype(str)
        grouped_nodes = sub_df.groupby("ccId")["clId"].apply(list).to_dict()

        sub_nodes = []
        sub_labels = []
        sub_cc_ids = []

        for _, row in cc_df.iterrows():
            cc_id = row["ccId"]
            if cc_id not in grouped_nodes:
                continue
            nodes = [node_map[n] for n in grouped_nodes[cc_id] if n in node_map]
            if not nodes:
                continue
            sub_nodes.append(nodes)
            sub_labels.append(lbl_map[row["ccLabel"]])
            sub_cc_ids.append(cc_id)

        if len(sub_nodes) == 0:
            raise ValueError(
                "No subgraphs found for AML dataset. Check input files in AML_BASE_PATH."
            )

        component_edges = None
        component_edges_path = os.path.join(base_path, "component_edges.csv")
        if os.path.exists(component_edges_path):
            component_edges = pd.read_csv(component_edges_path, usecols=["ccId", "feat0"])
            component_edges["ccId"] = component_edges["ccId"].astype(str)
            component_edges["feat0"] = component_edges["feat0"].astype(float)

        n = len(sub_nodes)
        if "ccStartTime" not in cc_df.columns:
            rng = np.random.default_rng(0)
            perm = rng.permutation(n)
            sub_nodes = [sub_nodes[i] for i in perm]
            sub_labels = [sub_labels[i] for i in perm]
            sub_cc_ids = [sub_cc_ids[i] for i in perm]

        train_ratio = float(os.environ.get("AML_TRAIN_RATIO", 0.7))
        val_only_ratio = float(os.environ.get("AML_VAL_RATIO", 0.15))
        split_mode = os.environ.get("AML_SPLIT_MODE", "temporal")
        mask = _build_aml_split_mask(sub_labels, train_ratio, val_only_ratio,
                                     split_mode)

        pos = pad_sequence([torch.tensor(i) for i in sub_nodes],
                           batch_first=True,
                           padding_value=-1)
        label = torch.tensor(sub_labels, dtype=torch.int64)

        x = torch.empty((len(node_map), 1, 0))
        graph = BaseGraph(x, edge_index, edge_attr, pos, label, mask)
        graph.edge_feature_cols = selected_edge_feature_cols
        if use_edge_features:
            graph.edge_attr = _normalize_edge_feature_tensor(
                graph.edge_attr, clip=edge_feature_clip)
        train_cc_ids = {cc_id for cc_id, split_id in zip(sub_cc_ids, mask.tolist())
                        if split_id == 0}
        val_cc_ids = train_cc_ids | {
            cc_id for cc_id, split_id in zip(sub_cc_ids, mask.tolist())
            if split_id == 1
        }
        graph.split_mode = split_mode
        _set_temporal_edge_masks(
            graph,
            edge_time,
            component_edges,
            train_cc_ids,
            val_cc_ids,
            train_ratio,
            train_ratio + val_only_ratio,
        )
        return graph
    else:
        raise NotImplementedError()
