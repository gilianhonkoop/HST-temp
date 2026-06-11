import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from data_base import SubgraphDataset


def _mask_edges_by_node_set(edge_index: torch.Tensor, node_set: torch.Tensor) -> torch.Tensor:
    src_ok = torch.isin(edge_index[0], node_set)
    dst_ok = torch.isin(edge_index[1], node_set)
    return src_ok & dst_ok


def _set_temporal_edge_masks(
    global_data: Data,
    edge_time: pd.Series,
    component_edges: pd.DataFrame,
    train_cc_ids,
    val_cc_ids,
    train_ratio: float,
    val_ratio: float,
) -> None:
    if edge_time is None or global_data.edge_index is None:
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

    global_data.edge_train_mask = torch.from_numpy(train_np)
    global_data.edge_val_mask = torch.from_numpy(val_np)
    global_data.edge_test_mask = torch.from_numpy(test_np)


class AMLHI(SubgraphDataset):
    """IBM AML HI-Small dataset loader for S2N."""

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 split_mode="temporal",
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        self.split_mode = str(split_mode).strip().lower().replace("-", "_")
        super().__init__(
            root=root,
            name=name,
            embedding_type=embedding_type,
            val_ratio=0.15 if val_ratio is None else val_ratio,
            test_ratio=0.15 if test_ratio is None else test_ratio,
            save_directed_edges=save_directed_edges,
            debug=debug,
            seed=seed,
            num_training_tails_to_tile_per_class=num_training_tails_to_tile_per_class,
            load_rwpe=load_rwpe,
            load_lepe=load_lepe,
            transform=transform,
            pre_transform=pre_transform,
            **kwargs,
        )

    def load(self):
        base_path = self.root
        if self.debug:
            print(f"[AMLHI] Loading from {base_path}")

        # 1) Global nodes
        nodes_df = pd.read_csv(os.path.join(base_path, "background_nodes.csv"))
        nodes_df["clId"] = nodes_df["clId"].astype(str)
        node_ids = nodes_df["clId"].tolist()
        node_map = {nid: i for i, nid in enumerate(node_ids)}

        feat_cols = [c for c in nodes_df.columns if c.startswith("feat")]
        if feat_cols:
            x = torch.tensor(nodes_df[feat_cols].to_numpy(), dtype=torch.float)
        else:
            x = torch.ones((len(node_map), 1), dtype=torch.float)

        # 2) Global edges
        edges_df = pd.read_csv(os.path.join(base_path, "background_edges.csv"))
        src_ids = edges_df["clId1"].astype(str)
        dst_ids = edges_df["clId2"].astype(str)
        valid = src_ids.isin(node_map) & dst_ids.isin(node_map)
        edges_df = edges_df[valid]
        edge_time = None
        if "feat0" in edges_df.columns:
            edge_time = edges_df["feat0"].astype(float).reset_index(drop=True)

        src = src_ids[valid].map(node_map).to_numpy(dtype=np.int64)
        dst = dst_ids[valid].map(node_map).to_numpy(dtype=np.int64)
        edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)

        efeat_cols = [c for c in edges_df.columns if c.startswith("feat")]
        if efeat_cols:
            edge_attr = torch.tensor(edges_df[efeat_cols].to_numpy(), dtype=torch.float)
        else:
            edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)

        self.global_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(node_map),
        )

        # 3) Components and labels (chronological order when available)
        cc_df = pd.read_csv(os.path.join(base_path, "connected_components.csv"))
        cc_df["ccId"] = cc_df["ccId"].astype(str)
        if "ccStartTime" in cc_df.columns:
            cc_df["ccStartTime"] = pd.to_datetime(cc_df["ccStartTime"], errors="coerce")
            cc_df = cc_df.sort_values(["ccStartTime", "ccId"]).reset_index(drop=True)
            ts = cc_df["ccStartTime"]
            cc_df["time"] = 0
            valid = ts.notna()
            cc_df.loc[valid, "time"] = (ts.loc[valid].astype("int64") // 10**9).astype(np.int64)

        label_col = "ccLabel" if "ccLabel" in cc_df.columns else "label"
        labels = sorted(cc_df[label_col].dropna().astype(str).unique())
        lbl_map = {lbl: i for i, lbl in enumerate(labels)}

        # 4) Nodes per component
        sub_df = pd.read_csv(os.path.join(base_path, "nodes.csv"))
        sub_df["ccId"] = sub_df["ccId"].astype(str)
        sub_df["clId"] = sub_df["clId"].astype(str)
        grouped_nodes = sub_df.groupby("ccId")["clId"].apply(list).to_dict()

        subgraphs: List[Data] = []
        for _, row in cc_df.iterrows():
            cc_id = row["ccId"]
            if cc_id not in grouped_nodes:
                continue

            sub_nodes = [node_map[n] for n in grouped_nodes[cc_id] if n in node_map]
            if not sub_nodes:
                continue

            node_tensor = torch.tensor(sub_nodes, dtype=torch.long)
            edge_mask = _mask_edges_by_node_set(edge_index, node_tensor)
            sub_edge_index = edge_index[:, edge_mask]
            y = torch.tensor([lbl_map[str(row[label_col])]], dtype=torch.long)

            d = Data(x=node_tensor.view(-1, 1), edge_index=sub_edge_index, y=y)
            d.cc_id = cc_id
            if "time" in cc_df.columns:
                d.preset_time = int(row["time"])
            subgraphs.append(d)

        component_edges_path = os.path.join(base_path, "component_edges.csv")
        component_edges = None
        if os.path.exists(component_edges_path):
            component_edges = pd.read_csv(component_edges_path, usecols=["ccId", "feat0"])
            component_edges["ccId"] = component_edges["ccId"].astype(str)
            component_edges["feat0"] = component_edges["feat0"].astype(float)

        # 5) Split with dataset-configured ratios. Default preserves the original
        # global chronological split; class_temporal preserves chronology per label.
        n = len(subgraphs)
        train_ratio = 1.0 - float(self.val_ratio) - float(self.test_ratio)
        val_cumulative_ratio = train_ratio + float(self.val_ratio)
        self.num_start = 0

        if self.split_mode in {"temporal", "global_temporal"}:
            self.num_train = int(train_ratio * n)
            self.num_val = int(float(self.val_ratio) * n)
            train_data = subgraphs[:self.num_train]
            val_data = subgraphs[self.num_train:self.num_train + self.num_val]
            test_data = subgraphs[self.num_train + self.num_val:]
        elif self.split_mode in {"class_temporal", "classwise_temporal", "stratified_temporal"}:
            train_data, val_data, test_data = [], [], []
            labels_present = sorted({int(d.y.view(-1)[0].item()) for d in subgraphs})
            for label in labels_present:
                class_subgraphs = [
                    d for d in subgraphs
                    if int(d.y.view(-1)[0].item()) == label
                ]
                class_n = len(class_subgraphs)
                class_train_end = int(train_ratio * class_n)
                class_val_end = class_train_end + int(float(self.val_ratio) * class_n)
                train_data.extend(class_subgraphs[:class_train_end])
                val_data.extend(class_subgraphs[class_train_end:class_val_end])
                test_data.extend(class_subgraphs[class_val_end:])

            train_data = sorted(train_data, key=lambda d: getattr(d, "preset_time", 0))
            val_data = sorted(val_data, key=lambda d: getattr(d, "preset_time", 0))
            test_data = sorted(test_data, key=lambda d: getattr(d, "preset_time", 0))
            subgraphs = train_data + val_data + test_data
            self.num_train = len(train_data)
            self.num_val = len(val_data)
        else:
            raise ValueError(
                "split_mode must be one of temporal, class_temporal, or "
                f"stratified_temporal; got {self.split_mode!r}."
            )

        train_cc_ids = {d.cc_id for d in train_data}
        val_cc_ids = train_cc_ids | {d.cc_id for d in val_data}
        _set_temporal_edge_masks(
            self.global_data,
            edge_time,
            component_edges,
            train_cc_ids,
            val_cc_ids,
            train_ratio,
            val_cumulative_ratio,
        )

        self.data, self.slices = self.collate(subgraphs)


# Compatibility alias
AMLDataset = AMLHI


class SAMLD(AMLHI):
    """SAML-D dataset loader using AML-normalized files."""

    pass
