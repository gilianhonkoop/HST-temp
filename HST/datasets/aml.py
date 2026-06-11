import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from typing import List, Tuple

from .base import BaseSubgraphDataset
from .utils import get_data_list_from_subgraphs

class AMLDataset(BaseSubgraphDataset):
    def load(self) -> Tuple[Data, List[Data]]:
        if self.verbose: print(f"Loading IBM AML dataset from {self.base_path}...")

        # 1. Load Nodes
        nodes_df = pd.read_csv(os.path.join(self.base_path, "background_nodes.csv"))
        nodes_df['clId'] = nodes_df['clId'].astype(str)
        node_map = {clId: i for i, clId in enumerate(nodes_df['clId'])}
        
        feat_cols = [c for c in nodes_df.columns if c.startswith('feat')]
        if feat_cols:
            x = torch.tensor(nodes_df[feat_cols].values, dtype=torch.float)
        else:
            # Some AML-format datasets, e.g. SAML-D, intentionally have no
            # native node features. Provide a neutral in-memory placeholder so
            # model code can still build node tensors without adding features
            # to background_nodes.csv.
            x = torch.ones((len(node_map), 1), dtype=torch.float)

        # 2. Load Edges
        edges_df = pd.read_csv(os.path.join(self.base_path, "background_edges.csv"))

        edges_df['clId1'] = edges_df['clId1'].astype(str)
        edges_df['clId2'] = edges_df['clId2'].astype(str)
        valid = edges_df['clId1'].isin(node_map) & edges_df['clId2'].isin(node_map)

        edges_df = edges_df.loc[valid].reset_index(drop=True)
        edge_time = edges_df['feat0'].astype(float).reset_index(drop=True)
        
        src = edges_df['clId1'].astype(str).map(node_map).to_numpy(dtype=np.int64)
        dst = edges_df['clId2'].astype(str).map(node_map).to_numpy(dtype=np.int64)
        edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
        
        efeat_cols = [c for c in edges_df.columns if c.startswith('feat')]
        edge_attr = torch.tensor(edges_df[efeat_cols].values, dtype=torch.float)

        global_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(node_map))

        # 3. Load Components & Temporal Ordering
        cc_df = pd.read_csv(os.path.join(self.base_path, "connected_components.csv"))
        cc_df['ccId'] = cc_df['ccId'].astype(str)
        
        if 'ccStartTime' in cc_df.columns:
            cc_df['ccStartTime'] = pd.to_datetime(cc_df['ccStartTime'], errors='coerce')
            cc_df = cc_df.sort_values('ccStartTime')
            # Convert to an integer timestamp (Unix epoch in seconds) for PyG loaders
            cc_df['time'] = (cc_df['ccStartTime'].astype('int64') // 10**9).astype(int)

        lbl_map = {l: i for i, l in enumerate(sorted(cc_df['ccLabel'].unique()))}
        
        # 4. Map Nodes to Components
        sub_df = pd.read_csv(os.path.join(self.base_path, "nodes.csv"))
        sub_df['ccId'] = sub_df['ccId'].astype(str)
        sub_df['clId'] = sub_df['clId'].astype(str)
        grouped_nodes = sub_df.groupby('ccId', sort=False)['clId'].apply(list).to_dict()

        component_edges_path = os.path.join(self.base_path, "component_edges.csv")
        grouped_edges = {}
        component_edges_for_split = None
        if os.path.exists(component_edges_path):
            ce_edges = pd.read_csv(component_edges_path, usecols=['ccId', 'clId1', 'clId2', 'feat0'])
            ce_edges['ccId'] = ce_edges['ccId'].astype(str)
            component_edges_for_split = ce_edges[['ccId', 'feat0']].copy()
            ce_edges['src'] = ce_edges['clId1'].astype(str).map(node_map)
            ce_edges['dst'] = ce_edges['clId2'].astype(str).map(node_map)
            ce_edges = ce_edges.dropna(subset=['src', 'dst'])
            ce_edges['src'] = ce_edges['src'].to_numpy(dtype=np.int64)
            ce_edges['dst'] = ce_edges['dst'].to_numpy(dtype=np.int64)
            for cc_id, group in ce_edges.groupby('ccId', sort=False):
                grouped_edges[cc_id] = np.vstack(
                    (
                        group['src'].to_numpy(dtype=np.int64, copy=False),
                        group['dst'].to_numpy(dtype=np.int64, copy=False),
                    )
                )

        sub_nodes, sub_ys, sub_times, sub_cc_ids = [], [], [], []
        for _, row in cc_df.iterrows():
            ccId = row['ccId']
            if ccId in grouped_nodes:
                nodes = [node_map[n] for n in grouped_nodes[ccId] if n in node_map]
                if nodes:
                    sub_nodes.append(nodes)
                    sub_ys.append(torch.tensor(lbl_map[row['ccLabel']], dtype=torch.long))
                    sub_cc_ids.append(ccId)
                    if 'time' in cc_df.columns:
                        sub_times.append(row['time'])

        if os.path.exists(component_edges_path):
            # Build subgraph objects from precomputed component edges. This avoids calling
            # torch_geometric.utils.subgraph once per component over the full global graph.
            subgraphs = []
            for nodes, y, cc_id in zip(sub_nodes, sub_ys, sub_cc_ids):
                x_index = torch.tensor(nodes, dtype=torch.long).view(-1, 1)
                if len(y.size()) == 0:
                    y = y.unsqueeze(0)
                else:
                    y = y.view(1, -1)

                edge_np = grouped_edges.get(cc_id)
                if edge_np is None:
                    sub_edge_index = torch.empty((2, 0), dtype=torch.long)
                else:
                    sub_edge_index = torch.from_numpy(edge_np).long()

                data = Data(
                    x=global_data.x[x_index.flatten()],
                    x_idx=x_index,
                    edge_index=sub_edge_index,
                    y=y,
                )
                data.cc_id = cc_id
                subgraphs.append(data)
        else:
            subgraphs = get_data_list_from_subgraphs(global_data, sub_nodes, sub_ys)
            for i, d in enumerate(subgraphs):
                d.cc_id = sub_cc_ids[i]

        if sub_times:
            for d, t in zip(subgraphs, sub_times):
                d.preset_time = t

        self._global_data = global_data
        # Keep raw temporal feature for full-edge temporal masking.
        self._edge_time = edge_time

        # Optional component-edge mapping used only to infer split proportions.
        if os.path.exists(component_edges_path):
            component_edges_for_split['feat0'] = component_edges_for_split['feat0'].astype(float)
            self._component_edges = component_edges_for_split
        else:
            self._component_edges = None

        if self.verbose: print(f"Done. Built {len(subgraphs)} subgraphs.")
        return global_data, subgraphs

    def _set_temporal_edge_masks(
        self,
        train_data: List[Data],
        val_data: List[Data],
        test_data: List[Data],
    ) -> None:
        global_data = getattr(self, "_global_data", None)
        edge_time = getattr(self, "_edge_time", None)
        if global_data is None or edge_time is None:
            return

        train_cc_ids = {d.cc_id for d in train_data if hasattr(d, 'cc_id')}
        val_cc_ids = train_cc_ids | {d.cc_id for d in val_data if hasattr(d, 'cc_id')}

        component_edges = getattr(self, "_component_edges", None)
        if component_edges is not None and len(component_edges) > 0:
            train_ratio = float(component_edges['ccId'].isin(train_cc_ids).mean())
            val_ratio = float(component_edges['ccId'].isin(val_cc_ids).mean())
        else:
            # Fallback to subgraph-level split ratios if component_edges are unavailable.
            total_sub = max(len(train_data) + len(val_data) + len(test_data), 1)
            train_ratio = float(len(train_data) / total_sub)
            val_ratio = float((len(train_data) + len(val_data)) / total_sub)

        edge_time_np = edge_time.to_numpy()
        total_edges = edge_time_np.shape[0]
        order = np.argsort(edge_time_np, kind='mergesort')
        train_cut = int(train_ratio * total_edges)
        val_cut = int(val_ratio * total_edges)
        train_cut = max(0, min(train_cut, total_edges))
        val_cut = max(train_cut, min(val_cut, total_edges))

        train_np = np.zeros(total_edges, dtype=bool)
        val_np = np.zeros(total_edges, dtype=bool)
        test_np = np.ones(total_edges, dtype=bool)
        if train_cut > 0:
            train_np[order[:train_cut]] = True
        if val_cut > 0:
            val_np[order[:val_cut]] = True

        train_mask = torch.from_numpy(train_np)
        val_mask = torch.from_numpy(val_np)
        test_mask = torch.from_numpy(test_np)

        if global_data.edge_index is not None and train_mask.numel() != global_data.edge_index.size(1):
            raise ValueError(
                "Temporal edge mask length does not match global edge_index. "
                f"mask={train_mask.numel()}, edges={global_data.edge_index.size(1)}"
            )

        global_data.edge_train_mask = train_mask
        global_data.edge_val_mask = val_mask
        global_data.edge_test_mask = test_mask

    def split(
        self,
        subgraphs: List[Data],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_mode: str = "temporal",
    ) -> Tuple[List[Data], List[Data], List[Data]]:
        if len(subgraphs) == 0:
            return [], [], []

        split_mode = str(split_mode).strip().lower().replace("-", "_")

        if split_mode in {"temporal", "global_temporal"}:
            n = len(subgraphs)
            t_end = int(train_ratio * n)
            v_end = t_end + int(val_ratio * n)
            train_data = subgraphs[:t_end]
            val_data = subgraphs[t_end:v_end]
            test_data = subgraphs[v_end:]
        elif split_mode in {"class_temporal", "classwise_temporal", "stratified_temporal"}:
            train_data, val_data, test_data = [], [], []
            labels = sorted({int(d.y.view(-1)[0].item()) for d in subgraphs})
            for label in labels:
                class_subgraphs = [
                    d for d in subgraphs
                    if int(d.y.view(-1)[0].item()) == label
                ]
                n = len(class_subgraphs)
                t_end = int(train_ratio * n)
                v_end = t_end + int(val_ratio * n)
                train_data.extend(class_subgraphs[:t_end])
                val_data.extend(class_subgraphs[t_end:v_end])
                test_data.extend(class_subgraphs[v_end:])

            train_data = sorted(train_data, key=lambda d: getattr(d, 'preset_time', 0))
            val_data = sorted(val_data, key=lambda d: getattr(d, 'preset_time', 0))
            test_data = sorted(test_data, key=lambda d: getattr(d, 'preset_time', 0))
        else:
            raise ValueError(
                "split_mode must be one of temporal, class_temporal, or "
                f"stratified_temporal; got {split_mode!r}."
            )

        self._set_temporal_edge_masks(train_data, val_data, test_data)
        
        return train_data, val_data, test_data
