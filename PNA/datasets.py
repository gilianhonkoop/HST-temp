import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected


@dataclass
class LoadedData:
    train: List[Data]
    val: List[Data]
    test: List[Data]
    num_node_features: int
    num_edge_features: int
    deg: torch.Tensor
    label_map: Dict[str, int]


def _zscore_edge_attrs(train: List[Data], val: List[Data], test: List[Data], eps: float, clip: Optional[float]) -> None:
    attrs = [d.edge_attr for d in train if getattr(d, "edge_attr", None) is not None and d.edge_attr.numel() > 0]
    if not attrs:
        return
    stacked = torch.cat(attrs, dim=0).float()
    mean = stacked.mean(dim=0, keepdim=True)
    std = stacked.std(dim=0, unbiased=False, keepdim=True).clamp_min(float(eps))

    for split in (train, val, test):
        for data in split:
            if getattr(data, "edge_attr", None) is None or data.edge_attr.numel() == 0:
                continue
            data.edge_attr = (data.edge_attr.float() - mean) / std
            if clip is not None:
                data.edge_attr = data.edge_attr.clamp(min=-float(clip), max=float(clip))


def _build_degree_histogram(data_list: List[Data], max_degree: Optional[int] = None) -> torch.Tensor:
    degrees = []
    for data in data_list:
        if data.edge_index.numel() == 0:
            degrees.append(torch.zeros(data.num_nodes, dtype=torch.long))
            continue
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        degrees.append(d.cpu().long())
    if not degrees:
        return torch.ones(1, dtype=torch.long)
    all_deg = torch.cat(degrees, dim=0)
    if max_degree is not None and int(max_degree) > 0:
        all_deg = all_deg.clamp(max=int(max_degree))
    hist = torch.bincount(all_deg, minlength=int(all_deg.max().item()) + 1)
    return hist.clamp_min(1)


def _split_subgraphs(
    subgraphs: List[Data],
    train_ratio: float,
    val_ratio: float,
    split_mode: str,
) -> Tuple[List[Data], List[Data], List[Data]]:
    split_mode = str(split_mode).strip().lower().replace("-", "_")
    if split_mode in {"temporal", "global_temporal"}:
        n = len(subgraphs)
        train_end = int(float(train_ratio) * n)
        val_end = train_end + int(float(val_ratio) * n)
        return subgraphs[:train_end], subgraphs[train_end:val_end], subgraphs[val_end:]

    if split_mode in {"class_temporal", "classwise_temporal", "stratified_temporal"}:
        train, val, test = [], [], []
        labels = sorted({int(d.raw_y.item()) for d in subgraphs})
        for label in labels:
            class_items = [d for d in subgraphs if int(d.raw_y.item()) == label]
            n = len(class_items)
            train_end = int(float(train_ratio) * n)
            val_end = train_end + int(float(val_ratio) * n)
            train.extend(class_items[:train_end])
            val.extend(class_items[train_end:val_end])
            test.extend(class_items[val_end:])
        key = lambda d: getattr(d, "preset_time", 0)
        return sorted(train, key=key), sorted(val, key=key), sorted(test, key=key)

    raise ValueError(
        "split_mode must be one of temporal, class_temporal, or stratified_temporal; "
        f"got {split_mode!r}"
    )


def _resolve_dataset_path(cfg: dict) -> str:
    dataset_type = str(cfg.get("dataset_type", "aml")).strip().lower()
    data_path = str(cfg.get("data_path", "/home/ghonkoop/data"))
    dataset_name = str(cfg.get("dataset_name", "HI-Small"))
    if os.path.isabs(dataset_name) and os.path.exists(dataset_name):
        return dataset_name
    if dataset_type in {"saml_d", "samld", "saml-d"}:
        if dataset_name in {"", ".", "SAML-D", "SAMLD", "saml-d"}:
            return os.path.join(data_path, "saml-d")
        return os.path.join(data_path, "saml-d", dataset_name)
    if dataset_type == "aml_patterns":
        return os.path.join(data_path, "aml_patterns", dataset_name)
    return os.path.join(data_path, "aml", dataset_name)


def _cache_path(base_path: str, cfg: dict) -> str:
    keys = {
        "node_features": cfg.get("node_features", "raw"),
        "add_degree_feature": bool(cfg.get("add_degree_feature", False)),
        "use_edge_features": bool(cfg.get("use_edge_features", True)),
        "edge_feature_normalize": cfg.get("edge_feature_normalize", "zscore"),
        "edge_feature_clip": cfg.get("edge_feature_clip", None),
        "add_reverse_edges": bool(cfg.get("add_reverse_edges", False)),
        "train_ratio": float(cfg.get("train_ratio", 0.7)),
        "val_ratio": float(cfg.get("val_ratio", 0.15)),
        "split_mode": str(cfg.get("split_mode", "temporal")),
        "illicit_label": cfg.get("illicit_label", None),
        "max_degree_hist": cfg.get("max_degree_hist", None),
    }
    safe = "_".join(f"{k}-{str(v).replace('/', '-').replace(' ', '')}" for k, v in keys.items())
    cache_dir = os.path.join(base_path, "pna_cache")
    return os.path.join(cache_dir, f"{safe}.pt")


def load_aml_dataset(cfg: dict) -> LoadedData:
    base_path = _resolve_dataset_path(cfg)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path not found: {base_path}")

    use_cache = bool(cfg.get("use_cache", True))
    cache_path = _cache_path(base_path, cfg)
    if use_cache and os.path.exists(cache_path):
        started = time.time()
        logging.info("Loading cached PNA dataset from %s", cache_path)
        try:
            loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(cache_path, map_location="cpu")
        logging.info("Loaded cached PNA dataset in %.1fs", time.time() - started)
        return loaded

    started_total = time.time()
    logging.info("Loading PNA dataset from %s", base_path)
    stage_started = time.time()
    node_df = pd.read_csv(os.path.join(base_path, "background_nodes.csv"))
    node_df["clId"] = node_df["clId"].astype(str)
    node_map = {node_id: i for i, node_id in enumerate(node_df["clId"])}
    node_feat_cols = [c for c in node_df.columns if c.startswith("feat")]
    if node_feat_cols and str(cfg.get("node_features", "raw")).lower() != "ones":
        global_x = torch.tensor(node_df[node_feat_cols].to_numpy(dtype=np.float32), dtype=torch.float)
    else:
        global_x = torch.ones((len(node_map), 1), dtype=torch.float)
    logging.info("Loaded %d nodes in %.1fs", len(node_map), time.time() - stage_started)

    stage_started = time.time()
    cc_df = pd.read_csv(os.path.join(base_path, "connected_components.csv"))
    cc_df["ccId"] = cc_df["ccId"].astype(str)
    label_col = "ccLabel" if "ccLabel" in cc_df.columns else "label"
    labels = sorted(cc_df[label_col].astype(str).dropna().unique().tolist())
    label_map = {label: i for i, label in enumerate(labels)}
    if "ccStartTime" in cc_df.columns:
        cc_df["ccStartTime"] = pd.to_datetime(cc_df["ccStartTime"], errors="coerce")
        cc_df = cc_df.sort_values(["ccStartTime", "ccId"]).reset_index(drop=True)
        valid = cc_df["ccStartTime"].notna()
        cc_df["time"] = 0
        cc_df.loc[valid, "time"] = (cc_df.loc[valid, "ccStartTime"].astype("int64") // 10**9).astype(np.int64)
    logging.info("Loaded %d connected components in %.1fs", len(cc_df), time.time() - stage_started)

    stage_started = time.time()
    nodes_df = pd.read_csv(os.path.join(base_path, "nodes.csv"))
    nodes_df["ccId"] = nodes_df["ccId"].astype(str)
    nodes_df["clId"] = nodes_df["clId"].astype(str)
    grouped_nodes = nodes_df.groupby("ccId", sort=False)["clId"].apply(list).to_dict()
    logging.info("Loaded component node membership in %.1fs", time.time() - stage_started)

    component_edges_path = os.path.join(base_path, "component_edges.csv")
    if not os.path.exists(component_edges_path):
        raise FileNotFoundError(
            f"{component_edges_path} is required for the PNA baseline. "
            "Rebuild the dataset with the current builder."
        )
    stage_started = time.time()
    component_edges = pd.read_csv(component_edges_path)
    component_edges["ccId"] = component_edges["ccId"].astype(str)
    component_edges["clId1"] = component_edges["clId1"].astype(str)
    component_edges["clId2"] = component_edges["clId2"].astype(str)
    edge_feat_cols = [c for c in component_edges.columns if c.startswith("feat")]
    use_edge_features = bool(cfg.get("use_edge_features", True))
    selected_edge_cols = edge_feat_cols if use_edge_features else []
    component_edges["src_global"] = component_edges["clId1"].map(node_map)
    component_edges["dst_global"] = component_edges["clId2"].map(node_map)
    component_edges = component_edges.dropna(subset=["src_global", "dst_global"])
    component_edges["src_global"] = component_edges["src_global"].astype(np.int64)
    component_edges["dst_global"] = component_edges["dst_global"].astype(np.int64)

    grouped_edges = {}
    edge_group_cols = ["src_global", "dst_global"] + selected_edge_cols
    for cc_id, group in component_edges.groupby("ccId", sort=False):
        src_global = group["src_global"].to_numpy(dtype=np.int64, copy=False)
        dst_global = group["dst_global"].to_numpy(dtype=np.int64, copy=False)
        attr = None
        if selected_edge_cols:
            attr = group[selected_edge_cols].to_numpy(dtype=np.float32, copy=True)
        grouped_edges[cc_id] = (src_global, dst_global, attr)
    logging.info(
        "Loaded and grouped %d component edges across %d components in %.1fs",
        len(component_edges),
        len(grouped_edges),
        time.time() - stage_started,
    )
    add_reverse_edges = bool(cfg.get("add_reverse_edges", False))
    add_degree_feature = bool(cfg.get("add_degree_feature", False))

    stage_started = time.time()
    subgraphs: List[Data] = []
    for i, (_, row) in enumerate(cc_df.iterrows(), start=1):
        cc_id = row["ccId"]
        raw_nodes = grouped_nodes.get(cc_id)
        if not raw_nodes:
            continue
        global_nodes = [node_map[n] for n in raw_nodes if n in node_map]
        if not global_nodes:
            continue

        local = {global_id: i for i, global_id in enumerate(global_nodes)}
        x = global_x[torch.tensor(global_nodes, dtype=torch.long)].clone()

        edge_group = grouped_edges.get(cc_id)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(selected_edge_cols)), dtype=torch.float) if selected_edge_cols else None
        if edge_group is not None and len(edge_group[0]) > 0:
            src_global, dst_global, attr = edge_group
            src_np = np.fromiter((local.get(int(v), -1) for v in src_global), dtype=np.int64, count=len(src_global))
            dst_np = np.fromiter((local.get(int(v), -1) for v in dst_global), dtype=np.int64, count=len(dst_global))
            valid_local = (src_np >= 0) & (dst_np >= 0)
            src_np = src_np[valid_local]
            dst_np = dst_np[valid_local]
            edge_index = torch.tensor(np.vstack((src_np, dst_np)), dtype=torch.long)
            if selected_edge_cols and attr is not None:
                edge_attr = torch.tensor(attr[valid_local], dtype=torch.float)

        if add_reverse_edges and edge_index.numel() > 0:
            edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr, reduce="add")

        if add_degree_feature:
            if edge_index.numel() == 0:
                deg = torch.zeros(x.size(0), 1)
            else:
                deg = degree(edge_index[1], num_nodes=x.size(0)).view(-1, 1)
            x = torch.cat([x, torch.log1p(deg)], dim=-1)

        raw_label = torch.tensor([label_map[str(row[label_col])]], dtype=torch.long)
        data = Data(
            x=x.float(),
            edge_index=edge_index,
            edge_attr=edge_attr.float() if edge_attr is not None else None,
            y=raw_label.clone(),
            raw_y=raw_label.clone(),
            num_nodes=x.size(0),
        )
        data.cc_id = cc_id
        if "time" in cc_df.columns:
            data.preset_time = int(row["time"])
        subgraphs.append(data)
        if i % 5000 == 0:
            logging.info("Built %d/%d PNA component graphs...", i, len(cc_df))

    if not subgraphs:
        raise ValueError(f"No subgraphs could be loaded from {base_path}")

    illicit_label = cfg.get("illicit_label", None)
    if illicit_label is None:
        illicit_label_id = 0
    elif isinstance(illicit_label, str) and not illicit_label.isdigit():
        illicit_label_id = label_map[illicit_label]
    else:
        illicit_label_id = int(illicit_label)

    for data in subgraphs:
        data.y = (data.raw_y.view(-1) == illicit_label_id).float()

    train, val, test = _split_subgraphs(
        subgraphs,
        train_ratio=float(cfg.get("train_ratio", 0.7)),
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        split_mode=str(cfg.get("split_mode", "temporal")),
    )

    edge_norm = str(cfg.get("edge_feature_normalize", "zscore")).lower()
    if use_edge_features and edge_norm in {"zscore", "standard", "standardize"}:
        logging.info("Normalizing edge features...")
        _zscore_edge_attrs(
            train,
            val,
            test,
            eps=float(cfg.get("edge_feature_norm_eps", 1e-6)),
            clip=cfg.get("edge_feature_clip", None),
        )

    max_degree_hist = cfg.get("max_degree_hist", None)
    deg_hist = _build_degree_histogram(train, max_degree=max_degree_hist)
    num_edge_features = 0
    for data in subgraphs:
        if getattr(data, "edge_attr", None) is not None:
            num_edge_features = int(data.edge_attr.size(-1))
            break

    loaded = LoadedData(
        train=train,
        val=val,
        test=test,
        num_node_features=int(subgraphs[0].x.size(-1)),
        num_edge_features=num_edge_features,
        deg=deg_hist,
        label_map=label_map,
    )

    logging.info(
        "Loaded %d/%d/%d train/val/test graphs | node_dim=%d edge_dim=%d labels=%s",
        len(train),
        len(val),
        len(test),
        int(subgraphs[0].x.size(-1)),
        num_edge_features,
        label_map,
    )
    logging.info("Finished PNA dataset build in %.1fs", time.time() - started_total)
    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(loaded, cache_path)
        logging.info("Saved cached PNA dataset to %s", cache_path)
    return loaded
