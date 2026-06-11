#!/usr/bin/env python3
"""
Build an AML-style usable subgraph dataset from SAML-D transactions.

Output format matches the local AML/Elliptic-compatible layout:
- background_nodes.csv     (clId, feat*)
- background_edges.csv     (clId1, clId2, feat*)
- nodes.csv                (ccId, clId)
- component_edges.csv      (ccId, clId1, clId2, feat*)
- connected_components.csv (ccId, ccLabel, feat*)
- metadata.json

Definitions mirror build_aml_subgraphs.py:
- A node is illicit if it appears in >=1 transaction with Is_laundering == 1.
- Illicit subgraphs are connected components in the illicit-only induced graph
  using undirected connectivity over edges between illicit nodes.
- Optional licit subgraphs are connected samples from the licit-only graph.
- By default, no node features are emitted because SAML-D has no native account
  metadata and transaction-derived account aggregates can leak future activity.
- Is_laundering is used only to form binary illicit/licit labels and is not
  emitted as a feature.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class BuildConfig:
    input_csv: Path
    output_dir: Path
    chunksize: int = 1_000_000
    include_singletons: bool = True
    normalize_features: bool = True
    add_licit_subgraphs: bool = True
    licit_ratio: float = 1.0
    licit_seed: int = 42
    licit_bfs_prob: float = 0.5
    max_component_size: int = 0
    dataset_name: str = "SAML-D"
    node_feature_mode: str = "none"


def _normalize_str_col(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _to_account_id(account_series: pd.Series) -> pd.Series:
    return _normalize_str_col(account_series)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _z_norm(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        mu = float(x.mean())
        sigma = float(x.std(ddof=0))
        if sigma > 0:
            df[c] = (x - mu) / sigma
        else:
            df[c] = 0.0
    return df


def _parse_ts(chunk: pd.DataFrame) -> pd.Series:
    date = _normalize_str_col(chunk["Date"])
    time = _normalize_str_col(chunk["Time"])
    return pd.to_datetime(
        date + " " + time,
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S",
    )


def _read_transaction_chunks(csv_path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    required_cols = [
        "Time",
        "Date",
        "Sender_account",
        "Receiver_account",
        "Amount",
        "Payment_currency",
        "Received_currency",
        "Sender_bank_location",
        "Receiver_bank_location",
        "Payment_type",
        "Is_laundering",
    ]

    for chunk in pd.read_csv(
        csv_path,
        usecols=required_cols,
        chunksize=chunksize,
        low_memory=False,
    ):
        yield chunk


def _pass1_find_illicit_nodes_and_maps(
    csv_path: Path,
    chunksize: int,
) -> Tuple[Set[str], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], float]:
    illicit_nodes: Set[str] = set()
    currencies: Set[str] = set()
    payment_types: Set[str] = set()
    sender_locations: Set[str] = set()
    receiver_locations: Set[str] = set()
    min_ts: pd.Timestamp | None = None

    print(f"[pass1] scanning SAML-D transactions: {csv_path}", flush=True)
    for i, chunk in enumerate(_read_transaction_chunks(csv_path, chunksize=chunksize)):
        if i % 5 == 0 and i > 0:
            print(f"  [pass1] Processed {i * chunksize} rows...", flush=True)

        recv_curr = _normalize_str_col(chunk["Received_currency"])
        pay_curr = _normalize_str_col(chunk["Payment_currency"])
        pay_type = _normalize_str_col(chunk["Payment_type"])
        sender_loc = _normalize_str_col(chunk["Sender_bank_location"])
        receiver_loc = _normalize_str_col(chunk["Receiver_bank_location"])

        currencies.update(recv_curr[recv_curr != ""].unique().tolist())
        currencies.update(pay_curr[pay_curr != ""].unique().tolist())
        payment_types.update(pay_type[pay_type != ""].unique().tolist())
        sender_locations.update(sender_loc[sender_loc != ""].unique().tolist())
        receiver_locations.update(receiver_loc[receiver_loc != ""].unique().tolist())

        src = _to_account_id(chunk["Sender_account"])
        dst = _to_account_id(chunk["Receiver_account"])
        laundering = _safe_numeric(chunk["Is_laundering"]).astype(np.int8)
        mask = laundering == 1
        if mask.any():
            illicit_nodes.update(src[mask].unique().tolist())
            illicit_nodes.update(dst[mask].unique().tolist())

        ts = _parse_ts(chunk)
        cur_min = ts.min()
        if pd.notna(cur_min):
            min_ts = cur_min if min_ts is None else min(min_ts, cur_min)

    if min_ts is None:
        raise RuntimeError("Could not parse Date/Time timestamps from SAML-D CSV.")

    start_time = datetime(min_ts.year, min_ts.month, min_ts.day)
    anchor_ts = start_time.timestamp() - 10.0

    return (
        illicit_nodes,
        {cur: i for i, cur in enumerate(sorted(currencies))},
        {pt: i for i, pt in enumerate(sorted(payment_types))},
        {loc: i for i, loc in enumerate(sorted(sender_locations))},
        {loc: i for i, loc in enumerate(sorted(receiver_locations))},
        anchor_ts,
    )


def _accumulate_node_features(
    node_ids: pd.Series,
    amounts: pd.Series,
    timestamps: pd.Series,
    count_acc: Dict[str, float],
    amount_acc: Dict[str, float],
    first_ts_acc: Dict[str, pd.Timestamp],
    last_ts_acc: Dict[str, pd.Timestamp],
) -> None:
    grouped = (
        pd.DataFrame({"node": node_ids, "amount": amounts, "ts": timestamps})
        .groupby("node", sort=False)
        .agg(
            cnt=("amount", "size"),
            amt=("amount", "sum"),
            first=("ts", "min"),
            last=("ts", "max"),
        )
    )

    for n, c in grouped["cnt"].items():
        count_acc[n] += float(c)
    for n, a in grouped["amt"].items():
        amount_acc[n] += float(a)

    for n, t in grouped["first"].dropna().items():
        prev = first_ts_acc.get(n)
        if prev is None or t < prev:
            first_ts_acc[n] = t
    for n, t in grouped["last"].dropna().items():
        prev = last_ts_acc.get(n)
        if prev is None or t > prev:
            last_ts_acc[n] = t


def _pass2_build_graph_and_features(
    csv_path: Path,
    illicit_nodes: Set[str],
    currency_to_id: Dict[str, int],
    payment_type_to_id: Dict[str, int],
    anchor_ts: float,
    chunksize: int,
    node_feature_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, nx.Graph, nx.Graph, Dict[str, pd.Timestamp], Dict[str, pd.Timestamp], int, int]:
    out_count: Dict[str, float] = defaultdict(float)
    in_count: Dict[str, float] = defaultdict(float)
    out_paid_sum: Dict[str, float] = defaultdict(float)
    in_recv_sum: Dict[str, float] = defaultdict(float)

    edge_clid1: List[str] = []
    edge_clid2: List[str] = []
    edge_feat0: List[float] = []
    edge_feat1: List[float] = []
    edge_feat2: List[int] = []
    edge_feat3: List[int] = []
    edge_feat4: List[float] = []
    edge_feat5: List[int] = []

    node_first_ts: Dict[str, pd.Timestamp] = {}
    node_last_ts: Dict[str, pd.Timestamp] = {}

    g_illicit = nx.Graph()
    g_illicit.add_nodes_from(illicit_nodes)
    g_licit = nx.Graph()

    illicit_lookup = illicit_nodes
    total_rows = 0
    kept_edge_rows = 0
    all_nodes_seen: Set[str] = set()

    print(f"[pass2] building SAML-D graph/features: {csv_path}", flush=True)
    for i, chunk in enumerate(_read_transaction_chunks(csv_path, chunksize=chunksize)):
        if i % 5 == 0 and i > 0:
            print(f"  [pass2] Processed {i * chunksize} rows...", flush=True)

        total_rows += int(len(chunk))
        ts = _parse_ts(chunk)
        src = _to_account_id(chunk["Sender_account"])
        dst = _to_account_id(chunk["Receiver_account"])
        all_nodes_seen.update(src.unique().tolist())
        all_nodes_seen.update(dst.unique().tolist())
        amount = _safe_numeric(chunk["Amount"])
        recv_curr = (
            _normalize_str_col(chunk["Received_currency"])
            .map(currency_to_id)
            .fillna(-1)
            .astype(int)
        )
        pay_curr = (
            _normalize_str_col(chunk["Payment_currency"])
            .map(currency_to_id)
            .fillna(-1)
            .astype(int)
        )
        pay_type = (
            _normalize_str_col(chunk["Payment_type"])
            .map(payment_type_to_id)
            .fillna(-1)
            .astype(int)
        )

        src_illicit = src.isin(illicit_lookup).to_numpy(dtype=bool)
        dst_illicit = dst.isin(illicit_lookup).to_numpy(dtype=bool)
        src_licit = ~src_illicit
        dst_licit = ~dst_illicit

        _accumulate_node_features(
            node_ids=src,
            amounts=amount,
            timestamps=ts,
            count_acc=out_count,
            amount_acc=out_paid_sum,
            first_ts_acc=node_first_ts,
            last_ts_acc=node_last_ts,
        )
        _accumulate_node_features(
            node_ids=dst,
            amounts=amount,
            timestamps=ts,
            count_acc=in_count,
            amount_acc=in_recv_sum,
            first_ts_acc=node_first_ts,
            last_ts_acc=node_last_ts,
        )

        keep_mask = (src_illicit & dst_illicit) | (src_licit & dst_licit)
        if keep_mask.any():
            kept_edge_rows += int(keep_mask.sum())
            ks = src[keep_mask].to_numpy()
            kd = dst[keep_mask].to_numpy()
            kamt = amount[keep_mask].to_numpy(dtype=float)
            krc = recv_curr[keep_mask].to_numpy(dtype=int)
            kpc = pay_curr[keep_mask].to_numpy(dtype=int)
            kpt = pay_type[keep_mask].to_numpy(dtype=int)

            kt = ts[keep_mask]
            ts_rel = np.zeros(len(kt), dtype=float)
            valid_ts = kt.notna().to_numpy(dtype=bool)
            if valid_ts.any():
                valid_epoch = kt[valid_ts].astype("int64").to_numpy(dtype=np.int64) / 1_000_000_000.0
                ts_rel[valid_ts] = valid_epoch - anchor_ts

            edge_clid1.extend(ks.tolist())
            edge_clid2.extend(kd.tolist())
            edge_feat0.extend(ts_rel.tolist())
            edge_feat1.extend(kamt.tolist())
            edge_feat2.extend(krc.tolist())
            edge_feat3.extend(kpt.tolist())
            edge_feat4.extend(kamt.tolist())
            edge_feat5.extend(kpc.tolist())

            illicit_pair_mask = src_illicit[keep_mask] & dst_illicit[keep_mask]
            licit_pair_mask = src_licit[keep_mask] & dst_licit[keep_mask]
            if illicit_pair_mask.any():
                g_illicit.add_edges_from(zip(ks[illicit_pair_mask].tolist(), kd[illicit_pair_mask].tolist()))
            if licit_pair_mask.any():
                g_licit.add_edges_from(zip(ks[licit_pair_mask].tolist(), kd[licit_pair_mask].tolist()))

    node_rows = []
    all_feature_nodes = all_nodes_seen | set(out_count.keys()) | set(in_count.keys())
    for n in sorted(all_feature_nodes):
        row = {"clId": n}
        if node_feature_mode == "activity":
            row.update(
                {
                    "feat0": out_count.get(n, 0.0),
                    "feat1": in_count.get(n, 0.0),
                    "feat2": out_paid_sum.get(n, 0.0),
                    "feat3": in_recv_sum.get(n, 0.0),
                }
            )
        elif node_feature_mode == "constant":
            row["feat0"] = 1.0
        elif node_feature_mode != "none":
            raise ValueError(f"Unknown node_feature_mode: {node_feature_mode}")
        node_rows.append(row)
    nodes_df = pd.DataFrame(node_rows)

    edges_df = pd.DataFrame(
        {
            "clId1": edge_clid1,
            "clId2": edge_clid2,
            "feat0": edge_feat0,
            "feat1": edge_feat1,
            "feat2": edge_feat2,
            "feat3": edge_feat3,
            "feat4": edge_feat4,
            "feat5": edge_feat5,
        }
    )

    return nodes_df, edges_df, g_illicit, g_licit, node_first_ts, node_last_ts, total_rows, kept_edge_rows


def _split_component_by_bfs(g: nx.Graph, comp: Set[str], max_size: int) -> List[Set[str]]:
    if max_size <= 0 or len(comp) <= max_size:
        return [comp]

    chunks: List[Set[str]] = []
    remaining = set(comp)
    while remaining:
        seed = next(iter(remaining))
        selected: Set[str] = {seed}
        queue = deque([seed])
        remaining.remove(seed)

        while queue and len(selected) < max_size:
            cur = queue.popleft()
            for nbr in g.neighbors(cur):
                if nbr not in remaining:
                    continue
                remaining.remove(nbr)
                selected.add(nbr)
                queue.append(nbr)
                if len(selected) >= max_size:
                    break

        chunks.append(selected)

    return chunks


def _connected_components_sets(g: nx.Graph, include_singletons: bool) -> List[Set[str]]:
    comps = [set(c) for c in nx.connected_components(g)]
    if not include_singletons:
        comps = [c for c in comps if len(c) >= 2]
    return comps


def _build_subgraph_tables(
    g: nx.Graph,
    node_first_ts: Dict[str, pd.Timestamp],
    node_last_ts: Dict[str, pd.Timestamp],
    include_singletons: bool,
    max_component_size: int,
    start_idx: int = 0,
    label: str = "illicit_component",
    cc_source: str = "component",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    components = _connected_components_sets(g, include_singletons=include_singletons)
    if max_component_size > 0:
        split_components: List[Set[str]] = []
        n_split_source = 0
        for comp in components:
            if len(comp) > max_component_size:
                n_split_source += 1
                split_components.extend(_split_component_by_bfs(g, comp, max_size=max_component_size))
            else:
                split_components.append(comp)
        components = split_components
        if not include_singletons:
            components = [c for c in components if len(c) >= 2]
        if n_split_source > 0:
            print(
                f"Split {n_split_source} oversized {label} component(s) into "
                f"{len(components)} total components (max_size={max_component_size}).",
                flush=True,
            )

    cc_rows = []
    node_cc_rows = []
    for i, comp in enumerate(components, start=start_idx):
        cc_id = f"SAMLCC_{i}"
        subg = g.subgraph(comp)
        first_ts_candidates = [node_first_ts[n] for n in comp if n in node_first_ts]
        last_ts_candidates = [node_last_ts[n] for n in comp if n in node_last_ts]
        cc_start_ts = min(first_ts_candidates) if len(first_ts_candidates) > 0 else pd.NaT
        cc_end_ts = max(last_ts_candidates) if len(last_ts_candidates) > 0 else pd.NaT

        cc_rows.append(
            {
                "ccId": cc_id,
                "ccLabel": label,
                "feat0": float(subg.number_of_nodes()),
                "feat1": float(subg.number_of_edges()),
                "ccStartTime": cc_start_ts.isoformat() if pd.notna(cc_start_ts) else "",
                "ccEndTime": cc_end_ts.isoformat() if pd.notna(cc_end_ts) else "",
                "ccSource": cc_source,
            }
        )
        for n in sorted(comp):
            node_cc_rows.append({"ccId": cc_id, "clId": n})

    return (
        pd.DataFrame(node_cc_rows, columns=["ccId", "clId"]),
        pd.DataFrame(
            cc_rows,
            columns=["ccId", "ccLabel", "feat0", "feat1", "ccStartTime", "ccEndTime", "ccSource"],
        ),
    )


def _sample_connected_subgraph_hybrid(
    g: nx.Graph,
    available_nodes: Set[str],
    available_nodes_pool: List[str],
    target_size: int,
    bfs_prob: float,
    rng: np.random.Generator,
) -> Set[str]:
    if target_size <= 0 or len(available_nodes) == 0:
        return set()

    seed: str | None = None
    while len(available_nodes_pool) > 0:
        idx = int(rng.integers(len(available_nodes_pool)))
        cand = available_nodes_pool[idx]
        if cand in available_nodes:
            seed = cand
            break
        available_nodes_pool[idx] = available_nodes_pool[-1]
        available_nodes_pool.pop()
    if seed is None:
        return set()

    selected: Set[str] = {seed}
    frontier = deque([seed])

    while frontier and len(selected) < target_size:
        cur = frontier.popleft() if rng.random() < bfs_prob else frontier.pop()
        nbrs = [
            n
            for n in g.neighbors(cur)
            if (n in available_nodes) and (n not in selected)
        ]
        if len(nbrs) == 0:
            continue
        rng.shuffle(nbrs)
        for n in nbrs:
            selected.add(n)
            frontier.append(n)
            if len(selected) >= target_size:
                break

    return selected


def _build_licit_subgraph_tables(
    g_licit: nx.Graph,
    target_sizes: List[int],
    node_first_ts: Dict[str, pd.Timestamp],
    node_last_ts: Dict[str, pd.Timestamp],
    start_idx: int,
    include_singletons: bool,
    bfs_prob: float,
    seed: int,
    max_component_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    rng = np.random.default_rng(seed)

    if g_licit.number_of_nodes() == 0 or len(target_sizes) == 0:
        return (
            pd.DataFrame(columns=["ccId", "clId"]),
            pd.DataFrame(columns=["ccId", "ccLabel", "feat0", "feat1", "ccStartTime", "ccEndTime", "ccSource"]),
            start_idx,
        )

    available_nodes: Set[str] = set(g_licit.nodes())
    available_nodes_pool: List[str] = list(available_nodes)
    cc_rows = []
    node_cc_rows = []
    cc_idx = start_idx
    target_idx = 0
    attempts = 0
    skipped_too_small = 0
    min_size = 1 if include_singletons else 2
    max_attempts = max(len(target_sizes) * 20, len(target_sizes) + g_licit.number_of_nodes())
    next_progress = 5000

    while target_idx < len(target_sizes):
        if len(available_nodes) == 0:
            break
        attempts += 1
        if len(cc_rows) >= next_progress:
            print(
                f"  [licit] sampled {len(cc_rows)}/{len(target_sizes)} subgraphs; "
                f"attempts={attempts}; available_nodes={len(available_nodes)}",
                flush=True,
            )
            next_progress += 5000
        if attempts > max_attempts:
            break

        tsize = target_sizes[target_idx]
        tsize = max(1 if include_singletons else 2, int(tsize))
        if max_component_size > 0:
            tsize = min(tsize, max_component_size)
        sampled = _sample_connected_subgraph_hybrid(
            g_licit,
            available_nodes=available_nodes,
            available_nodes_pool=available_nodes_pool,
            target_size=tsize,
            bfs_prob=bfs_prob,
            rng=rng,
        )

        if len(sampled) < min_size:
            skipped_too_small += 1
            if len(sampled) > 0:
                available_nodes.difference_update(sampled)
                if len(available_nodes_pool) > 2 * max(len(available_nodes), 1):
                    available_nodes_pool = list(available_nodes)
            continue

        cc_id = f"SAMLCC_{cc_idx}"
        cc_idx += 1
        subg = g_licit.subgraph(sampled)
        first_ts_candidates = [node_first_ts[n] for n in sampled if n in node_first_ts]
        last_ts_candidates = [node_last_ts[n] for n in sampled if n in node_last_ts]
        cc_start_ts = min(first_ts_candidates) if len(first_ts_candidates) > 0 else pd.NaT
        cc_end_ts = max(last_ts_candidates) if len(last_ts_candidates) > 0 else pd.NaT

        cc_rows.append(
            {
                "ccId": cc_id,
                "ccLabel": "licit_component",
                "feat0": float(subg.number_of_nodes()),
                "feat1": float(subg.number_of_edges()),
                "ccStartTime": cc_start_ts.isoformat() if pd.notna(cc_start_ts) else "",
                "ccEndTime": cc_end_ts.isoformat() if pd.notna(cc_end_ts) else "",
                "ccSource": "licit_sample",
            }
        )
        for n in sorted(sampled):
            node_cc_rows.append({"ccId": cc_id, "clId": n})

        available_nodes.difference_update(sampled)
        if len(available_nodes_pool) > 2 * max(len(available_nodes), 1):
            available_nodes_pool = list(available_nodes)
        target_idx += 1

    if len(cc_rows) < len(target_sizes):
        raise RuntimeError(
            "Could not sample requested licit subgraph count. "
            f"requested={len(target_sizes)}, sampled={len(cc_rows)}, attempts={attempts}, "
            f"skipped_too_small={skipped_too_small}, available_nodes={len(available_nodes)}, "
            f"include_singletons={include_singletons}. "
            "Try lowering LICIT_RATIO, enabling singletons, or allowing overlapping licit samples."
        )

    return (
        pd.DataFrame(node_cc_rows, columns=["ccId", "clId"]),
        pd.DataFrame(
            cc_rows,
            columns=["ccId", "ccLabel", "feat0", "feat1", "ccStartTime", "ccEndTime", "ccSource"],
        ),
        cc_idx,
    )


def _build_component_edges_df(edges_df: pd.DataFrame, nodes_cc_df: pd.DataFrame) -> pd.DataFrame:
    edge_columns = list(edges_df.columns)
    out_columns = ["ccId"] + edge_columns
    if edges_df.empty or nodes_cc_df.empty:
        return pd.DataFrame(columns=out_columns)

    membership = nodes_cc_df[["ccId", "clId"]].drop_duplicates()
    if not membership["clId"].duplicated().any():
        node_to_cc = dict(zip(membership["clId"], membership["ccId"]))
        cc_src = edges_df["clId1"].map(node_to_cc)
        cc_dst = edges_df["clId2"].map(node_to_cc)
        same_cc_mask = cc_src.notna() & (cc_src == cc_dst)
        if not same_cc_mask.any():
            return pd.DataFrame(columns=out_columns)
        component_edges_df = edges_df.loc[same_cc_mask].copy()
        component_edges_df.insert(0, "ccId", cc_src.loc[same_cc_mask].to_numpy())
        return component_edges_df.reset_index(drop=True)

    src_membership = membership.rename(columns={"clId": "clId1"})
    dst_membership = membership.rename(columns={"clId": "clId2"})
    component_edges_df = (
        edges_df.merge(src_membership, on="clId1", how="inner")
        .merge(dst_membership, on=["ccId", "clId2"], how="inner")
    )
    return component_edges_df[out_columns].reset_index(drop=True)


def _size_summary(cc_df: pd.DataFrame, label: str) -> dict[str, float | int]:
    if len(cc_df) == 0 or "ccLabel" not in cc_df.columns:
        return {"min": 0, "max": 0, "avg": 0.0, "median": 0.0}
    sizes = pd.to_numeric(
        cc_df.loc[cc_df["ccLabel"] == label, "feat0"],
        errors="coerce",
    ).dropna()
    if sizes.empty:
        return {"min": 0, "max": 0, "avg": 0.0, "median": 0.0}
    return {
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "avg": float(sizes.mean()),
        "median": float(sizes.median()),
    }


def _node_feature_description(node_feature_mode: str) -> dict[str, str]:
    if node_feature_mode == "activity":
        return {
            "clId": "SAML-D account identifier",
            "feat0": "outgoing_tx_count (zscore if enabled)",
            "feat1": "incoming_tx_count (zscore if enabled)",
            "feat2": "outgoing_amount_paid_sum from Amount (zscore if enabled)",
            "feat3": "incoming_amount_received_sum from Amount (zscore if enabled)",
        }
    if node_feature_mode == "constant":
        return {
            "clId": "SAML-D account identifier",
            "feat0": "constant 1.0 placeholder for featureless-node loaders",
        }
    if node_feature_mode == "none":
        return {
            "clId": "SAML-D account identifier",
        }
    raise ValueError(f"Unknown node_feature_mode: {node_feature_mode}")


def build_saml_d_subgraph_dataset(config: BuildConfig) -> None:
    input_csv = config.input_csv.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing SAML-D CSV: {input_csv}")

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        illicit_nodes,
        currency_to_id,
        payment_type_to_id,
        sender_location_to_id,
        receiver_location_to_id,
        anchor_ts,
    ) = _pass1_find_illicit_nodes_and_maps(input_csv, chunksize=config.chunksize)

    if not illicit_nodes:
        raise RuntimeError("No illicit nodes found (no Is_laundering == 1 transactions?).")

    print(f"Illicit nodes found: {len(illicit_nodes)}", flush=True)
    print(f"Currencies discovered: {len(currency_to_id)}", flush=True)
    print(f"Payment types discovered: {len(payment_type_to_id)}", flush=True)
    print(f"Sender locations discovered: {len(sender_location_to_id)}", flush=True)
    print(f"Receiver locations discovered: {len(receiver_location_to_id)}", flush=True)

    (
        nodes_df,
        edges_df,
        g_illicit,
        g_licit,
        node_first_ts,
        node_last_ts,
        total_transactions,
        kept_same_label_edges,
    ) = _pass2_build_graph_and_features(
        csv_path=input_csv,
        illicit_nodes=illicit_nodes,
        currency_to_id=currency_to_id,
        payment_type_to_id=payment_type_to_id,
        anchor_ts=anchor_ts,
        chunksize=config.chunksize,
        node_feature_mode=config.node_feature_mode,
    )

    nodes_cc_df, cc_df = _build_subgraph_tables(
        g_illicit,
        node_first_ts=node_first_ts,
        node_last_ts=node_last_ts,
        include_singletons=config.include_singletons,
        max_component_size=config.max_component_size,
        label="illicit_component",
        cc_source="illicit_component",
    )
    print(f"Illicit subgraphs: {len(cc_df)}", flush=True)
    print(f"Licit graph nodes available: {g_licit.number_of_nodes()}", flush=True)

    if config.add_licit_subgraphs and len(cc_df) > 0 and g_licit.number_of_nodes() > 0:
        illicit_sizes = cc_df["feat0"].astype(int).tolist()
        n_target = max(1, int(round(len(illicit_sizes) * max(config.licit_ratio, 0.0))))
        max_licit_samples = g_licit.number_of_nodes() if config.include_singletons else (g_licit.number_of_nodes() // 2)
        if n_target > max_licit_samples:
            print(
                f"Capping requested licit samples from {n_target} to {max_licit_samples} "
                "(node-availability bound).",
                flush=True,
            )
            n_target = max_licit_samples

        rng = np.random.default_rng(config.licit_seed)
        sampled_sizes = rng.choice(illicit_sizes, size=n_target, replace=True).tolist() if n_target > 0 else []
        print(
            f"Sampling licit subgraphs: target={n_target}, ratio={config.licit_ratio}, seed={config.licit_seed}",
            flush=True,
        )

        licit_nodes_cc_df, licit_cc_df, _ = _build_licit_subgraph_tables(
            g_licit,
            target_sizes=sampled_sizes,
            node_first_ts=node_first_ts,
            node_last_ts=node_last_ts,
            start_idx=len(cc_df),
            include_singletons=config.include_singletons,
            bfs_prob=config.licit_bfs_prob,
            seed=config.licit_seed,
            max_component_size=config.max_component_size,
        )
        if len(licit_cc_df) > 0:
            nodes_cc_df = pd.concat([nodes_cc_df, licit_nodes_cc_df], ignore_index=True)
            cc_df = pd.concat([cc_df, licit_cc_df], ignore_index=True)
            print(f"Added licit subgraphs: {len(licit_cc_df)}", flush=True)
        else:
            print("No licit subgraphs could be sampled with current settings.", flush=True)

    kept_nodes = set(nodes_cc_df["clId"].tolist())
    nodes_df = nodes_df[nodes_df["clId"].isin(kept_nodes)].reset_index(drop=True)
    if not edges_df.empty:
        edges_df = edges_df[
            edges_df["clId1"].isin(kept_nodes) & edges_df["clId2"].isin(kept_nodes)
        ].reset_index(drop=True)

    if config.normalize_features:
        if config.node_feature_mode == "activity":
            nodes_df = _z_norm(nodes_df, ["feat0", "feat1", "feat2", "feat3"])
        if not edges_df.empty:
            edges_df = _z_norm(edges_df, ["feat0", "feat1", "feat4"])

    component_edges_df = _build_component_edges_df(edges_df, nodes_cc_df)

    nodes_out = output_dir / "background_nodes.csv"
    edges_out = output_dir / "background_edges.csv"
    node_cc_out = output_dir / "nodes.csv"
    component_edges_out = output_dir / "component_edges.csv"
    cc_out = output_dir / "connected_components.csv"
    meta_out = output_dir / "metadata.json"

    nodes_df.to_csv(nodes_out, index=False)
    edges_df.to_csv(edges_out, index=False)
    nodes_cc_df.to_csv(node_cc_out, index=False)
    component_edges_df.to_csv(component_edges_out, index=False)
    cc_df.to_csv(cc_out, index=False)

    metadata = {
        "source_input_csv": str(input_csv),
        "dataset_name": config.dataset_name,
        "resolved_output_dir": str(output_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_transaction_files": 1,
        "num_transactions_read": int(total_transactions),
        "num_same_label_transaction_edges_before_component_filter": int(kept_same_label_edges),
        "num_illicit_nodes_before_component_filter": len(illicit_nodes),
        "num_nodes_kept": int(len(nodes_df)),
        "num_edges_kept": int(len(edges_df)),
        "num_component_edges": int(len(component_edges_df)),
        "num_subgraphs": int(len(cc_df)),
        "num_illicit_subgraphs": int((cc_df["ccLabel"] == "illicit_component").sum()) if len(cc_df) > 0 else 0,
        "num_licit_subgraphs": int((cc_df["ccLabel"] == "licit_component").sum()) if len(cc_df) > 0 else 0,
        "illicit_subgraph_sizes": _size_summary(cc_df, "illicit_component"),
        "licit_subgraph_sizes": _size_summary(cc_df, "licit_component"),
        "include_singletons": config.include_singletons,
        "normalize_features": config.normalize_features,
        "add_licit_subgraphs": config.add_licit_subgraphs,
        "licit_ratio": config.licit_ratio,
        "licit_seed": config.licit_seed,
        "licit_bfs_prob": config.licit_bfs_prob,
        "max_component_size": config.max_component_size,
        "node_feature_mode": config.node_feature_mode,
        "normalization_type": "zscore (continuous only)",
        "anchor_timestamp_epoch": anchor_ts,
        "currency_to_id": currency_to_id,
        "payment_type_to_id": payment_type_to_id,
        "sender_location_to_id": sender_location_to_id,
        "receiver_location_to_id": receiver_location_to_id,
        "edge_feature_description": {
            "feat0": "relative timestamp seconds from Date+Time (zscore if enabled)",
            "feat1": "amount_received proxy from Amount (zscore if enabled)",
            "feat2": "received_currency_id",
            "feat3": "payment_type_id",
            "feat4": "amount_paid proxy from Amount (zscore if enabled)",
            "feat5": "payment_currency_id",
        },
        "node_feature_description": _node_feature_description(config.node_feature_mode),
        "subgraph_feature_description": {
            "feat0": "num_nodes",
            "feat1": "num_internal_edges",
            "ccStartTime": "earliest transaction timestamp among nodes in component",
            "ccEndTime": "latest transaction timestamp among nodes in component",
            "ccSource": "source used to create component",
        },
        "component_edges_file": {
            "path": "component_edges.csv",
            "description": "Precomputed internal edges per connected component for faster loading.",
        },
        "notes": [
            "SAML-D has only transaction rows; no separate account metadata is used.",
            "Node illicit/licit flag is not included as a node feature.",
            "Default node_feature_mode=none emits no node features; transaction information lives on edges.",
            "Is_laundering is not included as an edge feature.",
            "Illicit subgraphs are connected components of the illicit-only graph.",
            "Licit subgraphs are sampled as connected sets from the licit-only graph using hybrid BFS/DFS.",
            "Transaction-level edges are kept to preserve temporal information.",
            "Sender/receiver bank locations are mapped in metadata but omitted from features to keep the AML edge feature contract.",
            "SAML-D has one Amount column, used for both AML-style amount_received and amount_paid feature slots.",
        ],
    }
    with meta_out.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print("Done.", flush=True)
    print(f"Saved: {nodes_out}", flush=True)
    print(f"Saved: {edges_out}", flush=True)
    print(f"Saved: {node_cc_out}", flush=True)
    print(f"Saved: {component_edges_out}", flush=True)
    print(f"Saved: {cc_out}", flush=True)
    print(f"Saved: {meta_out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an AML-style usable subgraph dataset from SAML-D transactions."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/home/ghonkoop/data/saml-d/SAML-D.csv"),
        help="Path to SAML-D.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ghonkoop/data/saml-d/usable"),
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per chunk while streaming the SAML-D transaction CSV.",
    )
    parser.add_argument(
        "--drop-singletons",
        action="store_true",
        help="Drop single illicit nodes (default keeps them to preserve maximum data).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable z-score normalization of continuous features.",
    )
    parser.add_argument(
        "--no-licit-subgraphs",
        action="store_true",
        help="Disable stage that samples licit subgraphs.",
    )
    parser.add_argument(
        "--licit-ratio",
        type=float,
        default=1.0,
        help="Target #licit subgraphs as ratio of #illicit subgraphs.",
    )
    parser.add_argument(
        "--licit-seed",
        type=int,
        default=42,
        help="Random seed for licit subgraph sampling.",
    )
    parser.add_argument(
        "--licit-bfs-prob",
        type=float,
        default=0.5,
        help="Probability of BFS step in hybrid BFS/DFS licit sampler (0..1).",
    )
    parser.add_argument(
        "--max-component-size",
        type=int,
        default=0,
        help="Split connected components larger than this many nodes into connected chunks; 0 disables splitting.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SAML-D",
        help="Name to store in metadata.json.",
    )
    parser.add_argument(
        "--node-feature-mode",
        choices=["none", "constant", "activity"],
        default="none",
        help=(
            "Node features to emit. 'none' writes only clId because SAML-D has no native "
            "account features; 'constant' writes feat0=1.0; 'activity' writes full-dataset "
            "transaction-derived aggregates for ablations."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = BuildConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        chunksize=int(args.chunksize),
        include_singletons=not args.drop_singletons,
        normalize_features=not args.no_normalize,
        add_licit_subgraphs=not args.no_licit_subgraphs,
        licit_ratio=float(args.licit_ratio),
        licit_seed=int(args.licit_seed),
        licit_bfs_prob=min(1.0, max(0.0, float(args.licit_bfs_prob))),
        max_component_size=max(0, int(args.max_component_size)),
        dataset_name=str(args.dataset_name),
        node_feature_mode=str(args.node_feature_mode),
    )
    build_saml_d_subgraph_dataset(cfg)
