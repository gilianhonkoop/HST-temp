#!/usr/bin/env python3
"""
Build an illicit-node subgraph dataset from IBM AML transaction CSVs.

Output format intentionally matches the CSV layout used by the Elliptic loader:
- background_nodes.csv     (clId, feat*)
- background_edges.csv     (clId1, clId2, feat*)
- nodes.csv                (ccId, clId)
- component_edges.csv      (ccId, clId1, clId2, feat*)
- connected_components.csv (ccId, ccLabel, feat*)
- metadata.json

Definition used:
- A node is illicit if it appears in >=1 transaction with Is Laundering == 1.
- Subgraphs are connected components in the illicit-only induced graph
  (undirected connectivity over edges between illicit nodes).

Important:
- 'Is Laundering' is NOT stored as an edge feature in output edges.
- No node-level illicit/licit indicator is stored.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from collections import defaultdict, deque

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class BuildConfig:
    input_dir: Path
    output_dir: Path
    dataset_name: str = ""
    chunksize: int = 1_000_000
    include_singletons: bool = True
    normalize_features: bool = True
    add_licit_subgraphs: bool = True
    licit_ratio: float = 1.0
    licit_seed: int = 42
    licit_bfs_prob: float = 0.5
    max_component_size: int = 0
    illicit_subgraph_mode: str = "components"


@dataclass
class PatternAttempt:
    pattern_type: str
    nodes: Set[str]
    first_ts: pd.Timestamp | None
    last_ts: pd.Timestamp | None


def _find_files(root: Path, suffix: str) -> List[Path]:
    return sorted([p for p in root.rglob("*.csv") if p.name.endswith(suffix)])


def _find_transaction_files(root: Path, dataset_name: str = "") -> List[Path]:
    # Supports both naming conventions:
    # - <prefix>_Transactions.csv
    # - <prefix>_Trans.csv
    if dataset_name.strip() != "":
        candidates = [
            root / f"{dataset_name}_Transactions.csv",
            root / f"{dataset_name}_Trans.csv",
        ]
        return [p for p in candidates if p.exists()]

    files = _find_files(root, "_Transactions.csv") + _find_files(root, "_Trans.csv")
    # keep deterministic order + dedupe
    return sorted(list({str(p): p for p in files}.values()), key=lambda p: str(p))


def _find_pattern_files(root: Path, dataset_name: str = "") -> List[Path]:
    if dataset_name.strip() != "":
        candidates = [
            root / f"{dataset_name}_Patterns.txt",
            root / f"{dataset_name}_Pattern.txt",
        ]
        return [p for p in candidates if p.exists()]

    files = sorted(root.rglob("*_Patterns.txt")) + sorted(root.rglob("*_Pattern.txt"))
    return sorted(list({str(p): p for p in files}.values()), key=lambda p: str(p))


def _normalize_str_col(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _to_node_id(bank_series: pd.Series, account_series: pd.Series) -> pd.Series:
    # FraudGT-style: bank is part of account identifier.
    bank = _normalize_str_col(bank_series)
    acct = _normalize_str_col(account_series)
    return bank + "::" + acct


def _normalize_pattern_bank_id(value: str) -> str:
    bank = str(value).strip()
    if bank.isdigit():
        bank = bank.lstrip("0") or "0"
    return bank


def _pattern_node_id(bank: str, account: str) -> str:
    return _normalize_pattern_bank_id(bank) + "::" + str(account).strip()


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


def _parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="%Y/%m/%d %H:%M")


def _parse_pattern_files(pattern_files: List[Path]) -> List[PatternAttempt]:
    attempts: List[PatternAttempt] = []

    current_type: str | None = None
    current_nodes: Set[str] = set()
    current_first_ts: pd.Timestamp | None = None
    current_last_ts: pd.Timestamp | None = None

    def finish_current() -> None:
        nonlocal current_type, current_nodes, current_first_ts, current_last_ts
        if current_type is not None and len(current_nodes) > 0:
            attempts.append(
                PatternAttempt(
                    pattern_type=current_type,
                    nodes=set(current_nodes),
                    first_ts=current_first_ts,
                    last_ts=current_last_ts,
                )
            )
        current_type = None
        current_nodes = set()
        current_first_ts = None
        current_last_ts = None

    for pattern_file in pattern_files:
        print(f"[patterns] parsing laundering attempts: {pattern_file}", flush=True)
        with pattern_file.open("r", encoding="utf-8", newline="") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line == "":
                    continue
                if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                    finish_current()
                    if " - " in line:
                        current_type = line.split(" - ", 1)[1].split(":", 1)[0].strip()
                    else:
                        current_type = "UNKNOWN"
                    current_nodes = set()
                    current_first_ts = None
                    current_last_ts = None
                    continue
                if line.startswith("END LAUNDERING ATTEMPT"):
                    finish_current()
                    continue
                if current_type is None:
                    continue

                row = next(csv.reader([line]))
                if len(row) < 11:
                    continue
                src = _pattern_node_id(row[1], row[2])
                dst = _pattern_node_id(row[3], row[4])
                current_nodes.add(src)
                current_nodes.add(dst)

                ts = pd.to_datetime(row[0].strip(), errors="coerce", format="%Y/%m/%d %H:%M")
                if pd.notna(ts):
                    current_first_ts = ts if current_first_ts is None else min(current_first_ts, ts)
                    current_last_ts = ts if current_last_ts is None else max(current_last_ts, ts)

        finish_current()

    return attempts


def _read_transaction_chunks(csv_path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    # Read by index to avoid duplicate-name ambiguity for the two "Account" columns.
    # Expected schema index:
    # 0 Timestamp
    # 1 From Bank
    # 2 Account (from)
    # 3 To Bank
    # 4 Account (to)
    # 5 Amount Received
    # 6 Receiving Currency
    # 7 Amount Paid
    # 8 Payment Currency
    # 9 Payment Format
    # 10 Is Laundering
    usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    names = [
        "timestamp",
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "is_laundering",
    ]

    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        header=0,
        names=names,
        chunksize=chunksize,
        low_memory=False,
    ):
        yield chunk


def _pass1_find_illicit_nodes_and_currency_map(
    tx_files: List[Path], chunksize: int
) -> Tuple[Set[str], Dict[str, int], Dict[str, int], float]:
    illicit_nodes: Set[str] = set()
    currencies: Set[str] = set()
    payment_formats: Set[str] = set()
    min_ts: pd.Timestamp | None = None

    for tx_file in tx_files:
        print(f"[pass1] scanning laundering flags: {tx_file}", flush=True)
        for i, chunk in enumerate(_read_transaction_chunks(tx_file, chunksize=chunksize)):
            if i % 5 == 0 and i > 0:
                print(f"  [pass1] Processed {i * chunksize} rows...", flush=True)
            recv_curr = _normalize_str_col(chunk["receiving_currency"])
            pay_curr = _normalize_str_col(chunk["payment_currency"])
            pay_fmt = _normalize_str_col(chunk["payment_format"])
            currencies.update(recv_curr[recv_curr != ""].unique().tolist())
            currencies.update(pay_curr[pay_curr != ""].unique().tolist())
            payment_formats.update(pay_fmt[pay_fmt != ""].unique().tolist())

            src = _to_node_id(chunk["from_bank"], chunk["from_account"])
            dst = _to_node_id(chunk["to_bank"], chunk["to_account"])
            laundering = _safe_numeric(chunk["is_laundering"]).astype(np.int8)
            mask = laundering == 1
            if mask.any():
                illicit_nodes.update(src[mask].unique().tolist())
                illicit_nodes.update(dst[mask].unique().tolist())

            ts = _parse_ts(chunk["timestamp"])
            cur_min = ts.min()
            if pd.notna(cur_min):
                min_ts = cur_min if min_ts is None else min(min_ts, cur_min)

    currency_to_id = {cur: i for i, cur in enumerate(sorted(currencies))}
    payfmt_to_id = {pf: i for i, pf in enumerate(sorted(payment_formats))}

    if min_ts is None:
        raise RuntimeError("Could not parse timestamps from transaction files.")
    start_time = datetime(min_ts.year, min_ts.month, min_ts.day)
    anchor_ts = start_time.timestamp() - 10.0

    return illicit_nodes, currency_to_id, payfmt_to_id, anchor_ts


def _accumulate_node_features(
    node_ids: pd.Series,
    amounts: pd.Series,
    timestamps: pd.Series,
    count_acc: Dict[str, float],
    amount_acc: Dict[str, float],
    first_ts_acc: Dict[str, pd.Timestamp],
    last_ts_acc: Dict[str, pd.Timestamp],
) -> None:
    """
    Aggregate node-level stats per chunk in vectorized pandas code.
    This avoids per-row Python loops on large datasets.
    """
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
    tx_files: List[Path],
    illicit_nodes: Set[str],
    currency_to_id: Dict[str, int],
    payfmt_to_id: Dict[str, int],
    anchor_ts: float,
    chunksize: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, nx.Graph, nx.Graph, Dict[str, pd.Timestamp], Dict[str, pd.Timestamp]]:
    """
    Returns
    -------
    nodes_df : columns [clId, feat0..]
    edges_df : columns [clId1, clId2, feat0..]  (NO laundering feature)
    g_illicit_undirected : connectivity graph used for CC extraction
    """

    # Node feature accumulators (all nodes)
    out_count: Dict[str, float] = defaultdict(float)
    in_count: Dict[str, float] = defaultdict(float)
    out_paid_sum: Dict[str, float] = defaultdict(float)
    in_recv_sum: Dict[str, float] = defaultdict(float)

    # Keep transaction-level edges (FraudGT-like temporal handling).
    # Column-wise lists are much faster and lighter than per-row dict objects.
    edge_clid1: List[str] = []
    edge_clid2: List[str] = []
    edge_feat0: List[float] = []
    edge_feat1: List[float] = []
    edge_feat2: List[int] = []
    edge_feat3: List[int] = []
    edge_feat4: List[float] = []
    edge_feat5: List[int] = []

    # Per-node temporal span over all transactions touching illicit nodes
    node_first_ts: Dict[str, pd.Timestamp] = {}
    node_last_ts: Dict[str, pd.Timestamp] = {}

    g = nx.Graph()
    g.add_nodes_from(illicit_nodes)
    g_licit = nx.Graph()

    illicit_lookup = illicit_nodes

    for tx_file in tx_files:
        print(f"[pass2] building graph/features: {tx_file}", flush=True)
        for i, chunk in enumerate(_read_transaction_chunks(tx_file, chunksize=chunksize)):
            if i % 5 == 0 and i > 0:
                print(f"  [pass2] Processed {i * chunksize} rows...", flush=True)
            ts = _parse_ts(chunk["timestamp"])
            src = _to_node_id(chunk["from_bank"], chunk["from_account"])
            dst = _to_node_id(chunk["to_bank"], chunk["to_account"])
            amount_paid = _safe_numeric(chunk["amount_paid"])
            amount_recv = _safe_numeric(chunk["amount_received"])
            recv_curr = (
                chunk["receiving_currency"]
                .fillna("")
                .astype(str)
                .str.strip()
                .map(currency_to_id)
                .fillna(-1)
                .astype(int)
            )
            pay_fmt = (
                chunk["payment_format"]
                .fillna("")
                .astype(str)
                .str.strip()
                .map(payfmt_to_id)
                .fillna(-1)
                .astype(int)
            )
            pay_curr = (
                chunk["payment_currency"]
                .fillna("")
                .astype(str)
                .str.strip()
                .map(currency_to_id)
                .fillna(-1)
                .astype(int)
            )

            src_illicit = src.isin(illicit_lookup).to_numpy(dtype=bool)
            dst_illicit = dst.isin(illicit_lookup).to_numpy(dtype=bool)
            src_licit = ~src_illicit
            dst_licit = ~dst_illicit

            # Node features over all transactions (grouped per chunk to avoid row loops).
            _accumulate_node_features(
                node_ids=src,
                amounts=amount_paid,
                timestamps=ts,
                count_acc=out_count,
                amount_acc=out_paid_sum,
                first_ts_acc=node_first_ts,
                last_ts_acc=node_last_ts,
            )
            _accumulate_node_features(
                node_ids=dst,
                amounts=amount_recv,
                timestamps=ts,
                count_acc=in_count,
                amount_acc=in_recv_sum,
                first_ts_acc=node_first_ts,
                last_ts_acc=node_last_ts,
            )

            # Keep transaction-level edges for illicit<->illicit or licit<->licit
            keep_mask = (src_illicit & dst_illicit) | (src_licit & dst_licit)
            if keep_mask.any():
                ks = src[keep_mask].to_numpy()
                kd = dst[keep_mask].to_numpy()
                kar = amount_recv[keep_mask].to_numpy(dtype=float)
                kap = amount_paid[keep_mask].to_numpy(dtype=float)
                krc = recv_curr[keep_mask].to_numpy(dtype=int)
                kpc = pay_curr[keep_mask].to_numpy(dtype=int)
                kpf = pay_fmt[keep_mask].to_numpy(dtype=int)

                kt = ts[keep_mask]
                ts_rel = np.zeros(len(kt), dtype=float)
                valid_ts = kt.notna().to_numpy(dtype=bool)
                if valid_ts.any():
                    valid_epoch = kt[valid_ts].astype("int64").to_numpy(dtype=np.int64) / 1_000_000_000.0
                    ts_rel[valid_ts] = valid_epoch - anchor_ts

                edge_clid1.extend(ks.tolist())
                edge_clid2.extend(kd.tolist())
                edge_feat0.extend(ts_rel.tolist())
                edge_feat1.extend(kar.tolist())
                edge_feat2.extend(krc.tolist())
                edge_feat3.extend(kpf.tolist())
                edge_feat4.extend(kap.tolist())
                edge_feat5.extend(kpc.tolist())

                illicit_pair_mask = src_illicit[keep_mask] & dst_illicit[keep_mask]
                licit_pair_mask = src_licit[keep_mask] & dst_licit[keep_mask]
                if illicit_pair_mask.any():
                    g.add_edges_from(zip(ks[illicit_pair_mask].tolist(), kd[illicit_pair_mask].tolist()))
                if licit_pair_mask.any():
                    g_licit.add_edges_from(zip(ks[licit_pair_mask].tolist(), kd[licit_pair_mask].tolist()))

    # Build node table
    node_rows = []
    all_feature_nodes = set(out_count.keys()) | set(in_count.keys())
    for n in sorted(all_feature_nodes):
        node_rows.append(
            {
                "clId": n,
                "feat0": out_count.get(n, 0.0),
                "feat1": in_count.get(n, 0.0),
                "feat2": out_paid_sum.get(n, 0.0),
                "feat3": in_recv_sum.get(n, 0.0),
            }
        )
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

    return nodes_df, edges_df, g, g_licit, node_first_ts, node_last_ts


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
    # Lazy cleanup: keep a reusable pool list and discard stale entries only when sampled.
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
        return pd.DataFrame(columns=["ccId", "clId"]), pd.DataFrame(columns=["ccId", "ccLabel", "feat0", "feat1", "ccStartTime", "ccEndTime"]), start_idx

    # For licit negatives we only need a non-overlapping connected sample capped
    # by target_size; splitting the full licit graph first can be very expensive.
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

        cc_id = f"AMLCC_{cc_idx}"
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
            }
        )
        for n in sorted(sampled):
            node_cc_rows.append({"ccId": cc_id, "clId": n})

        # Avoid overlap across sampled licit subgraphs
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

    return pd.DataFrame(node_cc_rows), pd.DataFrame(cc_rows), cc_idx


def _build_subgraph_tables(
    g_illicit: nx.Graph,
    node_first_ts: Dict[str, pd.Timestamp],
    node_last_ts: Dict[str, pd.Timestamp],
    include_singletons: bool,
    max_component_size: int,
    start_idx: int = 0,
    cc_source: str | None = None,
    cc_pattern_type: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    nodes_cc_df : columns [ccId, clId]
    cc_df       : columns [ccId, ccLabel, feat0, feat1]
                  feat0=num_nodes, feat1=num_internal_edges
    """
    components = _connected_components_sets(g_illicit, include_singletons=include_singletons)
    if max_component_size > 0:
        split_components: List[Set[str]] = []
        n_split_source = 0
        for comp in components:
            if len(comp) > max_component_size:
                n_split_source += 1
                split_components.extend(_split_component_by_bfs(g_illicit, comp, max_size=max_component_size))
            else:
                split_components.append(comp)
        components = split_components
        if not include_singletons:
            components = [c for c in components if len(c) >= 2]
        if n_split_source > 0:
            print(
                f"Split {n_split_source} oversized illicit component(s) into {len(components)} total components (max_size={max_component_size}).",
                flush=True,
            )

    cc_rows = []
    node_cc_rows = []

    for i, comp in enumerate(components, start=start_idx):
        cc_id = f"AMLCC_{i}"
        subg = g_illicit.subgraph(comp)
        first_ts_candidates = [node_first_ts[n] for n in comp if n in node_first_ts]
        last_ts_candidates = [node_last_ts[n] for n in comp if n in node_last_ts]
        cc_start_ts = min(first_ts_candidates) if len(first_ts_candidates) > 0 else pd.NaT
        cc_end_ts = max(last_ts_candidates) if len(last_ts_candidates) > 0 else pd.NaT

        row = {
            "ccId": cc_id,
            "ccLabel": "illicit_component",
            "feat0": float(subg.number_of_nodes()),
            "feat1": float(subg.number_of_edges()),
            "ccStartTime": cc_start_ts.isoformat() if pd.notna(cc_start_ts) else "",
            "ccEndTime": cc_end_ts.isoformat() if pd.notna(cc_end_ts) else "",
        }
        if cc_source is not None:
            row["ccSource"] = cc_source
            row["ccPatternType"] = cc_pattern_type
        cc_rows.append(row)
        for n in sorted(comp):
            node_cc_rows.append({"ccId": cc_id, "clId": n})

    return pd.DataFrame(node_cc_rows), pd.DataFrame(cc_rows)


def _build_pattern_aware_subgraph_tables(
    g_illicit: nx.Graph,
    pattern_attempts: List[PatternAttempt],
    node_first_ts: Dict[str, pd.Timestamp],
    node_last_ts: Dict[str, pd.Timestamp],
    include_singletons: bool,
    max_component_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    min_size = 1 if include_singletons else 2
    illicit_node_set = set(g_illicit.nodes())
    assigned_pattern_nodes: Set[str] = set()
    node_cc_rows = []
    cc_rows = []
    pattern_type_counts: Dict[str, int] = defaultdict(int)

    cc_idx = 0
    for attempt in pattern_attempts:
        nodes = set(n for n in attempt.nodes if n in illicit_node_set)
        if len(nodes) < min_size:
            continue

        cc_id = f"AMLCC_{cc_idx}"
        cc_idx += 1
        assigned_pattern_nodes.update(nodes)
        pattern_type_counts[attempt.pattern_type] += 1

        subg = g_illicit.subgraph(nodes)
        first_ts = attempt.first_ts
        last_ts = attempt.last_ts
        if first_ts is None or pd.isna(first_ts):
            first_ts_candidates = [node_first_ts[n] for n in nodes if n in node_first_ts]
            first_ts = min(first_ts_candidates) if len(first_ts_candidates) > 0 else pd.NaT
        if last_ts is None or pd.isna(last_ts):
            last_ts_candidates = [node_last_ts[n] for n in nodes if n in node_last_ts]
            last_ts = max(last_ts_candidates) if len(last_ts_candidates) > 0 else pd.NaT

        cc_rows.append(
            {
                "ccId": cc_id,
                "ccLabel": "illicit_component",
                "feat0": float(subg.number_of_nodes()),
                "feat1": float(subg.number_of_edges()),
                "ccStartTime": first_ts.isoformat() if pd.notna(first_ts) else "",
                "ccEndTime": last_ts.isoformat() if pd.notna(last_ts) else "",
                "ccSource": "pattern",
                "ccPatternType": attempt.pattern_type,
            }
        )
        for n in sorted(nodes):
            node_cc_rows.append({"ccId": cc_id, "clId": n})

    residual_nodes = illicit_node_set - assigned_pattern_nodes
    residual_g = g_illicit.subgraph(residual_nodes).copy()
    residual_nodes_cc_df, residual_cc_df = _build_subgraph_tables(
        residual_g,
        node_first_ts=node_first_ts,
        node_last_ts=node_last_ts,
        include_singletons=include_singletons,
        max_component_size=max_component_size,
        start_idx=cc_idx,
        cc_source="residual_laundering_component",
        cc_pattern_type="",
    )

    pattern_nodes_cc_df = pd.DataFrame(node_cc_rows, columns=["ccId", "clId"])
    pattern_cc_df = pd.DataFrame(
        cc_rows,
        columns=[
            "ccId",
            "ccLabel",
            "feat0",
            "feat1",
            "ccStartTime",
            "ccEndTime",
            "ccSource",
            "ccPatternType",
        ],
    )

    nodes_cc_df = pd.concat([pattern_nodes_cc_df, residual_nodes_cc_df], ignore_index=True)
    cc_df = pd.concat([pattern_cc_df, residual_cc_df], ignore_index=True)

    stats = {
        "num_pattern_attempts_read": len(pattern_attempts),
        "num_pattern_subgraphs_added": len(pattern_cc_df),
        "num_pattern_nodes_assigned": len(assigned_pattern_nodes),
        "num_residual_illicit_nodes": len(residual_nodes),
        "pattern_type_counts": dict(sorted(pattern_type_counts.items())),
    }
    return nodes_cc_df, cc_df, stats


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


def build_aml_subgraph_dataset(config: BuildConfig) -> None:
    final_output_dir = config.output_dir
    if config.dataset_name.strip() != "":
        ds = config.dataset_name.strip()
        if final_output_dir.name != ds:
            final_output_dir = config.output_dir / ds
    final_output_dir.mkdir(parents=True, exist_ok=True)

    tx_files = _find_transaction_files(config.input_dir, dataset_name=config.dataset_name)
    if not tx_files:
        raise FileNotFoundError(
            f"No transaction CSV found for dataset='{config.dataset_name}' under {config.input_dir}. "
            "Expected <name>_Trans.csv or <name>_Transactions.csv"
        )

    print(f"Found {len(tx_files)} transaction file(s).", flush=True)

    illicit_nodes, currency_to_id, payfmt_to_id, anchor_ts = _pass1_find_illicit_nodes_and_currency_map(
        tx_files,
        chunksize=config.chunksize,
    )
    if not illicit_nodes:
        raise RuntimeError("No illicit nodes found (no laundering transactions?).")
    print(f"Illicit nodes found: {len(illicit_nodes)}", flush=True)
    print(f"Currencies discovered: {len(currency_to_id)}", flush=True)
    print(f"Payment formats discovered: {len(payfmt_to_id)}", flush=True)

    nodes_df, edges_df, g_illicit, g_licit, node_first_ts, node_last_ts = _pass2_build_graph_and_features(
        tx_files=tx_files,
        illicit_nodes=illicit_nodes,
        currency_to_id=currency_to_id,
        payfmt_to_id=payfmt_to_id,
        anchor_ts=anchor_ts,
        chunksize=config.chunksize,
    )

    pattern_stats = {}
    if config.illicit_subgraph_mode == "patterns":
        pattern_files = _find_pattern_files(config.input_dir, dataset_name=config.dataset_name)
        if not pattern_files:
            raise FileNotFoundError(
                f"No pattern file found for dataset='{config.dataset_name}' under {config.input_dir}. "
                "Expected <name>_Patterns.txt or <name>_Pattern.txt"
            )
        pattern_attempts = _parse_pattern_files(pattern_files)
        nodes_cc_df, cc_df, pattern_stats = _build_pattern_aware_subgraph_tables(
            g_illicit,
            pattern_attempts=pattern_attempts,
            node_first_ts=node_first_ts,
            node_last_ts=node_last_ts,
            include_singletons=config.include_singletons,
            max_component_size=config.max_component_size,
        )
        print(
            "Pattern-aware illicit subgraphs: "
            f"patterns_added={pattern_stats['num_pattern_subgraphs_added']}, "
            f"residual_illicit_nodes={pattern_stats['num_residual_illicit_nodes']}",
            flush=True,
        )
    elif config.illicit_subgraph_mode == "components":
        nodes_cc_df, cc_df = _build_subgraph_tables(
            g_illicit,
            node_first_ts=node_first_ts,
            node_last_ts=node_last_ts,
            include_singletons=config.include_singletons,
            max_component_size=config.max_component_size,
        )
    else:
        raise ValueError(f"Unknown illicit_subgraph_mode: {config.illicit_subgraph_mode}")
    licit_nodes_count = max(0, int(len(nodes_df)) - len(illicit_nodes))
    print(f"Licit nodes available: {licit_nodes_count}", flush=True)

    # Optional second stage: build licit subgraphs with size profile similar to illicit ones
    if config.add_licit_subgraphs and len(cc_df) > 0 and g_licit.number_of_nodes() > 0:
        illicit_sizes = cc_df["feat0"].astype(int).tolist()
        n_target = max(1, int(round(len(illicit_sizes) * max(config.licit_ratio, 0.0))))
        # Non-overlapping samples cannot exceed the number of available nodes.
        max_licit_samples = g_licit.number_of_nodes() if config.include_singletons else (g_licit.number_of_nodes() // 2)
        if n_target > max_licit_samples:
            print(
                f"Capping requested licit samples from {n_target} to {max_licit_samples} (node-availability bound).",
                flush=True,
            )
            n_target = max_licit_samples
        if n_target <= 0:
            print("Skipping licit subgraph stage: no feasible sample budget.", flush=True)
        else:
            print(
                f"Sampling licit subgraphs: target={n_target}, ratio={config.licit_ratio}, seed={config.licit_seed}",
                flush=True,
            )
        rng = np.random.default_rng(config.licit_seed)
        sampled_sizes = rng.choice(illicit_sizes, size=n_target, replace=True).tolist() if n_target > 0 else []

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

    if "ccSource" in cc_df.columns:
        licit_source_mask = (cc_df["ccLabel"] == "licit_component") & cc_df["ccSource"].isna()
        cc_df.loc[licit_source_mask, "ccSource"] = "licit_sample"
        cc_df["ccSource"] = cc_df["ccSource"].fillna("")
    if "ccPatternType" in cc_df.columns:
        cc_df["ccPatternType"] = cc_df["ccPatternType"].fillna("")

    # Keep only nodes that are in retained components
    kept_nodes = set(nodes_cc_df["clId"].tolist())
    nodes_df = nodes_df[nodes_df["clId"].isin(kept_nodes)].reset_index(drop=True)
    if not edges_df.empty:
        edges_df = edges_df[
            edges_df["clId1"].isin(kept_nodes) & edges_df["clId2"].isin(kept_nodes)
        ].reset_index(drop=True)

    # Precompute per-component internal edges for faster loading.
    component_edges_df = _build_component_edges_df(edges_df, nodes_cc_df)

    if config.normalize_features:
        # FraudGT-style: z-score continuous features; keep categorical IDs untouched.
        nodes_df = _z_norm(nodes_df, ["feat0", "feat1", "feat2", "feat3"])
        if not edges_df.empty:
            edges_df = _z_norm(edges_df, ["feat0", "feat1", "feat4"])

    # Save in Elliptic-compatible naming convention
    nodes_out = final_output_dir / "background_nodes.csv"
    edges_out = final_output_dir / "background_edges.csv"
    node_cc_out = final_output_dir / "nodes.csv"
    component_edges_out = final_output_dir / "component_edges.csv"
    cc_out = final_output_dir / "connected_components.csv"
    meta_out = final_output_dir / "metadata.json"

    nodes_df.to_csv(nodes_out, index=False)
    edges_df.to_csv(edges_out, index=False)
    nodes_cc_df.to_csv(node_cc_out, index=False)
    component_edges_df.to_csv(component_edges_out, index=False)
    cc_df.to_csv(cc_out, index=False)

    metadata = {
        "source_input_dir": str(config.input_dir),
        "dataset_name": config.dataset_name,
        "resolved_output_dir": str(final_output_dir),
        "num_transaction_files": len(tx_files),
        "num_illicit_nodes_before_component_filter": len(illicit_nodes),
        "num_nodes_kept": int(len(nodes_df)),
        "num_edges_kept": int(len(edges_df)),
        "num_component_edges": int(len(component_edges_df)),
        "num_subgraphs": int(len(cc_df)),
        "num_illicit_subgraphs": int((cc_df["ccLabel"] == "illicit_component").sum()) if len(cc_df) > 0 else 0,
        "num_licit_subgraphs": int((cc_df["ccLabel"] == "licit_component").sum()) if len(cc_df) > 0 else 0,
        "illicit_subgraph_sizes": {
            "min": int(cc_df[cc_df["ccLabel"] == "illicit_component"]["feat0"].min()) if len(cc_df) > 0 and (cc_df["ccLabel"] == "illicit_component").any() else 0,
            "max": int(cc_df[cc_df["ccLabel"] == "illicit_component"]["feat0"].max()) if len(cc_df) > 0 and (cc_df["ccLabel"] == "illicit_component").any() else 0,
            "avg": float(cc_df[cc_df["ccLabel"] == "illicit_component"]["feat0"].mean()) if len(cc_df) > 0 and (cc_df["ccLabel"] == "illicit_component").any() else 0.0,
            "median": float(cc_df[cc_df["ccLabel"] == "illicit_component"]["feat0"].median()) if len(cc_df) > 0 and (cc_df["ccLabel"] == "illicit_component").any() else 0.0,
        },
        "licit_subgraph_sizes": {
            "min": int(cc_df[cc_df["ccLabel"] == "licit_component"]["feat0"].min()) if len(cc_df) > 0 and (cc_df["ccLabel"] == "licit_component").any() else 0,
            "max": int(cc_df[cc_df["ccLabel"] == "licit_component"]["feat0"].max()) if len(cc_df) > 0 and (cc_df["ccLabel"] == "licit_component").any() else 0,
            "avg": float(cc_df[cc_df["ccLabel"] == "licit_component"]["feat0"].mean()) if len(cc_df) > 0 and (cc_df["ccLabel"] == "licit_component").any() else 0.0,
            "median": float(cc_df[cc_df["ccLabel"] == "licit_component"]["feat0"].median()) if len(cc_df) > 0 and (cc_df["ccLabel"] == "licit_component").any() else 0.0,
        },
        "include_singletons": config.include_singletons,
        "normalize_features": config.normalize_features,
        "add_licit_subgraphs": config.add_licit_subgraphs,
        "licit_ratio": config.licit_ratio,
        "licit_seed": config.licit_seed,
        "licit_bfs_prob": config.licit_bfs_prob,
        "max_component_size": config.max_component_size,
        "illicit_subgraph_mode": config.illicit_subgraph_mode,
        "pattern_stats": pattern_stats,
        "normalization_type": "zscore (continuous only)",
        "anchor_timestamp_epoch": anchor_ts,
        "currency_to_id": currency_to_id,
        "payment_format_to_id": payfmt_to_id,
        "edge_feature_description": {
            "feat0": "relative timestamp seconds (zscore if enabled)",
            "feat1": "amount_received (zscore if enabled)",
            "feat2": "receiving_currency_id",
            "feat3": "payment_format_id",
            "feat4": "amount_paid (zscore if enabled)",
            "feat5": "payment_currency_id",
        },
        "node_feature_description": {
            "clId": "bank::account identifier",
            "feat0": "outgoing_tx_count (zscore if enabled)",
            "feat1": "incoming_tx_count (zscore if enabled)",
            "feat2": "outgoing_amount_paid_sum (zscore if enabled)",
            "feat3": "incoming_amount_received_sum (zscore if enabled)",
        },
        "subgraph_feature_description": {
            "feat0": "num_nodes",
            "feat1": "num_internal_edges",
            "ccStartTime": "earliest transaction timestamp among nodes in component",
            "ccEndTime": "latest transaction timestamp among nodes in component",
        },
        "component_edges_file": {
            "path": "component_edges.csv",
            "description": "Precomputed internal edges per connected component for faster loading.",
        },
        "notes": [
            "Node illicit/licit flag is not included as a node feature.",
            "Is Laundering is not included as an edge feature.",
            "When illicit_subgraph_mode=components, illicit subgraphs are connected components of the illicit-only graph.",
            "When illicit_subgraph_mode=patterns, pattern-file laundering attempts are emitted as individual illicit subgraphs and remaining illicit nodes are componentized.",
            "Components larger than max_component_size are split into connected BFS chunks when max_component_size > 0.",
            "Licit subgraphs are sampled as connected sets from licit-only graph using hybrid BFS/DFS.",
            "Transaction-level edges are kept to preserve temporal information.",
            "Node IDs include bank and account (bank::account), similar to FraudGT behavior.",
        ],
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Done.", flush=True)
    print(f"Saved: {nodes_out}", flush=True)
    print(f"Saved: {edges_out}", flush=True)
    print(f"Saved: {node_cc_out}", flush=True)
    print(f"Saved: {component_edges_out}", flush=True)
    print(f"Saved: {cc_out}", flush=True)
    print(f"Saved: {meta_out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build IBM AML illicit-node connected-component subgraph dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/scratch-shared/ghonkoop/data/aml"),
        help="Directory containing IBM AML *_Transactions.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ghonkoop/data/aml"),
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Dataset prefix, e.g. HI-Small, HI-Medium, HI-Large, LI-Small, ...",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per chunk while streaming large transaction CSVs.",
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
        help="Target #licit subgraphs as ratio of #illicit subgraphs (default 1.0).",
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
        "--illicit-subgraph-mode",
        choices=["components", "patterns"],
        default="components",
        help=(
            "How to form illicit subgraphs: 'components' uses the original illicit-node connected components; "
            "'patterns' emits each *_Patterns.txt laundering attempt as its own illicit subgraph, then "
            "componentizes remaining laundering-flagged nodes."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = BuildConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=str(args.dataset_name),
        chunksize=args.chunksize,
        include_singletons=not args.drop_singletons,
        normalize_features=not args.no_normalize,
        add_licit_subgraphs=not args.no_licit_subgraphs,
        licit_ratio=float(args.licit_ratio),
        licit_seed=int(args.licit_seed),
        licit_bfs_prob=min(1.0, max(0.0, float(args.licit_bfs_prob))),
        max_component_size=max(0, int(args.max_component_size)),
        illicit_subgraph_mode=str(args.illicit_subgraph_mode),
    )
    build_aml_subgraph_dataset(cfg)
