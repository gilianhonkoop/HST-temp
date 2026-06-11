import os
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Optional, Union, List, Callable, Tuple, Dict

import networkx as nx
import torch
import torch_sparse
from sklearn.decomposition import PCA
from termcolor import cprint
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import coalesce, to_undirected, is_undirected, remove_self_loops, dense_to_sparse, \
    add_remaining_self_loops, degree, to_dense_adj
from tqdm import tqdm
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min
from torch_sparse import spspmm

from data_utils import RelabelNodes, AddSelfLoopsV2, filter_living_edge_index
from utils import repr_kvs, try_getattr, func_normalize


class SubgraphToNode:
    _global_nxg = None
    _node_spl_mat = None
    _node_task_data_precursor = None

    MATRIX_TYPES = ['adj', 'A', 'sp_A']

    def __init__(self,
                 path: str,
                 global_data: Data,
                 subgraph_data_list: Optional[List[Data]] = None,
                 splits: Optional[List[int]] = None,
                 name: Optional[str] = None,
                 target_matrix: str = 'adj',
                 edge_aggr: Optional[Union[Callable[[Tensor], Tensor], str]] = None,
                 num_workers: int = 0,
                 is_weighted: bool = True,
                 original_edge_attr_mode: Optional[str] = None,
                 original_edge_attr_aggr: str = "mean",
                 original_edge_attr_normalize: Optional[str] = None,
                 original_edge_attr_norm_eps: float = 1e-6,
                 original_edge_attr_fill_value: float = 0.0,
                 matrix_type: str = 'adj',
                 verbose: int = 0,
                 undirected: Optional[bool] = None,
                 use_sub_edge_index: bool = False,
                 **kwargs,
                 ):
        """
        :param global_data: Single Data(edge_index=[2, *], x=[*, F])
        :param subgraph_data_list: List of Data(x=[*, 1], edge_index=[2, *], y=[1])
        :param splits: [num_train, num_train + num_val]
        :param node_spl_cutoff: Deprecated, used for methods based on shortest_path_length

          num_start
          ↓  [+] num_train
          ↓   ↓  [+] num_train + num_val
          ↓   ↓   ↓     num_subgraphs
          ↓   ↓   ↓     ↓
        @ @ @ # # + + +
        @ @ @ # # + + +
        @ @ @ # # + + +
        # # # # # + + +
        # # # # # + + +
        + + + + + + + +
        + + + + + + + +
        + + + + + + + +
        """
        # if matrix_type not in self.MATRIX_TYPES:
        #     raise ValueError(f"Wrong matrix_type: {matrix_type}")
        self.matrix_type = matrix_type
        self.global_data: Data = global_data
        self.subgraph_data_list: List[Data] = subgraph_data_list
        self.name: str = name
        self.path: Path = Path(path)
        self.splits = splits + [len(self.subgraph_data_list)]
        self.num_start = 0

        self.target_matrix = target_matrix
        self.edge_aggr = self.parse_edge_aggr(edge_aggr)
        self.original_edge_attr_mode = self.parse_original_edge_attr_mode(original_edge_attr_mode)
        self.original_edge_attr_aggr = original_edge_attr_aggr
        self.original_edge_attr_normalize = self.parse_original_edge_attr_normalize(original_edge_attr_normalize)
        self.original_edge_attr_norm_eps = float(original_edge_attr_norm_eps)
        self.original_edge_attr_fill_value = float(original_edge_attr_fill_value)
        self._original_edge_attr_lookup = None

        self.num_workers = num_workers
        self.is_weighted = is_weighted
        self.verbose = verbose

        self.N = global_data.num_nodes
        self.undirected = undirected if undirected is not None else is_undirected(global_data.edge_index)
        if self.undirected:
            self.global_data.edge_index, self.global_data.edge_attr = to_undirected(
                self.global_data.edge_index, self.global_data.edge_attr
            )

        # Pre-computation for some matrix
        self.node_spl_cutoff = None

        assert self.target_matrix in [
            "adjacent", "adjacent_with_self_loops", "adjacent_no_self_loops", "shortest_path"
        ]
        assert len(self.splits) >= 3
        self.path.mkdir(exist_ok=True, parents=True)

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.node_task_name}', path='{self.path}')"

    def parse_edge_aggr(self, edge_aggr):
        if isinstance(edge_aggr, str):
            return eval(edge_aggr)
        else:
            return edge_aggr or torch.min

    def parse_original_edge_attr_mode(self, mode):
        if mode in {None, False, "none", "None", "false", "False", "off", "Off"}:
            return None
        if mode not in {"append", "replace"}:
            raise ValueError(
                "original_edge_attr_mode must be one of null, 'append', or 'replace'. "
                f"Got {mode!r}."
            )
        return mode

    def parse_original_edge_attr_normalize(self, mode):
        if mode in {None, False, "none", "None", "false", "False", "off", "Off", "null"}:
            return None
        mode = str(mode).lower()
        if mode in {"zscore", "standard", "standardize"}:
            return "zscore"
        if mode in {"minmax", "min_max"}:
            return "minmax"
        raise ValueError(
            "original_edge_attr_normalize must be one of null, 'zscore', or 'minmax'. "
            f"Got {mode!r}."
        )

    @property
    def node_task_name(self):
        if self.target_matrix.startswith("adjacent"):
            return f"{self.name}-ADJ-{self.target_matrix}"
        else:
            return f"{self.name}-SP-EA-{self.edge_aggr.__name__}"

    @property
    def S(self):
        return len(self.subgraph_data_list)

    @property
    def global_nxg(self) -> nx.Graph:
        if self._global_nxg is None:
            self._global_nxg = to_networkx(self.global_data)
        return self._global_nxg

    def single_source_shortest_path_length_for_global_data(self, n):
        spl_dict = nx.single_source_shortest_path_length(
            self.global_nxg, n, cutoff=self.node_spl_cutoff)
        spl_list = [val for node, val in sorted(spl_dict.items(), key=lambda t: t[0])]
        return spl_list

    def all_pairs_shortest_path_length_for_global_data(self):
        if self.num_workers is not None:
            with mp.Pool(processes=self.num_workers) as pool:
                shortest_paths = pool.map(self.single_source_shortest_path_length_for_global_data,
                                          self.global_nxg.nodes)
        else:
            shortest_paths = [self.single_source_shortest_path_length_for_global_data(n)
                              for n in tqdm(self.global_nxg.nodes)]
        return torch.tensor(shortest_paths, dtype=torch.long)

    def node_spl_mat(self, save=True):
        path = self.path / f"{self.name}_spl_mat.pth"
        try:
            self._node_spl_mat = torch.load(path)
            cprint(f"Load: tensor of {self._node_spl_mat.size()} at {path}", "green")
            return self._node_spl_mat
        except FileNotFoundError:
            pass
        self._node_spl_mat = self.all_pairs_shortest_path_length_for_global_data()
        if save:
            torch.save(self._node_spl_mat, path)
            cprint(f"Saved: tensor of {self._node_spl_mat.size()} at {path}", "blue")
        return self._node_spl_mat

    @property
    def node_task_data_list(self) -> List[Data]:
        return self._node_task_data_list

    @property
    def num_subgraphs(self):
        return len(self.subgraph_data_list)

    def get_sparse_mapping_matrix_sxn(self,
                                      matrix_type: str,
                                      sub_x: Tensor,
                                      sub_batch: Tensor,
                                      global_edge_index: Tensor,
                                      summarized_edge_index: Optional[Tensor] = None,
                                      ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param matrix_type: See if clauses.
        :param sub_x: x_ids of all subgraphs as a Tensor of [sum |V_i|, 1]
        :param sub_batch: subgraph_ids of x_ids in a batch form as a Tensor of [sum |V_i|, 1]
        :param global_edge_index: edge_index of global graph as a Tensor of [2, E]
        :param summarized_edge_index: edge_index of summarized graph as a Tensor of [2, E_s]
        :return: an index and value tensor tuple of a sparse matrix where the size is [S, N]
        """
        # Originally,
        # Mapping matrix M (sxn) construction
        # batch = subgraph ids, x = node ids
        # m_index = torch.stack([self._node_task_data_precursor.batch,
        #                        self._node_task_data_precursor.x.squeeze(-1)]).long()
        m_index = torch.stack([sub_batch, sub_x]).long()

        if matrix_type == "unnormalized":
            mapping_matrix_value = torch.ones(m_index.size(1))
        elif matrix_type == "degree_normalized_by_sub":
            if summarized_edge_index is None:
                global_edge_value = torch.ones(global_edge_index.size(1))
                summarized_edge_index, _ = spspmm_quad(
                    m_index, global_edge_value, global_edge_index, global_edge_value, self.S, self.N, coalesced=True)

            d_value_n = degree(global_edge_index[0], num_nodes=self.N)
            d_value_s = degree(summarized_edge_index[0], num_nodes=sub_batch.max().item() + 1)
            mapping_matrix_value = torch.sqrt(d_value_n[m_index[1, :]] / d_value_s[m_index[0, :]])
        elif matrix_type == "sqrt_degree_normalized_by_sub":
            if summarized_edge_index is None:
                global_edge_value = torch.ones(global_edge_index.size(1))
                summarized_edge_index, _ = spspmm_quad(
                    m_index, global_edge_value, global_edge_index, global_edge_value, self.S, self.N, coalesced=True)

            a_index, _ = add_remaining_self_loops(self.global_data.edge_index)
            d_value_n = degree(a_index[0], num_nodes=self.N)
            d_value_s = degree(summarized_edge_index[0], num_nodes=sub_batch.max().item() + 1)
            mapping_matrix_value = torch.sqrt(d_value_n[mapping_matrix_index[1, :]]
                                              / d_value_s[mapping_matrix_index[0, :]])
        else:
            raise ValueError(f"Wrong matrix_type: {matrix_type}")

        # Mapping matrix M (sxn) construction
        # batch = subgraph ids, x = node ids
        # m_index = torch.stack([self._node_task_data_precursor.batch,
        #                        self._node_task_data_precursor.x.squeeze(-1)]).long()
        m_index = torch.stack([sub_batch, sub_x]).long()
        m_value = mapping_matrix_value

        if matrix_type == "unnormalized":
            return m_index, m_value

        # M[s, n] = sqrt( d_n / d_s )
        elif matrix_type == "sqrt_d_node_div_d_sub":
            if summarized_edge_index is None:
                global_edge_value = torch.ones(global_edge_index.size(1))
                summarized_edge_index, _ = spspmm_quad(
                    m_index, m_value, global_edge_index, global_edge_value, self.S, self.N, coalesced=True)

            d_index_s = torch.stack([torch.arange(self.S), torch.arange(self.S)])
            d_index_n = torch.stack([torch.arange(self.N), torch.arange(self.N)])
            d_value_s = 1 / torch.sqrt(degree(summarized_edge_index[0], num_nodes=self.S) + 1)
            d_value_n = torch.sqrt(degree(global_edge_index[0], num_nodes=self.N))

            # (s, s) * (s, n) --> (s, n)
            dsm_index, dsm_value = torch_sparse.spspmm(
                d_index_s, d_value_s, m_index, m_value, self.S, self.S, self.N, coalesced=True)

            # (s, n) * (n, n) --> (s, n)
            dsmdn_index, dsmdn_value = torch_sparse.spspmm(
                dsm_index, dsm_value, d_index_n, d_value_n, self.S, self.N, self.N, coalesced=True)

            return dsmdn_index, dsmdn_value

        # M[s, n] = 1 / sqrt( #nodes_s )
        elif matrix_type == "1_div_sqrt_num_nodes_in_sub":
            # num_nodes_per_subgraph
            nps_index_s = torch.stack([torch.arange(self.S), torch.arange(self.S)])
            nps_value_s = 1 / torch.sqrt(degree(sub_batch, num_nodes=self.S))

            # (s, s) * (s, n) --> (s, n)
            npsm_index, npsm_value = torch_sparse.spspmm(
                nps_index_s, nps_value_s, m_index, m_value, self.S, self.S, self.N, coalesced=True)

            return npsm_index, npsm_value

        else:
            raise ValueError(f"Wrong matrix_type: {matrix_type}")

    def get_ewmat_by_multiplying_adj(self, matrix_type: str) -> Tensor:
        cprint(f"Computing edge_weight_matrix by multiplying adjacent "
               f"(mmt={matrix_type}, target={self.target_matrix})", "blue")

        if self.target_matrix == "adjacent_with_self_loops":
            a_index, _ = add_remaining_self_loops(self.global_data.edge_index)
        elif self.target_matrix in {"adjacent_wo_self_loops", "adjacent_no_self_loops"}:
            a_index, _ = remove_self_loops(self.global_data.edge_index)
        else:
            raise ValueError(self.target_matrix)
        # NOTE: Avoid calling PyTorch/torch_sparse sparse-sparse matmul on the full (N x N)
        # adjacency; on some builds this intermittently triggers glibc heap corruption
        # ("mismatching next->prev_size" / "invalid chunk size") rather than a clean OOM.
        # We only need the adjacency induced by nodes that actually appear in subgraphs.
        a_index = a_index.detach().cpu().long()
        a_value = torch.ones(a_index.size(1), dtype=torch.float32)

        s_index, s_value = self.get_sparse_mapping_matrix_sxn(
            matrix_type=matrix_type,
            sub_x=self._node_task_data_precursor.x.squeeze(-1),
            sub_batch=self._node_task_data_precursor.batch,
            global_edge_index=self.global_data.edge_index,
        )
        s_index = s_index.detach().cpu().long()
        s_value = s_value.detach().cpu().to(torch.float32)

        # Restrict global adjacency to only the nodes that appear in any subgraph.
        sub_nodes = torch.unique(s_index[1])
        num_sub_nodes = int(sub_nodes.numel())
        if num_sub_nodes == 0:
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.float32),
                size=(self.num_subgraphs, self.num_subgraphs),
            )

        # Map global node ids -> [0, K) for induced-subgraph adjacency.
        # (Uses O(N) memory but prevents O(E log K) searchsorted cost.)
        global_to_local = torch.full((self.N,), -1, dtype=torch.long)
        global_to_local[sub_nodes] = torch.arange(num_sub_nodes, dtype=torch.long)

        # Build A_sub (K x K) from global edges whose endpoints are both in sub_nodes.
        local_u = global_to_local[a_index[0]]
        local_v = global_to_local[a_index[1]]
        mask = (local_u >= 0) & (local_v >= 0)
        local_u = local_u[mask]
        local_v = local_v[mask]
        a_sub_value = a_value[mask]

        # Build S_sub (S x K) by remapping S columns from global ids to local ids.
        s_col_local = global_to_local[s_index[1]]
        # All s_index cols must be present in sub_nodes; if not, drop them defensively.
        s_mask = s_col_local >= 0
        s_row = s_index[0][s_mask]
        s_col_local = s_col_local[s_mask]
        s_val = s_value[s_mask]

        try:
            import numpy as np
            import scipy.sparse as sp

            S_sp = sp.coo_matrix(
                (s_val.numpy(), (s_row.numpy(), s_col_local.numpy())),
                shape=(self.num_subgraphs, num_sub_nodes),
                dtype=np.float32,
            ).tocsr()

            A_sp = sp.coo_matrix(
                (a_sub_value.numpy(), (local_u.numpy(), local_v.numpy())),
                shape=(num_sub_nodes, num_sub_nodes),
                dtype=np.float32,
            ).tocsr()

            # EW (S x S) = S * A * S^T
            ew_sp = (S_sp @ A_sp) @ S_sp.T
            ew_coo = ew_sp.tocoo()

            ew_index = torch.from_numpy(
                np.vstack([ew_coo.row, ew_coo.col]).astype(np.int64)
            )
            ew_value = torch.from_numpy(ew_coo.data.astype(np.float32))
            return torch.sparse_coo_tensor(
                ew_index, ew_value, size=(self.num_subgraphs, self.num_subgraphs)
            ).coalesce()
        except Exception as e:
            # Fallback: keep everything in PyTorch. This can still be heavy/unstable on
            # some builds, so we only use it if SciPy is unavailable.
            cprint(f"[WARN] SciPy sparse matmul failed ({type(e).__name__}: {e}); falling back to torch.sparse.mm", "yellow")

            # Rebuild tensors in the induced space to keep dimensions small.
            s_coo = torch.sparse_coo_tensor(
                torch.stack([s_row, s_col_local], dim=0),
                s_val,
                (self.num_subgraphs, num_sub_nodes),
            ).coalesce()
            a_coo = torch.sparse_coo_tensor(
                torch.stack([local_u, local_v], dim=0),
                a_sub_value,
                (num_sub_nodes, num_sub_nodes),
            ).coalesce()
            sa_coo = torch.sparse.mm(s_coo, a_coo)
            ss_coo = torch.sparse.mm(sa_coo, s_coo.transpose(0, 1)).coalesce()
            return ss_coo

    def get_ewmat_by_aggregating_sub_spl_mat(self, save):
        # sub_spl_ij = min { d_uv | u \in S_i, v in S_j }
        node_spl_mat = self.node_spl_mat(save).float()
        sub_spl_mat = torch.full((self.S, self.S), fill_value=-1)
        for i, sub_data_i in enumerate(tqdm(self.subgraph_data_list,
                                            desc="get_ewmat_by_aggregating_sub_spl_mat")):
            for j, sub_data_j in enumerate(self.subgraph_data_list):
                if self.undirected and i <= j:
                    x_i = sub_data_i.x.squeeze(-1)
                    x_j = sub_data_j.x.squeeze(-1)
                    sub_spl = self.edge_aggr(node_spl_mat[x_i, :][:, x_j])
                    sub_spl_mat[i, j] = sub_spl
                    sub_spl_mat[j, i] = sub_spl

        # edge = 1 / (spl + 1) where 0 <= spl, then 0 < edge <= 1
        return 1 / (sub_spl_mat + 1)

    def get_original_edge_attr_lookup(self, matrix_type: str):
        if self._original_edge_attr_lookup is not None:
            return self._original_edge_attr_lookup

        edge_attr = getattr(self.global_data, "edge_attr", None)
        if edge_attr is None:
            self._original_edge_attr_lookup = None
            return None

        edge_attr = edge_attr.detach().cpu()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        if edge_attr.size(0) != self.global_data.edge_index.size(1):
            raise ValueError(
                "global_data.edge_attr must have one row per global edge to aggregate original edge attributes. "
                f"Got {edge_attr.size(0)} edge_attr rows for {self.global_data.edge_index.size(1)} edges."
            )

        aggr = str(self.original_edge_attr_aggr).lower()
        if aggr not in {"mean", "sum"}:
            raise ValueError(f"Unsupported original_edge_attr_aggr={self.original_edge_attr_aggr!r}. Use 'mean' or 'sum'.")

        s_index, s_value = self.get_sparse_mapping_matrix_sxn(
            matrix_type=matrix_type,
            sub_x=self._node_task_data_precursor.x.squeeze(-1),
            sub_batch=self._node_task_data_precursor.batch,
            global_edge_index=self.global_data.edge_index,
        )
        s_index = s_index.detach().cpu().long()
        s_value = s_value.detach().cpu().to(torch.float32)
        a_index = self.global_data.edge_index.detach().cpu().long()

        sub_nodes = torch.unique(s_index[1])
        if sub_nodes.numel() == 0:
            self._original_edge_attr_lookup = (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0, edge_attr.size(1)), dtype=torch.float32),
            )
            return self._original_edge_attr_lookup

        global_to_local = torch.full((self.N,), -1, dtype=torch.long)
        global_to_local[sub_nodes] = torch.arange(sub_nodes.numel(), dtype=torch.long)

        local_u = global_to_local[a_index[0]]
        local_v = global_to_local[a_index[1]]
        edge_mask = (local_u >= 0) & (local_v >= 0)
        local_u = local_u[edge_mask]
        local_v = local_v[edge_mask]
        edge_attr = edge_attr[edge_mask].to(torch.float32)

        s_col_local = global_to_local[s_index[1]]
        s_mask = s_col_local >= 0
        s_row = s_index[0][s_mask]
        s_col_local = s_col_local[s_mask]
        s_val = s_value[s_mask]

        try:
            import numpy as np
            import scipy.sparse as sp

            num_sub_nodes = int(sub_nodes.numel())
            S_sp = sp.coo_matrix(
                (s_val.numpy(), (s_row.numpy(), s_col_local.numpy())),
                shape=(self.num_subgraphs, num_sub_nodes),
                dtype=np.float32,
            ).tocsr()

            A_count = sp.coo_matrix(
                (np.ones(local_u.numel(), dtype=np.float32), (local_u.numpy(), local_v.numpy())),
                shape=(num_sub_nodes, num_sub_nodes),
                dtype=np.float32,
            ).tocsr()
            count_coo = ((S_sp @ A_count) @ S_sp.T).tocoo()
            keys = torch.from_numpy((count_coo.row.astype(np.int64) * self.num_subgraphs) + count_coo.col.astype(np.int64))
            values = torch.full(
                (count_coo.nnz, edge_attr.size(1)),
                fill_value=self.original_edge_attr_fill_value,
                dtype=torch.float32,
            )
            denom = count_coo.data.astype(np.float32)

            for feat_id in range(edge_attr.size(1)):
                A_feat = sp.coo_matrix(
                    (edge_attr[:, feat_id].numpy(), (local_u.numpy(), local_v.numpy())),
                    shape=(num_sub_nodes, num_sub_nodes),
                    dtype=np.float32,
                ).tocsr()
                feat_sp = ((S_sp @ A_feat) @ S_sp.T).tocsr()
                feat_values = feat_sp[count_coo.row, count_coo.col].A1.astype(np.float32)
                if aggr == "mean":
                    feat_values = feat_values / np.maximum(denom, 1e-12)
                values[:, feat_id] = torch.from_numpy(feat_values)

            order = torch.argsort(keys)
            self._original_edge_attr_lookup = (keys[order], values[order])
            return self._original_edge_attr_lookup
        except Exception as e:
            raise RuntimeError(
                f"Failed to aggregate original edge attributes into S2N edges: {type(e).__name__}: {e}"
            ) from e

    def lookup_original_edge_attr(self, edge_index: Tensor, s_0: int) -> Optional[Tensor]:
        lookup = self.get_original_edge_attr_lookup(self._original_edge_attr_matrix_type)
        if lookup is None:
            return None
        keys, values = lookup
        if keys.numel() == 0:
            return torch.empty((edge_index.size(1), values.size(1)), dtype=torch.float32)

        query = (edge_index[0].detach().cpu().long() + s_0) * self.num_subgraphs
        query = query + edge_index[1].detach().cpu().long() + s_0
        pos = torch.searchsorted(keys, query)
        in_bounds = pos < keys.numel()
        matched = torch.zeros(query.size(0), dtype=torch.bool)
        matched[in_bounds] = keys[pos[in_bounds]] == query[in_bounds]

        out = torch.full(
            (query.size(0), values.size(1)),
            fill_value=self.original_edge_attr_fill_value,
            dtype=torch.float32,
        )
        out[matched] = values[pos[matched]]
        return out

    def normalize_original_edge_attr(self, edge_attr: Tensor, state: Optional[Dict[str, Tensor]] = None):
        mode = self.original_edge_attr_normalize
        if mode is None or edge_attr is None:
            return edge_attr, state

        edge_attr = edge_attr.to(torch.float32)
        if edge_attr.numel() == 0:
            return edge_attr, state

        eps = self.original_edge_attr_norm_eps
        if state is None:
            if mode == "zscore":
                mean = edge_attr.mean(dim=0, keepdim=True)
                std = edge_attr.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
                state = {"mode": mode, "mean": mean, "std": std}
                cprint(
                    f"Fitted original edge_attr z-score normalization "
                    f"(mean={mean.flatten().tolist()}, std={std.flatten().tolist()})",
                    "blue",
                )
            elif mode == "minmax":
                v_min = edge_attr.min(dim=0, keepdim=True).values
                v_max = edge_attr.max(dim=0, keepdim=True).values
                scale = (v_max - v_min).clamp_min(eps)
                state = {"mode": mode, "min": v_min, "scale": scale}
                cprint(
                    f"Fitted original edge_attr min-max normalization "
                    f"(min={v_min.flatten().tolist()}, scale={scale.flatten().tolist()})",
                    "blue",
                )

        if state["mode"] == "zscore":
            return (edge_attr - state["mean"]) / state["std"], state
        if state["mode"] == "minmax":
            return (edge_attr - state["min"]) / state["scale"], state
        raise ValueError(f"Wrong original edge_attr normalization state: {state['mode']}")

    def has_node_task_data_precursor(self, matrix_type=None, use_sub_edge_index=False, **kwargs):
        name_key = repr_kvs(mmt=matrix_type, use_sei=use_sub_edge_index, **kwargs)
        path = self.path / f"{self.node_task_name}_node_task_data_precursor_{name_key}.pth"
        return path.is_file(), path

    def node_task_data_precursor(self, matrix_type=None, use_sub_edge_index=False, save=True, **kwargs):
        name_key = repr_kvs(mmt=matrix_type, use_sei=use_sub_edge_index, **kwargs)
        path = self.path / f"{self.node_task_name}_node_task_data_precursor_{name_key}.pth"
        try:
            self._node_task_data_precursor = torch.load(path)
            cprint(f"Load: {self._node_task_data_precursor} at {path}", "green")
            return self._node_task_data_precursor
        except FileNotFoundError:
            pass

        # Node aggregation: x, y, batch, ...
        # DataBatch(x=[16236, 1], y=[1591], split=[1591], batch=[16236], ptr=[1592])
        if use_sub_edge_index:
            rn_transform = RelabelNodes()
            self.subgraph_data_list = [rn_transform(d) for d in self.subgraph_data_list]
            self._node_task_data_precursor = Batch.from_data_list(self.subgraph_data_list)
        else:
            self._node_task_data_precursor = Batch.from_data_list(self.subgraph_data_list)

        # Row-wise sorting for using mapping_matrix_values without indices.
        batch, x = coalesce(torch.stack([self._node_task_data_precursor.batch,
                                         self._node_task_data_precursor.x.squeeze()]))
        # Relabel nodes in (sub)_edge_index based on coalesced batch and x
        if use_sub_edge_index:
            assert x.size(0) == self._node_task_data_precursor.x.squeeze().size(0)
            N = x.max().item() + 1
            bx = self._node_task_data_precursor.batch * N + self._node_task_data_precursor.x.squeeze()
            coalesce_bx = batch * N + x
            bx_to_idx = torch.full((bx.max().item() + 1,), fill_value=-1, dtype=torch.long)
            bx_to_idx[bx] = torch.arange(bx.size(0))
            idx_to_coalesce_idx = torch.full((bx.size(0),), fill_value=-1, dtype=torch.long)
            idx_to_coalesce_idx[bx_to_idx[coalesce_bx]] = torch.arange(bx.size(0))
            self._node_task_data_precursor.sub_edge_index = idx_to_coalesce_idx[
                self._node_task_data_precursor.edge_index]

        self._node_task_data_precursor.batch, self._node_task_data_precursor.x = batch, x.unsqueeze(1)
        del self._node_task_data_precursor.edge_index

        # Edge aggregation
        if self.target_matrix.startswith("adjacent"):
            self._node_task_data_precursor.edge_weight_matrix = self.get_ewmat_by_multiplying_adj(matrix_type)
        elif self.target_matrix == "shortest_path":
            self._node_task_data_precursor.edge_weight_matrix = self.get_ewmat_by_aggregating_sub_spl_mat(save)

        if save:
            torch.save(self._node_task_data_precursor, path)
            cprint(f"Saved: {self._node_task_data_precursor} at {path}", "blue")

        return self._node_task_data_precursor

    def node_task_data_splits(self,
                              mapping_matrix_type: str = None,
                              set_sub_x_weight: Optional[str] = "follow_mapping_matrix",
                              use_sub_edge_index: bool = False,
                              post_edge_normalize: Union[str, Callable, None] = None,
                              post_edge_normalize_args: Union[List, None] = None,
                              edge_thres: Union[float, Callable, List[float]] = 1.0,
                              use_consistent_processing=False,
                              save=True, load=True, is_custom_split=False, **kwargs) -> Tuple[Data, Data, Data]:
        """
        :return: Data(x=[N, 1], edge_index=[2, E], edge_attr=[E], y=[C], batch=[N])
            - N is the number of subgraphs = batch.sum()
            - edge_attr >= edge_thres
        """
        post_edge_normalize_args = post_edge_normalize_args or []
        if isinstance(post_edge_normalize, str):
            post_edge_normalize = func_normalize(post_edge_normalize, *post_edge_normalize_args)
        str_et = edge_thres.__name__ if isinstance(edge_thres, Callable) else edge_thres
        str_en = '-'.join(
            [post_edge_normalize.__name__ if isinstance(post_edge_normalize, Callable) else post_edge_normalize] +
            [str(round(a, 3)) for a in post_edge_normalize_args]  # todo: general repr for args
        ) if post_edge_normalize is not None else None

        name_key_kvs = dict(mmt=mapping_matrix_type, xw=set_sub_x_weight, sei=use_sub_edge_index,
                            et=str_et, en=str_en, ucp=use_consistent_processing)
        if self.original_edge_attr_mode is not None:
            name_key_kvs.update(
                oea=self.original_edge_attr_mode,
                oea_aggr=self.original_edge_attr_aggr,
                oea_norm=self.original_edge_attr_normalize,
                oea_eps=self.original_edge_attr_norm_eps,
                oea_fill=self.original_edge_attr_fill_value,
            )
        if not is_custom_split:
            name_key = repr_kvs(**name_key_kvs)
        else:
            name_key = repr_kvs(**name_key_kvs, splits="_".join([str(s) for s in self.splits]))

        path = self.path / f"{self.node_task_name}_node_task_data_{name_key}.pth"
        self._node_task_data_list = []
        try:
            if load:
                self._node_task_data_list = torch.load(path)
                cprint(f"Load: {self._node_task_data_list} at {path}", "green")
                return self._node_task_data_list
        except FileNotFoundError:
            pass

        node_task_data_precursor = self.node_task_data_precursor(mapping_matrix_type, use_sub_edge_index, **kwargs)
        self._original_edge_attr_matrix_type = mapping_matrix_type
        ew_mat = node_task_data_precursor.edge_weight_matrix

        train_val_test_splits = self.splits[-3:] if self.splits and len(self.splits) > 3 else self.splits

        if not isinstance(edge_thres, list):
            train_val_test_splits = train_val_test_splits or []
            edge_thres = [edge_thres] * len(train_val_test_splits)

        edge_norm_kws: Dict[str, Any] = {}
        original_edge_attr_norm_state: Optional[Dict[str, Tensor]] = None
        train_val_test_splits = train_val_test_splits or []
        for i, (s, et) in enumerate(zip(train_val_test_splits, edge_thres)):
            x, y, batch, ptr, sub_edge_index = try_getattr(node_task_data_precursor,
                                                           ["x", "y", "batch", "ptr", "sub_edge_index"],
                                                           default=None, as_dict=False)
            s_0, s_1 = self.num_start, self.num_start + s
            sub_x = x[ptr[s_0]:ptr[s_1], :]
            sub_batch = batch[ptr[s_0]:ptr[s_1]]
            sub_batch = sub_batch - sub_batch.min()  # if ptr[s_0] is not 0, sub_batch can be > 0.
            y = y[s_0:s_1]

            if sub_edge_index is not None:
                sub_edge_index = filter_living_edge_index(
                    sub_edge_index - ptr[s_0],  # sub_x is truncated when ptr[s_0] > 0.
                    num_nodes=sub_x.size(0), min_index=0)

            num_nodes = y.size(0)
            train_mask, eval_mask = None, None
            if i == 0 and torch.sum(y < 0) > 0:
                # Training samples contain coarsened nodes
                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                train_mask[y >= 0] = True
            if i > 0:
                eval_mask = torch.zeros(num_nodes, dtype=torch.bool)
                if train_val_test_splits:
                    eval_mask[train_val_test_splits[i - 1]:] = True

            if ew_mat.is_sparse:
                # Sparse-only path to avoid huge dense matrices.
                if post_edge_normalize is not None or isinstance(et, Callable):
                    raise ValueError("Sparse edge_weight_matrix requires numeric edge_thres and no post_edge_normalize.")

                ew_coo = ew_mat.coalesce()
                rows, cols = ew_coo.indices()
                vals = ew_coo.values()

                mask = (rows >= s_0) & (rows < s_1) & (cols >= s_0) & (cols < s_1)
                rows = rows[mask] - s_0
                cols = cols[mask] - s_0
                vals = vals[mask]

                et = et if not isinstance(et, Callable) else et(vals)
                if et is not None:
                    keep = vals >= et
                    rows, cols, vals = rows[keep], cols[keep], vals[keep]

                edge_index = torch.stack([rows, cols], dim=0)
                edge_attr = vals
            else:
                ew_mat_s_by_s = ew_mat.clone()[s_0:s_1, s_0:s_1]
                if post_edge_normalize is not None:
                    if use_consistent_processing:
                        ew_mat_s_by_s, edge_norm_kws = post_edge_normalize(ew_mat_s_by_s, **edge_norm_kws)
                    else:
                        ew_mat_s_by_s, edge_norm_kws = post_edge_normalize(ew_mat_s_by_s)
                # Remove ew_mat below than edge_thres
                et = et(ew_mat_s_by_s) if isinstance(et, Callable) else et
                ew_mat_s_by_s[ew_mat_s_by_s < et] = 0
                if i == 0:
                    self.print_mat_stat(ew_mat, "Summarizing edge_weight_matrix")
                self.print_mat_stat(ew_mat_s_by_s, f"Summarizing processed edge_weight_matrix ({i})")

                edge_index, edge_attr = dense_to_sparse(ew_mat_s_by_s)

            if self.original_edge_attr_mode is not None:
                original_edge_attr = self.lookup_original_edge_attr(edge_index, s_0)
                if original_edge_attr is not None:
                    original_edge_attr, original_edge_attr_norm_state = self.normalize_original_edge_attr(
                        original_edge_attr,
                        original_edge_attr_norm_state,
                    )
                    if self.original_edge_attr_mode == "append":
                        edge_attr = torch.cat([edge_attr.view(-1, 1), original_edge_attr], dim=-1)
                    elif self.original_edge_attr_mode == "replace":
                        edge_attr = original_edge_attr

            sub_x_weight = None
            if set_sub_x_weight is None:
                pass
            elif (self._mapping_matrix_value is not None) and set_sub_x_weight == "follow_mapping_matrix":
                sub_x_weight = self._mapping_matrix_value[ptr[s_0]:ptr[s_1]]
            elif "sqrt_d_node_div_d_sub" in set_sub_x_weight:
                if set_sub_x_weight == "sparse_sqrt_d_node_div_d_sub":
                    s_index = edge_index
                elif set_sub_x_weight == "original_sqrt_d_node_div_d_sub":
                    s_index, _ = dense_to_sparse(ew_mat[s_0:s_1, s_0:s_1])
                else:
                    raise ValueError(f"Wrong set_sub_x_weight: {set_sub_x_weight}")

                a_index, _ = add_remaining_self_loops(self.global_data.edge_index)
                _, sub_x_weight = self.get_sparse_mapping_matrix_sxn(
                    matrix_type="sqrt_d_node_div_d_sub",
                    sub_x=sub_x.squeeze(),
                    sub_batch=sub_batch,
                    global_edge_index=a_index,
                    summarized_edge_index=s_index,
                )

            edge_attr = edge_attr.view(-1, 1) if edge_attr.dim() == 1 else edge_attr

            self._node_task_data_list.append(Data(
                sub_x=sub_x, sub_batch=sub_batch,
                sub_x_weight=sub_x_weight, sub_edge_index=sub_edge_index,
                y=y, train_mask=train_mask, eval_mask=eval_mask,
                edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=num_nodes,
            ))

        if save:
            torch.save(self._node_task_data_list, path)
            cprint(f"Saved: {self._node_task_data_list} at {path}", "blue")

        return tuple(self._node_task_data_list)

    def node_task_add_sub_x_wl(self, s2n_data_list: List[Data],
                               separated_data_list: List[List[Data]]):
        num_layer = 3  # NOTE: num_layer is hard-coded.
        separated_wl_list = ReplaceXWithWL4Pattern(
            num_layers=num_layer,
            wl_step_to_use=-1,  # Last step
            wl_type_to_use="color",
            cache_path=(self.path / f"sub_wl_L={num_layer}.pth"),
            cumcat=True,
        )(separated_data_list)

        # todo: generalize & argnize
        """
        if self.name == "EMUser":
            reduce_dim = VarianceThreshold(5e-5)
        else:
            reduce_dim = VarianceThreshold(5e-4)
        """
        reduce_dim = PCA(n_components=128)

        for idx, (sep_wl_data, s2n_data) in enumerate(zip(separated_wl_list, s2n_data_list)):
            if idx == 0:
                reduce_dim.fit(sep_wl_data.x)
            s2n_data.sub_x_wl = torch.from_numpy(reduce_dim.transform(sep_wl_data.x)).float()
        return s2n_data_list

    @staticmethod
    def print_mat_stat(matrix, start=None, print_counter=False):

        def safe_quantile(t, q):
            try:
                return round(torch.quantile(t, q).item(), _decimal)
            except RuntimeError:
                return "NA"

        _decimal = 5
        _mean = lambda t: round(torch.mean(t).item(), _decimal)
        _std = lambda t: round(torch.std(t).item(), _decimal)
        _min = lambda t: round(torch.min(t).item(), _decimal)
        _median = lambda t: round(torch.median(t).item(), _decimal)
        _1q = lambda t: safe_quantile(t, 0.25)
        _3q = lambda t: safe_quantile(t, 0.75)

        _max = lambda t: round(torch.max(t).item(), _decimal)
        if start:
            cprint(start, "green")
        matrix_pos = matrix[matrix > 0]

        print(
            f"\tmean / std = {_mean(matrix)} / {_std(matrix)} \n"
            f"\tmin / 1q / median / 3q / max = {_min(matrix)} / {_1q(matrix)} / {_median(matrix)}"
            f" / {_3q(matrix)} / {_max(matrix)} \n"
            f"\tmean+ / std+ = {_mean(matrix_pos)} / {_std(matrix_pos)} \n"
            f"\tmin+ / 1q+ / median+ / 3q+ / max+ = {_min(matrix_pos)} / {_1q(matrix_pos)} / {_median(matrix_pos)}"
            f" / {_3q(matrix_pos)} / {_max(matrix_pos)} \n"
            f"\tN = {matrix.numel()}, N+ = {(matrix > 0).sum().item()}, "
            f"d = {(matrix > 0).sum().item() / matrix.numel()}"
        )
        if print_counter:
            print("\tCounters: ", Counter(matrix.flatten().tolist()))


def func_topk_thres(thres):
    def _func(x):
        k = int(x.numel() * thres)
        topk = torch.topk(x.flatten(), k, sorted=False).values
        return torch.min(topk).item()

    _func.__name__ = f"topk_{thres}"

    return _func


def dist_by_shared_nodes(node_spl_mat):
    non_shared_nodes = torch.count_nonzero(node_spl_mat)
    shared_nodes = node_spl_mat.numel() - non_shared_nodes
    # edge_weight = 1 / (1 + d) = 1 / (1 + -1 + (1 / shared_nodes)) = shared_nodes
    return -1 + (1 / shared_nodes)


def func_normalize(normalize_type: str, *args):
    def _func(matrix: Tensor, **kws) -> (Tensor, Dict):
        if len(kws) == 0:
            kws = {"mean": torch.mean(matrix),
                   "std": torch.std(matrix),
                   "mean_pos": torch.mean(matrix[matrix > 0]),
                   "std_pos": torch.std(matrix[matrix > 0]),
                   "max": torch.max(matrix)}
        if normalize_type == "standardize_then_thres_max_linear":
            assert len(args) == 1, f"Wrong args: {args}"
            thres = args[0]
            matrix = (matrix - kws["mean"]) / kws["std"]
            matrix = (matrix - thres) / (kws["max"] - thres)
        elif normalize_type == "standardize_then_trunc_thres_max_linear":
            assert len(args) == 2, f"Wrong args: {args}"
            assert args[1] > 0
            thres, trunc_diff = args[0], args[1]
            trunc_val = thres + trunc_diff
            matrix = (matrix - kws["mean"]) / kws["std"]
            matrix[matrix >= trunc_val] = trunc_val
            matrix = (matrix - thres) / (trunc_val - thres)
        elif normalize_type == "standardize_then_thres_max_power":
            assert len(args) == 2, f"Wrong args: {args}"
            thres, p = args[0], args[1]
            matrix = (matrix - kws["mean"]) / kws["std"]
            matrix = (matrix.relu_() ** p - thres ** p) / (kws["max"] ** p - thres ** p)
        elif normalize_type == "clamp_1":
            matrix[matrix >= 1.] = 1.
        elif normalize_type == "cut_mean_pos_k_std_pos_and_clamp_1":
            k = args[0]
            mean_k_std = kws["mean_pos"] + k * kws["std_pos"]
            matrix[matrix <= mean_k_std] = 0.
            matrix[matrix >= 1.] = 1.
        else:
            raise ValueError(f"Wrong type: {normalize_type}")
        return matrix, kws

    _func.__name__ = f"normalize_{normalize_type}"

    return _func


if __name__ == '__main__':

    from data_sub import HPOMetab, HPONeuro, PPIBP, EMUser, Density, Component, Coreness, CutRatio

    MODE = "PPIBP"
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, Component, Coreness, CutRatio
    PURPOSE = "MEASURE_TIME"
    # MANY, ONCE
    TARGET_MATRIX = "adjacent_with_self_loops"
    # adjacent_with_self_loops, adjacent_no_self_loops

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "glass"
    DEBUG = False

    if PURPOSE == "PRECURSOR":
        _cls = eval(MODE)
        dts = _cls(root=PATH, name=MODE, debug=DEBUG, embedding_type=E_TYPE,
                   num_training_tails_to_tile_per_class=80)
        _subgraph_data_list = dts.get_data_list_with_split_attr()
        _global_data = dts.global_data

        s2n = SubgraphToNode(
            _global_data, _subgraph_data_list,
            name=MODE,
            path=f"{PATH}/{MODE.upper()}/sub2node/",
            undirected=True,
            splits=dts.splits,
            target_matrix=TARGET_MATRIX,
        )
        s2n.node_task_data_precursor(matrix_type="unnormalized", use_sub_edge_index=True, ntt2tpc=80)
        s2n.node_task_data_precursor(matrix_type="unnormalized", use_sub_edge_index=False, ntt2tpc=80)
        exit()

    if MODE in ["HPOMetab", "PPIBP", "HPONeuro", "EMUser",
                "Density", "Component", "Coreness", "CutRatio"]:
        _cls = eval(MODE)
        dts = _cls(root=PATH, name=MODE, debug=DEBUG, embedding_type=E_TYPE)
        _subgraph_data_list = dts.get_data_list_with_split_attr()
        _global_data = dts.global_data

        s2n = SubgraphToNode(
            _global_data, _subgraph_data_list,
            name=MODE,
            path=f"{PATH}/{MODE.upper()}/sub2node/",
            undirected=True,
            splits=dts.splits,
            target_matrix=TARGET_MATRIX,
            edge_aggr=dist_by_shared_nodes,
        )
        print(s2n)
        """ Inverse sigmoid table 0.5 -- 0.95,
        inv_sig = [0.0, 0.201, 0.405, 0.619, 0.847, 1.099, 1.386, 1.735, 2.197, 2.944]
        """
        if PURPOSE == "MEASURE_TIME":
            import time

            t0 = time.time()
            ntds = s2n.node_task_data_splits(
                mapping_matrix_type="unnormalized",
                set_sub_x_weight=None,
                use_sub_edge_index=True,
                post_edge_normalize="standardize_then_trunc_thres_max_linear",
                post_edge_normalize_args=[2.1, 1.0],
                edge_thres=0.0,
                use_consistent_processing=True,
                save=False,
            )
            print((time.time() - t0) / 3)
        elif PURPOSE == "MANY_1":
            # standardize_then_thres_max_linear
            for i in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                      2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]:
                ntds = s2n.node_task_data_splits(
                    mapping_matrix_type="unnormalized",
                    post_edge_normalize="standardize_then_thres_max_linear",
                    post_edge_normalize_args=[i],
                    edge_thres=0.0,
                    use_consistent_processing=True,
                    save=True,
                )
                for _d in ntds:
                    print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))
                s2n._node_task_data_list = []  # flush
        elif PURPOSE == "MANY_2":
            # standardize_then_trunc_thres_max_linear, standardize_then_thres_max_power
            for i in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                      2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]:
                for j in [0.5, 1.0, 1.5, 2.0]:
                    ntds = s2n.node_task_data_splits(
                        mapping_matrix_type="unnormalized",
                        post_edge_normalize="standardize_then_trunc_thres_max_linear",
                        post_edge_normalize_args=[i, j],
                        edge_thres=0.0,
                        use_consistent_processing=True,
                        save=True,
                    )
                    for _d in ntds:
                        print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))
                    s2n._node_task_data_list = []  # flush
        elif PURPOSE == "MANY_3":
            # unnormalized, sqrt_d_node_div_d_sub, 1_div_sqrt_num_nodes_in_sub
            # cut_mean_pos_k_std_pos_and_clamp_1
            for i in [3.0, 2.0, 1.0, 0.0]:
                ntds = s2n.node_task_data_splits(
                    mapping_matrix_type="1_div_sqrt_num_nodes_in_sub",
                    post_edge_normalize="cut_mean_pos_k_std_pos_and_clamp_1",
                    post_edge_normalize_args=[i],
                    edge_thres=0.0,
                    use_consistent_processing=True,
                    save=True,
                )
                for _d in ntds:
                    print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))
                s2n._node_task_data_list = []  # flush

        elif PURPOSE == "MANY_4":
            # unnormalized, sqrt_d_node_div_d_sub, original_sqrt_d_node_div_d_sub
            # standardize_then_trunc_thres_max_linear
            for i in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                      2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]:
                for j in [0.5, 1.0, 1.5, 2.0]:
                    ntds = s2n.node_task_data_splits(
                        mapping_matrix_type="unnormalized",
                        set_sub_x_weight=None,
                        use_sub_edge_index=True,
                        post_edge_normalize="standardize_then_trunc_thres_max_linear",
                        post_edge_normalize_args=[i, j],
                        edge_thres=0.0,
                        use_consistent_processing=True,
                        save=True,
                    )
                    for _d in ntds:
                        print(_d)
                        print(f"\t- density: {_d.edge_index.size(1) / (_d.num_nodes ** 2)}")
                        if hasattr(_d, "sub_x_weight"):
                            _sub_x_weight_stats = repr_kvs(
                                min=torch.min(_d.sub_x_weight), max=torch.max(_d.sub_x_weight),
                                avg=torch.mean(_d.sub_x_weight), std=torch.std(_d.sub_x_weight), sep=", ")
                            print(f"\t- sub_x_weight: {_sub_x_weight_stats}")
                    s2n._node_task_data_list = []  # flush

        elif PURPOSE == "WEIGHT_DIST":
            ntdp = s2n.node_task_data_precursor(matrix_type="unnormalized", use_sub_edge_index=False, save=False)
            ewm = ntdp.edge_weight_matrix.flatten()
            ewm_pos = ewm[ewm > 0]

            s1_ewm_pos = (ewm_pos - torch.mean(ewm_pos)) / torch.std(ewm_pos)
            s2_ewm_pos = (ewm_pos - torch.mean(ewm)) / torch.std(ewm)

            plot_dis("hist", xs=torch.log(ewm_pos).tolist(), xlabel="log edge weights",
                     path="../_figures", key=f"{MODE}_ew", extension="png",
                     scales_kws={"yscale": "log"},
                     )

            plot_dis("kde", xs=s1_ewm_pos.tolist(), xlabel="edge weights",
                     path="../_figures", key=f"{MODE}_ew_s1", extension="png",
                     # scales_kws={"xscale": "log"},
                     )
            plot_dis("kde", xs=s2_ewm_pos.tolist(), xlabel="edge weights",
                     path="../_figures", key=f"{MODE}_ew_s2", extension="png",
                     # scales_kws={"xscale": "log"},
                     )

            plot_dis("kde", xs=ewm_pos.tolist(), xlabel="edge weights",
                     path="../_figures", key=f"{MODE}_ew", extension="png",
                     # scales_kws={"xscale": "log"},
                     )

        elif PURPOSE == "SUB_SIZE":
            szs = [s.x.size(0) for s in _subgraph_data_list]
            plot_dis("kde", xs=szs, xlabel="subgraph sizes",
                     path="../_figures", key=f"{MODE}_subgraph_sizes", extension="png",
                     # scales_kws={"xscale": "log"},
                     )
            plot_dis("hist", xs=szs, xlabel="subgraph sizes",
                     path="../_figures", key=f"{MODE}_subgraph_sizes", extension="png",
                     # scales_kws={"xscale": "log"},
                     )

        else:
            raise ValueError(f"Wrong purpose: {PURPOSE}")


def test_subgraph_to_node(is_weighted=True):
    from data_sub import PPIBP
    NAME = "PPIBP"
    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "glass"
    DEBUG = False

    dts: PPIBP = PPIBP(root=PATH, name=NAME, embedding_type=E_TYPE, debug=DEBUG)
    _global_data, _subgraph_data_list = dts.global_data, dts.tolist()

    if is_weighted:
        s2n = SubgraphToNode(
            _global_data, _subgraph_data_list,
            name="test",
            path="/mnt/nas2/GNN-DATA/SUBGRAPH/PPIBP/sub2node/",
            splits=dts.splits,
            num_start=dts.num_start,
            target_matrix='adjacent_with_self_loops',
            edge_aggr=dist_by_shared_nodes,
        )
    else:
        s2n = SubgraphToNode(
            _global_data, _subgraph_data_list,
            name="test",
            path="/mnt/nas2/GNN-DATA/SUBGRAPH/PPIBP/sub2node/",
            splits=dts.splits,
            num_start=dts.num_start,
            target_matrix='adjacent_with_self_loops',
        )

    # === Test cases ===
    if is_weighted:
        # 1. Unweighted mapping matrix (all ones)
        # 1-a. w/o sub_edge_index
        ntds = s2n.node_task_data_splits(
            mapping_matrix_type="unnormalized",
            use_sub_edge_index=False,
            save=True,
        )
        for _d in ntds:
            print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))

        # 1-b. w/ sub_edge_index
        ntds = s2n.node_task_data_splits(
            mapping_matrix_type="unnormalized",
            use_sub_edge_index=True,
            save=True,
        )
        for _d in ntds:
            print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))

        # 2. Degree-normalized mapping matrix
        # 2-a. w/o sub_edge_index
        ntds = s2n.node_task_data_splits(
            mapping_matrix_type="degree_normalized_by_sub",
            use_sub_edge_index=False,
            save=True,
        )
        for _d in ntds:
            print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))

        # 2-b. w/ sub_edge_index
        ntds = s2n.node_task_data_splits(
            mapping_matrix_type="degree_normalized_by_sub",
            use_sub_edge_index=True,
            save=True,
        )
        for _d in ntds:
            print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))

        # 3. Sqrt-degree-normalized mapping matrix
        # 3-a. w/o sub_edge_index, w/ summarized_edge_index
        ntds = s2n.node_task_data_splits(
            mapping_matrix_type="sqrt_degree_normalized_by_sub",
            use_sub_edge_index=False,
            save=True,
        )
        for _d in ntds:
            print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))

        # 3-b. w/ sub_edge_index, w/ summarized_edge_index
        ntds = s2n.node_task_data_splits(
            mapping_matrix_type="sqrt_degree_normalized_by_sub",
            use_sub_edge_index=True,
            save=True,
        )
        for _d in ntds:
            print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))

    else:
        ntds = s2n.node_task_data_splits(
            mapping_matrix_type="unnormalized",
            edge_thres=0.1,
        )
        for _d in ntds:
            print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))


def test_subgraph_to_node_wo_processing():
    from data_sub import PPIBP
    NAME = "PPIBP"
    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "glass"
    DEBUG = False

    dts: PPIBP = PPIBP(root=PATH, name=NAME, embedding_type=E_TYPE, debug=DEBUG)
    _global_data, _subgraph_data_list = dts.global_data, dts.get_data_list_with_split_attr()

    s2n = SubgraphToNode(
        _global_data, _subgraph_data_list,
        name=dts.name,
        path="/mnt/nas2/GNN-DATA/SUBGRAPH/PPIBP/sub2node/",
        splits=dts.splits,
        num_start=dts.num_start,
        target_matrix='adjacent_with_self_loops',
    )

    # === Test cases ===
    # 1. Unweighted mapping matrix (all ones)
    ntds = s2n.node_task_data_splits(
        mapping_matrix_type="unnormalized",
        edge_thres=0.1,
    )
    for _d in ntds:
        print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))


def test_subgraph_to_node_inconsistent_processing():
    from data_sub import PPIBP
    NAME = "PPIBP"
    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "glass"
    DEBUG = False

    dts: PPIBP = PPIBP(root=PATH, name=NAME, embedding_type=E_TYPE, debug=DEBUG)
    _global_data, _subgraph_data_list = dts.global_data, dts.get_data_list_with_split_attr()

    s2n = SubgraphToNode(
        _global_data, _subgraph_data_list,
        name=dts.name,
        path="/mnt/nas2/GNN-DATA/SUBGRAPH/PPIBP/sub2node/",
        splits=dts.splits,
        num_start=dts.num_start,
        target_matrix='adjacent_with_self_loops',
    )

    # === Test cases ===
    # 1. Unweighted mapping matrix (all ones)
    ntds = s2n.node_task_data_splits(
        mapping_matrix_type="unnormalized",
        edge_thres=0.1,
    )
    for _d in ntds:
        print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))


def test_subgraph_to_node_wl_add():
    from data_sub import PPIBP
    NAME = "PPIBP"
    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "glass"
    DEBUG = False

    dts: PPIBP = PPIBP(root=PATH, name=NAME, embedding_type=E_TYPE, debug=DEBUG)
    _global_data, _subgraph_data_list = dts.global_data, dts.get_data_list_with_split_attr()

    s2n = SubgraphToNode(
        _global_data, _subgraph_data_list,
        name=dts.name,
        path="/mnt/nas2/GNN-DATA/SUBGRAPH/PPIBP/sub2node/",
        splits=dts.splits,
        num_start=dts.num_start,
        target_matrix='adjacent_with_self_loops',
    )

    # === Test cases ===
    # 1. Unweighted mapping matrix (all ones)
    ntds = s2n.node_task_data_splits(
        mapping_matrix_type="unnormalized",
        edge_thres=0.1,
    )
    for _d in ntds:
        print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))

    ntds_wl = s2n.node_task_add_sub_x_wl(
        s2n_data_list=list(ntds),
        separated_data_list=list(dts.get_train_val_test_with_individual_relabeling()),
    )
    for _d in ntds_wl:
        print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))


def test_subgraph_to_node_custom_splits():
    from data_sub import PPIBP
    NAME = "PPIBP"
    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "glass"
    DEBUG = False

    dts: PPIBP = PPIBP(root=PATH, name=NAME, embedding_type=E_TYPE, debug=DEBUG)
    dts.set_num_start_train_by_num_train_per_class(10)
    _global_data, _subgraph_data_list = dts.global_data, dts.get_data_list_with_split_attr()

    s2n = SubgraphToNode(
        _global_data, _subgraph_data_list,
        name=dts.name,
        path="/mnt/nas2/GNN-DATA/SUBGRAPH/PPIBP/sub2node/",
        splits=dts.splits,
        num_start=dts.num_start,
        target_matrix='adjacent_with_self_loops',
    )

    # === Test cases ===
    # 1. Unweighted mapping matrix (all ones)
    ntds = s2n.node_task_data_splits(
        mapping_matrix_type="unnormalized",
        edge_thres=0.1,
    )
    for _d in ntds:
        print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))
