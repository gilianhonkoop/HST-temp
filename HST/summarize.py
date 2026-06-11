import warnings
import torch
from torch_geometric.data import Data
import argparse

import aggregators
import time

# Suppress warnings
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

import logging
logging.getLogger('pygsp').setLevel(logging.ERROR)
logging.getLogger('pygsp.graphs').setLevel(logging.ERROR)
logging.getLogger('pygsp.graphs.graph').setLevel(logging.ERROR)

class SubgraphSummarizer:
    def __init__(
        self,
        global_data,
        subgraph_data_list,
        node_aggregator=None,
        edge_aggregator=None,
        verbose=False,
        node_batch_size: int | None = None,
    ):
        """
        global_data: Data object containing the whole graph (edge_index, x)
        subgraph_data_list: List of Data objects representing individual subgraphs.
        node_aggregator: nn.Module implementation of NodeAggregator.
        edge_aggregator: nn.Module implementation of EdgeAggregator.
        """
        self.global_data = global_data
        self.subgraph_data_list = subgraph_data_list
        self.num_subgraphs = len(subgraph_data_list)
        self.num_global_nodes = global_data.num_nodes
        self.verbose = verbose
        self.node_batch_size = node_batch_size
        
        # Default to Mean aggregators if none provided
        self.node_aggregator = node_aggregator if node_aggregator is not None else aggregators.MeanNodeAggregator()
        self.edge_aggregator = edge_aggregator if edge_aggregator is not None else aggregators.MeanEdgeAggregator()
        
        # Caching: membership matrix and summarized graph can be reused
        self._m_index: torch.Tensor | None = None
        self._m_value: torch.Tensor | None = None
        self._cached_graph: Data | None = None

        # Enable graph-level caching only when both aggregators are frozen (no trainable params)
        def _frozen(m: torch.nn.Module | None) -> bool:
            if m is None:
                return True
            params = list(m.parameters())
            return len(params) == 0 or all(not p.requires_grad for p in params)

        self.cache_enabled = _frozen(self.node_aggregator) and _frozen(self.edge_aggregator)

        if hasattr(self.node_aggregator, 'prepare'):
            try:
                self.node_aggregator.prepare(self.global_data, self.subgraph_data_list)
            except Exception:
                pass

    def build_mapping_matrix(self):
        """
        Builds a sparse mapping matrix M of shape [S, N]
        where S is the number of subgraphs and N is the number of global nodes.
        M[s, n] = 1 if node n is in subgraph s.
        """
        sub_indices = []
        node_indices = []
        
        for s_idx, sub_data in enumerate(self.subgraph_data_list):
            # sub_data.x_idx contains the global node IDs for this subgraph
            n_ids = sub_data.x_idx.flatten().tolist()
            sub_indices.extend([s_idx] * len(n_ids))
            node_indices.extend(n_ids)
            
        # Place membership matrix on the same device as the global graph (prefer edge_index, else x)
        ref = self.global_data.edge_index if getattr(self.global_data, 'edge_index', None) is not None else self.global_data.x
        device = ref.device if ref is not None else torch.device('cpu')

        m_index = torch.tensor([sub_indices, node_indices], dtype=torch.long, device=device)
        m_value = torch.ones(m_index.size(1), dtype=torch.float, device=device)
        
        return m_index, m_value

    def summarize(self):
        """
        Returns a single PyG Data object where each node represents a summarized subgraph.
        """       
        # Fast path: reuse cached summarized graph if aggregators are frozen
        if self.cache_enabled and self._cached_graph is not None:
            return self._cached_graph

        start = time.time()
        sx = self._summarize_nodes()
        node_end = time.time()

        # Summarize Graph Structure (e.g. M * A * M^T)
        # Build or reuse membership matrix
        if self._m_index is None or self._m_value is None:
            self._m_index, self._m_value = self.build_mapping_matrix()
        m_index, m_value = self._m_index, self._m_value

        edge_start = time.time()
        s_edge_index, s_edge_attr = self.edge_aggregator(self.global_data, m_index, m_value, self.num_subgraphs)
        edge_end = time.time()
        
        # Gather Labels
        # Handle both single-label and multi-label cases properly
        sy = torch.stack([d.y.squeeze() for d in self.subgraph_data_list])
        
        # Gather Optional Temporal Attributes (crucial for LinkNeighborLoader sampling)
        times = []
        for d in self.subgraph_data_list:
            t = getattr(d, 'preset_time', None)
            if t is not None:
                times.append(int(t))
        stime = torch.tensor(times, dtype=torch.long) if len(times) == self.num_subgraphs else None
        
        summarized_graph = Data(
            x=sx,
            edge_index=s_edge_index,
            edge_attr=s_edge_attr,
            y=sy,
            time=stime,
            num_nodes=self.num_subgraphs
        )
        
        num_edges = summarized_graph.edge_index.size(1) if summarized_graph.edge_index is not None else 0
        # if self.verbose: print(f"Created summarized graph: {summarized_graph.num_nodes} nodes, {num_edges} edges.")

        end = time.time()

        if self.verbose:
            logging.info(f"Total summarization time: {end - start}, Node aggregation time: {node_end - start}, Edge aggregation time: {edge_end - edge_start}")

        # Update cache only when aggregators are frozen
        if self.cache_enabled:
            self._cached_graph = summarized_graph

        return summarized_graph

    def _summarize_nodes(self):
        if (
            self.node_batch_size is None
            or self.node_batch_size <= 0
            or self.num_subgraphs <= self.node_batch_size
        ):
            return self.node_aggregator(self.global_data, self.subgraph_data_list)

        chunks = []
        for start in range(0, self.num_subgraphs, self.node_batch_size):
            end = min(start + self.node_batch_size, self.num_subgraphs)
            chunks.append(self.node_aggregator(self.global_data, self.subgraph_data_list[start:end]))
        return torch.cat(chunks, dim=0)

def parse_args():
    parser = argparse.ArgumentParser(description="Summarize subgraphs into a supergraph.")
    parser.add_argument("--data-path", type=str, default="/home/ghonkoop/data", help="Base path to datasets")
    parser.add_argument("--dataset-name", type=str, choices=["ppi_bp", "em_user", "hpo_metab", "hpo_neuro"], 
                        default="ppi_bp", help="Name of the dataset")
    parser.add_argument("--dataset-type", type=str, choices=["elliptic", "subgnn"], default="subgnn",
                         help="Type of dataset")
    parser.add_argument("--embedding-type", type=str, choices=["glass", "gin", "graphsaint_gcn"], 
                        default="glass", help="Type of embeddings for subgnn")
    parser.add_argument("--node-pooling-mode", type=str, default="mean", 
                        help="Aggregation mode to use.")
    parser.add_argument("--edge-pooling-mode", type=str, default="mean", 
                        help="Aggregation mode to use.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def _mode_to_class_prefix(mode: str) -> str:
    """Convert mode strings like 'mean', 'attention_stats', 'attention-stats' to PascalCase."""
    parts = mode.replace("-", "_").split("_")
    parts = [p for p in parts if len(p) > 0]
    return "".join(p[0].upper() + p[1:] for p in parts)

def main():
    args = parse_args()
    from datasets.utils import init_dataset
    
    print(f"Loading dataset '{args.dataset_name}' of type '{args.dataset_type}' with embedding '{args.embedding_type}'...")

    ds = init_dataset(args.data_path, args.dataset_name, args.dataset_type, args.embedding_type, verbose=args.verbose)
    global_data, subgraphs_list = ds.load()

    print(f"Global Graph: {global_data.num_nodes} nodes, {global_data.num_edges if global_data.edge_index is not None else 0} edges\n")

    node_mode_capitalized = _mode_to_class_prefix(args.node_pooling_mode)
    edge_mode_capitalized = _mode_to_class_prefix(args.edge_pooling_mode)
    
    # Get correct Node Aggregator
    node_class_name = f"{node_mode_capitalized}NodeAggregator"
    if hasattr(aggregators, node_class_name):
        node_aggregator = getattr(aggregators, node_class_name)()
        if args.verbose: print(f"Loaded Node Aggregator: {node_class_name}")
    else:
        raise ValueError(f"{node_class_name} not found in aggregators. Please implement \
                 this class or choose a different node pooling mode.")

    # Get correct Edge Aggregator
    edge_class_name = f"{edge_mode_capitalized}EdgeAggregator"
    if hasattr(aggregators, edge_class_name):
        edge_aggregator = getattr(aggregators, edge_class_name)()
        if args.verbose: print(f"Loaded Edge Aggregator: {edge_class_name}")
    else:
        raise ValueError(f"{edge_class_name} not found in aggregators. Please implement \
                 this class or choose a different edge pooling mode.")

    # Summarize
    summarizer = SubgraphSummarizer(
        global_data, 
        subgraphs_list, 
        node_aggregator=node_aggregator, 
        edge_aggregator=edge_aggregator, 
        verbose=args.verbose
    )
    
    super_graph = summarizer.summarize()

    x_shape = torch.as_tensor(super_graph.x).shape
    y_shape = torch.as_tensor(super_graph.y).shape

    if summarizer.verbose | True:
        print("Super Graph nodes shape:", x_shape)
        print("Super Graph labels shape:", y_shape)
        print("Super Graph edge index shape:", super_graph.edge_index.shape 
              if super_graph.edge_index is not None else None)
        print("Super Graph edge attr shape:", super_graph.edge_attr.shape 
              if super_graph.edge_attr is not None else None)
        print(f"Is Directed: {super_graph.is_directed()}")
        print("Super graph keys:", super_graph.keys())
    
if __name__ == "__main__":
    main()
