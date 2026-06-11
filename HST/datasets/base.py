import os
from torch_geometric.data import Data

from typing import List, Literal, Tuple, Union, Any

DatasetType = Literal["elliptic", "subgnn", "ibm_aml", "aml", "saml_d"]
Embeddingtype = Literal["glass", "gin", "graphsaint_gcn"]

class BaseSubgraphDataset:
    def __init__(self, root: str, name: str, verbose: bool = False):
        self.root = root
        self.name = name
        self.verbose = verbose
        self.base_path = os.path.join(root, name)

    def load(self) -> Tuple[Data, List[Data]]:
        raise NotImplementedError

    def split(self, subgraphs: List[Data], train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[List[Data], List[Data], List[Data]]:
        raise NotImplementedError
