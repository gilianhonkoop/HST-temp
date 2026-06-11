import os
import re
from typing import List, Union

import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from termcolor import cprint
from torch_geometric.data import Data
from torch_geometric.transforms import LocalDegreeProfile
from torch_geometric.utils import subgraph, sort_edge_index, to_undirected
from tqdm import tqdm

from data_base import SubgraphDataset, SynGraphDataset
from aml import AMLHI

SynSubgraphGLASSDataset = SynGraphDataset

__all__ = [
    'HPOMetab',
    'HPONeuro',
    'PPIBP',
    'EMUser',
    'Density',
    'Component',
    'Coreness',
    'CutRatio',
    'AMLHI',
    'SubgraphDataset'
]


class HPOMetab(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class HPONeuro(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class EMUser(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class PPIBP(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Density(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Coreness(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class CutRatio(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Component(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


if __name__ == '__main__':

    FIND_SEED = False  # NOTE: If True, find_seed_that_makes_balanced_datasets

    NAME = "EMUser"
    # WLKSRandomTree
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, Component, Coreness, CutRatio

    USE_RWPE = False
    USE_LEPE = True

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    if NAME.startswith("WL"):
        E_TYPE = "no_embedding"
    elif NAME in ["Density", "Component", "Coreness", "CutRatio"]:
        E_TYPE = "ones_1/64/LEPE"
    else:
        E_TYPE = "glass"  # gin, graphsaint_gcn, glass

    DEBUG = False

    MORE_KWARGS = {}
    if USE_RWPE and NAME not in ["Density", "Component", "Coreness", "CutRatio"]:
        MORE_KWARGS["load_rwpe"] = True
    elif USE_LEPE and NAME not in ["Density", "Component", "Coreness", "CutRatio"]:
        MORE_KWARGS["load_lepe"] = True

    dts: SubgraphDataset = eval(NAME)(
        root=PATH,
        name=NAME,
        embedding_type=E_TYPE,
        debug=DEBUG,
        **MORE_KWARGS,
    )

    train_dts, val_dts, test_dts = dts.get_train_val_test()

    dts.print_summary()

    cprint("Train samples", "yellow")
    for i, b in enumerate(train_dts):
        print(b)
        if i >= 5:
            break

    cprint("Validation samples", "yellow")
    for i, b in enumerate(val_dts):
        print(b)
        if i >= 5:
            break

    cprint("global_data samples", "yellow")
    print(dts.global_data)
    print("Avg. degree: ", dts.global_data.edge_index.size(1) / dts.global_data.num_nodes)
    if hasattr(dts.global_data, "pe"):
        print("PE", dts.global_data.pe)

    cprint("All subgraph samples", "magenta")
    print(dts.data)
    try:
        for k, vs in dts.y_stat_dict().items():
            print(k, [round(v, 3) for v in vs])
            for v in vs:
                print(round(v, 3))
    except AttributeError:
        pass

from pytorch_lightning import LightningDataModule
from data_base import SubgraphDataset, SynGraphDataset
from aml import AMLHI

SynSubgraphGLASSDataset = SynGraphDataset

__all__ = [
    'HPOMetab',
    'HPONeuro',
    'PPIBP',
    'EMUser',
    'Density',
    'Component',
    'Coreness',
    'CutRatio',
    'AMLHI',
    'SubgraphDataset'
]


class HPOMetab(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class HPONeuro(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class EMUser(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class PPIBP(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Density(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Coreness(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class CutRatio(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Component(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


if __name__ == '__main__':

    FIND_SEED = False  # NOTE: If True, find_seed_that_makes_balanced_datasets will be performed

    NAME = "EMUser"
    # WLKSRandomTree
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, Component, Coreness, CutRatio

    USE_RWPE = False
    USE_LEPE = True

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    if NAME.startswith("WL"):
        E_TYPE = "no_embedding"
    elif NAME in ["Density", "Component", "Coreness", "CutRatio"]:
        E_TYPE = "ones_1/64/LEPE"
    else:
        E_TYPE = "glass"  # gin, graphsaint_gcn, glass

    DEBUG = False

    MORE_KWARGS = {}
    if USE_RWPE and NAME not in ["Density", "Component", "Coreness", "CutRatio"]:
        MORE_KWARGS["load_rwpe"] = True
    elif USE_LEPE and NAME not in ["Density", "Component", "Coreness", "CutRatio"]:
        MORE_KWARGS["load_lepe"] = True

    dts: SubgraphDataset = eval(NAME)(
        root=PATH,
        name=NAME,
        embedding_type=E_TYPE,
        debug=DEBUG,
        **MORE_KWARGS,
    )

    train_dts, val_dts, test_dts = dts.get_train_val_test()

    dts.print_summary()

    cprint("Train samples", "yellow")
    for i, b in enumerate(train_dts):
        print(b)
        if i >= 5:
            break

    cprint("Validation samples", "yellow")
    for i, b in enumerate(val_dts):
        print(b)
        if i >= 5:
            break

    cprint("global_data samples", "yellow")
    print(dts.global_data)
    print("Avg. degree: ", dts.global_data.edge_index.size(1) / dts.global_data.num_nodes)
    if hasattr(dts.global_data, "pe"):
        print("PE", dts.global_data.pe)

    cprint("All subgraph samples", "magenta")
    print(dts.data)
    try:
        for k, vs in dts.y_stat_dict().items():
            print(k, [round(v, 3) for v in vs])
            for v in vs:
                print(round(v, 3))
    except AttributeError:
        pass
