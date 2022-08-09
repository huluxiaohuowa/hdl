import typing as t

import numpy as np
import torch
from torch._C import dtype
import torch_geometric as tg
from torch_geometric.data import Dataset 

from hdl.features.graph.featurization import MolGraph


class MolDataset(Dataset):

    def __init__(
        self,
        smiles: t.List,
        labels: t.List,
        chiral_features: bool = False,
        global_chiral_features: bool = False,
    ):
        super(MolDataset, self).__init__()

        # self.split = list(range(len(smiles)))  # fix this
        # self.smiles = [smiles[i] for i in self.split]
        # self.labels = [labels[i] for i in self.split]
        self.smiles = smiles
        self.labels = labels
        # self.data_map = {k: v for k, v in zip(range(len(self.smiles)), self.split)}
        # self.args = args
        self.chiral_features = chiral_features
        self.global_chiral_features = global_chiral_features

        self.mean = np.mean(self.labels)
        self.std = np.std(self.labels)

    def process_key(self, key):
        smi = self.smiles[key]
        molgraph = MolGraph(
            smi,
            self.chiral_features,
            self.global_chiral_features
        )
        mol = self.molgraph2data(molgraph, key)
        return mol

    def molgraph2data(self, molgraph, key):
        data = tg.data.Data()
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        data.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        data.y = torch.tensor([self.labels[key]], dtype=torch.float)
        data.parity_atoms = torch.tensor(molgraph.parity_atoms, dtype=torch.long)
        data.parity_bond_index = torch.tensor(molgraph.parity_bond_index, dtype=torch.long)
        data.smiles = self.smiles[key]

        return data

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, key):
        return self.process_key(key)
