from os import path as osp
import typing as t

import torch.utils.data as tud
import pandas as pd

from jupyfuncs.dataframe import rm_index
from jupyfuncs.tensor import (
    label_to_onehot,
    label_to_tensor
)


def percent(x, *args, **kwargs):
    return x / 100
        

label_trans_dict = {
    'onehot': label_to_onehot,
    'tensor': label_to_tensor,
    'percent': percent 
}


class CSVDataset(tud.Dataset):
    def __init__(
        self,
        csv_file: str,
        splitter: str = ',',
        smiles_col: str = 'SMILES',
        target_cols: t.List[str] = [],
        num_classes: t.List[int] = [],
        target_transform: t.Union[str, t.List[str]] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.csv = osp.abspath(csv_file)
        df = pd.read_csv(
            self.csv,
            sep=splitter,
            **kwargs
        )
        self.df = rm_index(df)
        self.smiles_col = smiles_col
        self.target_cols = target_cols
        self.num_classes = num_classes
        if target_transform is not None:
            if not num_classes:
                self.num_classes = [1 for _ in range(len(target_cols))]
            else:
                assert len(self.num_classes) == len(target_cols)
            if isinstance(target_transform, str):
                self.target_transform = [label_trans_dict[target_transform]] * \
                    len(self.num_classes)
            elif isinstance(target_transform, t.Iterable):
                self.target_transform = [
                    label_trans_dict[target_trans]
                    for target_trans in target_transform
                ]
        else:
            self.target_transform = None
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.df)


class CSVRDataset(tud.Dataset):
    def __init__(
        self,
        csv_file: str,
        splitter: str,
        smiles_col: str,
        target_col: str = None,
        missing_label: str = None,
        target_transform: t.Union[str, t.List[str]] = None,
        **kwargs
    ) -> None:
        self.csv_file = csv_file 
        df = pd.read_csv(
            self.csv_file,
            sep=splitter,
            **kwargs
        )
        self.df = rm_index(df)
        self.smiles_col = smiles_col 
        self.target_col = target_col
        self.miss_label = missing_label
        if target_transform is not None:
            self.target_transform = label_trans_dict[target_transform]
 
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.df)