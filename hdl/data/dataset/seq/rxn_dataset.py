import typing as t

import numpy as np
import torch
import pkg_resources
from rxnfp.tokenization import (
    SmilesTokenizer,
    # convert_reaction_to_valid_features_batch,
    convert_reaction_to_valid_features,
)

from ..base_dataset import CSVDataset


class RXNCSVDataset(CSVDataset):
    def __init__(
        self,
        csv_file: str,
        max_len: int = 512,
        vocab_path: str = None,
        splitter: str = ',',
        smiles_col: str = 'SMILES',
        target_cols: t.List = [],
        **kwargs,
    ) -> None:
        super().__init__(
            csv_file,
            splitter=splitter,
            smiles_col=smiles_col,
            target_cols=target_cols,
            **kwargs
        )
        if vocab_path is None:
            vocab_path = pkg_resources.resource_filename(
                "rxnfp",
                "models/transformers/bert_ft/vocab.txt"
            )
        self.tokenizer = SmilesTokenizer(
            vocab_path, max_len=max_len
        )
 
    def __getitem__(self, index):
        # rxn_list = [self.df.loc[index][self.smiles_col]]
        rxn = self.df.loc[index][self.smiles_col]
        feats = convert_reaction_to_valid_features(
            rxn,
            self.tokenizer
        )
        X = [
            torch.tensor(feats.input_ids.astype(np.int64)),
            torch.tensor(feats.input_mask.astype(np.int64)),
            torch.tensor(feats.segment_ids.astype(np.int64))
        ]
        if any(self.target_cols):
            labels = self.df.loc[index][self.target_cols].tolist()
            y = torch.LongTensor(labels)
            
            return X, y
        else:
            return X
 