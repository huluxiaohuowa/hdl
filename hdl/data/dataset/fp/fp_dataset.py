import typing as t

import torch
from rdkit import Chem
# import torch.utils.data as tud

from hdl.data.dataset.base_dataset import CSVDataset
from hdl.features.fp.features_generators import (
    get_features_generator,
    get_available_features_generators,
    FP_BITS_DICT 
)
from hdl.features.fp.rxn import get_rxnrep_fingerprint


class RXNFPDataset(CSVDataset):
    def __init__(
        self,
        csv_file: str,
        splitter: str = ',',
        smiles_col: str = 'SMILES',
        reactant_cols: t.List[str] = [],
        product_cols: t.List[str] = [],
        target_cols: t.List[str] = [],
        condition_cols: t.List[str] = [],
        num_classes: t.List[int] = 1,
        target_transform: t.Union[str, t.List[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(
            csv_file, 
            splitter,
            smiles_col,
            target_cols,
            num_classes,
            target_transform,
            **kwargs
        )
        self.condition_cols = condition_cols
    
    def __getitem__(self, index):
        smiles = self.df.loc[index][self.smiles_col]
        fp = get_rxnrep_fingerprint([smiles])
        target = self.df.loc[index][self.target_cols]
        return fp, target


class FPDataset(CSVDataset):
    def __init__(
        self,
        csv_file: str,
        splitter: str,
        smiles_cols: t.List,
        target_cols: t.List = [],
        missing_labels: t.List = [],
        num_classes: t.List = [],
        target_transform: t.Union[str, t.List[str]] = None,
        fp_type: str = 'morgan_count',
        **kwargs
    ) -> None:
        super().__init__(
            csv_file,
            splitter=splitter,
            smiles_col=smiles_cols,
            target_cols=target_cols,
            num_classes=num_classes,
            target_transform=target_transform,
            **kwargs
        )
        self.smiles_cols = smiles_cols 
        assert fp_type in get_available_features_generators()
        self.fp_type = fp_type
        self.fp_generator = get_features_generator(self.fp_type)
        self.fp_numbits = FP_BITS_DICT[self.fp_type]
        self.missing_labels = missing_labels
    
    def __getitem__(self, index):
        smiles_list = self.df.loc[index][self.smiles_cols].tolist()

        fingerprint_list = list(
            map(
                lambda x: torch.LongTensor(self.fp_generator(Chem.MolFromSmiles(x))),
                smiles_list
            )
        )
        if any(self.target_cols):
            target_list = self.df.loc[index][self.target_cols].tolist()
            
            # process with missing label
            final_targets = []
            for target, missing_label in zip(target_list, self.missing_labels):
                if missing_label is not None and target == missing_label:
                    final_targets.append(float('nan'))
                else:
                    final_targets.append(target)
 
            if self.target_transform is None:
                return fingerprint_list, final_targets 
            else:
                # print(final_targets)
                target_tensors = [
                    trans(target, num_class, missing_label=float('nan'))
                    for trans, target, num_class in zip(
                        self.target_transform,
                        final_targets,
                        self.num_classes
                    )
                ]
                # print(target_tensors)
                return fingerprint_list, target_tensors, final_targets 
        else:
            return fingerprint_list
