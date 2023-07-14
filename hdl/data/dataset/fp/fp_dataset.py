import typing as t

import torch
from rdkit import Chem
# import torch.utils.data as tud

from hdl.data.dataset.base_dataset import CSVDataset, CSVRDataset
from hdl.features.fp.features_generators import (
    get_features_generator,
    get_available_features_generators,
    FP_BITS_DICT 
)
# from hdl.features.fp.rxn import get_rxnrep_fingerprint


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


class FPRDataset(CSVRDataset):
    def __init__(
        self,
        csv_file: str,
        splitter: str,
        smiles_col: str,
        target_col: str = None,
        missing_label: str = None,
        target_transform: t.Union[str, t.List[str]] = None,
        fp_type: str = 'morgan_count',
        **kwargs
    ) -> None:
        super().__init__(
            csv_file,
            splitter=splitter,
            smiles_col=smiles_col,
            target_col=target_col,
            target_transform=target_transform,
            missing_label=missing_label,
            **kwargs
        )
        assert fp_type in get_available_features_generators()
        self.fp_type = fp_type
        self.fp_generator = get_features_generator(self.fp_type)
        self.fp_numbits = FP_BITS_DICT[self.fp_type]
        self.missing_label = missing_label
    
    def __getitem__(self, index):
        smiles = self.df.loc[index][self.smiles_col]
        try:
            fp = torch.LongTensor(self.fp_generator(Chem.MolFromSmiles(smiles)))
        except Exception as _:
            fp = torch.zeros(self.fp_numbits).long()

        if self.target_col is not None: 
            target = self.df.loc[index][self.target_col]
            target = (target, )
            return fp, target
        else:
            return fp 
