import typing as t

import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

from hdl.data.dataset.graph.chiral import MolDataset
from hdl.data.dataset.samplers.chiral import StereoSampler 
from hdl.data.dataset.loaders.spliter import split_data


def get_chiralgraph_loader(
    data_path: str = None,
    smiles_list: t.List = [],
    label_list: t.List = [], 
    batch_size: int = 1,
    shuffle: bool = False,
    smiles_col: str = 'SMILES',
    label_col: str = 'label',
    num_workers: int = 10,
    shuffle_pairs: bool = False,
    chiral_features: bool = True,
    global_chiral_features: bool = True 
):

    if data_path is not None:
        data_df = pd.read_csv(data_path)

        # smiles = data_df.iloc[:, 0].values
        # labels = data_df.iloc[:, 1].values.astype(np.float32)
        smiles = data_df[smiles_col].tolist()
        labels = data_df[label_col].to_numpy()
    else:
        smiles = smiles_list
        labels = np.array(label_list)
    
    dataset = MolDataset(
        smiles=smiles,
        labels=labels,
        chiral_features=chiral_features,
        global_chiral_features=global_chiral_features
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=StereoSampler(dataset) if shuffle_pairs else None)
    return loader, dataset
    
    split_loader_list = []
    split_data_list = split_data(smiles, labels, split_type="random")
    for split_smiles, split_labels in split_data_list:
        dataset = MolDataset(
            smiles=split_smiles,
            labels=split_labels,
            chiral_features=chiral_features,
            global_chiral_features=global_chiral_features,
        )
    
    # train_dataset = dataset
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True,
                            sampler=StereoSampler(dataset) if shuffle_pairs else None)
        split_loader_list.append(loader)

    return split_loader_list, dataset