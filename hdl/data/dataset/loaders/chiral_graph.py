# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/data/dataset/loaders/chiral_graph.py
# 说明：DataLoader、collate 函数与采样器
# 模块功能：手性分子图数据集（MolDataset）到 PyTorch Geometric 数据加载器（DataLoader）的构建入口，可选按对映异构体成对采样。
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
    """把 SMILES 列表或 CSV 表构建成手性分子图数据集（MolDataset），并包装成 PyG 数据加载器（DataLoader）。

    data_path 非空时按 smiles_col/label_col 从 CSV 取列，否则用 smiles_list/label_list；
    chiral_features/global_chiral_features 决定是否附加局部与全局手性（chirality）特征。
    shuffle_pairs=True 时改用 StereoSampler 给出索引（此时 shuffle 需保持 False，加载器不允许同时指定两者），
    否则按 shuffle 打乱；合并成批（batching）由 PyG 默认 collate 完成：各图节点数累加，
    edge_index 按前面图的节点总数整体位移。返回 (loader, dataset)。
    """

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
    
    # 以下按 split_data 切分并逐份建加载器的分支位于 return 之后，实际不会执行
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