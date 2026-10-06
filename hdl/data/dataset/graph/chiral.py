# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/data/dataset/graph/chiral.py
# 说明：分子数据集构建与切分
# 模块功能：把 SMILES 列表按需构建为带手性（chirality）信息的分子图（molecular graph）数据集。
import typing as t

import numpy as np
import torch
from torch._C import dtype
import torch_geometric as tg
from torch_geometric.data import Dataset 

from hdl.features.graph.featurization import MolGraph


class MolDataset(Dataset):
    """torch_geometric Dataset 子类：按索引把 SMILES 惰性转为分子图，不做磁盘缓存。
    smiles/labels 为等长的分子串与标签列表；chiral_features、global_chiral_features
    控制 MolGraph 原子特征是否加入局部/全局手性编码；构造时用 numpy 计算标签均值与
    标准差（mean/std），供回归任务归一化。__getitem__ 返回 tg.data.Data。
    """

    def __init__(
        self,
        smiles: t.List,
        labels: t.List,
        chiral_features: bool = False,
        global_chiral_features: bool = False,
    ):
        """只登记数据与开关，不做任何建图：smiles/labels 为等长的分子串与标签列表，
        chiral_features/global_chiral_features 决定 MolGraph 是否加入局部/全局手性原子特征，
        另用 numpy 求出 labels 的 mean/std 供回归归一化。"""
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
        # 每次访问即时构建该索引的 MolGraph 并转成 Data，实现惰性加载
        smi = self.smiles[key]
        molgraph = MolGraph(
            smi,
            self.chiral_features,
            self.global_chiral_features
        )
        mol = self.molgraph2data(molgraph, key)
        return mol

    def molgraph2data(self, molgraph, key):
        """把 MolGraph 的特征数组逐字段转为 torch_geometric Data。"""
        data = tg.data.Data()
        # 原子特征矩阵 (原子数, 特征维)
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        # 边索引由 (边数, 2) 转置为 PyG 约定的 (2, 边数)
        data.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        # 每条有向边的键特征 (边数, 键特征维)
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        # 该样本的标签，形状 (1,)
        data.y = torch.tensor([self.labels[key]], dtype=torch.float)
        # 每个原子的四面体手性旋向：CW 为 +1、CCW 为 -1、非手性为 0
        data.parity_atoms = torch.tensor(molgraph.parity_atoms, dtype=torch.long)
        # 手性中心原子出边的索引序列，供手性消息传递按序聚合
        data.parity_bond_index = torch.tensor(molgraph.parity_bond_index, dtype=torch.long)
        # 保留原始 SMILES 便于溯源
        data.smiles = self.smiles[key]

        return data

    def __len__(self):
        """返回 SMILES 列表长度，即数据集样本数。"""
        return len(self.smiles)

    def __getitem__(self, key):
        """按索引 key 转交给 process_key，即时构建该分子的图并返回 Data（含 x、edge_index、edge_attr、y、parity_atoms、parity_bond_index、smiles）。"""
        return self.process_key(key)
