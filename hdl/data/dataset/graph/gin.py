# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/data/dataset/graph/gin.py
# 说明：分子数据集构建与切分
# 模块功能：面向图对比学习的分子图（molecular graph）数据集：SMILES 转图并随机掩蔽原子/键生成两个增强视图，附训练/验证/测试 DataLoader 封装。
# import os
import csv
import math
# import time
import random
# import networkx as nx
import numpy as np
from copy import deepcopy
import typing as t

import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# import torchvision.transforms as transforms

# from torch_scatter import scatter
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# import rdkit
from rdkit import Chem
# from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
# from rdkit.Chem import AllChem

from hdl.data.dataset.utils import read_smiles


__all__ = [
    "MoleculeDataset",
    "MoleculeDatasetWrapper"
]


# 原子序数/手性标记/键类型/键方向的取值表，特征以类别在该表中的下标表示
ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


class MoleculeDataset(Dataset):
    """torch_geometric Dataset 子类：经 read_smiles 读入 SMILES，访问时即时转图。
    data_path 为数据文件路径，file_type 取 'smi' 或 'csv'，smi_col_names 为多个 SMILES 列名，
    y_col_name 为标签列名。__getitem__ 对单个分子返回一对掩蔽视图 (data_i, data_j)；
    多 SMILES 列模式返回列表，配置 y 列时末元素为 float 标签。
    """
    def __init__(
        self,
        data_path,
        file_type: str = 'smi',
        smi_col_names: t.List = [],
        y_col_name: str = None, 
    ):
        """经 read_smiles 读 data_path（file_type 取 'smi' 或 'csv'）得到 self.smiles_data：
        单列模式为 SMILES 列表，多列模式每行为 smi_col_names 各列的 SMILES 加末列标签；
        smi_col_names/y_col_name 存为属性，决定 __getitem__ 返回列表还是 Data。"""
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(
            data_path=data_path,
            file_type=file_type,
            smi_col_names=smi_col_names,
            y_col_name=y_col_name
        )
        self.smi_col_names = smi_col_names
        self.y_col_name = y_col_name
    
    def __getitem__(
        self,
        idx: int
    ):
        if any(self.smi_col_names):
            # 前 len(smi_col_names) 列逐列转图
            item = [
                self.getitem(smiles)
                for smiles in self.smiles_data[idx][: len(self.smi_col_names)]
            ]
            # 配置了标签列则把行末列转为 float 追加到结果末尾
            if self.y_col_name is not None:
                item.append(float(self.smiles_data[idx][-1]))
            return item
        else:
            return self.getitem(self.smiles_data[idx])

    def getitem(self, smiles):
        """单个 SMILES 转分子图：拼原子特征 (N, 2)、双向边索引 (2, 2M)、边特征 (2M, 2)，
        再各随机掩蔽约 25% 的原子与键，返回增强视图对 (data_i, data_j)。"""
        mol = Chem.MolFromSmiles(smiles)
        # mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        # aromatic = []
        # sp, sp2, sp3, sp3d = [], [], [], []
        # num_hs = []
        # 逐原子记录原子序数与手性标记的类别下标
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())
            # aromatic.append(1 if atom.GetIsAromatic() else 0)
            # hybridization = atom.GetHybridization()
            # sp.append(1 if hybridization == HybridizationType.SP else 0)
            # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            # sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)

        # z = torch.tensor(atomic_number, dtype=torch.long)
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        # 两列类别下标按特征维拼接为 (原子数, 2) 的原子特征矩阵
        x = torch.cat([x1, x2], dim=-1)
        # x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, sp3d, num_hs],
        #                     dtype=torch.float).t().contiguous()
        # x = torch.cat([x1.to(torch.float), x2], dim=-1)

        row, col, edge_feat = [], [], []
        # 每条化学键拆成正反向两条有向边，各记一次 [键类型, 键方向] 特征
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        # 汇总为 (2, 2M) 边索引与 (2M, 2) 边特征
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        # random mask a subgraph of the molecule
        # 数据增强：按约 25% 比例随机选取要掩蔽的原子与键，构造两个视图
        num_mask_nodes = max([1, math.floor(0.25*N)])
        num_mask_edges = max([0, math.floor(0.25*M)])
        mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
        mask_nodes_j = random.sample(list(range(N)), num_mask_nodes)

        mask_edges_i_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_j_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_i = [2*i for i in mask_edges_i_single] + [2*i+1 for i in mask_edges_i_single]
        mask_edges_j = [2*i for i in mask_edges_j_single] + [2*i+1 for i in mask_edges_j_single]

        x_i = deepcopy(x)
        # 被掩蔽原子的特征置为预留的未知原子类别 len(ATOM_LIST)
        for atom_idx in mask_nodes_i:
            x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        # 剔除被掩蔽的有向边并压缩边列表（视图 j 同理）
        edge_index_i = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
        edge_attr_i = torch.zeros((2*(M-num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M):
            if bond_idx not in mask_edges_i:
                edge_index_i[:,count] = edge_index[:,bond_idx]
                edge_attr_i[count,:] = edge_attr[bond_idx,:]
                count += 1
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

        x_j = deepcopy(x)
        for atom_idx in mask_nodes_j:
            x_j[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_j = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
        edge_attr_j = torch.zeros((2*(M-num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M):
            if bond_idx not in mask_edges_j:
                edge_index_j[:,count] = edge_index[:,bond_idx]
                edge_attr_j[count,:] = edge_attr[bond_idx,:]
                count += 1
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
        
        return data_i, data_j

    def __len__(self):
        """返回 self.smiles_data 的长度，即样本数。"""
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    """数据集加载器封装（非 Dataset）：持有数据路径与批大小等配置，
    构建 MoleculeDataset 并产出训练/验证或测试的 torch_geometric DataLoader。"""
    def __init__(
        self,
        batch_size,
        num_workers,
        valid_size,
        data_path,
        file_type: str = 'smi',
        smi_col_names: t.List = [], 
        y_col_name: str = None,
    ):
        """只存配置、不建数据集：batch_size 组批大小、num_workers 加载子进程数、valid_size 验证集比例，
        data_path/file_type/smi_col_names/y_col_name 为构建 MoleculeDataset 的读取参数，
        后续 get_data_loaders/get_test_loader 依据这些属性建 Dataset 与 DataLoader。"""
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.file_type = file_type
        self.smi_col_names = smi_col_names
        self.y_col_name = y_col_name

    def get_data_loaders(self):
        # 仅传 data_path 建数据集（其余配置走默认值），再随机切分训练/验证加载器
        train_dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader
    
    def get_test_loader(self, shuffle=False):
        # 用完整列名配置构建测试数据集并组批
        test_dataset = MoleculeDataset(
            data_path=self.data_path,
            file_type=self.file_type,
            smi_col_names=self.smi_col_names,
            y_col_name=self.y_col_name
        )
        test_loader = self.get_test_data_loader(
            test_dataset,
            shuffle=shuffle
        )
        return test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        """随机打乱全部索引，按 valid_size 比例切出验证集，其余为训练集，
        各用 SubsetRandomSampler 组批（drop_last 丢弃末尾不完整批）。"""
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader

    def get_test_data_loader(
        self,
        test_dataset,
        shuffle=False
    ):
        """按 batch_size 对测试集组批，drop_last=False 保留末尾不完整批。"""
        # num_test = len(test_dataset)
        # indices = list(range(num_test))
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=shuffle
        )
        return test_loader
