# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/data/dataset/base_dataset.py
# 说明：分子数据集构建与切分
# 模块功能：CSV 表格数据集基类，负责读表、列名配置与标签变换注册表，具体取样本交由子类实现。
from os import path as osp
import typing as t

import torch.utils.data as tud
import pandas as pd

from jupyfuncs.dl.dataframe import rm_index
from jupyfuncs.dl.tensor import (
    label_to_onehot,
    label_to_tensor
)


def percent(x, *args, **kwargs):
    """把数值按百分比换算为小数（x/100），忽略变换接口传入的多余参数。"""
    return x / 100
        

# 标签变换名称 -> 函数的注册表，target_transform 传名称即在此查表
label_trans_dict = {
    'onehot': label_to_onehot,
    'tensor': label_to_tensor,
    'percent': percent 
}


class CSVDataset(tud.Dataset):
    """多列 SMILES、多任务标签（multi-task labels）的 CSV 数据集基类，继承 torch.utils.data.Dataset。
    csv_file 为表格路径，splitter 为分隔符，smiles_col 指定 SMILES 列（可传列名列表），
    target_cols 为各任务标签列名，num_classes 为对应类别数，target_transform 按名称逐列配置标签变换。
    __getitem__ 由子类实现，__len__ 返回表行数。
    """
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
        # 按分隔符读入 CSV，**kwargs 透传给 pandas.read_csv
        df = pd.read_csv(
            self.csv,
            sep=splitter,
            **kwargs
        )
        # 删除列名以 Unnamed 开头的冗余索引列
        self.df = rm_index(df)
        self.smiles_col = smiles_col
        self.target_cols = target_cols
        self.num_classes = num_classes
        if target_transform is not None:
            # 未给类别数时按每个标签列 1 维处理（回归或单维标签）
            if not num_classes:
                self.num_classes = [1 for _ in range(len(target_cols))]
            else:
                assert len(self.num_classes) == len(target_cols)
            # 传单个字符串则所有列共用同一变换，传可迭代对象则逐列映射
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
    """单 SMILES 列、单标签列的 CSV 数据集基类，继承 torch.utils.data.Dataset。
    target_col 为标签列名（None 表示无标签），missing_label 为缺失值占位符，
    target_transform 只接受单个名称并在 label_trans_dict 中查表。
    __getitem__ 由子类实现，__len__ 返回表行数。
    """
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
        # 记录原始 CSV 路径（未做绝对化处理）
        self.csv_file = csv_file 
        # 按分隔符读入 CSV 并去掉 Unnamed 冗余索引列
        df = pd.read_csv(
            self.csv_file,
            sep=splitter,
            **kwargs
        )
        self.df = rm_index(df)
        self.smiles_col = smiles_col 
        self.target_col = target_col
        # 缺失值占位符字符串存于 miss_label，供子类判定并处理缺失标签
        self.miss_label = missing_label
        if target_transform is not None:
            # 单标签列数据集仅配置一个变换函数
            self.target_transform = label_trans_dict[target_transform]
 
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.df)