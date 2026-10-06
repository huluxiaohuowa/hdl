# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/data/dataset/loaders/collate_funcs/fp.py
# 说明：DataLoader、collate 函数与采样器
# 模块功能：指纹（fingerprint）样本的批处理合并函数（collate_fn），把若干样本按列堆叠成批量张量。
r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""
import typing as t

import numpy as np
import pandas as pd
import torch


# 视为整数标量的类型集合，目前只被下面注释掉的 LongTensor 分支引用
int_types = (
    int,
    np.int32,
    np.int64,
    pd.Int16Dtype,
    pd.Int32Dtype,
    pd.Int64Dtype,
    # torch.int32
)


def fp_collate(batch):
    """把批内若干指纹（fingerprint）样本合并成一个批（batch）：按样本元组的位置转置，再逐列堆叠成张量。

    样本来自 FPDataset，形如 (指纹列表,) / (指纹列表, 原始标签) / (指纹列表, 变换后标签, 原始标签)。
    zip(*batch) 先得到每个位置的批内元组，对第 0 位再 zip 一次即按 SMILES 列对齐，
    torch.vstack 把同列的 batch_size 个一维指纹堆成 (批大小, 指纹位数) 的 float 张量。
    原始标签按任务列转置：取值为首元素不可迭代的标量时转成 1D float 张量，否则原样留作 list。
    返回随样本段数变化：3 段返回 (fps, target_tensors, targets_list)，其中 target_tensors 由第 1 位
    各任务的变换后标签（one-hot 等）vstack 成张量；2 段返回 (fps, targets, targets_list)；
    1 段（无标签）只返回 fps。
    """
    transposed = list(zip(*batch))

    # fps
    fps = list(zip(*transposed[0]))
    fps = [torch.vstack(fp).float() for fp in fps]
    if len(transposed) == 1:
        return fps

    # target_list
    targets = list(zip(*transposed[-1]))
    targets_list = []
    for target_labels in targets:
        if not isinstance(target_labels[0], t.Iterable):
            target_labels = torch.Tensor(target_labels)
        else:
            target_labels = list(target_labels)
        # if isinstance(target_labels[0], int_types):
        #     target_labels = torch.LongTensor(target_labels)
        targets_list.append(target_labels) 
 
    # target_tensors
    if len(transposed) == 3:
        target_tensors = list(zip(*transposed[1])) 
        target_tensors = [
            torch.vstack(target_tensor).float()
            for target_tensor in target_tensors
        ]
    
        return fps, target_tensors, targets_list
    else:
        return fps, targets, targets_list 