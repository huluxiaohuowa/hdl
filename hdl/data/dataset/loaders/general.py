# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/data/dataset/loaders/general.py
# 说明：DataLoader、collate 函数与采样器
# 模块功能：通用数据加载器（DataLoader）子类 Loader，默认用 fp_collate 做批处理合并（batching）。
import typing as t

import torch.utils.data as tud

from hdl.data.dataset.loaders.collate_funcs.fp import fp_collate


class Loader(tud.DataLoader):
    """指纹（fingerprint）数据集用的数据加载器（DataLoader）：在 torch 版本之上固定批大小、打乱（shuffle）
    与子进程数（num_workers）默认值，并把 collate_fn 默认设为 fp_collate，可按参数替换。"""
    def __init__(
        self,
        dataset,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 12,
        collate_fn: t.Callable = fp_collate
    ):
        """无新增状态，只是给 tud.DataLoader 填默认值：dataset 为指纹数据集，batch_size 默认 128、
        shuffle 默认 True、num_workers 默认 12，collate_fn 默认 fp_collate（按列把批内指纹堆成
        (批大小, 指纹位数) 张量并整理标签），各参数原样传给父类。"""
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )