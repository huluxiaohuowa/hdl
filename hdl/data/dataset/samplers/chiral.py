# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/data/dataset/samplers/chiral.py
# 说明：分子数据集构建与切分
# 模块功能：手性（chirality）图数据集的成对采样器（Sampler），以相邻两条样本为一组打乱索引。
from itertools import chain

import numpy as np
from torch.utils.data.sampler import Sampler


class StereoSampler(Sampler):
    """成对打乱的采样器（Sampler）：数据源需按手性对（同一分子的两个对映体（enantiomer）相邻存放）组织，
    采样时只打乱对的先后，不拆开对，使一对样本仍相邻进入同一批（batch）。"""

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        """把索引切成 [i, i+1] 的成对组，np.random.shuffle 打乱组序后 chain 展平为索引序列。"""
        groups = [[i, i + 1] for i in range(0, len(self.data_source), 2)]
        np.random.shuffle(groups)
        indices = list(chain(*groups))
        return iter(indices)

    def __len__(self):
        return len(self.data_source)