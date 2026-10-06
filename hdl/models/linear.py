# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/models/linear.py
# 说明：神经网络模型定义与注册表
# 模块功能：基于分子指纹（fingerprint）线性组合的迭代式多任务多分类基线模型 MMIterLinear。
import typing as t

import torch
from torch import nn
from torch.nn import functional as nnfunc
import numpy as np

from hdl.layers.general.linear import (
    BNReLULinearBlock,
    BNReLULinear
)
# from hdl.ops.utils import get_activation


class MMIterLinear(nn.Module):
    """指纹（fingerprint）线性基线：三组指纹各经一个线性层，按 X = W3·fp3 - (W1·fp1 + W2·fp2) 组合成描述子，
    再逐个任务用 MLP 分类头预测；iterative=True 时把上一任务结果（真值或预测）拼回 X 形成迭代式条件预测。
    forward 输入 (fp1, fp2, fp3) 三元张量，返回 {目标名: (batch, 类别数)} 字典。"""
    # 注册表中的模型名
    _NAME = 'mumc_linear'

    def __init__(
        self,
        num_fp_bits: int,
        num_in_feats: int,
        nums_classes: t.List[int] = [3, 3],
        target_names: t.List[str] = None,
        hidden_size: int = 128,
        num_hidden_layers: int = 10,
        activation: str = 'elu',
        out_act: str = 'softmax',
        hard_select: bool = False,
        iterative: bool = True,
        **kwargs,
    ):
        """num_fp_bits：输入指纹位长；num_in_feats：线性映射后的特征维；nums_classes：各任务的类别数列表；
        target_names：各任务输出在结果字典中的键名；hidden_size/num_hidden_layers：任务 MLP 塔的隐层宽与层数；
        activation：塔内激活名；out_act：输出激活名或按任务的列表；hard_select：迭代时用硬 Gumbel-Softmax 采样；
        iterative：是否把前序任务输出拼进后续任务输入。"""
        super().__init__()

        # target_names 缺省时用任务下标作为结果字典的键
        if target_names is None:
            self.target_names = list(range(len(nums_classes)))
        else:
            self.target_names = target_names
        
        self.init_args = {
            'num_fp_bits': num_fp_bits,
            'num_in_feats': num_in_feats,
            'nums_classes': nums_classes,
            'target_names': target_names,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'activation': activation,
            'out_act': out_act,
            'hard_select': hard_select,
            'iterative': iterative,
            **kwargs
        }
        self.hard_select = hard_select
        self.iterative = iterative
        # 默认冻结全部分类头，由 freeze_classifier 开关逐个解冻
        self._freeze_classifier = [True] * len(target_names)
        
        # 三组指纹各自一个线性层映射到 num_in_feats
        # self.w1 = BNReLULinear(num_fp_bits, num_in_feats, activation)
        self.w1 = nn.Linear(num_fp_bits, num_in_feats)
        # self.w2 = BNReLULinear(num_fp_bits, num_in_feats, activation)
        self.w2 = nn.Linear(num_fp_bits, num_in_feats)
        # self.w3 = BNReLULinear(num_fp_bits, num_in_feats, activation)
        self.w3 = nn.Linear(num_fp_bits, num_in_feats)

        nums_in_feats = [num_in_feats]
        if iterative:
            # 迭代模式：后续任务输入维 = num_in_feats + 前序任务类别数的累加和（丢掉末项）
            nums_in_feats.extend(nums_classes)
            nums_in_feats = np.cumsum(np.array(nums_in_feats, dtype=np.int))[:-1]
        else:
            # 独立模式：每个任务输入维都是 num_in_feats
            nums_in_feats = nums_in_feats * len(nums_classes)
        
        if isinstance(out_act, str):
            self.out_acts = [out_act] * len(nums_classes)
        else:
            self.out_acts = out_act

        # 每任务一个分类头：BN-ReLU-Linear 塔（num_in -> hidden_size -> hidden_size）再接输出层（hidden_size -> 该任务类别数 + out_act）
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                BNReLULinearBlock(
                    num_in,
                    hidden_size,
                    num_hidden_layers,
                    hidden_size,
                    activation,
                    **kwargs
                ),
                BNReLULinear(
                    hidden_size,
                    num_out,
                    out_act,
                    **kwargs
                )
            )
            for num_in, num_out, out_act in zip(
                nums_in_feats, nums_classes, self.out_acts
            )
        ])

    @property
    def freeze_classifier(self):
        """按任务是否冻结的布尔列表（与 target_names 一一对应）。"""
        return self._freeze_classifier

    @freeze_classifier.setter
    def freeze_classifier(self, freeze: t.List = []):
        """设置冻结列表，并同步把对应分类头的参数 requires_grad 取反。"""
        self._freeze_classifier = freeze
        self.change_classifier_grad([not f for f in freeze])

    def change_classifier_grad(self, requires_grads: t.List = []):
        """按 requires_grads 逐任务开关分类头参数的梯度。"""
        for requires_grad, classifier in zip(requires_grads, self.classifiers):
            for param in classifier.parameters():
                param.requires_grad = requires_grad
    
    def forward(self, fps, target_tensors=None, teach=True):
        """fps: (fp1, fp2, fp3) 三组 (batch, num_fp_bits) 指纹；target_tensors: 各任务真值标签列表（teach 时拼回输入）；
        teach: 是否用真值做教师强制（teacher forcing），False 时用预测概率或硬 Gumbel 采样。
        返回 {目标名: (batch, 类别数)}。"""
        result_dict = {}
        # 三组指纹各自线性映射后相减：X = W3·fp3 - (W1·fp1 + W2·fp2)
        fp1, fp2, fp3 = fps
        fp1 = self.w1(fp1)
        fp2 = self.w2(fp2)
        fp3 = self.w3(fp3)
        X = fp3 - (fp1 + fp2)
        if target_tensors is None:
            # 未给真值时按任务数占位，predict（teach=False）路径不需要真值
            target_tensors = [None] * len(self.target_names)
        for classifier, target_name, target_tensor in zip(
            self.classifiers, self.target_names, target_tensors
        ):
            result = classifier(X)
            result_dict[target_name] = result
            if self.iterative:
                # 迭代式：把本任务信息拼回 X 作为下一任务输入 —— teach 用真值标签，否则用预测概率；
                # hard_select 时用硬 Gumbel-Softmax 采样代替软概率
                if teach:
                    assert target_tensors is not None
                    X = torch.cat((X, target_tensor), -1) 
                else: 
                    if not self.hard_select:
                        X = torch.cat((X, result), -1)
                    else:
                        X = torch.cat(
                            (X, nnfunc.gumbel_softmax(result, tau=1, hard=True)),
                            -1
                        )
        return result_dict
