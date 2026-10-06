# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/metric_loss/multi_label.py
# 说明：损失函数与评估指标
# 模块功能：多标签分类（multi-label classification）的 BP-MLL 损失，惩罚负标签得分高于正标签的成对项。
import torch
from torch import Tensor


class BPMLLLoss(torch.nn.Module):
    """BP-MLL 多标签损失：对每个样本枚举（正标签 k, 负标签 l）对，累加 exp(-c_k + c_l)，再除以 |Y_i| 与 |Ȳ_i| 的 bias 次幂。"""
    def __init__(self, reduction: str = 'mean', bias=(1, 1)):
        # bias 两个正整数分别是 |Y_i| 与 |Ȳ_i| 归一化项的指数
        super(BPMLLLoss, self).__init__()
        self.bias = bias
        self.reduction = reduction
        assert len(self.bias) == 2 \
            and all(map(lambda x: isinstance(x, int) and x > 0, bias)), \
            "bias must be positive integers"

    def forward(self, c: Tensor, y: Tensor) -> Tensor:
        r"""
        计算 BP-MLL 损失：c 为各标签得分 (batch_size, n_labels)，y 为 0/1 多标签矩阵；
        返回标量（reduction='mean'）或形状 (batch_size,) 的逐样本损失（reduction='none'）。
        compute the loss, which has the form:
        L = \sum_{i=1}^{m} \frac{1}{|Y_i| \cdot |\bar{Y}_i|} \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
        :param c: prediction tensor, size: batch_size * n_labels
        :param y: target tensor, size: batch_size * n_labels
        :return: size: scalar tensor
        """
        y = y.float()
        # 1/0 标签取反，得到负标签掩码
        y_bar = -y + 1
        # 正、负标签数分别取 bias 次幂，作为逐样本归一化的分母
        y_norm = torch.pow(y.sum(dim=(1,)), self.bias[0])
        y_bar_norm = torch.pow(y_bar.sum(dim=(1,)), self.bias[1])
        assert torch.all(y_norm != 0) or torch.all(y_bar_norm != 0), \
            "an instance cannot have none or all the labels"
        # 逐样本：正负标签对的指数项之和除以 |Y_i|·|Ȳ_i|
        loss = 1 / torch.mul(y_norm, y_bar_norm) \
            * self.pairwise_sub_exp(y, y_bar, c)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'none':
            return loss  

    def pairwise_sub_exp(self, y: Tensor, y_bar: Tensor, c: Tensor) -> Tensor:
        r"""
        对每个样本枚举正标签 j 与负标签 k 的配对，返回形状 (batch_size,) 的成对指数项之和。
        compute \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
        """
        # truth_matrix[i, j, k] 为 1 当且仅当 j 是样本 i 的正标签且 k 是其负标签
        truth_matrix = y.unsqueeze(2).float() @ y_bar.unsqueeze(1).float()
        # exp_matrix[i, j, k] = exp(c[i, k] - c[i, j])，负标签得分高于正标签时该项变大
        exp_matrix = torch.exp(c.unsqueeze(1) - c.unsqueeze(2))
        # 仅保留正-负配对对应的指数项，再对 (j, k) 两个标签维求和
        return (torch.mul(truth_matrix, exp_matrix)).sum(dim=(1, 2))
