# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/models/norm_flows.py
# 说明：神经网络模型定义与注册表
# 模块功能：可微归一化流（normalizing flow）模型外壳，把基底分布（prior）与若干可逆流层串起来做密度计算与采样。
import torch
import torch.nn as nn


class NormalizingFlowModel(nn.Module):
    """归一化流（normalizing flow）模型：prior 为已知对数概率可算的基底分布（如标准正态），
    flows 为依次级联的可逆流层（如耦合层 coupling layer）。
    forward 做数据 x -> 噪声 z 的正向变换并累加对数行列式（log det Jacobian），inverse 反向 z -> x，sample 由 prior 采样反变换出数据。"""

    def __init__(self, prior, flows):
        """prior：基底分布模块（需实现 log_prob 与 sample）；flows：正序级联的可逆流层列表，
        每层需实现 forward/inverse 并返回 (变换结果, 逐样本 log det)。"""
        super().__init__()
        self.prior = prior
        # 流层按 forward 方向顺序级联，inverse 时逆序调用
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        """正向（数据 -> 噪声）：x 形状 (m, d)，依次过每个 flow.forward 得到 (z, 该步 log det)，
        累加得到总 log_det，并返回 (z, prior 在 z 上的对数概率, log_det)，形状均为 (m,)。"""
        m, _ = x.shape
        log_det = torch.zeros(m)
        # 正向按 flows 顺序级联：每个流层返回 (变换后的 x, 该层逐样本 log det Jacobian)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z):
        """反向（噪声 -> 数据）：flows 逆序调用各自的 flow.inverse，逐步累加对数行列式。
        z 形状 (m, d)，返回 (x, log_det)，log_det 形状 (m,)。"""
        m, _ = z.shape
        log_det = torch.zeros(m)
        # 反向：flows 逆序调用各自的 flow.inverse，累加逐样本 log det Jacobian
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        """从 prior 采 n_samples 个噪声向量，再反向变换得到样本 x，形状 (n_samples, d)。"""
        z = self.prior.sample((n_samples,))
        x, _ = self.inverse(z)
        return x