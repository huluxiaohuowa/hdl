# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/layers/graph/tetra.py
# 说明：图神经网络层（GCN/GIN/Transformer/手性图）
# 模块功能：四面体（tetrahedral）手性中心的消息更新层集合，替代把 4 个邻居直接求和的聚合方式：
#           输入统一为 [num_chiral_centers, 4, hidden]（每个手性中心及其 4 个邻居嵌入），
#           输出 [num_chiral_centers, hidden]。三种实现均只对四面体的 12 个偶置换
#           （保持手性/绕向不变的旋转操作）做对称化，从而区分 R/S 构型。

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class TetraPermuter(nn.Module):
    """置换型手性更新层：继承 nn.Module。
    4 个邻居槽位各有一个独立线性变换 W_bs，对 12 个偶置换分别应用后再求和，
    最后经 mlp_out 输出。输入 [N, 4, hidden]，输出 [N, hidden]。
    """

    def __init__(
        self,
        hidden,
        # device
    ):
        """Args:
            hidden (int): 隐藏维度，输入末维与输出维度相同
        """
        super(TetraPermuter, self).__init__()

        # 4 份结构相同但参数独立的 Linear(hidden, hidden)，分别绑定到 4 个邻居槽位
        self.W_bs = nn.ModuleList([copy.deepcopy(nn.Linear(hidden, hidden)) for _ in range(4)])
        # self.device = device
        self.drop = nn.Dropout(p=0.2)
        self.reset_parameters()
        # 输出 MLP：Linear -> BatchNorm1d -> ReLU -> Linear（BatchNorm 在两层线性之间）
        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))

        # 12 个 4 元偶置换（经校验全部为偶置换），即四面体的旋转群：奇置换会翻转手性，故不列入
        self.tetra_perms = torch.tensor([[0, 1, 2, 3],

                                         [0, 2, 3, 1],
                                         [0, 3, 1, 2],
                                         [1, 0, 3, 2],
                                         [1, 2, 0, 3],
                                         [1, 3, 2, 0],
                                         [2, 0, 1, 3],
                                         [2, 1, 3, 0],
                                         [2, 3, 0, 1],
                                         [3, 0, 2, 1],
                                         [3, 1, 0, 2],
                                         [3, 2, 1, 0]])

    def reset_parameters(self):
        """权重初始化：对 4 个槽位 Linear 做 xavier 均匀初始化，gain 依次 0.5/1.0/1.5/2.0，
        使不同邻居槽位的初始尺度有所区分。
        """
        gain = 0.5
        for W_b in self.W_bs:
            nn.init.xavier_uniform_(W_b.weight, gain=gain)
            gain += 0.5

    def forward(self, x):

        """前向：对 12 个偶置换分别计算邻居消息并求和。
        Args:
            x: [N, 4, hidden]，N 个手性中心各自的 4 个邻居嵌入（已按 RDKit 键序排列）
        Returns:
            [N, hidden] 手性中心更新后的表示
        """

        # x[:, self.tetra_perms, :] -> [N, 12, 4, hidden]（按每个置换重排邻居轴）
        # torch.split(..., 1, dim=-2) 拆出 4 个槽位 [N, 12, 1, hidden]，分别过各自的 Linear 后取 tanh
        nei_messages_list = [self.drop(F.tanh(l(t))) for l, t in zip(self.W_bs, torch.split(x[:, self.tetra_perms, :], 1, dim=-2))]
        # 4 槽位拼回 [N,12,4,hidden] -> 对槽位求和得 [N,12,hidden] -> relu/dropout -> 对 12 个置换求和得 [N,hidden]
        nei_messages = torch.sum(self.drop(F.relu(torch.cat(nei_messages_list, dim=-2).sum(dim=-2))), dim=-2)

        # 除以 3 缩放求和量级，再过输出 MLP（含 BatchNorm1d）
        return self.mlp_out(nei_messages / 3.)


class ConcatTetraPermuter(nn.Module):
    """拼接型手性更新层：继承 nn.Module。
    与 TetraPermuter 相比不做逐槽位独立变换，而是把 12 个偶置换下的 4 个邻居嵌入沿 hidden 维拼接
    （4*hidden）后用单个 Linear 压回 hidden，再对置换求和。输入 [N, 4, hidden]，输出 [N, hidden]。
    """

    def __init__(
        self,
        hidden,
        # device
    ):
        """Args:
            hidden (int): 隐藏维度
        """
        super(ConcatTetraPermuter, self).__init__()

        # 一次性吃下 4 个邻居拼接后的向量：Linear(hidden*4 -> hidden)
        self.W_bs = nn.Linear(hidden * 4, hidden)
        torch.nn.init.xavier_normal_(self.W_bs.weight, gain=1.0)
        self.hidden = hidden
        # self.device = device
        self.drop = nn.Dropout(p=0.2)
        # 输出 MLP：Linear -> BatchNorm1d -> ReLU -> Linear
        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))

        # 与 TetraPermuter 相同的 12 个偶置换（四面体旋转操作）
        tetra_perms = torch.tensor([
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [1, 0, 3, 2],
            [1, 2, 0, 3],
            [1, 3, 2, 0],
            [2, 0, 1, 3],
            [2, 1, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 2, 1],
            [3, 1, 0, 2],
            [3, 2, 1, 0]
        ])
        self.register_buffer('tetra_perms', tetra_perms)

    def forward(self, x):

        """前向：12 个偶置换下把 4 个邻居拼接后线性压缩，再对置换求和。
        Args:
            x: [N, 4, hidden]，N 个手性中心的 4 个邻居嵌入
        Returns:
            [N, hidden] 更新后的手性中心表示
        """

        # x[:, tetra_perms, :] -> [N,12,4,hidden]，flatten(start_dim=2) -> [N,12,4*hidden]
        # 经 W_bs -> [N,12,hidden]，再 relu + dropout
        nei_messages = self.drop(
            F.relu(
                self.W_bs(
                    x[
                        :,
                        self.tetra_perms,
                        :
                    ].flatten(start_dim=2)
                )
            )
        )
        # 对 12 个置换维（dim=-2）求和并除以 3 缩放 -> [N, hidden]
        nei_messages_sum = nei_messages.sum(dim=-2) / 3.
        if nei_messages_sum.size(0) == 1:
            # N==1 时复制成 2 个样本，避免训练模式下 BatchNorm1d 因 batch 维为 1 报错，算完再取第 0 行
            nei_messages_sum_repeat = torch.repeat_interleave(nei_messages_sum, 2, dim=0)
            return self.mlp_out(nei_messages_sum_repeat)[:1, ...]
        return self.mlp_out(nei_messages_sum)


class TetraDifferencesProduct(nn.Module):
    """差积型手性更新层：继承 nn.Module。
    对 4 个邻居嵌入取全部 6 个两两差 (h_i - h_j) 并连乘，得到对奇置换变号、对偶置换不变
    （即保持手性）的反对称（antisymmetric）表示；再开 6 次根控制量级并过输出 MLP。
    输入 [N, 4, hidden]，输出 [N, hidden]。
    """

    def __init__(
        self,
        hidden
    ):
        """Args:
            hidden (int): 隐藏维度
        """
        super(TetraDifferencesProduct, self).__init__()

        # 输出 MLP：Linear -> BatchNorm1d -> ReLU -> Linear
        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))
        # 4 个邻居槽位的索引 [4]，注册为 buffer 随模型移动到设备
        self.register_buffer('indices', torch.arange(4))

    def forward(self, x):
        """前向：按两两差连乘构造反对称手性消息。
        Args:
            x: [N, 4, hidden]，N 个手性中心的 4 个邻居嵌入
        Returns:
            [N, hidden] 更新后的手性中心表示
        """

        # indices = torch.arange(4).to(x.device)
        # 逐个取出第 i 个邻居槽位，得到 4 个 [N, hidden] 张量
        message_tetra_nbs = [
            x.index_select(dim=1, index=i).squeeze(1)
            for i in self.indices
        ]
        # 连乘的初始值：全 1 张量 [N, hidden]
        message_tetra = torch.ones_like(message_tetra_nbs[0])

        # note: this will zero out reps for chiral centers with multiple carbon neighbors on first pass
        # 遍历 i<j 共 6 个配对，把 6 个 (h_i - h_j) 因子逐元素相乘；交换任意两个邻居会使整体变号
        for i in range(4):
            for j in range(i + 1, 4):
                message_tetra = torch.mul(message_tetra, (message_tetra_nbs[i] - message_tetra_nbs[j]))
        # 保留符号、对绝对值加 eps 后开 6 次根（对应 6 个乘积因子），把量级压回单因子尺度
        message_tetra = torch.sign(message_tetra) * torch.pow(torch.abs(message_tetra) + 1e-6, 1 / 6)
        return self.mlp_out(message_tetra)


# def get_tetra_update(
#     hidden_size,
#     device,
#     message,
# ):

#     if message == 'tetra_permute':
#         return TetraPermuter(hidden_size, device)
#     elif message == 'tetra_permute_concat':
#         return ConcatTetraPermuter(hidden_size, device)
#     elif message == 'tetra_pd':
#         return TetraDifferencesProduct(hidden_size)
#     else:
#         raise ValueError("Invalid message type.")


# 手性更新层注册表：键为超参 message 的取值，值为对应类；chiral_graph/chiral_gnn 按名实例化
TETRA_UPDATE_DICT = {
    'tetra_permute': TetraPermuter,
    'tetra_permute_concat': ConcatTetraPermuter,
    'tetra_pd': TetraDifferencesProduct
}