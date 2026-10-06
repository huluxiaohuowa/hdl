# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/metric_loss/nt_xent.py
# 说明：损失函数与评估指标
# 模块功能：对比学习（contrastive learning）中的 NT-Xent 损失，即归一化温度缩放交叉熵损失。
import torch
import numpy as np


class NTXentLoss(torch.nn.Module):
    """NT-Xent 损失：同一实例的两个视图构成正样本对，批内其余表示构成负样本，
    以交叉熵的形式让正样本对的相似度在温度系数 tau 下相对负样本最大。"""

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        """Args:
            device: 掩码与标签张量所在计算设备。
            batch_size (int): 单视图批大小 N，拼接后共 2N 个表示。
            temperature (float): 温度系数 tau，logits 除以后参与交叉熵。
            use_cosine_similarity (bool): True 用余弦相似度，False 用点积相似度。
        """
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        # 掩码在构造时一次性算好：True 表示该位置可作为当前锚点的负样本
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        """按开关选择相似度函数：True 返回余弦相似度封装，False 返回张量点积封装。"""
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        """构造 (2N, 2N) 布尔掩码：剔除单位对角（自身相似度）与偏移 ±N 的两条对角
        （同一实例的另一视图，即正样本对），其余位置视为负样本。

        Returns:
            torch.Tensor: 形状 (2N, 2N) 的布尔张量，True 表示可作负样本。
        """
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        """点积相似度：对 x 的每一行与 y 的每一行求内积，返回形状 (N, M) 的相似度矩阵。"""
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        """余弦相似度：逐对比较 x 的行与 y 的行，返回形状 (N, M) 的相似度矩阵。"""
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        """前向计算 NT-Xent 损失。

        Args:
            zis (torch.Tensor): 视图 i 的表示，形状 (N, C)。
            zjs (torch.Tensor): 视图 j 的表示，形状 (N, C)，与 zis 按位置构成正样本对。

        Returns:
            torch.Tensor: 标量损失，已对 2N 个锚点取平均。
        """
        # 沿第 0 维拼接成 2N 个表示：第 t 个与第 t+N 个互为正样本
        representations = torch.cat([zjs, zis], dim=0)

        # 相似度矩阵，形状 (2N, 2N)
        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        # 正样本项：取偏离主对角 ±N 的两条对角线，得到每个锚点在其配对视图上的相似度
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        # 负样本项：用掩码从相似度矩阵挑出非自身、非配对的位置，每个锚点 2N-2 个
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # 正样本相似度放在第 0 列（即标签位置），其余列为负样本相似度
        logits = torch.cat((positives, negatives), dim=1)
        # 温度系数 tau（temperature）缩放，tau 越小 softmax 分布越尖锐
        logits /= self.temperature

        # 所有锚点的正确类别都是 0，用交叉熵（reduction=sum）拉近正样本、推远负样本
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        # 交叉熵为求和归约，这里按 2N 个锚点取平均得到标量损失
        return loss / (2 * self.batch_size)
