# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/layers/graph/chiral_graph.py
# 说明：图神经网络层（GCN/GIN/Transformer/手性图）
# 模块功能：手性（chirality）感知的图卷积层集合，均基于 torch_geometric.nn.MessagePassing 做消息传递（message passing）。
#           与标准实现的区别：对四面体手性中心（parity_atoms 非 0 的原子）不用求和聚合邻居，
#           而是改用 tetra.py 中的置换/差积更新层（TETRA_UPDATE_DICT[message]）替换其聚合结果，
#           从而让节点表示区分 R/S 立体构型。edge_attr 在这些层中已是稠密边嵌入 [num_edges, hidden_size]。

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tetra import (
    # get_tetra_update,
    TETRA_UPDATE_DICT
)


class GCNConv(MessagePassing):
    """手性感知的图卷积（GCN）层：继承 MessagePassing，聚合方式显式取 add（求和）。
    边不做更新（edge_attr 原样返回），只更新节点；对称归一化（symmetric normalization）在 forward 中手工计算。
    输入 x [num_nodes, hidden_size]、edge_index [2, num_edges]、edge_attr [num_edges, hidden_size]、
    parity_atoms [num_nodes]（+1 顺时针 CW / -1 逆时针 CCW / 0 非手性），
    输出 (batch_norm 后的节点表示 [num_nodes, hidden_size], 未改动的 edge_attr)。
    """
    def __init__(
        self,
        # args,
        hidden_size,
        tetra,
        message
    ):
        """Args:
            hidden_size (int): 节点/边的隐藏维度，输入输出同维
            tetra (bool): 是否对手性中心启用手性置换更新
            message (str): 手性更新层的名称，作为 TETRA_UPDATE_DICT 的键
        """
        # aggr='add'：邻居消息按目标节点求和聚合（非 mean/max）
        super(GCNConv, self).__init__(aggr='add')
        # 节点特征自身的线性变换，先于邻域聚合执行
        self.linear = nn.Linear(hidden_size, hidden_size)
        # 归一化（LayerNorm 位置的替代物）：放在残差相加之后、对整层输出做批归一化
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.tetra = tetra  # bool
        if self.tetra:
            # self.tetra_update = get_tetra_update(args)
            # 按名字实例化四面体手性更新层（置换/拼接置换/差积三种之一）
            self.tetra_update = TETRA_UPDATE_DICT[message](hidden_size)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        parity_atoms
    ):
        """前向：线性变换 -> 对称归一化邻域聚合 -> 手性中心覆盖 -> 残差 -> 批归一化。
        Args:
            x: 节点特征 [num_nodes, hidden_size]
            edge_index: 边索引 [2, num_edges]，第 0 行 row 为源节点、第 1 行 col 为目标节点
            edge_attr: 边嵌入 [num_edges, hidden_size]
            parity_atoms: 每个原子的手性标记 [num_nodes]，0 表示非手性中心
        Returns:
            (节点表示 [num_nodes, hidden_size], 原样返回的 edge_attr [num_edges, hidden_size])
        """

        # no edge updates
        x = self.linear(x)

        # Compute normalization
        # 目标节点出现次数即入度；+1 等价于给每个节点补一条自环（self-loop）边参与归一化
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        # 对称归一化系数 norm_e = 1/sqrt(deg[row] * deg[col])，逐边标量 [num_edges]
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

        if self.tetra:
            # 取出手性中心节点下标（parity 非 0），无手性原子时跳过
            tetra_ids = parity_atoms.nonzero().squeeze(1)
            if tetra_ids.nelement() != 0:
                # 用四面体置换更新覆盖这些节点的求和聚合结果
                x_new[tetra_ids] = self.tetra_message(x, edge_index, edge_attr, tetra_ids, parity_atoms)
        # 残差（residual）连接：聚合结果 + 本层线性变换后输入的 relu(x)（x 已在函数开头过 self.linear）
        x = x_new + F.relu(x)

        # 输出前做 BatchNorm1d；边特征不更新，原样传出供下一层使用
        return self.batch_norm(x), edge_attr

    def message(self, x_j, edge_attr, norm):
        """单边消息：邻居（源节点 j）特征加边嵌入后经 ReLU，再乘对称归一化系数。
        返回 [num_edges, hidden_size]，随后按目标节点求和。
        """
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def tetra_message(self, x, edge_index, edge_attr, tetra_ids, parity_atoms):
        """手性中心专用消息：收集每个手性原子的 4 个邻居，按四面体旋转群做置换/差积更新。
        Args:
            x: 已线性变换的节点特征 [num_nodes, hidden_size]
            edge_index / edge_attr: 边索引与边嵌入
            tetra_ids: 手性中心节点下标 [num_chiral]
            parity_atoms: 手性标记 [num_nodes]，用于判断绕向
        Returns:
            [num_chiral, hidden_size]，已乘上手性伪度归一化系数
        """

        row, col = edge_index
        # 对每个手性中心 i，取所有 col==i 的边的源节点 row -> [1, 4]，拼接成 [num_chiral, 4] 邻居下标
        tetra_nei_ids = torch.cat([row[col == i].unsqueeze(0) for i in range(x.size(0)) if i in tetra_ids])

        # calculate pseudo tetra degree aligned with GCN method
        # 用手性中心的 4 个邻居各自的度取倒数平方根后再求均值，乘 0.5 作为该中心的缩放系数 [num_chiral]
        deg = degree(col, x.size(0), dtype=x.dtype)
        t_deg = deg[tetra_nei_ids]
        t_deg_inv_sqrt = t_deg.pow(-0.5)
        t_norm = 0.5 * t_deg_inv_sqrt.mean(dim=1)

        # switch entries for -1 rdkit labels
        # RDKit 逆时针（CCW, parity=-1）的中心的邻居顺序与顺时针相反，交换前两个槽位统一绕向
        ccw_mask = parity_atoms[tetra_ids] == -1
        tetra_nei_ids[ccw_mask] = tetra_nei_ids.clone()[ccw_mask][:, [1, 0, 2, 3]]

        # calculate reps
        # 组装 (邻居, 中心) 顶点对 [2, num_chiral*4]，用于在稀疏边表中回查对应边
        edge_ids = torch.cat([tetra_nei_ids.view(1, -1), tetra_ids.repeat_interleave(4).unsqueeze(0)], dim=0)
        # dense_edge_attr = to_dense_adj(edge_index, batch=None, edge_attr=edge_attr).squeeze(0)
        # edge_reps = dense_edge_attr[edge_ids[0], edge_ids[1], :].view(tetra_nei_ids.size(0), 4, -1)
        # 逐对与 edge_index.t()（[num_edges, 2] 的 (源, 目标) 行）比较，取首个匹配边的行号
        attr_ids = [torch.where((a == edge_index.t()).all(dim=1))[0] for a in edge_ids.t()]
        # 边嵌入重塑为 [num_chiral, 4, hidden_size]
        edge_reps = edge_attr[attr_ids, :].view(tetra_nei_ids.size(0), 4, -1)
        # 邻居节点特征与对应边嵌入相加，得到四面体更新的输入 [num_chiral, 4, hidden_size]
        reps = x[tetra_nei_ids] + edge_reps

        # tetra_update 内部对 4 个邻居槽位做置换或差积，输出 [num_chiral, hidden_size]
        return t_norm.unsqueeze(-1) * self.tetra_update(reps)


class GINEConv(MessagePassing):
    """手性感知的图同构网络（GIN with Edge features, GINE）卷积层：继承 MessagePassing，聚合为 add（求和）。
    与 GCNConv 的区别：自身信息用可学习系数 (1+eps) 加权后与邻域求和一起过 MLP（GIN 的 sum 形式），
    自环不靠边索引补齐而是由该加权项承担；边特征以加性方式进入消息。
    输入 x [num_nodes, hidden_size]、edge_index [2, num_edges]、edge_attr [num_edges, hidden_size]、
    parity_atoms [num_nodes]，输出 (节点表示 [num_nodes, hidden_size], 未改动的 edge_attr)。
    """
    def __init__(
        self,
        # args,
        hidden_size,
        tetra,
        message
    ):
        """Args:
            hidden_size (int): 隐藏维度
            tetra (bool): 是否启用手性中心的置换更新
            message (str): TETRA_UPDATE_DICT 中手性更新类的键名
        """
        # aggr="add"：邻居消息求和聚合
        super(GINEConv, self).__init__(aggr="add")
        # GIN 的可学习 epsilon，控制自身特征相对邻域求和的权重，初始化为 0（即系数 1）
        self.eps = nn.Parameter(torch.Tensor([0]))
        # 节点更新 MLP：hidden -> 2*hidden，中间 BatchNorm1d + ReLU，再回到 hidden
        self.mlp = nn.Sequential(nn.Linear(hidden_size, 2 * hidden_size),
                                 nn.BatchNorm1d(2 * hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(2 * hidden_size, hidden_size))
        # 整层输出再做一次 BatchNorm1d（在 MLP 之后）
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.tetra = tetra
        if self.tetra:
            # self.tetra_update = get_tetra_update(args)
            self.tetra_update = TETRA_UPDATE_DICT[message](hidden_size)

    def forward(self, x, edge_index, edge_attr, parity_atoms):
        """前向：邻域聚合 -> 手性中心覆盖 -> (1+eps)·x + 聚合 过 MLP -> BatchNorm。
        Args:
            x: 节点特征 [num_nodes, hidden_size]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边嵌入 [num_edges, hidden_size]
            parity_atoms: 手性标记 [num_nodes]，非 0 行为四面体手性中心
        Returns:
            (节点表示 [num_nodes, hidden_size], 原样返回的 edge_attr)
        """
        # no edge updates
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.tetra:
            # 手性中心的求和结果被置换/差积更新的输出整体替换
            tetra_ids = parity_atoms.nonzero().squeeze(1)
            if tetra_ids.nelement() != 0:
                x_new[tetra_ids] = self.tetra_message(x, edge_index, edge_attr, tetra_ids, parity_atoms)

        # GIN 更新式：自身特征加权 (1+eps) 加上邻域求和结果，再过 MLP
        x = self.mlp((1 + self.eps) * x + x_new)
        return self.batch_norm(x), edge_attr

    def message(self, x_j, edge_attr):
        """单边消息：邻居特征与边嵌入相加后过 ReLU，形状 [num_edges, hidden_size]。"""
        return F.relu(x_j + edge_attr)

    def tetra_message(self, x, edge_index, edge_attr, tetra_ids, parity_atoms):
        """手性中心消息：与 GCNConv 版本同名，但不乘度归一化系数，直接返回四面体更新结果。
        Returns:
            [num_chiral, hidden_size]
        """

        row, col = edge_index
        # 收集每个手性中心的 4 个邻居节点下标 -> [num_chiral, 4]
        tetra_nei_ids = torch.cat([row[col == i].unsqueeze(0) for i in range(x.size(0)) if i in tetra_ids])

        # switch entries for -1 rdkit labels
        # 逆时针（CCW, -1）中心交换前两个邻居槽位，使 4 个邻居按统一绕向排列
        ccw_mask = parity_atoms[tetra_ids] == -1
        tetra_nei_ids[ccw_mask] = tetra_nei_ids.clone()[ccw_mask][:, [1, 0, 2, 3]]

        # calculate reps
        # (邻居, 中心) 顶点对 [2, num_chiral*4]，用于回查边在 edge_index 中的行号
        edge_ids = torch.cat([tetra_nei_ids.view(1, -1), tetra_ids.repeat_interleave(4).unsqueeze(0)], dim=0)
        # dense_edge_attr = to_dense_adj(edge_index, batch=None, edge_attr=edge_attr).squeeze(0)
        # edge_reps = dense_edge_attr[edge_ids[0], edge_ids[1], :].view(tetra_nei_ids.size(0), 4, -1)
        attr_ids = [torch.where((a == edge_index.t()).all(dim=1))[0] for a in edge_ids.t()]
        # 取对应边嵌入并重塑为 [num_chiral, 4, hidden_size]
        edge_reps = edge_attr[attr_ids, :].view(tetra_nei_ids.size(0), 4, -1)
        # 邻居节点特征 + 边嵌入，作为四面体更新层的输入
        reps = x[tetra_nei_ids] + edge_reps

        return self.tetra_update(reps)


class DMPNNConv(MessagePassing):
    """方向性消息传递神经网络（Directed Message Passing Neural Network, D-MPNN）卷积层：继承 MessagePassing，聚合 add。
    消息定义在边（化学键）上而非节点上：节点表示只是各入边消息的和，真正的状态更新发生在边上，
    并通过减去反向边消息避免信息沿同一条键来回传播。边不更新自身特征维度，hidden_size 始终不变。
    输入 x [num_nodes, hidden_size]（仅手性分支用到）、edge_index [2, num_edges]（键按正/反成对排列）、
    edge_attr [num_edges, hidden_size]（当前边状态）、parity_atoms [num_nodes]、parity_bond_index [num_chiral*4]，
    输出 (入边消息按节点求和 [num_nodes, hidden_size], 更新后的边状态 [num_edges, hidden_size])。
    """
    def __init__(
        self,
        # args,
        hidden_size,
        tetra,
        message
    ):
        """Args:
            hidden_size (int): 边/节点消息的隐藏维度
            tetra (bool): 是否启用手性中心的四面体更新
            message (str): 手性更新类名称（TETRA_UPDATE_DICT 的键）
        """
        # aggr='add'：同一目标节点上的入边消息求和
        super(DMPNNConv, self).__init__(aggr='add')
        # 边消息的线性投影，作用于 edge_attr 而非节点特征
        self.lin = nn.Linear(hidden_size, hidden_size)
        # 边状态更新：Linear -> BatchNorm1d -> ReLU（末层仍为 hidden_size 维，可直接作为下一层边状态）
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.ReLU())
        self.tetra = tetra
        if self.tetra:
            # self.tetra_update = get_tetra_update(args)
            self.tetra_update = TETRA_UPDATE_DICT[message](hidden_size)

    def forward(self, x, edge_index, edge_attr, parity_atoms, parity_bond_index):
        """前向：聚合入边消息 -> 手性中心覆盖 -> 减去反向边消息更新边状态。
        Args:
            x: 节点特征 [num_nodes, hidden_size]（只在手性分支中被传入）
            edge_index: 边索引 [2, num_edges]，第 0 行 row 为源、第 1 行 col 为目标
            edge_attr: 当前边状态 [num_edges, hidden_size]，num_edges 为偶数且正/反边相邻成对
            parity_atoms: 手性标记 [num_nodes]
            parity_bond_index: 各手性中心 4 条入边在 edge_attr 中的行号 [num_chiral*4]
        Returns:
            (节点级入边消息和 [num_nodes, hidden_size], 新边状态 [num_edges, hidden_size])
        """
        row, col = edge_index
        # x=None：消息只来自边特征；按目标节点求和后得到每个节点的入边消息和 [num_nodes, hidden_size]
        a_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)

        if self.tetra:
            # 手性节点的求和消息改由 4 条入边的四面体更新给出
            tetra_ids = parity_atoms.nonzero().squeeze(1)
            if tetra_ids.nelement() != 0:
                a_message[tetra_ids] = self.tetra_message(x, edge_index, edge_attr, tetra_ids, parity_atoms, parity_bond_index)

        # 把边按 (正向, 反向) 成对 reshape 后 flip 交换，得到每条边对应的反向边状态 [num_edges, hidden_size]
        rev_message = torch.flip(edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]).view(edge_attr.size(0), -1)
        # D-MPNN 核心更新：边 (row->col) 的新状态 = 源节点入边消息和 - 该边反向消息，再过 Linear/BN/ReLU
        return a_message, self.mlp(a_message[row] - rev_message)

    def message(self, x_j, edge_attr):
        """单边消息：只由边特征经线性投影 + ReLU 得到，形状 [num_edges, hidden_size]。
        形参 x_j 保留但不参与计算（D-MPNN 的消息载体是边而非源节点）。
        """
        return F.relu(self.lin(edge_attr))

    def tetra_message(self, x, edge_index, edge_attr, tetra_ids, parity_atoms, parity_bond_index):
        """手性中心的边基消息：直接用数据侧排好绕向的 4 条入边索引取边嵌入做四面体更新。
        parity_bond_index 在 featurizer 中已对 CCW（-1）中心交换过前两项，故此处无需再调序。
        Returns:
            [num_chiral, hidden_size]
        """
        # 按 [num_chiral*4] 行号取边状态，重塑为 [num_chiral, 4, hidden_size]
        edge_reps = edge_attr[parity_bond_index, :].view(parity_bond_index.size(0)//4, 4, -1)

        # 上面已 return，以下语句不会被执行（历史实现，按节点/边索引回查边的另一套写法）
        return self.tetra_update(edge_reps)
        # print('1')
        row, col = edge_index

        # 收集每个手性中心 4 条入边在 edge_index 中的列位置
        col_ids = torch.cat(
            [(col == i).nonzero() for i in tetra_ids]
        ).squeeze().unsqueeze(0)
        tetra_nei_ids = row[col_ids].reshape(-1, 4)
        
        # tetra_nei_ids = torch.cat([
        #     row[col == i].unsqueeze(0)  
        #     for i in tetra_ids
        # ])

        # print('2')
        # switch entries for -1 rdkit labels
        ccw_mask = parity_atoms[tetra_ids] == -1
        tetra_nei_ids[ccw_mask] = tetra_nei_ids.clone()[ccw_mask][:, [1, 0, 2, 3]]

        # calculate reps
        edge_ids = torch.cat([tetra_nei_ids.view(1, -1), tetra_ids.repeat_interleave(4).unsqueeze(0)], dim=0)
        # dense_edge_attr = to_dense_adj(edge_index, batch=None, edge_attr=edge_attr).squeeze(0)
        # edge_reps = dense_edge_attr[edge_ids[0], edge_ids[1], :].view(tetra_nei_ids.size(0), 4, -1)
        # edge_index_T = edge_index.t()
        # edge_ids_T = edge_ids.t()

        # attr_ids = [
        #     torch.where(
        #         (a == edge_index_T).all(dim=1)
        #     )[0]
        #     for a in edge_ids_T
        # ]
        # attr_ids = torch.cat([(edge_index_T == i).nonzero() for i in edge_ids_T])[:, 0].unique()

        edge_index_T = edge_index.t()
        edge_ids_T = edge_ids.t()        
        
        # 笛卡尔积批量匹配：把 edge_index 与待查 (邻居, 中心) 顶点对的两列分别组合，替代逐条 torch.where
        c0 = torch.cartesian_prod(
            edge_index_T[:, 0], edge_ids_T[:, 0]
        )
        c1 = torch.cartesian_prod(
            edge_index_T[:, 1], edge_ids_T[:, 1]
        )
        # 源、目标两端点差之和为 0 表示该组合命中一条已有边
        diff = torch.abs(c0[:, 0] - c0[:, 1]) \
            + torch.abs(c1[:, 0] - c1[:, 1])
        
        # 命中位置整除每个手性中心的边数（4），换回待查边对自身的序号
        attr_ids = torch.div(
            (diff == 0).nonzero(as_tuple=True)[0],
            edge_ids.size(1),
            rounding_mode='floor'
        )

        edge_reps = edge_attr[attr_ids, :].view(tetra_nei_ids.size(0), 4, -1)

        return self.tetra_update(edge_reps)