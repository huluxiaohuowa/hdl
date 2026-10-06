# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/models/chiral_gnn.py
# 说明：神经网络模型定义与注册表
# 模块功能：手性图神经网络（chiral GNN）分子性质预测模型，按 gnn_type 堆叠 GCN/GIN/有向消息传递（D-MPNN）卷积并注入四面体手性信息。
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from hdl.layers.graph.chiral_graph import (
    GCNConv,
    GINEConv,
    DMPNNConv,
    # get_tetra_update,
)

from hdl.layers.graph.tetra import (
    # get_tetra_update,
    TETRA_UPDATE_DICT
)


class GNN(nn.Module):
    """手性图神经网络（chiral GNN）分子性质预测模型：depth 层图卷积在原子或化学键特征上迭代消息传递，
    再经图池化读出层（readout）把节点特征聚成图级向量，最后单层线性映射输出。
    forward 输入批量图 data，输出形状 (图数, 输出维) 的 sigmoid 概率。"""
    def __init__(
        self,
        # args,
        num_node_features: int = 48,
        num_edge_features: int = 7,
        depth: int = 15,
        hidden_size: int = 128,
        dropout: float = 0.1,
        gnn_type: str = 'dmpnn',
        graph_pool: str = 'mean',
        tetra: bool = True,
        task: str = 'classification', 
        output_num: int = None,
        message: str = 'tetra_permute_concat',
        include_vars: bool = False,
    ):
        """num_node_features/num_edge_features：原子、化学键输入特征维；depth：图卷积层数；
        hidden_size：隐藏特征维；dropout： Dropout 比例；gnn_type：卷积类型（gin/gcn/dmpnn）；
        graph_pool：读出池化方式（sum/mean/max/attn/set2set）；tetra：是否启用四面体手性（chirality）更新；
        task：regression 或 classification；output_num：输出维数；message：手性聚合方式（TETRA_UPDATE_DICT 键）；
        include_vars：是否在输出上额外给一列用于方差。"""
        super(GNN, self).__init__()

        # 保存构造参数，供 save_model 写入 checkpoint 以便之后按原参数重建模型
        self.init_args = {
            "num_node_features": num_node_features,
            "num_edge_features": num_edge_features,
            "depth": depth,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "gnn_type": gnn_type,
            "graph_pool": graph_pool,
            "tetra": tetra,
            "task": task,
            "message": message,
            "include_vars": include_vars
        }

        self.depth = depth
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.graph_pool = graph_pool
        self.tetra = tetra
        self.task = task
        self.out_dim = output_num
        self.include_vars = include_vars

        if self.gnn_type == 'dmpnn':
            # D-MPNN：隐状态在化学键上，初始键表征 = 源端原子特征与键特征拼接后的线性映射 + ReLU
            self.edge_init = nn.Linear(61, self.hidden_size)
            self.edge_to_node = DMPNNConv(
                hidden_size=hidden_size,
                tetra=tetra,
                message=message
            )
        else:
            # GCN/GIN：原子特征与键特征（13 维）各自线性投影到 hidden_size
            self.node_init = nn.Linear(num_node_features, self.hidden_size)
            self.edge_init = nn.Linear(13, self.hidden_size)

        # layers
        # 深度为 depth 的图卷积堆叠（ModuleList），按 gnn_type 选 GINEConv / GCNConv / DMPNNConv
        self.convs = torch.nn.ModuleList()

        for _ in range(self.depth):
            # 每层按 gnn_type 实例化一个带手性信息通道的图卷积（hidden_size 进出，层内相加在 forward 中完成）
            if self.gnn_type == 'gin':
                self.convs.append(GINEConv(
                    hidden_size=hidden_size,
                    tetra=tetra,
                    message=message
                ))
            elif self.gnn_type == 'gcn':
                self.convs.append(GCNConv(
                    hidden_size=hidden_size,
                    tetra=tetra,
                    message=message
                ))
            elif self.gnn_type == 'dmpnn':
                self.convs.append(DMPNNConv(
                    hidden_size=hidden_size,
                    tetra=tetra,
                    message=message
                ))
            else:
                ValueError('Undefined GNN type called {}'.format(self.gnn_type))

        # graph pooling
        # 手性（四面体）更新模块：按 message 名字从 TETRA_UPDATE_DICT 取工厂类构造
        if self.tetra:
            self.tetra_update = TETRA_UPDATE_DICT[message](hidden_size)
            # self.tetra_update = get_tetra_update(args)

        # 读出层（readout）：图池化把原子/键级特征聚成图级向量；attn 为门控注意力池化，set2set 输出维度翻倍
        if self.graph_pool == "sum":
            self.pool = global_add_pool
        elif self.graph_pool == "mean":
            self.pool = global_mean_pool
        elif self.graph_pool == "max":
            self.pool = global_max_pool
        elif self.graph_pool == "attn":
            # 注意力池化门控网络：hidden -> 2*hidden -> BN -> ReLU -> 1，逐节点打分后加权求和
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 2 * self.hidden_size),
                                            torch.nn.BatchNorm1d(2 * self.hidden_size),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(2 * self.hidden_size, 1)))
        elif self.graph_pool == "set2set":
            self.pool = Set2Set(self.hidden_size, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        # ffn
        # 输出层：单线性映射；set2set 读出维度翻倍故乘 mult；include_vars 固定输出 2 列，否则取 output_num 或 1
        self.mult = 2 if self.graph_pool == "set2set" else 1
        if self.include_vars:
            out_dim = 2
        elif self.out_dim:
            out_dim = self.out_dim
        else:
            out_dim = 1
        self.ffn = nn.Linear(self.mult * self.hidden_size, out_dim)

    def forward(self, data):
        """data 为 torch_geometric 批量图：x (原子特征)、edge_index (2, 边数)、edge_attr (边数, 键特征)、
        batch (节点所属图编号)、parity_atoms (原子手性标记)、parity_bond_index (手性中心对应键索引)。
        返回 (图数, 输出维) 概率；include_vars 时返回 (mean, var)。"""
        x, edge_index, edge_attr, batch, parity_atoms, parity_bond_index = data.x, data.edge_index, data.edge_attr, data.batch, data.parity_atoms, data.parity_bond_index

        if self.gnn_type == 'dmpnn':
            # 初始键表征：源端原子特征拼接键特征后 ReLU 投影
            row, col = edge_index
            edge_attr = torch.cat([x[row], edge_attr], dim=1)
            edge_attr = F.relu(self.edge_init(edge_attr))
        else:
            # 原子与键特征各自 ReLU 投影到 hidden_size
            x = F.relu(self.node_init(x))
            edge_attr = F.relu(self.edge_init(edge_attr))

        # 逐层缓存节点/键表征，作为下一层卷积的输入（dmpnn 只追加键表征，节点输入始终为初始 x）
        x_list = [x]
        edge_attr_list = [edge_attr]

        # convolutions
        # 逐层图卷积；除末层外先 ReLU 再 Dropout（末层不接 ReLU），最后把本层卷积原始输出加回激活后的结果
        for layer_idx in range(self.depth):

            x_h, edge_attr_h = self.convs[layer_idx](x_list[-1], edge_index, edge_attr_list[-1], parity_atoms, parity_bond_index)
            # dmpnn 的隐状态留在化学键上，其余类型留在原子上
            h = edge_attr_h if self.gnn_type == 'dmpnn' else x_h

            if layer_idx == self.depth - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            if self.gnn_type == 'dmpnn':
                # 相加：本层激活后的键表征 + 本层卷积原始键输出
                h += edge_attr_h
                edge_attr_list.append(h)
            else:
                # 相加：本层激活后的节点表征 + 本层卷积原始节点输出
                h += x_h
                x_list.append(h)

        # dmpnn edge -> node aggregation
        # D-MPNN 隐状态在键上，最后再做一次边→节点聚合得到原子表征
        if self.gnn_type == 'dmpnn':
            h, _ = self.edge_to_node(x_list[-1], edge_index, h, parity_atoms, parity_bond_index)

        # 读出层池化（按 batch 索引聚成图级向量）后经 ffn 线性映射，再 sigmoid 压缩到 0~1；回归与分类走同一出口
        if self.task == 'regression':
            output = torch.sigmoid(self.ffn(self.pool(h, batch)))
        elif self.task == 'classification':

            output = torch.sigmoid(self.ffn(self.pool(h, batch)))
        # mean = output[:, 0]       
        if not self.include_vars:
            return output
        else:
            # include_vars：对输出第 2 列做 softplus 后拆成均值与方差两项返回
            mean, var = F.softplus(output[:, 1])
            return mean, var