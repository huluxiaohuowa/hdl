# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/layers/graph/gin.py
# 说明：图神经网络层（GCN/GIN/Transformer/手性图）
# 模块功能：带边特征的图同构网络（Graph Isomorphism Network with Edge features, GINE）卷积层，
#           把键/芳族边类型等离散类别嵌入（nn.Embedding）加到邻居消息上后做求和聚合，再经 MLP 更新节点。

import torch
from torch import nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

# 下列常量是离散特征字典的大小，决定 nn.Embedding 的行数（类别数），必须与数据侧 featurizer 对齐
# 原子类别数：原子序数 1~118 共 118 类，再加 1 个掩码（mask token）类，用于子图掩码预训练
num_atom_type = 119  # including the extra mask tokens
# 原子手性标记类别数：未指定 / 四面体顺时针 / 四面体逆时针
num_chirality_tag = 3

# 键（化学键）类别数：单键、双键、三键、芳族键共 4 类，再加 1 类自环（self-loop）边（索引 4）
num_bond_type = 5  # including aromatic and self-loop edge
# 键方向类别数：NONE / ENDUPRIGHT / ENDDOWNRIGHT，用于立体化学双键构型
num_bond_direction = 3 


class GINEConv(MessagePassing):
    """GINE 图卷积层：继承 torch_geometric.nn.MessagePassing（消息传递, message passing 基类）。
    消息形式为 h_j + e_ij（邻居节点嵌入加边嵌入），聚合为基类默认的求和（add），
    节点更新为 MLP(聚合结果 + 自身信息经由自环边传入)。
    输入 x [num_nodes, emb_dim]、edge_index [2, num_edges]、edge_attr [num_edges, 2]（两列离散类别索引），
    输出更新后的节点嵌入 [num_nodes, emb_dim]。
    """
    def __init__(self, emb_dim):
        # 基类未指定 aggr，沿用其默认聚合方式（求和 add）
        super(GINEConv, self).__init__()
        # 节点更新 MLP：emb_dim -> 2*emb_dim -> emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2 * emb_dim, emb_dim)
        )
        # 边类别嵌入表：键类型（5 类，含自环）与键方向（3 类）各查一个 emb_dim 向量
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        """前向：补自环 → 边类别索引查嵌入 → 消息传递 → MLP 更新。
        Args:
            x: 节点嵌入 [num_nodes, emb_dim]
            edge_index: 边索引 [2, num_edges]（每条化学键存正反两条有向边）
            edge_attr: 边的离散类别索引 [num_edges, 2]，列 0 为键类型、列 1 为键方向
        Returns:
            节点嵌入 [num_nodes, emb_dim]
        """
        # add self loops in the edge space
        # 在边表尾部追加 num_nodes 条自环（self-loop）边，只取返回的边索引
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        # 为自环边构造类别特征：键类型列填 4（自环专用类别），键方向列保持 0（NONE）
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # 拼接后边特征形状变为 [num_edges + num_nodes, 2]，与追加自环后的边索引一一对应
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # 两张嵌入表逐边相加，得到边向量表示 [num_edges + num_nodes, emb_dim]
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # propagate 按目标节点把 message 结果做求和聚合，再调用 update
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        """单边消息：源节点（邻居 j）嵌入与边嵌入逐元素相加，形状 [num_edges, emb_dim]。
        边嵌入作为加性项注入消息，使键类型/方向信息参与聚合。
        """
        return x_j + edge_attr

    def update(self, aggr_out):
        """节点更新：聚合后的邻域表示过 MLP，输出 [num_nodes, emb_dim]。"""
        return self.mlp(aggr_out)