# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/models/ginet.py
# 说明：神经网络模型定义与注册表
# 模块功能：图同构网络（GIN）分子编码器 GINet 与多分子图分支（每个 SMILES 列一张图）拼接后接 MLP 读出的 GINMLPR，用于分子/反应性质预测。
import torch
from torch import nn
import torch.nn.functional as F

# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from hdl.layers.graph.gin import GINEConv
from hdl.layers.general.linear import (
    # BNReLULinear,
    BNReLULinearBlock,
)
from hdl.models.utils import load_model
from hdl.ops.utils import get_activation


# 本模块对外导出的模型类
__all__ = [
    "GINet",
    "GINMLPR",
]


# 原子类型（含 mask token）、手性标记、键类型（含芳香与自环）与键方向的类别数，用于嵌入查表
num_atom_type = 119  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5  # including aromatic and self-loop edge
num_bond_direction = 3 


class GINet(nn.Module):
    """
    图同构网络（GIN）分子编码器：原子类型与手性标记查表得初始节点特征，num_layer 层 GINEConv + BatchNorm 堆叠，
    池化读出层（readout）后接线性层。forward 返回 (图级特征 h, 降维表征 out)。
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(
        self,
        num_layer=5,
        emb_dim=300,
        feat_dim=512,
        drop_ratio=0,
        pool='mean'
    ):
        """num_layer：GIN 卷积层数；emb_dim：节点嵌入维；feat_dim：池化后图级特征维；
        drop_ratio：Dropout 比例；pool：读出池化方式（mean/max/add）。"""
        super(GINet, self).__init__()
        # 保存构造参数，供 checkpoint 存取与按名重建模型
        self.init_args = {
            'num_layer': num_layer,
            'emb_dim': emb_dim,
            'feat_dim': feat_dim,
            'drop_ratio': drop_ratio,
            'pool': pool
        }
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        # 原子类型与手性标记两张查表用 xavier 均匀分布初始化
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        # num_layer 个 GINEConv 堆叠，内部为「邻居求和聚合 -> MLP」的图同构更新
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        # 每层一个 BatchNorm1d，与上面的卷积层一一对应
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
        # 读出层（readout）：按图编号 batch 把节点特征池化为图级向量，再线性映射到 feat_dim；
        # out_lin 为 feat_dim -> feat_dim -> ReLU -> feat_dim//2 的降维头
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
        self.feat_lin = nn.Linear(
            self.emb_dim,
            self.feat_dim
        )

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(
                self.feat_dim,
                self.feat_dim // 2
            )
        )

    def forward(self, data):
        """data 为批量图：x[:,0] 原子类型、x[:,1] 手性标记，edge_index/edge_attr 为键拓扑与键类型，
        data.batch 节点所属图编号。
        返回 h (图数, feat_dim) 图级特征与 out (图数, feat_dim//2) 降维表征。"""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # 节点初始特征 = 原子类型嵌入 + 手性标记嵌入
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        # num_layer 层「GIN 卷积 -> BatchNorm -> (ReLU) -> Dropout」堆叠，末层不接 ReLU
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        # 图池化读出 -> feat_lin 得图级特征 h -> out_lin 得降维表征 out
        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        
        return h, out


class GINMLPR(nn.Module):
    """多分子图（每个 SMILES 列一张图）性质预测模型：num_smiles 个独立 GINet 分支各自出图级表征，
    横向拼接后送入 BN-ReLU-Linear 多层感知机（MLP）塔，再经 sigmoid 输出 (batch, out_dim) 预测值。
    ckpt_file 非空时在构造末尾调用 load_ckpt，为每个 GIN 分支载入预训练权重。"""
    def __init__(
        self,
        num_layer=5,
        emb_dim=300,
        feat_dim=512,
        out_dim=1,
        drop_ratio=0,
        pool='mean',
        ckpt_file: str = None,
        num_smiles: int = 1,
    ) -> None:
        """num_layer/emb_dim/feat_dim/drop_ratio/pool：透传给每个 GINet 分支；out_dim：最终预测维数；
        ckpt_file：各 GIN 分支的预训练权重路径；num_smiles：并行的分子图分支数（即参与反应的 SMILES 数）。"""
        super().__init__()
        self.init_args = {
            "num_layer": num_layer,
            "emb_dim": emb_dim,
            "feat_dim": feat_dim,
            "out_dim": out_dim,
            "drop_ratio": drop_ratio,
            "pool": pool,
            "ckpt_file": ckpt_file,
            "num_smiles": num_smiles
        }
        # num_smiles 个结构相同、参数独立的 GIN 分支，每个负责一张分子图
        self.gins = nn.ModuleList([])
        for _ in range(num_smiles):
            self.gins.append(
                GINet(
                    num_layer=num_layer,
                    emb_dim=emb_dim,
                    feat_dim=feat_dim,
                    drop_ratio=drop_ratio,
                    pool=pool,
                )
            )
        self.ckpt_file = ckpt_file
        self.num_smiles = num_smiles

        # 各分支图级表征拼接后送入 MLP 塔：输入维 feat_dim//2 * num_smiles，输出 out_dim，再接 sigmoid
        self.ffn = BNReLULinearBlock(
            in_features=feat_dim // 2 * num_smiles,
            out_features=out_dim,
            num_layers=num_layer,
            hidden_size=feat_dim // 2
        )
        self.out_act = get_activation('sigmoid')

        # 构造时若给了权重路径则立即加载预训练分支权重
        if ckpt_file is not None:
            self.load_ckpt()

    def load_ckpt(self):
        """把 ckpt_file 中的权重依次载入每个 GIN 分支（复用 hdl.models.utils.load_model）。"""
        if self.ckpt_file is not None:
            for i in range(self.num_smiles):
                load_model(
                    self.ckpt_file,
                    model=self.gins[i]
                )
 
    def forward(
        self,
        data 
    ):
        """data 为长度 num_smiles 的序列，每个元素对应一张分子图的批量数据（取 data_i[0] 作为 torch_geometric 批量图）。
        各 GIN 分支的降维表征按列拼接成 (batch, feat_dim//2 * num_smiles)，经 MLP 塔与 sigmoid 返回 (batch, out_dim)。"""
        out_list = []
        for data_i, gin in zip(data, self.gins):
            out_list.append(gin(data_i[0])[1])
        out = torch.hstack(out_list)  # (batch_size, feat_dim//2 * num_smiles)
        out = self.ffn(out)
        out = self.out_act(out)

        return out
            