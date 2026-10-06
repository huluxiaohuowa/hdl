# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/layers/graph/gcn.py
# 说明：图神经网络层（GCN/GIN/Transformer/手性图）
# 模块功能：图卷积网络（Graph Convolutional Network, GCN）的最小封装层，
#           直接用 torch_geometric 的 GCNConv 做一层节点特征到节点特征的邻域聚合。
from torch import nn
from torch_geometric.nn import GCNConv


class GraphConv(nn.Module):
    """图卷积层：继承 torch.nn.Module，内部只包一个 torch_geometric GCNConv。
    GCNConv 自带自环（self-loop）添加与对称归一化，因此这里等价于一层标准 GCN 传播。
    """
    def __init__(self, num_features, num_out_features):
        # Init parent
        super(GraphConv, self).__init__()

        # GCN layers
        # num_features: 输入节点特征维度；num_out_features: 输出节点特征维度
        self.conv = GCNConv(num_features, num_out_features) 

    def forward(self, x, edge_index):
        """前向：单次图卷积传播。
        Args:
            x: 节点特征 [num_nodes, num_features]
            edge_index: 边索引（邻接表）[2, num_edges]，第 0 行为源节点、第 1 行为目标节点
        Returns:
            隐藏节点表示 [num_nodes, num_out_features]
        """

        hidden = self.conv(x, edge_index)
        return hidden
