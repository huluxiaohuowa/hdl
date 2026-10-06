# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/layers/graph/transformer.py
# 说明：图神经网络层（GCN/GIN/Transformer/手性图）
# 模块功能：图 Transformer（graph transformer）多头注意力（multi-head attention）卷积层的本地实现，
#           下方英文 docstring 与公式说明保留自上游实现，本文件只补充中文注释。
import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class TransformerConv(MessagePassing):
    # 中文：图 Transformer 注意力卷积层，继承 torch_geometric.nn.MessagePassing（消息传递基类，聚合方式默认 add）。
    # 输入：x 为节点特征 [num_nodes, in_channels]（也可为源/目标二元组 PairTensor）、
    #       edge_index 边索引 [2, num_edges]、可选 edge_attr 边特征 [num_edges, edge_dim]；
    # 输出：concat=True 时 [num_nodes, heads*out_channels]，concat=False 时对多头取均值得 [num_nodes, out_channels]。
    # 边特征以加性方式注入 key（影响注意力权重）与 value（影响聚合结果）。
    # 下方英文 docstring 保留自上游实现。
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:
    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (bool, optional): If set, will combine aggregation and
            skip information via
            .. math::
                \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}
            with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
            [ \mathbf{x}_i, \mathbf{m}_i, \mathbf{x}_i - \mathbf{m}_i ])`
            (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). Edge features are added to the keys after
            linear transformation, that is, prior to computing the
            attention dot product. They are also added to final values
            after the same linear transformation. The model is:
            .. math::
                \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
                \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
                \right),
            where the attention coefficients :math:`\alpha_{i,j}` are now
            computed via:
            .. math::
                \alpha_{i,j} = \textrm{softmax} \left(
                \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                {\sqrt{d}} \right)
            (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output and the
            option  :attr:`beta` is set to :obj:`False`. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int,
                                                     int]], out_channels: int,
                 heads: int = 1, concat: bool = True, beta: bool = False,
                 dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, root_weight: bool = True, **kwargs):
        """初始化各线性投影与开关。
        Args:
            in_channels: 输入维度，int 或 (源维度, 目标维度) 二元组
            out_channels: 每个注意力头（head）的输出维度
            heads: 多头注意力（multi-head attention）的头数
            concat: True 拼接多头，False 对多头求均值
            beta: True 时用可学习门控在聚合结果与自身变换之间加权（跳过连接, skip）
            edge_dim: 边特征维度，None 表示不使用边特征
            root_weight: False 时不加入自身（root）节点变换项
        """
        # 未显式指定时，消息聚合方式取 add（求和）
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        # beta 门控依赖 root 项，关闭 root_weight 时 beta 一并失效
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # key/query/value 三个投影，各输出 heads*out_channels；key 来自源节点 j、query 来自目标节点 i
        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            # 边特征投影：加到 key 上参与注意力点积，同时加到 value 上参与聚合
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            # 无边的特征时不注册参数，message 中据此跳过边注入
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            # 自身（root）节点的跳过变换，维度需匹配 heads*out_channels
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                # 门控：由 [聚合结果, 自身变换, 二者之差] 压成单个标量 logit
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            # 多头取均值后维度降为 out_channels，跳过投影随之变窄
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        """重新初始化各投影权重（未注册参数时对应分支跳过）。"""
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None):
        """"""
        # 中文：前向。x [num_nodes, in_channels]（或源/目标二元组）、edge_index [2, num_edges]、
        # edge_attr [num_edges, edge_dim] 或 None；返回 [num_nodes, heads*out_channels]（concat）
        # 或 [num_nodes, out_channels]（concat=False）。

        if isinstance(x, Tensor):
            # 单一节点特征时，源侧与目标侧共用同一张量（二部图输入才需要分开传）
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        # 消息传递：得到 [num_nodes, heads, out_channels] 的邻域聚合结果
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.concat:
            # 多头拼接：摊平头维到最后一维 -> [num_nodes, heads * out_channels]
            out = out.view(
                -1,
                self.heads * self.out_channels
            )
        else:
            # 多头平均：对头维（dim=1）求均值 -> [num_nodes, out_channels]
            out = out.mean(dim=1)

        if self.root_weight:
            # 自身（root）节点的线性变换，等价于跳过连接（skip connection）项
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                # beta 门控：由 [聚合结果, 自身项, 二者之差] 逐节点算出 sigmoid 权重，再在两者间凸组合
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                # 无门控时直接相加，构成残差（residual）连接
                out += x_r

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        """单边消息：缩放点积注意力（scaled dot-product attention）+ 值加权。
        Args:
            x_i: 目标节点特征 [num_edges, in_channels]（由 edge_index[1] gather）
            x_j: 源节点（邻居）特征 [num_edges, in_channels]（由 edge_index[0] gather）
            edge_attr: 边特征 [num_edges, edge_dim] 或 None
            index: 每条边对应的目标节点下标 [num_edges]，用于按节点分段 softmax
            ptr / size_i: 分段起点 / 每段边数，用于归一化
        Returns:
            [num_edges, heads, out_channels]，已乘以注意力权重
        """

        # query 来自目标节点 i、key 来自源节点 j，各自投影后拆成 [num_edges, heads, out_channels]
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:
            # 有边特征时：边嵌入加到 key 上，直接参与注意力打分
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key += edge_attr

        # 注意力分数：逐头点积再除以 sqrt(out_channels)（按单头维度缩放）-> [num_edges, heads]
        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        # 本实现不构造稠密 mask：稀疏 edge_index 天然限制只能看见一跳邻居；
        # softmax 以 index（目标节点）分组，在"同一目标节点的所有入边"这一维上归一化，而非特征维
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # value 投影；边嵌入同样加到 value 上，使边信息进入聚合结果
        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        if edge_attr is not None:
            out += edge_attr

        # 在最后一维广播注意力权重（head 后补 1 以便逐元素相乘）-> [num_edges, heads, out_channels]
        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        # 打印输入/输出维度与头数，便于检查层配置
        return '{}({}, {}, heads={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.heads
        )