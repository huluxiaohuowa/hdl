# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/layers/general/linear.py
# 说明：通用神经网络层
# 模块功能：通用全连接与图级层组件：BN+线性+激活的基本块（BNReLULinear 系列）、节点-边交互的 Weave 层与其堆叠、
#           DenseNet 式稠密连接网络、图级求和/平均池化，以及接在编码器上的多任务多分类头。
#           图输入统一使用稠化/稀疏邻接张量 adj 与节点特征矩阵，而非 edge_index。
import typing as t

import torch
from torch import nn
from torch.autograd import Function
import torch_scatter
from torch.utils import checkpoint as tuc

from hdl.ops.utils import get_activation

# 本模块对外导出的层类清单
__all__ = [
    "WeaveLayer",
    "DenseNet",
    "AvgPooling",
    "SumPooling",
    "CasualWeave",
    "DenseLayer"
]


def _bn_function_factory(bn_module):
    """构造瓶颈（bottleneck）前向函数：先把多个特征张量沿最后一维拼接，再整体送入 bn_module。
    单独提取成函数是为了配合梯度检查点（gradient checkpointing）时反向重算复用同一拼接逻辑。
    """
    def bn_function(*inputs):
        # inputs 为若干 [..., in_features] 张量，concat 后形状 [..., sum(in_features)]
        concated_features = torch.cat(inputs, -1)
        bottleneck_output = bn_module(concated_features)
        return bottleneck_output

    return bn_function


class BNReLULinear(nn.Module):
    """
    批归一化 -> 线性 -> 激活 的基本块：继承 nn.Module，输入输出为 [..., in_features] -> [..., out_features]。
    Linear layer with bn->relu->linear architecture
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = 'elu',
        **kwargs
    ):
        """
        构建 BatchNorm1d(in_features) -> Linear(in_features, out_features, bias=False) -> 激活函数 的串行模块。
        Args:
            in_features (int):
                The number of input features
            out_features (int):
                The number of output features
            activation (str):
                The type of activation unit to use in this module,
                default to elu
        """
        super(BNReLULinear, self).__init__()
        # 归一化在最前（Pre-Norm 结构），线性层不带 bias（偏移已由前置 BatchNorm 承担）
        self.bn_relu_linear = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(
                in_features,
                out_features,
                bias=False
            ),
            get_activation(
                activation,
                inplace=True,
                **kwargs
            )
        )

    def forward(self, x):
        # 前向：输入 [..., in_features]，输出 [..., out_features]
        """The forward method"""
        return self.bn_relu_linear(x)


class SelectAdd(Function):
    """
    按行索引选择后相加的省显存实现：继承 torch.autograd.Function（autograd 算子基类）。
    计算 a + b[indices]（必要时 a 也先按 indices_a 选择），反向用 scatter_add 把梯度按索引段累加。
    Implement the memory efficient version of `a + b.index_select(indices)`
    """

    def __init__(self,
                 indices: torch.Tensor,
                 indices_a: torch.Tensor = None):
        """
        记录两个行索引：indices 用于选择 b，indices_a 用于（可选）选择 a。
        Initializer
        Args:
            indices (torch.Tensor): The indices to select the object `b`
            indices_a (torch.Tensor or None):
                The indices to select the object `a`. Default to None
        """
        self._indices = indices
        self._indices_a = indices_a

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        前向：按行索引取子集再逐行相加；传了 indices_a 时结果行数为 len(indices_a)，否则为 len(indices)。
        The forward pass
        Args:
            a (torch.Tensor)
            b (torch.Tensor): The input tensors
        Returns:
            torch.Tensor:
                The output tensor
        """
        if self._indices_a is not None:
            # 两侧都按索引展开成边级张量后相加：a[indices_a] + b[indices]
            return (a.index_select(dim=0, index=self._indices_a) +
                    b.index_select(dim=0, index=self._indices))
        else:
            # a 已是边级张量，只需把 b 按索引展开
            return a + b.index_select(dim=0, index=self._indices)

    def backward(self, grad_output):
        """反向：把输出梯度按索引散射累加回 a、b 的每一行，返回 (grad_a, grad_b)。"""
        # For the input a
        if self._indices_a is not None:
            # 同一源行被多条边引用时，其梯度需累加（scatter_add 的分段求和）
            grad_a = torch_scatter.scatter_add(grad_output,
                                               index=self._indices_a,
                                               dim=0)
        else:
            # If a is not index selected, simply clone the gradient
            grad_a = grad_output.clone()
        # For the input b, perform a segment sum
        grad_b = torch_scatter.scatter_add(grad_output,
                                           index=self._indices,
                                           dim=0)
        return grad_a, grad_b


class WeaveLayer(nn.Module):
    """节点-边交互层（Weave 层）：继承 nn.Module，把节点特征广播成边上的多组投影，
    再按边聚合回节点。邻接关系由稀疏邻接张量 adj 提供，不使用 edge_index。
    输入 n_feat [num_nodes, num_in_feat]、adj（COO 稀疏邻接矩阵，非零元即边），
    输出节点特征 [num_nodes, num_out_feat]。
    """
    def __init__(
        self,
        num_in_feat: int,
        num_out_feat: int,
        activation: str = 'relu',
        is_first_layer: bool = False
    ):
        """Args:
            num_in_feat (int): 输入节点特征维度
            num_out_feat (int): 输出节点/边特征维度
            activation (str): 激活函数名，交给 get_activation 解析
            is_first_layer (bool): 首层标记，True 时广播用裸 Linear（输入还未归一化）
        """
        super().__init__()
        self.num_in_feat = num_in_feat
        self.num_out_feat = num_out_feat
        self.activation = activation
        # Broadcasting node features to edges
        # 广播投影输出 5*num_out_feat，后续按 5 段切分：自身、起点求和项、终点求和项、起点最大值项、终点最大值项
        if is_first_layer:
            self.broadcast = nn.Linear(self.num_in_feat,
                                       self.num_out_feat * 5)
        else:
            self.broadcast = BNReLULinear(self.num_in_feat,
                                          self.num_out_feat * 5,
                                          self.activation)
        # Gather edge features to node
        # 边->节点的归一化+激活，仅作用于求和分支（最大值分支不过 BN）
        self.gather = nn.Sequential(nn.BatchNorm1d(self.num_out_feat),
                                    get_activation(self.activation,
                                                       inplace=True))

        # Update node features
        # 输入为 [最大值, 求和] 拼接后的 2*num_out_feat，压回 num_out_feat
        self.update = BNReLULinear(self.num_out_feat * 2,
                                   self.num_out_feat,
                                   self.activation)

    def forward(
        self,
        n_feat: torch.Tensor,
        adj: torch.Tensor
    ):
        """前向：广播到边 -> 边级两种聚合（求和 / 最大值）回节点 -> 与自身分支相加。
        Args:
            n_feat (torch.Tensor): 节点特征 [num_nodes, num_in_feat]
            adj (torch.Tensor): COO 稀疏邻接矩阵，_indices() 给出边的 (起点, 终点) 下标
        Returns:
            torch.Tensor: 更新后的节点特征 [num_nodes, num_out_feat]
        """
        node_broadcast = self.broadcast(n_feat)
        # 沿特征维切成 5 段，每段 [num_nodes, num_out_feat]
        (self_features,
         begin_features_sum,
         end_features_sum,
         begin_features_max,
         end_features_max) = torch.split(node_broadcast,
                                         self.num_out_feat,
                                         dim=-1)
        # 稀疏邻接的三元组下标：begin_ids 为边起点、end_ids 为边终点，长度均为 num_edges
        edge_info = adj._indices()
        begin_ids, end_ids = edge_info[0, :], edge_info[1, :]
        # 边上消息 = 起点节点投影 + 终点节点投影（max/sum 两组各算一次），形状 [num_edges, num_out_feat]
        edge_features_max = SelectAdd(end_ids,
                                      begin_ids)(begin_features_max,
                                                 end_features_max)
        edge_features_sum = SelectAdd(end_ids,
                                      begin_ids)(begin_features_sum,
                                                 end_features_sum)
        edge_gathered_sum = self.gather(edge_features_sum)
        # 求和聚合：把边特征按起点节点累加 -> [num_nodes, num_out_feat]
        edge_gathered_sum = torch_scatter.scatter_add(edge_gathered_sum,
                                                      begin_ids,
                                                      dim=0)
        min_val = edge_features_max.min()
        # 先整体平移到非负再做 scatter_max（结果再减回 min_val），使每段取到真实最大值
        edge_gathered_max = edge_features_max - min_val
        edge_gathered_max = torch_scatter.scatter_max(edge_gathered_max,
                                                      begin_ids,
                                                      dim=0)[0]
        edge_gathered_max = edge_gathered_max + min_val
        # 两种聚合方式（max 与 sum）沿特征维拼接 -> [num_nodes, 2*num_out_feat]
        edge_gathered = torch.cat([edge_gathered_max,
                                   edge_gathered_sum],
                                  dim=-1)
        node_update = self.update(edge_gathered)
        # 与广播的自身分支相加，构成跳过连接（skip connection）
        outputs = self_features + node_update
        return outputs


class CasualWeave(nn.Module):
    """Weave 层的堆叠（链式前向）：继承 nn.Module，按 hidden_sizes 依次改变节点特征维度，
    同一邻接矩阵 adj 在每层复用。输入 [num_nodes, num_feat]，输出 [num_nodes, hidden_sizes[-1]]。
    """
    def __init__(
        self,
        num_feat: int,
        hidden_sizes: t.Iterable,
        activation: str = 'elu'
    ):
        """Args:
            num_feat (int): 初始节点特征维度
            hidden_sizes (t.Iterable): 各层的输出维度，逐层首尾相接
            activation (str): 激活函数名
        """
        super().__init__()
        self.num_feat = num_feat
        self.hidden_sizes = list(hidden_sizes)
        self.activation = activation

        layers = []
        # 相邻两个 hidden size 组成一层的 (输入维度, 输出维度)
        for i, (in_feat, out_feat) in enumerate(
            zip(
                [self.num_feat, ] +
                list(self.hidden_sizes)[:-1],  # in_features
                self.hidden_sizes  # out_features
            )
        ):
            if i == 0:
                # 第一层输入未归一化，broadcast 用裸 Linear（is_first_layer=True）
                layers.append(
                    WeaveLayer(
                        in_feat,
                        out_feat,
                        self.activation,
                        True
                    )
                )
            else:
                layers.append(
                    WeaveLayer(
                        in_feat,
                        out_feat,
                        self.activation
                    )
                )
            self.layers = nn.ModuleList(layers)

    def forward(
        self,
        feat: torch.Tensor,
        adj: torch.Tensor
    ):
        """前向：节点特征依次通过各 WeaveLayer，维度按 hidden_sizes 演化，返回最后一层输出。"""
        feat_out = feat
        for layer in self.layers:
            feat_out = layer(
                feat_out,
                adj
            )
        return feat_out


class DenseLayer(nn.Module):
    """稠密连接（DenseNet）中的一个层：继承 nn.Module，先把前面所有层的输出按特征维拼接并过瓶颈网络
    （BNReLULinear），再交给一个 WeaveLayer。瓶颈用梯度检查点（gradient checkpointing）以时间换显存。
    输入为特征张量列表（各 [num_nodes, *]），输出 [num_nodes, num_out_feat]。
    """
    def __init__(
        self,
        num_in_feat: int,
        num_botnec_feat: int,
        num_out_feat: int,
        activation: str = 'elu',
    ):
        """Args:
            num_in_feat (int): 拼接后的输入特征总维度
            num_botnec_feat (int): 瓶颈输出维度
            num_out_feat (int): 本层新增输出维度（稠密连接的增长率 growth rate）
            activation (str): 激活函数名
        """
        super().__init__()
        self.num_in_feat = num_in_feat
        self.num_out_feat = num_out_feat
        self.num_botnec_feat = num_botnec_feat
        self.activation = activation

        # 瓶颈层：把拼接后的多源特征压到 num_botnec_feat
        self.bottlenec = BNReLULinear(
            self.num_in_feat,
            self.num_botnec_feat,
            self.activation
        )

        # 主体图卷积：Weave 层，节点-边交互后输出 num_out_feat
        self.weave = WeaveLayer(
            self.num_botnec_feat,
            self.num_out_feat,
            self.activation
        )

    def forward(
        self,
        ls_feat: t.List[torch.Tensor],
        adj: torch.Tensor,
    ):
        """前向：ls_feat 中各 [num_nodes, *] 特征沿最后一维拼接 -> 瓶颈 -> Weave 层。
        Args:
            ls_feat (t.List[torch.Tensor]): 之前所有层的节点特征
            adj (torch.Tensor): COO 稀疏邻接矩阵
        Returns:
            torch.Tensor: 本层新增特征 [num_nodes, num_out_feat]
        """
        bn_fn = _bn_function_factory(self.bottlenec)
        # checkpoint 不保存中间激活，反向时重算前向以节省显存
        feat = tuc.checkpoint(bn_fn, *ls_feat)
        return self.weave(
            feat,
            adj
        )


class DenseNet(nn.Module):
    """稠密连接的图网络：继承 nn.Module，结构为 CasualWeave 干层 + 若干 DenseLayer + 末端 BNReLULinear。
    每个 DenseLayer 的输入是干层输出与之前各层输出的拼接，故特征维度逐层增长 num_k_feat。
    输入 feat [num_nodes, num_feat]、adj（稀疏邻接），输出 [num_nodes, num_out_feat]。
    """
    def __init__(
        self,
        num_feat: int,
        casual_hidden_sizes: t.Iterable,
        num_botnec_feat: int,
        num_k_feat: int,
        num_dense_layers: int,
        num_out_feat: int,
        activation: str = 'elu'
    ):
        """Args:
            num_feat (int): 初始节点特征维度
            casual_hidden_sizes (t.Iterable): 干层 CasualWeave 的逐层维度
            num_botnec_feat (int): 每个稠密层瓶颈维度
            num_k_feat (int): 稠密层增长率（每层新增特征维度）
            num_dense_layers (int): 稠密层数
            num_out_feat (int): 最终输出维度
            activation (str): 激活函数名
        """
        super().__init__()
        self.num_feat = num_feat
        self.num_dense_layers = num_dense_layers
        self.casual_hidden_sizes = list(casual_hidden_sizes)
        self.num_out_feat = num_out_feat
        self.activation = activation
        self.num_k_feat = num_k_feat
        self.num_botnec_feat = num_botnec_feat
        # 干层：先做若干轮 Weave 节点-边交互，输出维度为 casual_hidden_sizes[-1]
        self.casual = CasualWeave(
            self.num_feat,
            self.casual_hidden_sizes,
            self.activation
        )
        dense_layers = []
        # 第 i 个稠密层的输入维度 = 干层输出 + 前 i 层各自贡献的 num_k_feat
        for i in range(self.num_dense_layers):
            dense_layers.append(
                DenseLayer(
                    self.casual_hidden_sizes[-1] + i * self.num_k_feat,
                    self.num_botnec_feat,
                    self.num_k_feat,
                    self.activation
                )
            )
        self.dense_layers = nn.ModuleList(dense_layers)

        # 末端把干层与全部稠密层特征（已拼接）压到 num_out_feat
        self.output = BNReLULinear(
            (
                self.casual_hidden_sizes[-1] +
                self.num_dense_layers * self.num_k_feat
            ),
            self.num_out_feat,
            self.activation
        )

    def forward(
        self,
        feat,
        adj
    ):
        """前向：干层 -> 逐个稠密层（输出累积到 ls_feat）-> 全部特征拼接 -> 输出投影。
        Args:
            feat: 节点特征 [num_nodes, num_feat]
            adj: COO 稀疏邻接矩阵
        Returns:
            [num_nodes, num_out_feat]
        """
        feat = self.casual(
            feat,
            adj
        )
        # ls_feat 累积干层输出与每个稠密层的输出，体现稠密连接（dense connection）
        ls_feat = [feat, ]
        for dense_layer in self.dense_layers:
            feat_i = dense_layer(
                ls_feat,
                adj
            )
            ls_feat.append(feat_i)
        # 沿特征维拼接全部中间表示 -> [num_nodes, 干层维度 + 层数*num_k_feat]
        feat_cat = torch.cat(ls_feat, dim=-1)
        return self.output(feat_cat)


class _Pooling(nn.Module):
    """图级池化（graph pooling）基类：继承 nn.Module，先做 BatchNorm+激活，再按 ids 分段聚合，
    把节点级特征 [num_nodes, in_features] 压成图级特征 [num_seg, in_features]。子类通过 pooling_op 指定聚合方式。
    """
    def __init__(
        self,
        in_features: int,
        pooling_op: t.Callable = torch_scatter.scatter_mean,
        activation: str = 'elu'
    ):
        # 构造归一化+激活子模块，并保存分段聚合函数（mean / add 等 torch_scatter 算子）
        """Summary
        Args:
            in_features (int): Description
            pooling_op (t.Callable, optional): Description
            activation (str, optional): Description
        """
        super(_Pooling, self).__init__()
        # 池化前先做 BatchNorm1d + 激活（Pre-Norm），保证进入聚合的节点特征已归一化
        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(in_features),
            get_activation(activation, inplace=True)
        )
        self.pooling_op = pooling_op

    def forward(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        num_seg: int = None
    ) -> torch.Tensor:
        """
        前向：先归一化激活，再用 torch_scatter 的分段聚合把同一图（ids 相同）的节点合并为一行。
        Args:
            x (torch.Tensor): The input tensor, size=[N, in_features]
            ids (torch.Tensor): A tensor of type `torch.long`, size=[N, ]
            num_seg (int): The number of segments (graphs)
        Returns:
            torch.Tensor: Output tensor with size=[num_seg, in_features]
        """

        # performing batch_normalization and activation
        x_bn = self.bn_relu(x)  # size=[N, in_features]

        # performing segment operation
        # 按 ids（batch 向量）分段聚合；dim_size 指定输出行数即图数
        x_pooled = self.pooling_op(
            x_bn,
            dim=0,
            index=ids,
            dim_size=num_seg
        )  # size=[num_seg, in_features]

        return x_pooled


class AvgPooling(_Pooling):
    # 中文：图级平均池化（average pooling）层，继承 _Pooling；输入 [num_nodes, in_features]，输出 [num_graphs, in_features]。
    """Average pooling layer for graph"""

    def __init__(
        self,
        in_features: int,
        activation: str = 'elu'
    ):
        # 只把聚合算子指定为分段求均值 scatter_mean，其余逻辑沿用 _Pooling
        """ Performing graph level average pooling (with bn_relu)
        Args:
            in_features (int):
                The number of input features
            activation (str):
                The type of activation function to use, default to elu
        """
        super(AvgPooling, self).__init__(
            in_features,
            activation=activation,
            pooling_op=torch_scatter.scatter_mean
        )


class SumPooling(_Pooling):
    # 中文：图级求和池化（sum pooling）层，继承 _Pooling；与 AvgPooling 只差聚合算子为 scatter_add。
    """Sum pooling layer for graph"""

    def __init__(
        self,
        in_features: int,
        activation: str = 'elu'
    ):
        # 聚合算子指定为分段求和 scatter_add，输出图级表示随图内节点数线性增长
        """ Performing graph level sum pooling (with bn_relu)
        Args:
            in_features (int):
                The number of input features
            activation (str):
                The type of activation function to use, default to elu
        """
        super(SumPooling, self).__init__(
            in_features,
            activation=activation,
            pooling_op=torch_scatter.scatter_add
        )


class BNReLULinearBlock(nn.Module):
    """多层感知机（MLP）块：继承 nn.Module，由 num_layers 个 BNReLULinear 串联
    （1 个输入层 + num_layers-2 个隐藏层 + 1 个输出层，中间维度均为 hidden_size，要求 num_layers >= 2）。
    输入 [batch, in_features]，输出 [batch, out_features]。
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_size: int,
        activation: str = 'elu',
        # out_act: str = 'sigmoid',
        **kwargs
    ):
        """Args:
            in_features / out_features (int): 输入、输出维度
            num_layers (int): BN+Linear+激活 块的总层数
            hidden_size (int): 中间层宽度
            activation (str): 激活函数名
        """
        super().__init__()
        
        # 输入层：in_features -> hidden_size
        input_brl = BNReLULinear(
            in_features,
            hidden_size,
            activation
        )

        # 中间层：hidden_size -> hidden_size，共 num_layers-2 个
        btn_brl = [
            BNReLULinear(
                hidden_size,
                hidden_size,
                activation
            )
            for _ in range(num_layers - 2)
        ]

        # 输出层：hidden_size -> out_features
        output_brl = BNReLULinear(
            hidden_size,
            out_features,
            activation,
        )
        # self.out_act = get_activation(out_act, **kwargs)
            
        self.brl_block = nn.Sequential(
            input_brl,
            *btn_brl,
            output_brl,
            # self.out_act
        )

    def forward(self, X):
        """前向：整段 MLP 一次通过，不在末层施加额外输出激活。"""
        return self.brl_block(X)


class MultiTaskMultiClassBlock(nn.Module):
    """多任务多分类头（multi-task multi-class）：继承 nn.Module，编码器 + 每个任务一条独立分类 MLP。
    取编码器输出的第 0 个 token（[CLS] 位）作为图/序列级表示，再分发给 nums_classes 个 BNReLULinearBlock。
    输入 X 为传给 encoder 的位置参数元组，encoder 返回 (last_hidden_state [...], ...)；
    输出为长度为 len(nums_classes) 的列表，第 k 项形状 [batch, nums_classes[k]]。
    """
    _NAME = 'rxn_trans'

    def __init__(
        self,
        encoder: nn.Module = None,
        nums_classes: t.List[int] = [3, 3],
        hidden_size: int = 128,
        num_hidden_layers: int = 10,
        activation: str = 'elu',
        out_act: str = 'softmax',
        **kwargs,
    ):
        """Args:
            encoder (nn.Module): 特征编码器，其输出维度需为 256（分类器输入维度硬编码为 256）
            nums_classes (t.List[int]): 每个任务的类别数，决定分类器个数与输出宽度
            hidden_size (int): 各任务 MLP 的中间宽度
            num_hidden_layers (int): 各任务 MLP 的层数
            activation (str): MLP 内部激活函数名
            out_act (str 或 list): 评估阶段的输出激活（如 softmax），训练阶段不施加
        """
        super().__init__()
        # 保存构造参数，便于序列化/复现模型
        self.init_args = {
            'encoder': encoder,
            'nums_classes': nums_classes,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'activation': activation,
            'out_act': out_act,
            **kwargs
        }
        if isinstance(out_act, str):
            # 单个激活名按任务数复制
            self.out_acts = [out_act] * len(nums_classes)
        else:
            self.out_acts = out_act
        self.out_act_funcs = nn.ModuleList(
            [get_activation(act, **kwargs) for act in self.out_acts]
        )

        self.encoder = encoder
        # 默认冻结编码器（由 freeze_encoder setter 控制 requires_grad）
        self._freeze_encoder = True 
        # 每个任务一条独立 MLP：256 -> hidden_size(× num_hidden_layers) -> num_class
        self.classifiers = nn.ModuleList([
            BNReLULinearBlock(
                256,
                num_class,
                num_hidden_layers,
                hidden_size,
                activation,
                # out_action,
                **kwargs
            )
            for num_class in nums_classes
        ])
    
    @property
    def freeze_encoder(self):
        """当前是否冻结编码器参数。"""
        return self._freeze_encoder
    
    @freeze_encoder.setter
    def freeze_encoder(self, freeze: bool):
        """设置冻结开关，同时同步更新编码器参数的 requires_grad。"""
        self._freeze_encoder = freeze
        self.change_encoder_grad(not freeze)
    
    def change_encoder_grad(self, requires_grad: bool):
        """遍历编码器参数设置 requires_grad（True 参与训练 / False 冻结）。"""
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad
 
    def forward(self, X):
        """前向：编码器取 [CLS] 表示 -> 各任务分类器。
        Args:
            X (tuple): 传给 encoder 的位置参数
        Returns:
            t.List[torch.Tensor]：每个任务的预测 [batch, num_class]
        """
        # encoder(*X)[0] 为 [batch, seq_len, 256]，[:, 0, :] 取首个 token 作为整体表示
        embeddings = self.encoder(*X)[0][:, 0, :]
        if self.training:
            # 训练时返回未归一化的 logits（配合带 label smoothing 的损失更稳定）
            outputs = [
                classifier(embeddings)
                for classifier in self.classifiers
            ]
        else:
            # 评估时套上输出激活（如 softmax）给出概率
            outputs = [
                act(classifier(embeddings))
                for classifier, act in zip(self.classifiers, self.out_act_funcs)
            ]
        
        return outputs


class MuMcHardBlock(nn.Module):
    """多任务多分类头（硬参数共享, hard parameter sharing 版本）：继承 nn.Module。
    与 MultiTaskMultiClassBlock 的区别：所有任务先共用一条深层 MLP 抽取共享表示，
    再由每个任务各自一个浅层 BNReLULinear 输出头分类。
    输入 X 为 encoder 的位置参数元组，输出为 len(nums_classes) 个 [batch, nums_classes[k]] 张量。
    """
    _NAME = 'rxn_trans_hard'

    def __init__(
        self,
        encoder: nn.Module = None,
        nums_classes: t.List[int] = [3, 3],
        hidden_size: int = 128,
        num_hidden_layers: int = 10,
        activation: str = 'elu',
        out_act: str = 'softmax',
        **kwargs,
    ):
        """Args:
            encoder (nn.Module): 特征编码器，输出维度需为 256（共享塔输入维度硬编码为 256）
            nums_classes (t.List[int]): 各任务类别数
            hidden_size (int): 共享塔宽度，同时作为输出头的输入维度
            num_hidden_layers (int): 共享塔层数
            activation (str): MLP 内部激活函数名
            out_act (str 或 list): 评估阶段的输出激活
        """
        super().__init__()
        self.init_args = {
            'encoder': encoder,
            'nums_classes': nums_classes,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'activation': activation,
            'out_act': out_act,
            **kwargs
        }
        if isinstance(out_act, str):
            self.out_acts = [out_act] * len(nums_classes)
        else:
            self.out_acts = out_act
        self.out_act_funcs = nn.ModuleList(
            [get_activation(act, **kwargs) for act in self.out_acts]
        )

        self.encoder = encoder
        self._freeze_encoder = True 
        # 共享塔：256 -> hidden_size(× num_hidden_layers)
        self.classifier = BNReLULinearBlock(
            256,
            hidden_size,
            num_hidden_layers,
            hidden_size,
            activation,
            # out_action,
            **kwargs
        )
        
        # 任务输出头：每个任务一个 hidden_size -> num_classes 的 BNReLULinear
        self.out_layers = nn.ModuleList([
            BNReLULinear(
                hidden_size,
                num_classes
            )
            for num_classes in nums_classes
        ])

    @property
    def freeze_encoder(self):
        """当前是否冻结编码器参数。"""
        return self._freeze_encoder
    
    @freeze_encoder.setter
    def freeze_encoder(self, freeze: bool):
        """设置冻结开关，同时同步更新编码器参数的 requires_grad。"""
        self._freeze_encoder = freeze
        self.change_encoder_grad(not freeze)
    
    def change_encoder_grad(self, requires_grad: bool):
        """遍历编码器参数设置 requires_grad。"""
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad
 
    def forward(self, X):
        """前向：[CLS] 表示 -> 共享塔 -> 各任务输出头。
        Args:
            X (tuple): 传给 encoder 的位置参数
        Returns:
            t.List[torch.Tensor]：每个任务的预测 [batch, num_class]
        """
        embeddings = self.encoder(*X)[0][:, 0, :]
        # 先过所有任务共享的深层 MLP，得到任务无关的中间表示
        embeddings = self.classifier(embeddings)

        if self.training:
            # 训练时输出各任务 logits
            outputs = [
                out_layer(embeddings)
                for out_layer in self.out_layers
            ]
        else:
            # 评估时再套输出激活
            outputs = [
                act(out_layer(embeddings))
                for out_layer, act in zip(self.out_layers, self.out_act_funcs)
            ]
        
        return outputs
