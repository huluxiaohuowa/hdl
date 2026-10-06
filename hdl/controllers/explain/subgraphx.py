# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/controllers/explain/subgraphx.py
# 说明：模型归因解释（Shapley / 子图）
# 模块功能：子图归因（SubgraphX）：用蒙特卡洛树搜索（MCTS）在分子图/图上搜索使模型预测改变最大的关键子图，奖励由 shapley.py 中的夏普利值（Shapley value）函数提供
import os
import math
import copy
from typing import List, Tuple, Dict
from typing import Callable, Union, Optional
from collections import Counter
from functools import partial
from textwrap import wrap

import torch
import numpy as np
import networkx as nx
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import remove_self_loops

from .shapley import GnnNetsGC2valueFunc, GnnNetsNC2valueFunc, \
    gnn_score, mc_shapley, l_shapley, mc_l_shapley, NC_mc_l_shapley, sparsity


def find_closest_node_result(results, max_nodes):
    # 从搜索得到的所有节点里，挑出子图规模不超过 max_nodes 且奖励 P 最高的那个作为解释结果
    """ return the highest reward tree_node with its subgraph is smaller than max_nodes """
    results = sorted(results, key=lambda x: len(x.coalition))

    result_node = results[0]
    for result_idx in range(len(results)):
        x = results[result_idx]
        if len(x.coalition) <= max_nodes and x.P > result_node.P:
            result_node = x
    return result_node


def reward_func(reward_method, value_func, node_idx=None,
                local_radius=4, sample_num=100,
                subgraph_building_method='zero_filling'):
    """按名字把归因打分函数包装成树节点奖励函数（reward）：gnn_score 直接取子图得分，mc_shapley/l_shapley/mc_l_shapley/nc_mc_l_shapley 取相应夏普利值（Shapley value）近似；其余名字抛 NotImplementedError"""
    if reward_method.lower() == 'gnn_score':
        return partial(gnn_score,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method)

    elif reward_method.lower() == 'mc_shapley':
        return partial(mc_shapley,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    elif reward_method.lower() == 'l_shapley':
        return partial(l_shapley,
                       local_radius=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method)

    elif reward_method.lower() == 'mc_l_shapley':
        return partial(mc_l_shapley,
                       local_radius=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    elif reward_method.lower() == 'nc_mc_l_shapley':
        assert node_idx is not None, " Wrong node idx input "
        return partial(NC_mc_l_shapley,
                       node_idx=node_idx,
                       local_radius=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    else:
        raise NotImplementedError


def k_hop_subgraph_with_default_whole_graph(
        edge_index, node_idx=None, num_hops=3, relabel_nodes=False,
        num_nodes=None, flow='source_to_target'):
    # 与 k-hop 子图（k_hop_subgraph）的差别：node_idx 为 None 时改为从节点 0 出发反复并入邻居，直到集合不再增长，即取该连通分量
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    # flow 决定 row/col 的取法：source_to_target 时 col 为源节点、row 为目标节点
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    inv = None

    # node_idx 为空时走"全图"分支：从节点 0 开始逐跳并入邻居，集合稳定后即整个连通分量
    if node_idx is None:
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while 1:
            # 用当前集合标记节点，再沿边收集邻居，直到不再新增节点
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break
    else:
        # 把中心节点统一成张量形式（标量/列表/张量都支持）
        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
        elif isinstance(node_idx, torch.Tensor) and len(node_idx.shape) == 0:
            node_idx = torch.tensor([node_idx])
        else:
            node_idx = node_idx.to(row.device)

        # 逐跳扩展 num_hops 层：每轮只用上一步新增的节点做邻居查询
        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        # unique 的逆映射 inv 记录中心节点在新编号中的位置
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

    # 由最终节点集合导出保留的边：两端都在集合内才保留
    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        # 重编号：子图内节点映射为从 0 开始的连续索引
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask  # subset: key new node idx; value original node idx


def compute_scores(score_func, children):
    """为一批子节点补算奖励：已算过（P 不为 0）的直接复用缓存值，否则用 score_func 对 (coalition, data) 求分"""
    results = []
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition, child.data)
        else:
            score = child.P
        results.append(score)
    return results


class PlotUtils(object):
    """可视化（visualization）工具类：按数据集名把解释出的节点子图画出来，分子/句子/B-A 形状等走各自的绘制分支"""
    def __init__(self, dataset_name, is_show=True):
        """入参：dataset_name 数据集名（决定绘图分支与节点标注来源）；is_show 是否调用 plt.show() 弹窗显示"""
        self.dataset_name = dataset_name
        self.is_show = is_show

    def plot(self, graph, nodelist, figname, title_sentence=None, **kwargs):
        # 按数据集名分发到具体绘图函数，未知名字抛 NotImplementedError
        """ plot function for different dataset """
        if self.dataset_name.lower() in ['ba_2motifs', 'ba_lrp']:
            self.plot_ba2motifs(graph, nodelist, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['mutag'] + list(MoleculeNet.names.keys()):
            x = kwargs.get('x')
            self.plot_molecule(graph, nodelist, x, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle']:
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            self.plot_bashapes(graph, nodelist, y, node_idx, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['graph_sst2', 'graph_sst5', 'twitter']:
            words = kwargs.get('words')
            self.plot_sentence(graph, nodelist, words=words, title_sentence=title_sentence, figname=figname)
        else:
            raise NotImplementedError

    def plot_subgraph(self,
                      graph,
                      nodelist,
                      colors: Union[None, str, List[str]] = '#FFA500',
                      labels=None,
                      edge_color='gray',
                      edgelist=None,
                      subgraph_edge_color='black',
                      title_sentence=None,
                      figname=None):
        """通用子图绘制：按 kamada-kawai 布局画全部节点与灰色边，再把 nodelist（解释出的子图节点）内部边用更粗的黑色边高亮；figname 非空则存图并在显示后关闭画布"""

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=6,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if figname is not None:
            plt.savefig(figname)
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_subgraph_with_nodes(self,
                                 graph,
                                 nodelist,
                                 node_idx,
                                 colors='#FFA500',
                                 labels=None,
                                 edge_color='gray',
                                 edgelist=None,
                                 subgraph_edge_color='black',
                                 title_sentence=None,
                                 figname=None):
        """子图绘制并额外突出目标节点：node_idx 对应的节点用更大的尺寸与自身颜色单独描一遍，用于节点分类任务中标出被解释的原子"""
        node_idx = int(node_idx)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]

        pos = nx.kamada_kawai_layout(graph)  # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)
        if isinstance(colors, list):
            list_indices = int(np.where(np.array(graph.nodes()) == node_idx)[0])
            node_idx_color = colors[list_indices]
        else:
            node_idx_color = colors

        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[node_idx],
                               node_color=node_idx_color,
                               node_size=600)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_sentence(self, graph, nodelist, words, edgelist=None, title_sentence=None, figname=None):
        """句子图绘制：节点标签用 words 里的单词，解释出的 nodelist 节点及其内部边用黄色高亮，标题为拼接后的句子"""
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(graph, pos_coalition,
                                   nodelist=nodelist,
                                   node_color='yellow',
                                   node_shape='o',
                                   node_size=500)
            if edgelist is None:
                edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                            if n_frm in nodelist and n_to in nodelist]
                nx.draw_networkx_edges(graph, pos=pos_coalition, edgelist=edgelist, width=5, edge_color='yellow')

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color='grey')
        nx.draw_networkx_labels(graph, pos, words_dict)

        plt.axis('off')
        plt.title('\n'.join(wrap(' '.join(words), width=50)))
        if title_sentence is not None:
            string = '\n'.join(wrap(' '.join(words), width=50))
            string += '\n'.join(wrap(title_sentence, width=60))
            plt.title(string)
        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_ba2motifs(self,
                       graph,
                       nodelist,
                       edgelist=None,
                       title_sentence=None,
                       figname=None):
        """B-A-2-motifs 数据集绘图入口：直接复用 plot_subgraph（无节点标签）"""
        return self.plot_subgraph(graph, nodelist,
                                  edgelist=edgelist,
                                  title_sentence=title_sentence,
                                  figname=figname)

    def plot_molecule(self,
                      graph,
                      nodelist,
                      x,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        # 分子图绘制：把节点特征还原为元素符号与配色，再交给 plot_subgraph 高亮解释子图
        # collect the text information and node color
        if self.dataset_name == 'mutag':
            # MUTAG 为独热（one-hot） atom 特征，按列下标查自定义原子字典
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        elif self.dataset_name in MoleculeNet.names.keys():
            # MoleculeNet 数据集第 0 列是原子序数，用 RDKit 周期表转成元素符号作标签
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            node_labels = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                           for k, v in element_idxs.items()}
            node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
            colors = [node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()]
        else:
            raise NotImplementedError

        self.plot_subgraph(graph, nodelist,
                           colors=colors,
                           labels=node_labels,
                           edgelist=edgelist,
                           edge_color='gray',
                           subgraph_edge_color='black',
                           title_sentence=title_sentence,
                           figname=figname)

    def plot_bashapes(self,
                      graph,
                      nodelist,
                      y,
                      node_idx,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        """B-A shapes 类数据集绘图：用节点属性 y 取颜色，按 plot_subgraph_with_nodes 标出目标节点"""
        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph,
                                      nodelist,
                                      node_idx,
                                      colors,
                                      edgelist=edgelist,
                                      title_sentence=title_sentence,
                                      figname=figname,
                                      subgraph_edge_color='black')


class MCTSNode(object):
    """蒙特卡洛树搜索（MCTS）的树节点：保存一个候选子图（coalition 节点集合）及其统计量 W/N/P，用于选择与回传"""
    def __init__(self, coalition: list = None, data: Data = None, ori_graph: nx.Graph = None,
                 c_puct: float = 10.0, W: float = 0, N: int = 0, P: float = 0,
                 load_dict: Optional[Dict] = None, device='cpu'):
        """入参：coalition 当前子图保留的节点编号；data 该子图对应的批次数据；ori_graph 原图（networkx，用于绘图）；c_puct 探索权重；W 累计价值；N 访问次数；P 先验/奖励分数；load_dict 非空时从中恢复节点状态"""
        self.data = data
        self.coalition = coalition
        self.ori_graph = ori_graph
        self.device = device
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)
        if load_dict is not None:
            # 从序列化字典恢复节点，便于复用之前的搜索记录
            self.load_info(load_dict)

    def Q(self):
        """平均价值 Q = W/N，未被访问过（N=0）时返回 0"""
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        """探索项 U = c_puct * P * sqrt(n) / (1 + N)，n 为兄弟节点的总访问数"""
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)

    @property
    def info(self):
        """可序列化属性字典：data 先搬回 CPU，便于 torch.save 存盘"""
        info_dict = {
            'data': self.data.to('cpu'),
            'coalition': self.coalition,
            'ori_graph': self.ori_graph,
            'W': self.W,
            'N': self.N,
            'P': self.P
        }
        return info_dict

    def load_info(self, info_dict):
        """从 info 字典恢复统计量与子图数据（data 搬回 self.device），children 置空后返回自身"""
        self.W = info_dict['W']
        self.N = info_dict['N']
        self.P = info_dict['P']
        self.coalition = info_dict['coalition']
        self.ori_graph = info_dict['ori_graph']
        self.data = info_dict['data'].to(self.device)
        self.children = []
        return self


class MCTS(object):
    r"""
    蒙特卡洛树搜索（MCTS）：从全图节点集合出发，每次删掉一个原子并保留最大连通分量来扩展子节点，用 score_func（夏普利值/子图得分）作奖励，按 Q+U 选择下探，最终返回按奖励排序的候选子图列表。
    Monte Carlo Tree Search Method
    Args:
        X (:obj:`torch.Tensor`): Input node features
        edge_index (:obj:`torch.Tensor`): The edge indices.
        num_hops (:obj:`int`): The number of hops :math:`k`.
        n_rollout (:obj:`int`): The number of sequence to build the monte carlo tree.
        min_atoms (:obj:`int`): The number of atoms for the subgraph in the monte carlo tree leaf node.
        c_puct (:obj:`float`): The hyper-parameter to encourage exploration while searching.
        expand_atoms (:obj:`int`): The number of children to expand.
        high2low (:obj:`bool`): Whether to expand children tree node from high degree nodes to low degree nodes.
        node_idx (:obj:`int`): The target node index to extract the neighborhood.
        score_func (:obj:`Callable`): The reward function for tree node, such as mc_shapely and mc_l_shapely.
    """
    def __init__(self, X: torch.Tensor, edge_index: torch.Tensor, num_hops: int,
                 n_rollout: int = 10, min_atoms: int = 3, c_puct: float = 10.0,
                 expand_atoms: int = 14, high2low: bool = False,
                 node_idx: int = None, score_func: Callable = None, device='cpu'):

        # X/edge_index 为单张图的节点特征与边；data 打包成 batch 供特征函数（value function）使用
        self.X = X
        self.edge_index = edge_index
        self.device = device
        self.num_hops = num_hops
        self.data = Data(x=self.X, edge_index=self.edge_index)
        # 建 networkx 图时去掉自环，避免自环影响连通分量与邻居扩展
        graph_data = Data(x=self.X, edge_index=remove_self_loops(self.edge_index)[0])
        self.graph = to_networkx(graph_data, to_undirected=True)
        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.score_func = score_func
        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low
        self.new_node_idx = None

        # extract the sub-graph and change the node indices.
        # 节点分类任务：先取 node_idx 的 num_hops 邻域并重编号，new_node_idx 是目标节点在新图中的编号
        if node_idx is not None:
            self.ori_node_idx = node_idx
            self.ori_graph = copy.copy(self.graph)
            x, edge_index, subset, edge_mask, kwargs = \
                self.__subgraph__(node_idx, self.X, self.edge_index, self.num_hops)
            self.data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
            self.graph = self.ori_graph.subgraph(subset.tolist())
            mapping = {int(v): k for k, v in enumerate(subset)}
            self.graph = nx.relabel_nodes(self.graph, mapping)
            self.new_node_idx = torch.where(subset == self.ori_node_idx)[0].item()
            self.num_nodes = self.graph.number_of_nodes()
            self.subset = subset

        # 根节点为全部节点的集合，state_map 用节点列表的字符串形式做去重索引
        self.root_coalition = sorted([node for node in range(self.num_nodes)])
        self.MCTSNodeClass = partial(MCTSNode, data=self.data, ori_graph=self.graph,
                                     c_puct=self.c_puct, device=self.device)
        self.root = self.MCTSNodeClass(self.root_coalition)
        self.state_map = {str(self.root.coalition): self.root}

    def set_score_func(self, score_func):
        """替换树节点的奖励函数（score_func），SubgraphX 在节点分类分支里用它延迟绑定 node_idx"""
        self.score_func = score_func

    @staticmethod
    def __subgraph__(node_idx, x, edge_index, num_hops, **kwargs):
        """取中心节点 num_hops 跳邻域的子图并重编号：返回子图特征 x、新 edge_index、邻域节点集合 subset、边掩码与按尺寸过滤后的 kwargs"""
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            edge_index, node_idx, num_hops, relabel_nodes=True, num_nodes=num_nodes)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, subset, edge_mask, kwargs

    def mcts_rollout(self, tree_node):
        """一次 rollout：节点子图已小于 min_atoms 时直接返回其奖励；首次访问则扩展子节点（逐个删原子），再按 Q+U 选最优子节点递归下探，并把返回值回传到该子节点的 W/N"""
        cur_graph_coalition = tree_node.coalition
        # 叶子条件：子图原子数已少于阈值，不再继续剪枝
        if len(cur_graph_coalition) <= self.min_atoms:
            return tree_node.P

        # Expand if this node has never been visited
        # 扩展：按节点度数排序（high2low=True 时高度数在前），最多取 expand_atoms 个待剪原子
        if len(tree_node.children) == 0:
            node_degree_list = list(self.graph.subgraph(cur_graph_coalition).degree)
            node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=self.high2low)
            all_nodes = [x[0] for x in node_degree_list]

            # 节点分类时不能剪掉目标节点 new_node_idx
            if self.new_node_idx:
                expand_nodes = [node for node in all_nodes if node != self.new_node_idx]
            else:
                expand_nodes = all_nodes

            if len(all_nodes) > self.expand_atoms:
                expand_nodes = expand_nodes[:self.expand_atoms]

            for each_node in expand_nodes:
                # for each node, pruning it and get the remaining sub-graph
                # here we check the resulting sub-graphs and only keep the largest one
                # 剪掉该原子后取连通分量：图分类保留节点最多的那一块，节点分类保留含目标节点的那一块
                subgraph_coalition = [node for node in all_nodes if node != each_node]

                subgraphs = [self.graph.subgraph(c)
                             for c in nx.connected_components(self.graph.subgraph(subgraph_coalition))]

                if self.new_node_idx:
                    for sub in subgraphs:
                        if self.new_node_idx in list(sub.nodes()):
                            main_sub = sub
                else:
                    main_sub = subgraphs[0]

                    for sub in subgraphs:
                        if sub.number_of_nodes() > main_sub.number_of_nodes():
                            main_sub = sub

                new_graph_coalition = sorted(list(main_sub.nodes()))

                # check the state map and merge the same sub-graph
                # 同一子图（节点集合相同）复用已在 state_map 中的节点，避免重复搜索与重复计分
                find_same = False
                for old_graph_node in self.state_map.values():
                    if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                        new_node = old_graph_node
                        find_same = True

                if not find_same:
                    new_node = self.MCTSNodeClass(new_graph_coalition)
                    self.state_map[str(new_graph_coalition)] = new_node

                find_same_child = False
                for cur_child in tree_node.children:
                    if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                        find_same_child = True

                if not find_same_child:
                    tree_node.children.append(new_node)

            # 为新扩展的子节点批量算奖励（已有 P 的复用缓存），写入 child.P 作为先验
            scores = compute_scores(self.score_func, tree_node.children)
            for child, score in zip(tree_node.children, scores):
                child.P = score

        # 选择（selection）：在子节点中取 Q + U 最大者，即平均价值与探索项之和（PUCT 风格）
        sum_count = sum([c.N for c in tree_node.children])
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        v = self.mcts_rollout(selected_node)
        # 回传（backup）：把子树返回值累加到被选节点的 W 并增加访问次数 N
        selected_node.W += v
        selected_node.N += 1
        return v

    def mcts(self, verbose=True):
        """执行 n_rollout 次 rollout 遍历搜索树，返回 state_map 中所有被探索过的子图节点，按奖励 P 从高到低排序作为解释候选列表"""
        if verbose:
            print(f"The nodes in graph is {self.graph.number_of_nodes()}")
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root)
            if verbose:
                print(f"At the {rollout_idx} rollout, {len(self.state_map)} states that have been explored.")

        explanations = [node for _, node in self.state_map.items()]
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        return explanations


class SubgraphX(object):
    r"""
    子图归因器（explainer）：对输入图逐个类别调用 MCTS 搜索关键子图，输出解释节点列表与"保留子图 / 去掉子图 / 原图 / 稀疏度"四项打分，可选存盘与绘图。
    The implementation of paper
    `On Explainability of Graph Neural Networks via Subgraph Explorations <https://arxiv.org/abs/2102.05152>`_.
    Args:
        model (:obj:`torch.nn.Module`): The target model prepared to explain
        num_classes(:obj:`int`): Number of classes for the datasets
        num_hops(:obj:`int`, :obj:`None`): The number of hops to extract neighborhood of target node
          (default: :obj:`None`)
        explain_graph(:obj:`bool`): Whether to explain graph classification model (default: :obj:`True`)
        rollout(:obj:`int`): Number of iteration to get the prediction
        min_atoms(:obj:`int`): Number of atoms of the leaf node in search tree
        c_puct(:obj:`float`): The hyperparameter which encourages the exploration
        expand_atoms(:obj:`int`): The number of atoms to expand
          when extend the child nodes in the search tree
        high2low(:obj:`bool`): Whether to expand children nodes from high degree to low degree when
          extend the child nodes in the search tree (default: :obj:`False`)
        local_radius(:obj:`int`): Number of local radius to calculate :obj:`l_shapley`, :obj:`mc_l_shapley`
        sample_num(:obj:`int`): Sampling time of monte carlo sampling approximation for
          :obj:`mc_shapley`, :obj:`mc_l_shapley` (default: :obj:`mc_l_shapley`)
        reward_method(:obj:`str`): The command string to select the
        subgraph_building_method(:obj:`str`): The command string for different subgraph building method,
          such as :obj:`zero_filling`, :obj:`split` (default: :obj:`zero_filling`)
        save_dir(:obj:`str`, :obj:`None`): Root directory to save the explanation results (default: :obj:`None`)
        filename(:obj:`str`): The filename of results
        vis(:obj:`bool`): Whether to show the visualization (default: :obj:`True`)
    Example:
        >>> # For graph classification task
        >>> subgraphx = SubgraphX(model=model, num_classes=2)
        >>> _, explanation_results, related_preds = subgraphx(x, edge_index)
    """
    def __init__(self, model, num_classes: int, device, num_hops: Optional[int] = None, verbose: bool = False,
                 explain_graph: bool = True, rollout: int = 20, min_atoms: int = 5, c_puct: float = 10.0,
                 expand_atoms=14, high2low=False, local_radius=4, sample_num=100, reward_method='mc_l_shapley',
                 subgraph_building_method='zero_filling', save_dir: Optional[str] = None,
                 filename: str = 'example', vis: bool = True):
        """入参：model 待解释的图神经网络；num_classes 类别数（决定对多少类各搜一次）；device 计算设备；num_hops 邻域跳数（None 时按模型里 MessagePassing 层数推断）；explain_graph True 为图分类、False 为节点分类；rollout/min_atoms/c_puct/expand_atoms/high2low 为 MCTS 超参；local_radius/sample_num 为夏普利值（Shapley value）奖励的超参；reward_method 与 subgraph_building_method 选择奖励算法与子图构造方式；save_dir/filename 控制结果存盘；vis 是否绘图"""

        self.model = model
        # 解释阶段把模型切到评估模式（eval）并搬到目标设备
        self.model.eval()
        self.device = device
        self.model.to(self.device)
        self.num_classes = num_classes
        self.num_hops = self.update_num_hops(num_hops)
        self.explain_graph = explain_graph
        self.verbose = verbose

        # mcts hyper-parameters
        self.rollout = rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low

        # reward function hyper-parameters
        self.local_radius = local_radius
        self.sample_num = sample_num
        self.reward_method = reward_method
        self.subgraph_building_method = subgraph_building_method

        # saving and visualization
        self.vis = vis
        self.save_dir = save_dir
        self.filename = filename
        self.save = True if self.save_dir is not None else False

    def update_num_hops(self, num_hops):
        """num_hops 未指定时，统计模型中 MessagePassing 消息传递层的层数作为感受野跳数"""
        if num_hops is not None:
            return num_hops

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def get_reward_func(self, value_func, node_idx=None):
        """构造树节点的奖励函数：图分类时忽略 node_idx，节点分类时必须提供；具体算法由 self.reward_method 决定"""
        if self.explain_graph:
            node_idx = None
        else:
            assert node_idx is not None
        return reward_func(reward_method=self.reward_method,
                           value_func=value_func,
                           node_idx=node_idx,
                           local_radius=self.local_radius,
                           sample_num=self.sample_num,
                           subgraph_building_method=self.subgraph_building_method)

    def get_mcts_class(self, x, edge_index, node_idx: int = None, score_func: Callable = None):
        """按已保存的 MCTS 超参实例化搜索器（MCTS）：图分类时 node_idx 置 None（对整图搜索），节点分类时以其邻域子图搜索"""
        if self.explain_graph:
            node_idx = None
        else:
            assert node_idx is not None
        return MCTS(x, edge_index,
                    node_idx=node_idx,
                    device=self.device,
                    score_func=score_func,
                    num_hops=self.num_hops,
                    n_rollout=self.rollout,
                    min_atoms=self.min_atoms,
                    c_puct=self.c_puct,
                    expand_atoms=self.expand_atoms,
                    high2low=self.high2low)

    def visualization(self, results: list,
                      max_nodes: int, plot_utils: PlotUtils, words: Optional[list] = None,
                      y: Optional[Tensor] = None, title_sentence: Optional[str] = None,
                      vis_name: Optional[str] = None):
        """绘图：先用 find_closest_node_result 在 results 里取规模不超过 max_nodes 的最优子图，再按图分类/节点分类分支调用 PlotUtils.plot；设置了 save_dir 时图名默认为 <filename>.png"""
        if self.save:
            if vis_name is None:
                vis_name = f"{self.filename}.png"
        else:
            vis_name = None
        tree_node_x = find_closest_node_result(results, max_nodes=max_nodes)
        if self.explain_graph:
            if words is not None:
                plot_utils.plot(tree_node_x.ori_graph,
                                tree_node_x.coalition,
                                words=words,
                                title_sentence=title_sentence,
                                figname=vis_name)
            else:
                plot_utils.plot(tree_node_x.ori_graph,
                                tree_node_x.coalition,
                                x=tree_node_x.data.x,
                                title_sentence=title_sentence,
                                figname=vis_name)
        else:
            subset = self.mcts_state_map.subset
            subgraph_y = y[subset].to('cpu')
            subgraph_y = torch.tensor([subgraph_y[node].item()
                                       for node in tree_node_x.ori_graph.nodes()])
            plot_utils.plot(tree_node_x.ori_graph,
                            tree_node_x.coalition,
                            node_idx=self.mcts_state_map.new_node_idx,
                            title_sentence=title_sentence,
                            y=subgraph_y,
                            figname=vis_name)

    def read_from_MCTSInfo_list(self, MCTSInfo_list):
        """把存盘的节点字典列表还原成 MCTSNode 对象：支持单层（单个标签）与两层（按标签分组）两种结构"""
        if isinstance(MCTSInfo_list[0], dict):
            ret_list = [MCTSNode(device=self.device).load_info(node_info) for node_info in MCTSInfo_list]
        elif isinstance(MCTSInfo_list[0][0], dict):
            ret_list = []
            for single_label_MCTSInfo_list in MCTSInfo_list:
                single_label_ret_list = [MCTSNode(device=self.device).load_info(node_info) for node_info in single_label_MCTSInfo_list]
                ret_list.append(single_label_ret_list)
        return ret_list

    def write_from_MCTSNode_list(self, MCTSNode_list):
        """把 MCTSNode 列表转成可 torch.save 的字典列表（调用 node.info），结构与 read_from_MCTSInfo_list 对应"""
        if isinstance(MCTSNode_list[0], MCTSNode):
            ret_list = [node.info for node in MCTSNode_list]
        elif isinstance(MCTSNode_list[0][0], MCTSNode):
            ret_list = []
            for single_label_MCTSNode_list in MCTSNode_list:
                single_label_ret_list = [node.info for node in single_label_MCTSNode_list]
                ret_list.append(single_label_ret_list)
        return ret_list

    def explain(self, x: Tensor, edge_index: Tensor, label: int,
                max_nodes: int = 5,
                node_idx: Optional[int] = None,
                saved_MCTSInfo_list: Optional[List[List]] = None):
        """对单个目标类别 label 做一次子图搜索归因：
        x 为节点特征、edge_index 为边；saved_MCTSInfo_list 非空时直接复用已存盘的搜索节点而不重跑 MCTS。
        返回 (results 各候选子图节点的字典列表, related_pred 字典)；
        related_pred 含 masked（保留关键子图时的目标类得分）、maskout（去掉关键子图后的得分）、
        origin（原图该节点/整图的目标类概率）、sparsity（解释子图的稀疏度）"""

        # 先取原图（或原节点）的类别概率，作为 origin 参照值（此处前向未包 torch.no_grad）
        probs = self.model(x, edge_index).squeeze().softmax(dim=-1)
        if self.explain_graph:
            if saved_MCTSInfo_list:
                results = self.read_from_MCTSInfo_list(saved_MCTSInfo_list)

            if not saved_MCTSInfo_list:
                # 图分类：以目标类概率为特征函数（value function），用夏普利值（Shapley value）奖励驱动 MCTS
                value_func = GnnNetsGC2valueFunc(self.model, target_class=label)
                payoff_func = self.get_reward_func(value_func)
                self.mcts_state_map = self.get_mcts_class(x, edge_index, score_func=payoff_func)
                results = self.mcts_state_map.mcts(verbose=self.verbose)

            # l sharply score
            # 重新构造特征函数，供后面 masked/maskout 两种掩码打分使用
            value_func = GnnNetsGC2valueFunc(self.model, target_class=label)
            tree_node_x = find_closest_node_result(results, max_nodes=max_nodes)

        else:
            if saved_MCTSInfo_list:
                results = self.read_from_MCTSInfo_list(saved_MCTSInfo_list)

            self.mcts_state_map = self.get_mcts_class(x, edge_index, node_idx=node_idx)
            self.new_node_idx = self.mcts_state_map.new_node_idx
            # mcts will extract the subgraph and relabel the nodes
            # 节点分类：MCTS 已把邻域子图重编号，故用新编号的 new_node_idx 取节点概率
            value_func = GnnNetsNC2valueFunc(self.model,
                                             node_idx=self.mcts_state_map.new_node_idx,
                                             target_class=label)

            if not saved_MCTSInfo_list:
                payoff_func = self.get_reward_func(value_func,
                                                   node_idx=self.mcts_state_map.new_node_idx)
                self.mcts_state_map.set_score_func(payoff_func)
                results = self.mcts_state_map.mcts(verbose=self.verbose)

            tree_node_x = find_closest_node_result(results, max_nodes=max_nodes)

        # keep the important structure
        # 掩码方案一：只保留选中子图的节点
        masked_node_list = [node for node in range(tree_node_x.data.x.shape[0])
                            if node in tree_node_x.coalition]

        # remove the important structure, for node_classification,
        # remain the node_idx when remove the important structure
        # 掩码方案二：保留子图以外的节点，节点分类时把目标节点本身也留下
        maskout_node_list = [node for node in range(tree_node_x.data.x.shape[0])
                             if node not in tree_node_x.coalition]
        if not self.explain_graph:
            maskout_node_list += [self.new_node_idx]

        # 两种掩码子图各自前向打分，差值越大说明该子图越关键
        masked_score = gnn_score(masked_node_list,
                                 tree_node_x.data,
                                 value_func=value_func,
                                 subgraph_building_method=self.subgraph_building_method)

        maskout_score = gnn_score(maskout_node_list,
                                  tree_node_x.data,
                                  value_func=value_func,
                                  subgraph_building_method=self.subgraph_building_method)

        # 解释子图的稀疏度（sparsity）：越接近 1 表示用越少的节点/边就复现了预测
        sparsity_score = sparsity(masked_node_list, tree_node_x.data,
                                  subgraph_building_method=self.subgraph_building_method)

        results = self.write_from_MCTSNode_list(results)
        # 汇总四项指标：保留子图 / 移除子图 / 原图 / 稀疏度
        related_pred = {'masked': masked_score,
                        'maskout': maskout_score,
                        'origin': probs[node_idx, label].item(),
                        'sparsity': sparsity_score}

        return results, related_pred

    def __call__(self, x: Tensor, edge_index: Tensor, **kwargs)\
            -> Tuple[None, List, List[Dict]]:
        # 对每个类别各调一次 explain；若 save_dir 下已有 <filename>.pt 则载入作为 saved_MCTSInfo_list 复用搜索结果，结束时把结果写回该文件
        r""" explain the GNN behavior for the graph using SubgraphX method
        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            kwargs(:obj:`Dict`):
              The additional parameters
                - node_idx (:obj:`int`, :obj:`None`): The target node index when explain node classification task
                - max_nodes (:obj:`int`, :obj:`None`): The number of nodes in the final explanation results
        :rtype: (:obj:`None`, List[torch.Tensor], List[Dict])
        """
        node_idx = kwargs.get('node_idx')
        max_nodes = kwargs.get('max_nodes')   # default max subgraph size

        # collect all the class index
        # 每个类别单独包一层张量作为 label 传入 explain
        labels = tuple(label for label in range(self.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

        related_preds = []
        explanation_results = []
        saved_results = None
        if self.save:
            if os.path.isfile(os.path.join(self.save_dir, f"{self.filename}.pt")):
                saved_results = torch.load(os.path.join(self.save_dir, f"{self.filename}.pt"))

        for label_idx, label in enumerate(ex_labels):
            results, related_pred = self.explain(x, edge_index,
                                                 label=label,
                                                 max_nodes=max_nodes,
                                                 node_idx=node_idx,
                                                 saved_MCTSInfo_list=saved_results)
            related_preds.append(related_pred)
            explanation_results.append(results)

        if self.save:
            torch.save(explanation_results,
                       os.path.join(self.save_dir, f"{self.filename}.pt"))

        return None, explanation_results, related_preds