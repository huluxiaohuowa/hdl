# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/controllers/explain/shapley.py
# 说明：模型归因解释（Shapley / 子图）
# 模块功能：图神经网络归因（attribution）的夏普利值（Shapley value）族实现：把节点集合当作联盟（coalition），用掩码子图的模型打分作为特征函数，给出精确枚举与蒙特卡洛（Monte Carlo）采样两类近似
import copy
import torch
import numpy as np
from scipy.special import comb
from itertools import combinations
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Batch, Dataset, DataLoader


def GnnNetsGC2valueFunc(gnnNets, target_class):
    """构造图分类的特征函数（value function）：输入打包后的图批次，返回目标类 target_class 的 softmax 概率"""
    def value_func(batch):
        with torch.no_grad():
            # 关闭梯度只做前向，取概率向量第 target_class 列
            logits = gnnNets(data=batch)
            probs = F.softmax(logits, dim=-1)
            score = probs[:, target_class]
        return score
    return value_func


def GnnNetsNC2valueFunc(gnnNets_NC, node_idx, target_class):
    """构造节点分类的特征函数（value function）：把逐节点概率重排为 (图数, 节点数, 类别数)，返回指定 node_idx 上目标类的概率"""
    def value_func(data):
        with torch.no_grad():
            logits = gnnNets_NC(data=data)
            probs = F.softmax(logits, dim=-1)
            # select the corresponding node prob through the node idx on all the sampling graphs
            # 按批次内的图重新整形，再取目标节点位置的概率
            batch_size = data.batch.max() + 1
            probs = probs.reshape(batch_size, -1, probs.shape[-1])
            score = probs[:, node_idx, target_class]
            return score
    return value_func


def get_graph_build_func(build_method):
    """按名字（不区分大小写）选择子图构造函数：zero_filling 置零掩码、split 按掩码切边，其他名字抛 NotImplementedError"""
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    else:
        raise NotImplementedError


class MarginalSubgraphDataset(Dataset):
    """边际子图数据集（torch_geometric Dataset）：把一组 exclude/include 节点掩码包装成样本，每个索引返回"排除联盟的图"与"包含联盟的图"两个 Data，供批量算边际贡献"""
    def __init__(self, data, exclude_mask, include_mask, subgraph_build_func):
        """入参：data 原始单图；exclude_mask/include_mask 形状 (掩码数, 节点数) 的 0/1 数组（转成 float 张量存到原图设备）；subgraph_build_func 由掩码重建 x 与 edge_index 的函数"""
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device

        self.label = data.y
        # 每行掩码对应一个联盟采样点：先构造"排除"版本，再把 coalition 位置置 1 得"包含"版本
        self.exclude_mask = torch.tensor(exclude_mask).type(torch.float32).to(self.device)
        self.include_mask = torch.tensor(include_mask).type(torch.float32).to(self.device)
        self.subgraph_build_func = subgraph_build_func

    def __len__(self):
        """样本数等于掩码行数"""
        return self.exclude_mask.shape[0]

    def __getitem__(self, idx):
        """第 idx 个联盟采样点：分别用 exclude/include 掩码重建子图，返回一对 Data（不含标签，标签由特征函数在整批上计算）"""
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.exclude_mask[idx])
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.include_mask[idx])
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        return exclude_data, include_data


def marginal_contribution(data: Data, exclude_mask: np.array, include_mask: np.array,
                          value_func, subgraph_build_func):
    # 按成对掩码批量前向，返回每个联盟采样点的特征函数差值（包含 - 排除），即夏普利值（Shapley value）的边际贡献项
    """ Calculate the marginal value for each pair. Here exclude_mask and include_mask are node mask. """
    marginal_subgraph_dataset = MarginalSubgraphDataset(data, exclude_mask, include_mask, subgraph_build_func)
    # 固定 batch_size=256、不打乱，保证输出顺序与掩码顺序一致
    dataloader = DataLoader(marginal_subgraph_dataset, batch_size=256, shuffle=False, num_workers=0)

    marginal_contribution_list = []

    for exclude_data, include_data in dataloader:
        # 同一批内分别对两种掩码图打分，差值即该子集 S 上加入 coalition 的边际值
        exclude_values = value_func(exclude_data)
        include_values = value_func(include_data)
        margin_values = include_values - exclude_values
        marginal_contribution_list.append(margin_values)

    marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
    return marginal_contributions


def graph_build_zero_filling(X, edge_index, node_mask: np.array):
    # 置零填充式子图构造：保留全部节点与边，把掩码为 0 的节点特征整体乘 0（抹掉信息但不改变图结构）
    """ subgraph building through masking the unselected nodes with zero features """
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(X, edge_index, node_mask: np.array):
    # 切分式子图构造：节点特征全部保留，只保留两端都被掩码选中的边，从而把子图从原图中拆出来
    """ subgraph building through spliting the selected nodes from the original graph """
    ret_X = X
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return ret_X, ret_edge_index


def l_shapley(coalition: list, data: Data, local_radius: int,
              value_func: str, subgraph_building_method='zero_filling'):
    # 局部夏普利值（L-Shapley）：只在 coalition 的 local_radius 跳邻域内枚举外围节点的所有子集，按组合数加权求边际贡献
    """ shapley value where players are local neighbor nodes """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    # 以 coalition 为种子，反复并入邻居，扩展到 local_radius-1 跳的局部区域
    local_region = copy.copy(coalition)
    for k in range(local_radius - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    set_exclude_masks = []
    set_include_masks = []
    # 博弈玩家：局部区域内除 coalition 之外的节点
    nodes_around = [node for node in local_region if node not in coalition]
    num_nodes_around = len(nodes_around)

    # 枚举外围节点的所有子集作为子集合 S：S 内的局部节点保留（掩码 1），其余局部节点为 0（被掩掉）
    for subset_len in range(0, num_nodes_around + 1):
        node_exclude_subsets = combinations(nodes_around, subset_len)
        for node_exclude_subset in node_exclude_subsets:
            set_exclude_mask = np.ones(num_nodes)
            set_exclude_mask[local_region] = 0.0
            if node_exclude_subset:
                set_exclude_mask[list(node_exclude_subset)] = 1.0
            # 包含联盟的掩码 = 子集 S 的掩码再打开 coalition 对应位置
            set_include_mask = set_exclude_mask.copy()
            set_include_mask[coalition] = 1.0

            set_exclude_masks.append(set_exclude_mask)
            set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    # Shapley 权重 1 / (C(p, |S|) * (p - |S|))：p 为玩家数，S 为该掩码下已就位的玩家数
    num_players = len(nodes_around) + 1
    num_player_in_set = num_players - 1 + len(coalition) - (1 - exclude_mask).sum(axis=1)
    p = num_players
    S = num_player_in_set
    coeffs = torch.tensor(1.0 / comb(p, S) / (p - S + 1e-6))

    # 对所有 (S, S∪coalition) 掩码对做批量前向取边际贡献，再按权重求和得到该联盟的归因分数
    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    l_shapley_value = (marginal_contributions.squeeze().cpu() * coeffs).sum().item()
    return l_shapley_value


def mc_shapley(coalition: list, data: Data,
               value_func: str, subgraph_building_method='zero_filling',
               sample_num=1000) -> float:
    # 蒙特卡洛（Monte Carlo）采样近似的夏普利值（Shapley value）：sample_num 次随机排列取前驱集合，边际贡献取平均
    """ monte carlo sampling approximation of the shapley value """
    subset_build_func = get_graph_build_func(subgraph_building_method)

    num_nodes = data.num_nodes
    node_indices = np.arange(num_nodes)
    # 用一个越界编号充当占位符，排列中它的位置就是联盟的插入点
    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []

    # 每次采样一个随机排列，占位符之前的节点集合即为子集合 S（基线：仅保留 S 中的节点，其余掩掉）
    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in node_indices if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.zeros(num_nodes)
        set_exclude_mask[selected_nodes] = 1.0
        # 包含联盟的掩码：在 S 的基础上再打开 coalition
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = marginal_contribution(data, exclude_mask, include_mask, value_func, subset_build_func)
    # 采样边际贡献的均值作为近似夏普利值
    mc_shapley_value = marginal_contributions.mean().item()

    return mc_shapley_value


def mc_l_shapley(coalition: list, data: Data, local_radius: int,
                 value_func: str, subgraph_building_method='zero_filling',
                 sample_num=1000) -> float:
    # 局部夏普利值（L-Shapley）的蒙特卡洛（Monte Carlo）近似：随机排列只在局部邻域内采样，掩码约定与 l_shapley 一致（区域外节点始终保留）
    """ monte carlo sampling approximation of the l_shapley value """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    # 从 coalition 出发扩展 local_radius-1 跳得到局部区域
    local_region = copy.copy(coalition)
    for k in range(local_radius - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    # 每次随机排列局部区域内的其他节点，占位符之前的节点作为已在集合 S 中的邻居
    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in local_region if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        # 基线掩码：局部区域整体关掉，仅选中的 S 重新打开；区域外节点保持为 1（不参与博弈）
        set_exclude_mask = np.ones(num_nodes)
        set_exclude_mask[local_region] = 0.0
        set_exclude_mask[selected_nodes] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    # 采样边际贡献取均值，不乘 Shapley 组合权重
    mc_l_shapley_value = (marginal_contributions).mean().item()
    return mc_l_shapley_value


def gnn_score(coalition: list, data: Data, value_func: str,
              subgraph_building_method='zero_filling') -> torch.Tensor:
    # 只保留 coalition 对应节点（其余按所选方式掩掉）后模型的打分，用作子图得分/奖励
    """ the value of subgraph with selected nodes """
    num_nodes = data.num_nodes
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    mask = torch.zeros(num_nodes).type(torch.float32).to(data.x.device)
    mask[coalition] = 1.0
    ret_x, ret_edge_index = subgraph_build_func(data.x, data.edge_index, mask)
    # 单张图打包成 batch，以匹配特征函数（value function）的批量输入
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    score = value_func(mask_data)
    # get the score of predicted class for graph or specific node idx
    return score.item()


def NC_mc_l_shapley(coalition: list, data: Data, local_radius: int,
                    value_func: str, node_idx: int = -1,
                    subgraph_building_method='zero_filling', sample_num=1000) -> float:
    # 节点分类版局部夏普利值（L-Shapley）蒙特卡洛近似：在 mc_l_shapley 基础上额外把目标节点 node_idx 同时在两种掩码下保留，衡量的是 coalition 相对"保留目标节点"基线的贡献
    """ monte carlo approximation of l_shapley where the target node is kept in both subgraph """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    local_region = copy.copy(coalition)
    for k in range(local_radius - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in local_region if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        # 基线掩码：关掉整个局部区域，只保留排列中前驱节点与目标节点 node_idx
        set_exclude_mask = np.ones(num_nodes)
        set_exclude_mask[local_region] = 0.0
        set_exclude_mask[selected_nodes] = 1.0
        if node_idx != -1:
            set_exclude_mask[node_idx] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0  # include the node_idx

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    # 采样边际贡献均值即近似分数
    mc_l_shapley_value = (marginal_contributions).mean().item()
    return mc_l_shapley_value


def sparsity(coalition: list, data: Data, subgraph_building_method='zero_filling'):
    """解释子图的稀疏度（sparsity）：zero_filling 按被掩掉的节点比例计算，split 按被丢掉的边比例计算，数值越大说明保留的结构越少"""
    if subgraph_building_method == 'zero_filling':
        # 1 - 保留节点数 / 总节点数
        return 1.0 - len(coalition) / data.num_nodes

    elif subgraph_building_method == 'split':
        row, col = data.edge_index
        node_mask = torch.zeros(data.x.shape[0])
        node_mask[coalition] = 1.0
        # 只统计两端都在 coalition 内的边，用被移除的边占比衡量稀疏度
        edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
        return 1.0 - edge_mask.sum() / edge_mask.shape[0]