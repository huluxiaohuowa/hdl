# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/data/dataset/loaders/spliter.py
# 说明：DataLoader、collate 函数与采样器
# 模块功能：数据切分函数 split_data，支持随机（random）与按骨架均衡（scaffold balanced）两种策略，返回训练/验证/测试三份。
from typing import DefaultDict, Tuple
from random import Random
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from hdl.data.dataset.graph.chiral import MolDataset


def split_data(
    smis: Tuple[str],
    labels: Tuple,
    split_type: str = "random",
    sizes: Tuple[float, float, float] = (0.8, 0.2, 0.0),
    seed: int = 999,
    num_folds: int = 1,
    balanced: bool = True,
    args=None,
) -> Tuple[Tuple[str], Tuple[str], Tuple[str]]:
    """按 split_type 把 smis/labels 切成 (train, val, test) 三份并返回。

    split_type 支持："random"（随机切分）与 "scaffold_balanced"（按 Murcko 骨架分组后整组装箱，
    即按骨架切分（scaffold split）的均衡变体），其他取值不匹配任何分支。
    sizes 是 (train, val, test) 占比三元组：random 分支按 int(占比 × 样本数) 取两段索引边界。
    seed 用于 Random 洗牌；num_folds 与 args 在函数体内未被使用，实现里没有交叉验证折（k-fold）循环。
    balanced=True 时把骨架组按大小分成大/小两堆各自打乱后拼接；balanced=False 时按组大小降序排列，
    大骨架优先进入 train。
    返回结构随策略不同：random 每份是 [smiles, labels] 二元列表；scaffold_balanced 每份只有 smiles 列表。
    """
    random = Random(seed)

    if split_type == "random":
        # 洗牌后的索引按 sizes 算出的两段边界切成 train/val/test
        indices = list(range(len(smis)))
        random.shuffle(indices)

        train_size = int(sizes[0] * len(smis))
        train_val_size = int((sizes[0] + sizes[1]) * len(smis))
        train = [
            [smis[i] for i in indices[:train_size]],
            [labels[i] for i in indices[:train_size]],
        ]
        val = [
            [smis[i] for i in indices[train_size:train_val_size]],
            [labels[i] for i in indices[train_size:train_val_size]],
        ]
        test = [
            [smis[i] for i in indices[train_val_size:]],
            [labels[i] for i in indices[train_val_size:]],
        ]
    elif split_type == "scaffold_balanced":
        # 三份的目标条数（浮点）：该式用的名字 data 在本模块未定义
        train_size, val_size, test_size = (
            sizes[0] * len(data),
            sizes[1] * len(data),
            sizes[2] * len(data),
        )
        train, val, test = [], [], []
        train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0
        scaffold_to_indices = defaultdict(set)
        rdmols = [Chem.MolFromSmiles(s) for s in smis]
        for i, rdmol in enumerate(rdmols):
            # 按不含手性的 Murcko 骨架聚索引组：同一骨架的样本整组进入同一份，避免骨架泄漏（scaffold leakage）
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=rdmol, includeChirality=False
            )
            scaffold_to_indices[scaffold].add(i)
        if balanced:
            # 长度超过 val_size/2 或 test_size/2 的骨架组算大组，其余算小组，各自打乱后拼接（大组在前）
            index_sets = list(scaffold_to_indices.values())
            big_index_sets = []
            small_index_sets = []
            for index_set in index_sets:
                if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                    big_index_sets.append(index_set)
                else:
                    small_index_sets.append(index_set)
            random.seed(seed)
            random.shuffle(big_index_sets)
            random.shuffle(small_index_sets)
            index_sets = big_index_sets + small_index_sets
        else:
            # balanced=False：不洗牌，骨架组按大小降序排列后依次装箱
            index_sets = sorted(
                list(scaffold_to_indices.values()),
                key=lambda index_set: len(index_set),
                reverse=True,
            )
        for index_set in index_sets:
            # 贪心装箱：整组塞进第一个还装得下它的分区，装不进 val 就全部落到 test
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1
        train = [smis[i] for i in train]
        val = [smis[i] for i in val]
        test = [smis[i] for i in test]
    return train, val, test
