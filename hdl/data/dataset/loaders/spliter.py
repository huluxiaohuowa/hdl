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
    random = Random(seed)

    if split_type == "random":
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
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=rdmol, includeChirality=False
            )
            scaffold_to_indices[scaffold].add(i)
        if balanced:
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
            index_sets = sorted(
                list(scaffold_to_indices.values()),
                key=lambda index_set: len(index_set),
                reverse=True,
            )
        for index_set in index_sets:
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
