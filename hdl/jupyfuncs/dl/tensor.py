import typing as t
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def spmmsp(
    sp1: torch.sparse.Tensor,
    sp2: torch.sparse.Tensor
) -> torch.sparse.Tensor:
    from torch_sparse import spspmm
    assert sp1.size(-1) == sp2.size(0) and sp1.is_sparse and sp2.is_sparse
    m = sp1.size(0)
    k = sp2.size(0)
    n = sp2.size(-1)
    indices, values = spspmm(
        sp1.indices(), sp1.values(),
        sp2.indices(), sp2.values(),
        m, k, n
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        torch.Size([m, n])
    )


def label_to_onehot(ls, class_num, missing_label=-1):
    """
    example:
    >>>label_to_onehot([2,3,-1],6,-1)
    array([[ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.],
       [nan, nan, nan, nan]])
    :param ls:
    :param class_num:
    :param missing_label:
    :return:
    """
    if isinstance(ls, torch.Tensor):
        bool_t = ls == missing_label
        clamp_t = torch.clamp(ls, min=0)
        full_tensor = torch.zeros(ls.numel(), class_num)
        full_tensor = full_tensor.scatter_(1, clamp_t.reshape(-1, 1), 1)
        full_tensor[bool_t] = 0
        return full_tensor
    elif isinstance(ls, t.List):
        ls = np.array(ls, dtype=np.int32)
        bool_array = ls == missing_label
        arr = np.zeros((ls.size, ls.max() + 1))
        arr[np.arange(ls.size), ls] = 1
        arr[bool_array] = 0
        return arr
    elif not isinstance(ls, t.Iterable):
        arr = torch.zeros(class_num)
        if ls != missing_label and not np.isnan(ls) and ls is not None:
            arr[int(ls)] = 1
        return arr


def onehot_to_label(tensor):
    if isinstance(tensor, torch.Tensor):
        return torch.argmax(tensor, dim=-1)
    elif isinstance(tensor, np.ndarray):
        return np.argmax(tensor, axis=-1)


def label_to_tensor(
    label,
    num_classes,
    missing_label=-1,
    device=torch.device('cpu')
):
    if isinstance(label, t.List) and not any(label):
        return torch.zeros(num_classes).to(device)
    elif isinstance(label, t.List) and isinstance(label[0], t.Iterable):
        max_length = max([len(_l) for _l in label])
        index = [_l + _l[-1:] * (max_length - len(_l)) for _l in label]
        tensor_list = []
#         tensor = torch.zeros(len(label), num_classes, device=device)
        for _idx in index:
            _tensor = torch.zeros(num_classes).to(device)
            _idx = torch.LongTensor(_idx)
            _tensor = _tensor.scatter(0, _idx, 1)
            tensor_list.append(_tensor)
        
        return torch.vstack(tensor_list).to(device)
    else:
        if label == missing_label or np.isnan(label):
            return torch.zeros(num_classes)
        tensor = torch.zeros(num_classes).to(device)
        tensor[int(label)] = 1
        return tensor


def tensor_to_label(tensor, threshold=0.5):
    label_list, label_dict = [], defaultdict(list)
    labels = (tensor > threshold).nonzero(as_tuple=False)
    for label in labels:
        label_dict[label[0].item()].append(label[1].item())
    for _, label_value in label_dict.items():
        label_list.append(label_value)
    return label_list 


def get_dist_matrix(
    a: np.ndarray, b: np.ndarray
):
    return cdist(a, b)
    # aSumSquare = np.sum(np.square(a), axis=1)
    # bSumSquare = np.sum(np.square(b), axis=1)
    # mul = np.dot(a, b.T)
    # dists = np.sqrt(aSumSquare[:, np.newaxis] + bSumSquare - 2 * mul)
    # return dists


def get_valid_indices(labels):
    if isinstance(labels, torch.Tensor):
        nan_indices = torch.isnan(labels)
        valid_indices = (
            nan_indices == False
        ).nonzero(as_tuple=True)[0]
    else:
        target_pd = pd.array(labels)
        nan_indices = pd.isna(target_pd)
        valid_indices = torch.LongTensor(
            np.where(nan_indices == False)[0]
        )
    return valid_indices


def smooth_max(
    tensor: torch.Tensor,
    inf_k: int = None,
    **kwargs
):
    if inf_k is None:
        inf_k = 10
    max_value = torch.log(
        torch.sum(
            torch.exp(tensor * inf_k),
            **kwargs
        )
    ) / inf_k
    return max_value


def list_df(listA, listB):
    retB = list(set(listA).intersection(set(listB)))

    retC = list(set(listA).union(set(listB)))

    retD = list(set(listB).difference(set(listA)))

    retE = list(set(listA).difference(set(listB)))
    return retB, retC, retD, retE

