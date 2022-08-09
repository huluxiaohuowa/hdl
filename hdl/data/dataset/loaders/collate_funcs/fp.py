r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""
import typing as t

import numpy as np
import pandas as pd
import torch


int_types = (
    int,
    np.int32,
    np.int64,
    pd.Int16Dtype,
    pd.Int32Dtype,
    pd.Int64Dtype,
    # torch.int32
)


def fp_collate(batch):
    transposed = list(zip(*batch))

    # fps
    fps = list(zip(*transposed[0]))
    fps = [torch.vstack(fp).float() for fp in fps]
    if len(transposed) == 1:
        return fps

    # target_list
    targets = list(zip(*transposed[-1]))
    targets_list = []
    for target_labels in targets:
        if not isinstance(target_labels[0], t.Iterable):
            target_labels = torch.Tensor(target_labels)
        else:
            target_labels = list(target_labels)
        # if isinstance(target_labels[0], int_types):
        #     target_labels = torch.LongTensor(target_labels)
        targets_list.append(target_labels) 
 
    # target_tensors
    if len(transposed) == 3:
        target_tensors = list(zip(*transposed[1])) 
        target_tensors = [
            torch.vstack(target_tensor).float()
            for target_tensor in target_tensors
        ]
    
        return fps, target_tensors, targets_list
    else:
        return fps, targets, targets_list 