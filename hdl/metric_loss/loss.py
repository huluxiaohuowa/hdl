import typing as t

import torch
from torch import nn

from .multi_label import BPMLLLoss


def get_lossfunc(
    name: str,
    *args,
    **kwargs
) -> t.Callable:
    """Get loss function by name

    Args:
        name (str): The name of the loss function

    Returns:
        t.Callable: the loss function
    """
    name = name.lower()
    if name == 'bce':
        return nn.BCELoss(*args, **kwargs)
    elif name == 'ce':
        return nn.CrossEntropyLoss(*args, **kwargs)
    elif name == 'mse':
        return nn.MSELoss(*args, **kwargs)
    elif name == 'bpmll':
        return BPMLLLoss(*args, **kwargs)
    elif name == 'nll':
        return nn.GaussianNLLLoss(*args, **kwargs)


def mtmc_loss(
    y_preds: t.Iterable,
    y_trues: t.Iterable,
    loss_names: t.Iterable[str] = None,
    individual: bool = False,
    task_weights: t.List = None,
    device=torch.device('cpu'),
    **kwargs
):
    num_tasks = len(y_preds)
    if loss_names is None: 
        loss_func = nn.CrossEntropyLoss()
        loss_funcs = [loss_func] * num_tasks
    elif isinstance(loss_names, str):
        loss_func = get_lossfunc(loss_names, **kwargs)
        loss_funcs = [loss_func] * num_tasks
    else:
        loss_funcs = [
            get_lossfunc(loss_str)
            for loss_str in loss_names
        ]

    if task_weights is None:
        task_weights = torch.ones(num_tasks).to(device)
    else:
        assert len(task_weights) == num_tasks
        task_weights = torch.FloatTensor(task_weights).to(device)
    
    loss_values = [
        loss_func(y_pred, y_true)
        for y_pred, y_true, loss_func in zip(
            y_preds, y_trues, loss_funcs
        )
    ]
 
    loss_final = sum([
        loss_value * task_weight
        for loss_value, task_weight in zip(loss_values, task_weights)
    ])
    # loss_final = sum(loss_values) / num_tasks
    if not individual:
        return loss_final
    else:
        loss_list = [loss_value for loss_value in loss_values]
        return (loss_final, loss_list)