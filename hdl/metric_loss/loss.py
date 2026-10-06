# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/metric_loss/loss.py
# 说明：损失函数与评估指标
# 模块功能：损失函数工厂 get_lossfunc（按名称实例化损失模块）与多任务损失（multi-task loss）加权聚合 mtmc_loss。
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
    按名称返回损失函数实例（注册表式工厂）。名称转小写后匹配：
    bce=BCELoss、ce=CrossEntropyLoss、mse=MSELoss、bpmll=BPMLLLoss（多标签损失）、
    nll=GaussianNLLLoss（高斯负对数似然）；名称不匹配时隐式返回 None。
    *args/**kwargs 直接透传给对应的损失构造函数。

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
    """多任务损失聚合：逐任务调用各自的损失函数，再按 task_weights 加权求和。

    Args:
        y_preds (Iterable): 各任务的预测张量序列，长度为任务数。
        y_trues (Iterable): 各任务的标签张量序列。
        loss_names (Iterable[str] | str, optional): 损失名称；None 时全部用 CrossEntropyLoss，
            字符串时全部任务共用该损失，列表时按顺序逐任务指定。
        individual (bool): 是否同时返回各任务的单独损失列表。
        task_weights (List, optional): 各任务权重，长度需等于任务数；缺省为全 1。
        device: 权重张量所在设备。
        kwargs: 透传给 get_lossfunc。

    Returns:
        加权总损失（标量张量）；individual=True 时返回 (总损失, 各任务损失列表)。
    """
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
        # 每个任务用自己的损失函数计算一项
        loss_func(y_pred, y_true)
        for y_pred, y_true, loss_func in zip(
            y_preds, y_trues, loss_funcs
        )
    ]
 
    # 各任务损失乘以权重后相加得到总损失（未做任务数平均）
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