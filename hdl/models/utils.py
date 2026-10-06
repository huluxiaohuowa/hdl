# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/models/utils.py
# 说明：神经网络模型定义与注册表
# 模块功能：模型 checkpoint 存取工具，保存/恢复权重、优化器状态、epoch 与 loss，并可按注册表名重建模型。
import typing as t

import torch
from torch import nn


def save_model(
    model: t.Union[nn.Module, nn.DataParallel],
    save_dir: str = "./model.ckpt",
    epoch: int = 0,
    optimizer: torch.optim.Optimizer = None,
    loss: float = None,
) -> None:
    """保存 checkpoint 到 save_dir：写入模型自身的 init_args（供按名重建）、epoch、模型权重、
    优化器状态与 loss；DataParallel 模型取其 .module 的权重以免多出 module. 前缀。"""
    # DataParallel 取内层 module 的权重，避免保存出多余的 module. 前缀
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    # 优化器可选：无优化器时该字段存 None
    if optimizer is None:
        optim_params = None
    else:
        optim_params = optimizer.state_dict()
    torch.save(
        {
            'init_args': model.init_args,
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optim_params,
            'loss': loss,
        },
        save_dir
    )


def load_model(
    save_dir: str,
    model_name: str = None,
    model: t.Union[nn.Module, nn.DataParallel] = None,
    optimizer: torch.optim.Optimizer = None,
    train: bool = False,
) -> t.Tuple[
    t.Union[nn.Module, nn.DataParallel],
    torch.optim.Optimizer,
    int,
    float
]:
    """从 save_dir 读回 checkpoint：model 为 None 时用 checkpoint 的 init_args 和 model_name（MODEL_DICT 键）新建模型并载入权重；
    传入 DataParallel 时先把权重键补上 module. 前缀再载入；optimizer 非 None 时一并恢复其状态；
    train 决定切到 train() 还是 eval()。返回 (model, optimizer, epoch, loss)。"""
    from .model_dict import MODEL_DICT
    checkpoint = torch.load(save_dir)
    if model is None:
        # 按 checkpoint 记录的 init_args 与注册表名重建模型，再直接载入权重
        init_args = checkpoint['init_args']
        assert model_name is not None
        model = MODEL_DICT[model_name](**init_args)
        model.load_state_dict(
            checkpoint['model_state_dict'],
        )

    elif isinstance(model, nn.DataParallel):
        # 权重键补 module. 前缀以匹配 DataParallel 包装后的参数名（并修正 features.module. 的特例）
        state_dict = checkpoint['model_state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(
            checkpoint['model_state_dict'],
        )

    if optimizer is not None:
        # 一并恢复优化器动量等状态，便于断点续训
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)

    # train=True 切训练模式（启用 Dropout/BatchNorm 统计更新），否则切推理模式
    if train:
        model.train()
    else:
        model.eval()

    return model, optimizer, epoch, loss
