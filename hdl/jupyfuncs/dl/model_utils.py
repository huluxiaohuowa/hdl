# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/dl/model_utils.py
# 说明：深度学习张量与模型辅助工具
# 模块功能：模型检查点（checkpoint）的保存与加载，同时记录 init_args、epoch、优化器状态与损失。
import torch
from torch import nn


def save_model(
    model,
    save_dir,
    epoch=0,
    optimizer=None,
    loss=None,
):
    """Save the model and related training information to a specified directory.
    保存单个检查点文件：模型需带 init_args 属性；nn.DataParallel 包装时保存内层 module 的 state_dict。
    
    Args:
        model: The model to be saved.
        save_dir: The directory where the model will be saved.
        epoch (int): The current epoch number (default is 0).
        optimizer: The optimizer used for training (default is None).
        loss: The loss value (default is None).
    """
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    if optimizer is None:
        optim_params = None
    else:
        optim_params = optimizer.state_dict()
    # 检查点字典：重建模型所需的 init_args + 训练状态（epoch/权重/优化器/损失）
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
    save_dir,
    model_class=None,
    model=None,
    optimizer=None,
    train=False,
):
    """Load a saved model from the specified directory.
    载入检查点：未传入 model 时用 checkpoint 的 init_args 重建模型；传入 DataParallel 模型时把权重键名补上 module. 前缀。
    
    Args:
        save_dir (str): The directory where the model checkpoint is saved.
        model_class (torch.nn.Module, optional): The class of the model to be loaded. Defaults to None.
        model (torch.nn.Module, optional): The model to load the state_dict into. Defaults to None.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state_dict into. Defaults to None.
        train (bool, optional): Whether to set the model to training mode. Defaults to False.
    
    Returns:
        tuple: A tuple containing the loaded model, optimizer, epoch, and loss.
    """
    # from .model_dict import MODEL_DICT
    checkpoint = torch.load(save_dir)
    # 未给 model 时：按保存的 init_args 重新构造模型并载入权重
    if model is None:
        init_args = checkpoint['init_args']
        assert model_class is not None
        model = model_class(**init_args)
        model.load_state_dict( 
            checkpoint['model_state_dict'], 
        )
    
    elif isinstance(model, nn.DataParallel):
        state_dict = checkpoint['model_state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        # 权重键名补 module. 前缀，并把 features.module. 归一化为 module.features.
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
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # train=True 切训练模式，否则切评估模式
    if train:
        model.train()
    else:
        model.eval()

    return model, optimizer, epoch, loss