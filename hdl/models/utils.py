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
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
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
    from .model_dict import MODEL_DICT
    checkpoint = torch.load(save_dir)
    if model is None:
        init_args = checkpoint['init_args']
        assert model_name is not None
        model = MODEL_DICT[model_name](**init_args)
        model.load_state_dict( 
            checkpoint['model_state_dict'], 
        )
    
    elif isinstance(model, nn.DataParallel):
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
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)

    if train:
        model.train()
    else:
        model.eval()

    return model, optimizer, epoch, loss