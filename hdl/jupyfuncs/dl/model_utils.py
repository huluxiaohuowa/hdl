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

    if train:
        model.train()
    else:
        model.eval()

    return model, optimizer, epoch, loss