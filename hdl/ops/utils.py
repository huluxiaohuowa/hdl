import typing as t

# import torch
from torch import nn


__all__ = [
    'get_activation'
]


def get_activation(
    name: str,
    **kwargs
) -> t.Callable:
    """ Get activation module by name
    Args:
        name (str): The name of the activation function (relu, elu, selu)
        args, kwargs: Other parameters
    Returns:
        nn.Module: The activation module
    """
    name = name.lower()
    if name == 'relu':
        inplace = kwargs.get('inplace', False)
        return nn.ReLU(inplace=inplace)
    elif name == 'elu':
        alpha = kwargs.get('alpha', 1.)
        inplace = kwargs.get('inplace', False)
        return nn.ELU(alpha=alpha, inplace=inplace)
    elif name == 'selu':
        inplace = kwargs.get('inplace', False)
        return nn.SELU(inplace=inplace)
    elif name == 'softmax':
        dim = kwargs.get('dim', -1)
        return nn.Softmax(dim=dim)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'none':
        return 
    else:
        raise ValueError('Activation not implemented')
