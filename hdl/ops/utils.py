# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/ops/utils.py
# 说明：自定义算子 Python 绑定
# 模块功能：按名称返回激活函数（activation）模块的工厂，供网络按字符串配置激活层。
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
    名称转小写后匹配 relu/elu/selu/softmax/sigmoid/none，对应 nn.ReLU、nn.ELU、nn.SELU、nn.Softmax、
    nn.Sigmoid 与 None；kwargs 中读取 inplace、alpha、dim 等可选项，未匹配的名称抛 ValueError。

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
