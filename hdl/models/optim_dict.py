# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/models/optim_dict.py
# 说明：神经网络模型定义与注册表
# 模块功能：优化器注册表 OPTIM_DICT，按优化器名取到对应的 torch 优化器类（工厂函数）。
from torch.optim import (
    Adadelta,
    Adam,
    SGD,
    RMSprop,
)
from hdl.optims.nadam import Nadam


# 优化器名 -> 优化器类（工厂函数）：训练器用 OPTIM_DICT[optimizer_name](params) 建优化器，
# params 为 [{**optimizer_kwargs, 'params': 模型参数}] 参数组列表。
# 键含义：adam=Adam 自适应矩估计；adadelta=Adadelta 免学习率自适应；sgd=随机梯度下降；
# rmsprop=RMSProp 均方根传播；nadam=Nesterov 动量 Adam（hdl 自定义优化器）。
OPTIM_DICT = {
    'adam': Adam,
    'adadelta': Adadelta,
    'sgd': SGD,
    'rmsprop': RMSprop,
    'nadam': Nadam,
}