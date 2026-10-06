# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/models/model_dict.py
# 说明：神经网络模型定义与注册表
# 模块功能：模型注册表 MODEL_DICT，按模型名取到对应的模型类（构造器），供训练器与 load_model 按名实例化模型。
from hdl.layers.general.linear import (
    MultiTaskMultiClassBlock,
    MuMcHardBlock
)
from .linear import MMIterLinear
from .chiral_gnn import GNN
from .ginet import GINet
from .ginet import GINMLPR


# 模型名 -> 模型类（工厂构造器）：训练器用 MODEL_DICT[model_name](**model_init_args) 建模型，
# load_model 也用同一张表按 checkpoint 保存的 init_args 重建模型。
# 键含义：rxn_trans=BERT 编码 + 每任务独立 MLP 头的反应多任务多分类；rxn_trans_hard=共享隐层塔 + 各任务输出层的硬共享版本；
# mmiter_linear=指纹线性迭代式多任务基线；chiral_gnn=手性图神经网络分子性质预测；
# ginet=图同构网络分子编码器；ginmlpr=多张分子图 GIN + MLP 回归头。
MODEL_DICT = {
    'rxn_trans': MultiTaskMultiClassBlock,
    'rxn_trans_hard': MuMcHardBlock,
    'mmiter_linear': MMIterLinear,
    'chiral_gnn': GNN,
    'ginet': GINet,
    'ginmlpr': GINMLPR
}