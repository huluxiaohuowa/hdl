# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/models/rxn.py
# 说明：神经网络模型定义与注册表
# 模块功能：反应（reaction）预测模型装配：加载 rxnfp 微调 BERT 作反应式编码器，再接多任务多分类头并放到可用设备。
import pkg_resources
from transformers import BertModel
import torch
# from torch import nn

from hdl.layers.general.linear import (
    MultiTaskMultiClassBlock,
    MuMcHardBlock
)
# from hdl.data.seq.rxn import rxn_model


def get_rxn_model(
    model_path: str = None
):
    """加载反应 SMILES 序列编码器：model_path 缺省时从 rxnfp 包资源目录 models/transformers/bert_ft 读取微调 BERT，
    返回 eval() 且置于 CPU 的 BertModel。"""
    if model_path is None:
        model_path = pkg_resources.resource_filename(
            "rxnfp",
            "models/transformers/bert_ft"
        )
        # 从本地 rxnfp 包目录读取微调权重，推理模式并放 CPU
        model = BertModel.from_pretrained(model_path)
        model = model.eval().cpu()

    return model


# 默认 encoder 参数在函数定义时（即模块导入时）就调用 get_rxn_model() 完成 BERT 加载
def build_rxn_mu(
    nums_classes,
    hard=False,
    hidden_size=128,
    nums_hidden_layers=10,
    encoder=get_rxn_model(),
    # freeze_encoder=True,
    device_id: int = 0,
    **kwargs
):
    """装配反应多任务多分类模型：encoder 提供反应式序列嵌入，nums_classes 给出各任务的类别数，
    hidden_size/nums_hidden_layers 控制分类塔的宽与深；hard=False 用每任务独立 MLP 塔，True 用共享塔后接各任务输出层。
    按 CUDA 可用性选择设备，返回 (model, device)。"""
    if not hard:
        model = MultiTaskMultiClassBlock(
            encoder=encoder,
            nums_classes=nums_classes,
            hidden_size=hidden_size,
            num_hidden_layers=nums_hidden_layers,
            # freeze_encoder=freeze_encoder,
            **kwargs
        )
    else:
        model = MuMcHardBlock(
            encoder=encoder,
            nums_classes=nums_classes,
            hidden_size=hidden_size,
            num_hidden_layers=nums_hidden_layers,
            # freeze_encoder=freeze_encoder,
            **kwargs
        )
    device = torch.device(f'cuda:{device_id}') \
        if torch.cuda.is_available() \
        else torch.device('cpu')

    # 把编码器与各任务分类头一起搬到选中设备
    model = model.to(device)
    
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    return model, device