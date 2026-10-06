# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/data/dataset/loaders/collate_funcs/rxn.py
# 说明：DataLoader、collate 函数与采样器
# 模块功能：反应（reaction）序列批的构造入口，用 rxnfp 词元器（tokenizer）把整批反应 SMILES 转成 BERT 输入张量。
from rxnfp.tokenization import (
    SmilesTokenizer,
    convert_reaction_to_valid_features_batch,
)
import torch
import numpy as np
import pkg_resources


__all__ = [
    'collate_rxn',
]


def collate_rxn(
    rxn_list,
    labels,
    vocab_path: str = None,
    max_len: int = 512
):
    """把整批反应 SMILES 一次性词元化（tokenization）成模型输入，接口不是逐样本合并的 collate_fn。

    vocab_path 为 None 时用 pkg_resources 取 rxnfp 包内置的 bert_ft/vocab.txt 词表（vocab），
    与 max_len（序列长度上限）一起构造 SmilesTokenizer。
    convert_reaction_to_valid_features_batch 对 rxn_list 逐条编码并 padding（填充）成批内等长的词元序列，
    input_ids、input_mask、segment_ids 三个数组各转成 int64 张量，按该顺序放进列表 X；
    input_mask 标记有效词元位置，补位处为 0 由模型侧忽略，segment_ids 区分别反应各段。
    y 是 labels 直接转成的 LongTensor（类别索引）。返回 (X, y)。
    """
    if vocab_path is None:
        vocab_path = pkg_resources.resource_filename(
            "rxnfp",
            "models/transformers/bert_ft/vocab.txt"
        )
    tokenizer = SmilesTokenizer(
        vocab_path, max_len=max_len
    )

    feats = convert_reaction_to_valid_features_batch(
        rxn_list,
        tokenizer
    )
    X = [
        torch.tensor(feats.input_ids.astype(np.int64)),
        torch.tensor(feats.input_mask.astype(np.int64)),
        torch.tensor(feats.segment_ids.astype(np.int64))
    ]
    y = torch.LongTensor(labels)
    return X, y
