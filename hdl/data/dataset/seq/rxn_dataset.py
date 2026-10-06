# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/data/dataset/seq/rxn_dataset.py
# 说明：分子数据集构建与切分
# 模块功能：反应（reaction）SMILES 序列数据集，把 CSV 每行反应经 rxnfp 词元器编码成 BERT 输入张量与类别标签索引。
import typing as t

import numpy as np
import torch
import pkg_resources
from rxnfp.tokenization import (
    SmilesTokenizer,
    # convert_reaction_to_valid_features_batch,
    convert_reaction_to_valid_features,
)

from ..base_dataset import CSVDataset


class RXNCSVDataset(CSVDataset):
    """继承 CSVDataset 的反应序列数据集：整表读入 CSV，逐行把反应 SMILES 词元化（tokenization）为张量样本。"""
    def __init__(
        self,
        csv_file: str,
        max_len: int = 512,
        vocab_path: str = None,
        splitter: str = ',',
        smiles_col: str = 'SMILES',
        target_cols: t.List = [],
        **kwargs,
    ) -> None:
        """csv_file 以 splitter 为分隔符读入整表，smiles_col 指定反应列、target_cols 指定标签列名列表；
        vocab_path 为 None 时用 pkg_resources 取 rxnfp 包内置的 bert_ft/vocab.txt 词表（vocab），
        连同 max_len（词元序列长度上限）一起构造 self.tokenizer。"""
        super().__init__(
            csv_file,
            splitter=splitter,
            smiles_col=smiles_col,
            target_cols=target_cols,
            **kwargs
        )
        if vocab_path is None:
            vocab_path = pkg_resources.resource_filename(
                "rxnfp",
                "models/transformers/bert_ft/vocab.txt"
            )
        self.tokenizer = SmilesTokenizer(
            vocab_path, max_len=max_len
        )
 
    def __getitem__(self, index):
        # rxn_list = [self.df.loc[index][self.smiles_col]]
        rxn = self.df.loc[index][self.smiles_col]
        feats = convert_reaction_to_valid_features(
            rxn,
            self.tokenizer
        )
        X = [
            torch.tensor(feats.input_ids.astype(np.int64)),
            torch.tensor(feats.input_mask.astype(np.int64)),
            torch.tensor(feats.segment_ids.astype(np.int64))
        ]
        if any(self.target_cols):
            labels = self.df.loc[index][self.target_cols].tolist()
            y = torch.LongTensor(labels)
            
            return X, y
        else:
            return X
 