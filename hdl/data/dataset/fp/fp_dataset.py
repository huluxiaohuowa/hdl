# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/data/dataset/fp/fp_dataset.py
# 说明：分子数据集构建与切分
# 模块功能：从 CSV 读取 SMILES 并按需计算分子指纹（fingerprint）的数据集类。
import typing as t

import torch
from rdkit import Chem
# import torch.utils.data as tud

from hdl.data.dataset.base_dataset import CSVDataset, CSVRDataset
from hdl.features.fp.features_generators import (
    get_features_generator,
    get_available_features_generators,
    FP_BITS_DICT 
)
# from hdl.features.fp.rxn import get_rxnrep_fingerprint


class FPDataset(CSVDataset):
    """多 SMILES 列、多任务标签的指纹数据集，继承 CSVDataset。
    fp_type 指定指纹类型（如 morgan_count），missing_labels 为各标签列的缺失值占位符。
    __getitem__ 返回 fingerprint_list（每条 SMILES 一个 LongTensor 指纹）；
    有 target_cols 时另返回变换后的标签张量与原始标签列表。
    """
    def __init__(
        self,
        csv_file: str,
        splitter: str,
        smiles_cols: t.List,
        target_cols: t.List = [],
        missing_labels: t.List = [],
        num_classes: t.List = [],
        target_transform: t.Union[str, t.List[str]] = None,
        fp_type: str = 'morgan_count',
        **kwargs
    ) -> None:
        super().__init__(
            csv_file,
            splitter=splitter,
            smiles_col=smiles_cols,
            target_cols=target_cols,
            num_classes=num_classes,
            target_transform=target_transform,
            **kwargs
        )
        self.smiles_cols = smiles_cols 
        # 指纹类型须在特征生成器注册表中；fp_numbits 为该类型的指纹位长
        assert fp_type in get_available_features_generators()
        self.fp_type = fp_type
        self.fp_generator = get_features_generator(self.fp_type)
        self.fp_numbits = FP_BITS_DICT[self.fp_type]
        self.missing_labels = missing_labels
    
    def __getitem__(self, index):
        # 取该行的各 SMILES 列
        smiles_list = self.df.loc[index][self.smiles_cols].tolist()

        # SMILES 逐条解析为 RDKit Mol 后即时生成指纹并转为整型张量
        fingerprint_list = list(
            map(
                lambda x: torch.LongTensor(self.fp_generator(Chem.MolFromSmiles(x))),
                smiles_list
            )
        )
        if any(self.target_cols):
            target_list = self.df.loc[index][self.target_cols].tolist()
            
            # process with missing label
            # 标签命中缺失值占位符时改写为 NaN，标记为缺失
            final_targets = []
            for target, missing_label in zip(target_list, self.missing_labels):
                if missing_label is not None and target == missing_label:
                    final_targets.append(float('nan'))
                else:
                    final_targets.append(target)
 
            # 无变换时返回（指纹列表, 原始标签列表）
            if self.target_transform is None:
                return fingerprint_list, final_targets 
            else:
                # print(final_targets)
                # 逐列套用注册的标签变换，缺失标签以 NaN 作为 missing_label 参与编码
                target_tensors = [
                    trans(target, num_class, missing_label=float('nan'))
                    for trans, target, num_class in zip(
                        self.target_transform,
                        final_targets,
                        self.num_classes
                    )
                ]
                # print(target_tensors)
                return fingerprint_list, target_tensors, final_targets 
        else:
            return fingerprint_list


class FPRDataset(CSVRDataset):
    """单 SMILES 列、单标签列的指纹数据集，继承 CSVRDataset。
    __getitem__ 返回 (指纹 LongTensor, (标签,))；无 target_col 时仅返回指纹张量。
    """
    def __init__(
        self,
        csv_file: str,
        splitter: str,
        smiles_col: str,
        target_col: str = None,
        missing_label: str = None,
        target_transform: t.Union[str, t.List[str]] = None,
        fp_type: str = 'morgan_count',
        **kwargs
    ) -> None:
        super().__init__(
            csv_file,
            splitter=splitter,
            smiles_col=smiles_col,
            target_col=target_col,
            target_transform=target_transform,
            missing_label=missing_label,
            **kwargs
        )
        # 校验指纹类型并取对应的生成器与位长
        assert fp_type in get_available_features_generators()
        self.fp_type = fp_type
        self.fp_generator = get_features_generator(self.fp_type)
        self.fp_numbits = FP_BITS_DICT[self.fp_type]
        self.missing_label = missing_label
    
    def __getitem__(self, index):
        # 取该行单个 SMILES
        smiles = self.df.loc[index][self.smiles_col]
        try:
            fp = torch.LongTensor(self.fp_generator(Chem.MolFromSmiles(smiles)))
        except Exception as _:
            # 解析或计算失败时回退为 fp_numbits 位全零指纹
            fp = torch.zeros(self.fp_numbits).long()

        if self.target_col is not None: 
            target = self.df.loc[index][self.target_col]
            target = (target, )
            return fp, target
        else:
            return fp 
