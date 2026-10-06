# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/features/utils/utils.py
# 说明：特征工程通用工具
# 模块功能：按扩展名读写已算好的分子特征文件（.npz/.npy/.csv/.pkl/.sdf），并提供 SMILES 逐 token 切分正则
import csv
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools


# SMILES 逐 token 切分用的正则：匹配 [...] 括号原子、Br/Cl、N/O/S/P/F/I、小写芳香原子、
# 各类键符号、%NN 环闭合编号与数字
SMI_REGEX_PATTERN = \
    r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.\|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""


def save_features(path: str, features: List[np.ndarray]) -> None:
    """
    以压缩 .npz 保存特征，数组名为 "features"。

    Saves features to a compressed :code:`.npz` file with array name "features".

    :param path: Path to a :code:`.npz` file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    按扩展名加载分子级特征文件，返回形状 (分子数, 特征维) 的 2D numpy 数组。

    Loads features saved in a variety of formats.

    Supported formats:

    * :code:`.npz` compressed (assumes features are saved with name "features")
    * .npy
    * :code:`.csv` / :code:`.txt` (assumes comma-separated features with a header and with one line per molecule)
    * :code:`.pkl` / :code:`.pckl` / :code:`.pickle` containing a sparse numpy array

    .. note::

       All formats assume that the SMILES loaded elsewhere in the code are in the same
       order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size :code:`(num_molecules, features_size)` containing the features.
    """
    extension = os.path.splitext(path)[1]

    # 依据扩展名选择读取方式：npz 取 "features" 键，npy 直接读数组
    if extension == '.npz':
        features = np.load(path)['features']
    elif extension == '.npy':
        features = np.load(path)
    elif extension in ['.csv', '.txt']:
        # CSV/TXT 每行一个分子，逐元素转 float
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            features = np.array([[float(value) for value in row] for row in reader])
    elif extension in ['.pkl', '.pckl', '.pickle']:
        # pickle 内存的是稀疏矩阵列表，逐条转稠密后压平
        with open(path, 'rb') as f:
            features = np.array([np.squeeze(np.array(feat.todense())) for feat in pickle.load(f)])
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


def load_valid_atom_or_bond_features(path: str, smiles: List[str]) -> List[np.ndarray]:
    """
    加载逐原子/逐键（atom/bond）描述符，返回每个分子一个 2D 数组的列表。
    Args:
        smiles: SMILES 顺序列表，.sdf 分支用它对齐（reindex）读取到的特征行。

    Loads features saved in a variety of formats.

    Supported formats:

    * :code:`.npz` descriptors are saved as 2D array for each molecule in the order of that in the data.csv
    * :code:`.pkl` / :code:`.pckl` / :code:`.pickle` containing a pandas dataframe with smiles as index and numpy array of descriptors as columns
    * :code:'.sdf' containing all mol blocks with descriptors as entries

    :param path: Path to file containing atomwise features.
    :return: A list of 2D array.
    """

    extension = os.path.splitext(path)[1]

    # npz：文件内每个 key 存一个分子的 2D 描述符数组，按 key 顺序收集
    if extension == '.npz':
        container = np.load(path)
        features = [container[key] for key in container]

    elif extension in ['.pkl', '.pckl', '.pickle']:
        # DataFrame 的行对应分子、列对应描述符；按单元格维数选择堆叠或拼接
        features_df = pd.read_pickle(path)
        if features_df.iloc[0, 0].ndim == 1:
            # 每格是一维向量（逐原子一个值），按列 stack 成 (原子数, 描述符数)
            features = features_df.apply(lambda x: np.stack(x.tolist(), axis=1), axis=1).tolist()
        elif features_df.iloc[0, 0].ndim == 2:
            # 每格已是 2D，沿列方向拼接成更宽的特征矩阵
            features = features_df.apply(lambda x: np.concatenate(x.tolist(), axis=1), axis=1).tolist()
        else:
            raise ValueError(f'Atom/bond descriptors input {path} format not supported')

    elif extension == '.sdf':
        # 读 SDF 的性质字段成表，丢掉 ID 与 RDKit Mol 列，用 SMILES 作行索引
        features_df = PandasTools.LoadSDF(path).drop(['ID', 'ROMol'], axis=1).set_index('SMILES')

        # 同一 SMILES 只保留第一条，避免 reindex 产生重复行
        features_df = features_df[~features_df.index.duplicated()]

        # locate atomic descriptors columns
        # 只保留值为逗号分隔字符串的列，即逐原子（atomwise）描述符列
        features_df = features_df.iloc[:, features_df.iloc[0, :].apply(lambda x: isinstance(x, str) and ',' in x).to_list()]
        features_df = features_df.reindex(smiles)
        if features_df.isnull().any().any():
            raise ValueError('Invalid custom atomic descriptors file, Nan found in data')

        # 去掉换行并按逗号切分，转成 float 数组（每个元素对应一个原子）
        features_df = features_df.applymap(lambda x: np.array(x.replace('\r', '').replace('\n', '').split(',')).astype(float))

        # 沿列方向 stack，得到每个分子形状 (原子数, 描述符数) 的 2D 数组
        features = features_df.apply(lambda x: np.stack(x.tolist(), axis=1), axis=1).tolist()

    else:
        raise ValueError(f'Extension "{extension}" is not supported.')

    return features
