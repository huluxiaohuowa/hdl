# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/dl/fp.py
# 说明：深度学习张量与模型辅助工具
# 模块功能：从 SMILES 生成分子指纹（molecular fingerprint）/描述符的函数集，并用 fp_dict 按名称分发。
from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


__all__ = [
    'get_fp',
]


def get_rdnorm_fp(smiles):
    """归一化 RDKit 2D 描述符：用 descriptastorus 的 RDKit2DNormalized 处理 SMILES，取 process 返回值除首项外的数值转 numpy 数组。"""
    from descriptastorus.descriptors import rdNormalizedDescriptors
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]
    arr = np.array(features)
    return arr


def get_maccs_fp(smiles):
    """MACCS 结构键指纹：解析 SMILES 后把置位的键索引写成 0/1 数组（长度 167），解析异常时打印并返回全零。"""
    arr = np.zeros(167)
    try:
        mol = Chem.MolFromSmiles(smiles)
        vec = MACCSkeys.GenMACCSKeys(mol)
        bv = list(vec.GetOnBits())
        arr[bv] = 1
    except Exception as e:
        print(e)
    return arr


def get_morgan_fp(smiles):
    """Morgan 指纹（位向量型）：半径 2、1024 位，把置位索引写成 0/1 的 numpy 数组。"""
    mol = Chem.MolFromSmiles(smiles)
    vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    bv = list(vec.GetOnBits())
    arr = np.zeros(1024)
    arr[bv] = 1
    return arr


# 名称到指纹生成函数的注册表：键 rdnorm/maccs/morgan，值为接受 smiles 的可调用对象
fp_dict = {
    'rdnorm': get_rdnorm_fp,
    'maccs': get_maccs_fp,
    'morgan': get_morgan_fp
}


def get_fp(smiles, fp='maccs'):
    """按 fp 名称（默认 maccs）从 fp_dict 取对应函数，返回 smiles 的分子指纹向量。"""
    return fp_dict[fp](smiles)