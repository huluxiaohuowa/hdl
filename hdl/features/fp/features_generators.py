# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/features/fp/features_generators.py
# 说明：分子指纹特征生成
# 模块功能：以名称注册表管理分子指纹生成器（Morgan/ECFP、MACCS、RDKit 2D 描述符等），把 SMILES 或 RDKit Mol 转成固定长度的 numpy 特征向量
from typing import Callable, List, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


# 分子输入类型别名：SMILES 字符串或 RDKit Mol 对象
Molecule = Union[str, Chem.Mol]
# 特征生成器类型别名：输入分子、返回 1D numpy 特征数组的可调用对象
FeaturesGenerator = Callable[[Molecule], np.ndarray]


# 全局注册表：生成器名称 -> 特征生成器函数，按名称取用
FEATURES_GENERATOR_REGISTRY = {}


# 各特征名对应的输出维度（位向量长度/描述符个数），供下游按名称确定特征形状
FP_BITS_DICT = {
    'maccs': 167,
    'morgan': 2048,
    'morgan_count': 2048,
    'rdkit_2d_normalized': 200,
    'rdkit_2d': 200,
}


def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """
    装饰器工厂：把特征生成器按名称登记到全局注册表 FEATURES_GENERATOR_REGISTRY。

    Creates a decorator which registers a features generator in a global dictionary to enable access by name.

    :param features_generator_name: The name to use to access the features generator.
    :return: A decorator which will add a features generator to the registry using the specified name.
    """
    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        """以闭包捕获的 features_generator_name 为键，把 features_generator 写入 FEATURES_GENERATOR_REGISTRY，随后原样返回该函数（仅登记，不包装行为）。"""
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    """
    按名称从注册表取出特征生成器；名称未注册则抛 ValueError。

    Gets a registered features generator by name.

    :param features_generator_name: The name of the features generator.
    :return: The desired features generator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


# 返回当前已注册的特征生成器名称列表（如 morgan/maccs/rdkit_2d 等）
def get_available_features_generators() -> List[str]:
    """Returns a list of names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())


# 扩展连通性指纹（ECFP / Morgan fingerprint）默认半径（2，即 ECFP4）与默认位数（2048）
MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048


@register_features_generator('morgan')
def morgan_binary_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    生成二值（binary）Morgan 指纹。

    Generates a binary Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the binary Morgan fingerprint.
    """
    # 字符串输入先解析为 RDKit Mol 对象
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    # 计算子结构哈希后的位向量（长度 num_bits，默认 2048）
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    # 把 RDKit 位向量转成 numpy 一维 0/1 数组（ConvertToNumpyArray 会按位向量长度重填）
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


@register_features_generator('morgan_count')
def morgan_counts_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    生成计数（counts）型 Morgan 指纹。

    Generates a counts-based Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the counts-based Morgan fingerprint.
    """
    # 字符串输入先解析为 RDKit Mol 对象
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    # 哈希到 num_bits 个桶，每位存该子结构的出现次数（非 0/1）
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    # 转为 numpy 一维计数数组（长度 num_bits）
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


# MACCS 密钥（MACCS keys）结构指纹生成器
@register_features_generator('maccs')
def macss_features_generator(
    mol
) -> np.ndarray:
    """生成 MACCS keys 指纹，返回长度 167 的 0/1 一维 numpy 数组。"""
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    # MACCS keys 固定 167 个子结构键，与 FP_BITS_DICT['maccs'] 一致
    vec = MACCSkeys.GenMACCSKeys(mol)
    # 只取出被置位的下标，再在 167 维全零向量上置 1
    bv = list(vec.GetOnBits())
    arr = np.zeros(167)
    arr[bv] = 1
    return arr


# RDKit 2D 描述符依赖 descriptastorus；缺失时退化为同名占位实现（调用即抛 ImportError）
try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """
        生成 RDKit 2D 描述符特征。

        Generates RDKit 2D features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features.
        """
        # descriptastorus 只接受 SMILES 输入，故先写成含立体信息的 SMILES
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        # process 返回 [SMILES, 200 个描述符]，[1:] 去掉首列只留数值特征
        features = generator.process(smiles)[1:]

        return features

    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
        """
        生成经过归一化的 RDKit 2D 描述符特征。

        Generates RDKit 2D normalized features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        # RDKit2DNormalized 内部按预置分布对 200 个描述符做归一化
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]

        return features
except ImportError:
    # 占位实现：保持注册名可用，但调用时提示需安装 descriptastorus
    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
        raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                          '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D features.')

    # 占位实现：归一化 2D 特征同样依赖 descriptastorus
    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
        """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
        raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                          '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D normalized features.')


@register_features_generator('e3fp')
def e3fp_features_generator(mol: Molecule) -> np.ndarray:
    """
    E3FP 三维分子指纹生成器：当前为占位实现，直接返回 NotImplemented。

    E3FP is a 3D molecular fingerprinting method inspired by Extended Connectivity FingerPrints (ECFP),

    [LINK](https://pubs.acs.org/doi/10.1021/acs.jmedchem.7b00696)
    Axen SD, Huang XP, Caceres EL, Gendelev L, Roth BL, Keiser MJ. 
    A Simple Representation Of Three-Dimensional Molecular Structure. 
    J. Med. Chem. 60 (17): 7393–7409 (2017).

    The source code: https://github.com/keiserlab/e3fp

    :param mol: A molecule(i.e., either a SMILES or an RDKit molecule).
    :return: A 1D numpy array containing the E3FP fingerprints
    """
    return NotImplemented


@register_features_generator('whales')
def whales_features_generator(mol: Molecule) -> np.ndarray:
    """
    WHALES 描述符（WHALES descriptors）生成器：当前为占位实现，直接返回 NotImplemented。

    WHALES descriptor is a Weighted Holistic Atom Localization and Entity Shape (WHALES) 
    descriptors starting from an rdkit supplier file.

    [LINK](https://www.nature.com/articles/s42004-018-0043-x)
    Francesca Grisoni, Daniel Merk, Viviana Consonni, Jan A. Hiss, 
    Sara Giani Tagliabue, Roberto Todeschini & Gisbert Schneider 
    "Scaffold hopping from natural products to synthetic mimetics by 
    holistic molecular similarity", Nature Communications Chemistry 1, 44, 2018.

    The source code: https://github.com/grisoniFr/whales_descriptors

    :param mol: A molecule(i.e., either a SMILES or an RDKit molecule).
    :return: A 2D numpy array containing the WHALES descriptors.
    """
    return NotImplemented


@register_features_generator('selfies')
def selfies_features_generator(mol) -> np.ndarray:
    """
    SELFIES（Self-Referencing Embedded Strings）字符串表示：当前为占位实现，直接返回 NotImplemented。

    Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation

    A main objective is to use SELFIES as direct input into machine learning models,
    in particular in generative models, for the generation of molecular graphs
    which are syntactically and semantically valid.

    [LINK](https://iopscience.iop.org/article/10.1088/2632-2153/aba947)
    Mario Krenn et al 2020 Mach. Learn.: Sci. Technol. 1 045024

    The source code: https://github.com/aspuru-guzik-group/selfies

    :param mol: A molecule(i.e., either a SMILES or an RDKit molecule).
    :return: A 1D numpy array containing the symbols of input molecule in SELFIES style.
    """
    return NotImplemented


# 下面是自定义特征生成器的写法模板（仅字符串常量，不参与执行）
"""
Custom features generator template.

Note: The name you use to register the features generator is the name
you will specify on the command line when using the --features_generator <name> flag.
Ex. python train.py ... --features_generator custom ...

@register_features_generator('custom')
def custom_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    features = np.array([0, 0, 1])

    return features
"""
