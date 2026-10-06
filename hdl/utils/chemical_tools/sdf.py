# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/utils/chemical_tools/sdf.py
# 说明：化学结构文件与分子查询工具
# 模块功能：把 SDF 结构文件逐条读出，转成含 SMILES 与性质列的 pandas DataFrame
from rdkit import Chem
import pandas as pd


def sdf2df(
    sdf_file,
    id_col: str = 'Molecule Name',
    target_col: str = 'Average △G (kcal/mol)'
):
    """
    读取 SDF（Structure-Data File）并汇总表为 DataFrame。
    Args:
        sdf_file: SDF 文件路径，按分子块（mol block）逐条读取。
        id_col: 作为标识列的名称，读出后重命名为 name。
        target_col: 作为标签列的性质字段名，读出后重命名为 y。
    Returns:
        pandas DataFrame：每行一个分子，列为 SDF 原有性质字段加上新增的 smiles、y、name。
    """
    supp = Chem.SDMolSupplier(sdf_file)
    mol_dict_list = []
    for mol in supp:
        # 每个分子块：取规范化 SMILES 与全部性质字段
        smiles = Chem.MolToSmiles(mol)
        mol_dict = mol.GetPropsAsDict()
        mol_dict['smiles'] = smiles
        mol_dict_list.append(mol_dict)
        # 把指定的性质列改名为建模通用列名 y / name（改的是同一个 dict 引用）
        mol_dict['y'] = mol_dict.pop(target_col)
        mol_dict['name'] = mol_dict.pop(id_col)
    df = pd.DataFrame(mol_dict_list)
    return df