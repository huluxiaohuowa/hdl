# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/data/dataset/utils.py
# 说明：分子数据集构建与切分
# 模块功能：数据集侧的文本读取工具，把 SMILES 文件/CSV 表读成 Python 列表，供 MoleculeDataset 等数据集构造使用
import typing as t
import csv



def read_smiles(
    data_path: str,
    file_type: str = 'smi',
    smi_col_names: t.List = [],
    y_col_name: str = 'None',
):
    """从文件读取 SMILES（可按列名取多个结构列并附带标签列）。

    Args:
        data_path (str): 数据文件路径。
        file_type (str): 'smi' 按 CSV 逐行取末列作为 SMILES；'csv' 按 smi_col_names 指定的列名取值。
        smi_col_names (list): CSV 模式下的 SMILES 列名列表，非空才走 CSV 分支。
        y_col_name (str): CSV 模式下的标签列名；为 None 时不追加标签。

    Returns:
        list: smi 模式为 [smiles, ...]；csv 模式为 [[smi1, smi2, ..., label], ...]。
        两种 file_type 都不匹配时返回空列表。
    """
    smiles_data = []
    if file_type == 'smi':
        # 默认约定：SMILES 写在每行最后一列，故取 row[-1]
        with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                smiles = row[-1]
                smiles_data.append(smiles)
    elif file_type == 'csv' and any(smi_col_names):
        # for _ in smi_col_names:
        #     smiles_data.append([])
        # 按表头读成字典行，便于用列名（而非位置）取多个结构列
        with open(data_path, 'r') as theFile:
            reader = csv.DictReader(theFile)
            for line in reader:
                # line is { 'workers': 'w0', 'constant': 7.334, 'age': -1.406, ... }
                # e.g. print( line[ 'workers' ] ) yields 'w0'
                smiles_data_i = [line[i] for i in smi_col_names]
                if y_col_name is not None:
                    # 标签追加在行尾，与数据集侧按最后一位取 y 的约定一致
                    smiles_data_i.append(line[y_col_name])
                smiles_data.append(smiles_data_i)
    return smiles_data