# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/utils/general/glob.py
# 说明：通用运行与路径工具
import subprocess


def get_num_lines(file):
    """
    用外部 wc 命令统计行数：subprocess 执行 wc -l 并取输出的第一个字段转 int，故行数按换行符计数（末行没有换行符时不计入），文件不存在时 wc 的非零退出码抛 CalledProcessError。
    Get the number of lines in a given file.
    Args:
        file (str): The path to the file.
    Returns:
        int: The number of lines in the file.
    """

    num_lines = subprocess.check_output(
        ['wc', '-l', file]
    ).split()[0]
    return int(num_lines)


def str_from_line(file, line, split=False):
    """
    用外部 sed 只抽取文件的一行并可选取首列：命令为 sed "Nq;d"（N = line+1，故入参 line 是 0 基行号），读到目标行即退出、不整篇读入；
    输出是 bytes 时 decode 后 strip 掉行尾换行与空白；split=True 时按任意连续空白切分只返回第一段（第一列）。
    文件不存在时 sed 的非零退出码抛 CalledProcessError，目标行号越界则返回空字符串。
    Extracts a specific line from a file and optionally splits the line.
    Args:
        file (str): The path to the file from which to extract the line.
        line (int): The line number to extract (0-based index).
        split (bool, optional): If True, splits the line at the first space or tab and returns the first part. Defaults to False.
    Returns:
        str: The extracted line, optionally split at the first space or tab.
    """
    smi = subprocess.check_output(
        # ['sed','-n', f'{str(i+1)}p', file]
        ["sed", f"{str(line + 1)}q;d", file]
    )
    if isinstance(smi, bytes):
        smi = smi.decode().strip()
    if split:
        if ' ' or '\t' in smi:
            smi = smi.split()[0]
    return smi