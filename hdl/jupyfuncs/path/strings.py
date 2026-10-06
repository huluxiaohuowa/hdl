# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/path/strings.py
# 说明：路径与字符串处理工具
import re
import subprocess


def get_n_tokens(
    paragraph,
    model: str = None
):
    """Get the number of tokens in a paragraph using a specified model.
    估算文本词元数：model 为 None 时走内置的 CJK 粗略估计——用正则把中日韩（及 h/H）字符替换成 ' a '，
    再按空白切分取段数，即约等于 CJK 字符个数加原有英文词数；给出模型名时改用 tiktoken.encoding_for_model(model)
    的真实词表编码后取长度（首次调用才 import tiktoken，需要联网下载词表时会在此发生）。
    
    Args:
        paragraph (str): The input paragraph to tokenize.
        model (str): The name of the model to use for tokenization. If None, a default CJK tokenization will be used.
    
    Returns:
        int: The number of tokens in the paragraph based on the specified model or default CJK tokenization.
    """
    if model is None:
        cjk_regex = re.compile(u'[\u1100-\uFFFDh]+?')
        trimed_cjk = cjk_regex.sub( ' a ', paragraph, 0)
        return len(trimed_cjk.split())
    else:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(paragraph))
        return num_tokens


def str_from_line(file, line, split=False):
    """Retrieve a specific line from a file and process it.
    用外部 sed 只取文件的一行：执行 sed "Nq;d"（N = line+1，行号从 1 开始，故入参 line 是 0 基索引），
    读到第 N 行即退出，因此大文件不必整篇载入；输出为 bytes 时 decode 并 strip 掉行尾换行与空白。
    split=True 时再按任意空白切分只返回首段（取第一列，常用于 SMILES 文件只留结构式）；
    此处的内层判断只要第一个字面量为真即成立，故 split=True 必走切分分支。
    
    Args:
        file (str): The path to the file.
        line (int): The line number to retrieve (starting from 0).
        split (bool, optional): If True, split the line by space or tab and return the first element. Defaults to False.
    
    Returns:
        str: The content of the specified line from the file.
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


def splitted_strs_from_line(
    file: str,
    idx: int
) -> list:
    """Return a list of strings obtained by splitting the line at the specified index from the given file.
    取文件指定行的所有字段：调 str_from_line 用外部 sed 抽出第 idx 行（0 基索引，split 未开启故保留整行），再 .split() 按任意连续空白切成词元列表；
    空行、空文件或 idx 越界都得到空列表，每列都是 str，列数由该行决定（show/plot.py 把指标日志的每行按 [自变量, 指标名, 指标值] 三列使用）。
    
        Args:
            file (str): The file path.
            idx (int): The index of the line to split.
    
        Returns:
            List: A list of strings obtained by splitting the line at the specified index.
    """
    return str_from_line(file, idx).split()
