# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/dl/dataframe.py
# 说明：深度学习张量与模型辅助工具
# 模块功能：通用 pandas 工具（删除匹配列名的列、打乱行序），与化学或神经网络逻辑无关。
def rm_index(df):
    """Remove columns with 'Unnamed' in their names from the DataFrame.
    
        Args:
            df (pandas.DataFrame): The input DataFrame.
    
        Returns:
            pandas.DataFrame: DataFrame with columns containing 'Unnamed' removed.
    """
    return df.loc[:, ~df.columns.str.match('Unnamed')]


def rm_col(df, col_name):
    """按 col_name 对列名做正则匹配（df.columns.str.match，前缀匹配），返回丢弃匹配到的列、其余列保持原顺序的新 DataFrame。

    Remove a column from a DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        col_name (str): The name of the column to be removed.
    
    Returns:
        pandas.DataFrame: A new DataFrame with the specified column removed.
    """
    return df.loc[:, ~df.columns.str.match(col_name)]


def shuffle_df(df):
    """用 df.sample(frac=1) 随机重排全部行，再 reset_index(drop=True) 丢弃旧索引，返回行序打乱、列不变的新 DataFrame。

    Shuffle the rows of a DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame to shuffle.
    
    Returns:
        pandas.DataFrame: A new DataFrame with rows shuffled.
    
    Example:
        shuffled_df = shuffle_df(df)
    """
    return df.sample(frac=1).reset_index(drop=True)
