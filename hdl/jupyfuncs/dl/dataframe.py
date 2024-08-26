def rm_index(df):
    """Remove columns with 'Unnamed' in their names from the DataFrame.
    
        Args:
            df (pandas.DataFrame): The input DataFrame.
    
        Returns:
            pandas.DataFrame: DataFrame with columns containing 'Unnamed' removed.
    """
    return df.loc[:, ~df.columns.str.match('Unnamed')]


def rm_col(df, col_name):
    """Remove a column from a DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        col_name (str): The name of the column to be removed.
    
    Returns:
        pandas.DataFrame: A new DataFrame with the specified column removed.
    """
    return df.loc[:, ~df.columns.str.match(col_name)]


def shuffle_df(df):
    """Shuffle the rows of a DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame to shuffle.
    
    Returns:
        pandas.DataFrame: A new DataFrame with rows shuffled.
    
    Example:
        shuffled_df = shuffle_df(df)
    """
    return df.sample(frac=1).reset_index(drop=True)
