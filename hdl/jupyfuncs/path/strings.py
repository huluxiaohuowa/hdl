import re
import subprocess


def get_n_tokens(
    paragraph,
    model: str = None
):
    """Get the number of tokens in a paragraph using a specified model.
    
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
    
        Args:
            file (str): The file path.
            idx (int): The index of the line to split.
    
        Returns:
            List: A list of strings obtained by splitting the line at the specified index.
    """
    return str_from_line(file, idx).split()
