import subprocess


def get_num_lines(file):
    """
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