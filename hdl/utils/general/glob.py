import subprocess


def get_num_lines(file):
    num_lines = subprocess.check_output(
        ['wc', '-l', file]
    ).split()[0]
    return int(num_lines)


def str_from_line(file, line, split=False):
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