import sys
from os import path as osp
from IPython.core.display import HTML

__all__ = [
    'in_jupyter',
    'tqdm',
    'trange',
    'tnrange',
    'NO_WHITE',
]


NO_WHITE = HTML("""
    <style>
    .jp-OutputArea-prompt:empty {
    padding: 0;
    border: 0;
    }
    </style>
    """)


def in_jupyter():
    """Check if the code is running in a Jupyter notebook.
    
        Returns:
            bool: True if running in Jupyter notebook, False otherwise.
    """

    which = True if 'ipykernel_launcher.py' in sys.argv[0] else False
    return which

def in_docker():
    """Check if the code is running inside a Docker container.
    
        Returns:
            bool: True if running inside a Docker container, False otherwise.
    """
    return osp.exists('/.dockerenv')


if in_jupyter():
    from tqdm.notebook import tqdm
    from tqdm.notebook import trange
    from tqdm.notebook import tnrange
else:
    from tqdm import tqdm
    from tqdm import trange
    from tqdm import tnrange