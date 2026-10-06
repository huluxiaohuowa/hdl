# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/show/pbar.py
# 说明：进度条与绘图展示工具
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
    判断当前进程是否跑在 Jupyter 内核里：只看 sys.argv[0] 是否为 ipykernel_launcher.py；本模块在导入时就用它的返回值决定 tqdm/trange/tnrange 取自 notebook 版还是控制台版。
    
        Returns:
            bool: True if running in Jupyter notebook, False otherwise.
    """

    which = True if 'ipykernel_launcher.py' in sys.argv[0] else False
    return which

def in_docker():
    """Check if the code is running inside a Docker container.
    判断是否运行在 Docker 容器内：容器根目录会有镜像不携带的 /.dockerenv 标记文件，据此用 osp.exists 探测（只读取该路径是否存在，不改动文件系统）。
    
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