# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/path/glob.py
# 说明：路径与字符串处理工具
import os
import typing as t
import inspect
import fnmatch
import linecache
import time
import gc
import psutil
from os import path as osp
import pathlib
import sys
import importlib
import subprocess
import re
from pathlib import Path

import multiprocess as mp

import importlib.resources as pkg_resources
import json


def in_jupyter():
    """Check if the code is running in a Jupyter notebook.
    判断当前进程是否跑在 Jupyter 内核里：只看 sys.argv[0] 是否为 ipykernel_launcher.py，脚本、单元测试等场景一律返回 False。

        Returns:
            bool: True if running in Jupyter notebook, False otherwise.
    """

    which = True if 'ipykernel_launcher.py' in sys.argv[0] else False
    return which


def in_docker():
    """Check if the code is running inside a Docker container.
    判断是否运行在 Docker 容器内：容器根目录会有镜像不携带的 /.dockerenv 标记文件，据此用 osp.exists 探测（只读该路径，不改动文件系统）。

        Returns:
            bool: True if running inside a Docker container, False otherwise.
    """
    return osp.exists('/.dockerenv')


def get_files(
    dir_path,
    file_types: list = ["txt"]
):
    """Get a list of files with specific file extensions in the given directory path.

    Args:
        dir_path (str): The path to the target directory.

    Returns:
        list: A list of absolute file paths that have file extensions such as .md, .doc, .docx, .pdf, .csv, or .txt.
    """
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        filenames = [f for f in filenames if not f[0] == '.']
        dirnames[:] = [d for d in dirnames if not d[0] == '.']
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(file_types):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
    return file_list


def get_dataset_file(filename):
    """Get dataset file.
    按文件名读取随包分发的 JSON 数据集：用 importlib.resources 定位资源包 jupyfuncs.datasets 下的 filename，open 后 json.load 返回解析结果（只读，不修改磁盘文件）；资源包名与当前目录布局不符时 pkg_resources.path 会直接抛错。

    Args:
        filename (str): The name of the dataset file.

    Returns:
        dict: The data loaded from the dataset file.
    """
    with pkg_resources.path('jupyfuncs.datasets', filename) as file_path:
        with open(file_path, 'r') as f:
            data = json.load(f)
    return data


def recursive_glob(treeroot, pattern):
    """Recursively searches for files matching a specified pattern starting from the given directory.
    递归通配查找：os.walk 遍历 treeroot 下每一层目录，用 fnmatch 只对文件名（不含路径）做 pattern 匹配，命中项拼回所在目录加入结果；结果为列表，路径前缀跟随 treeroot 是相对还是绝对。

    Args:
        treeroot (str): The root directory to start the search from.
        pattern (str): The pattern to match the files against.

    Returns:
        list: A list of file paths that match the specified pattern.
    """
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results


def makedirs(path: str, isfile: bool = False) -> None:
    """Creates a directory given a path to either a directory or file.
    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.
    按需创建多级目录：isfile=True 时先取 dirname 只建文件的父目录；dirname 对纯文件名会得空串，此处空串直接跳过以免 os.makedirs 报错；exist_ok=True 表示目录已存在时静默返回，无返回值。


    Args:
        path (str): Path to a directory or file.
        isfile (bool, optional): Whether the provided path is a directory or file.Defaults to False.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def get_current_dir():
    """Return the current directory path."""
    # 返回本模块 glob.py 所在目录的绝对路径（由当前帧的源文件推导），不是进程的工作目录 os.getcwd()
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def get_num_lines(file):
    """Get the number of lines in a file.
    用外部 wc 命令统计文件行数：subprocess 执行 wc -l 并取输出第一个字段转 int，因此行数按换行符计数（末行无换行符则少计一行），文件不存在时 wc 的报错以 CalledProcessError 抛出。

    Args:
        file (str): The path to the file.

    Returns:
        int: The number of lines in the file.
    """
    num_lines = subprocess.check_output(
        ['wc', '-l', file]
    ).split()[0]
    return int(num_lines)




def chunkify_file(
    fname,
    size=1024 * 1024 * 1000,
    skiplines=-1
):
    """
    把大文本文件按字节切成近似 size 字节且首尾都落在整行边界上的分块表，供多进程各读一段：
    以二进制打开，skiplines>0 时先用 readline 跳过开头若干行（分块从跳过后的偏移开始）；
    每块先记录 chunkStart，再 seek 前进 size 字节并 readline 把块尾推到换行之后，
    返回 [(chunkStart, 块字节数, fname), ...]，越过 os.path.getsize 得到的文件末尾即停止（只读不改文件）。
    function to divide a large text file into chunks each having size ~= size so that the chunks are line aligned

    Params :
        fname : path to the file to be chunked
        size : size of each chink is ~> this
        skiplines : number of lines in the begining to skip, -1 means don't skip any lines
    Returns :
        start and end position of chunks in Bytes
    """
    chunks = []
    fileEnd = os.path.getsize(fname)
    with open(fname, "rb") as f:
        if(skiplines > 0):
            for i in range(skiplines):
                f.readline()

        chunkEnd = f.tell()
        count = 0
        while True:
            chunkStart = chunkEnd
            f.seek(f.tell() + size, os.SEEK_SET)
            f.readline()  # make this chunk line aligned
            chunkEnd = f.tell()
            chunks.append((chunkStart, chunkEnd - chunkStart, fname))
            count += 1

            if chunkEnd > fileEnd:
                break
    return chunks


def parallel_apply_line_by_line_chunk(chunk_data):
    """
    工作进程侧的分块处理器（供 pool.map 调用，func_apply 必须可 pickle）：
    从 chunk_data 前 4 项解出 (起始偏移, 字节长度, 文件路径, func_apply)，其余项作为 func_apply 的附加参数；
    以二进制打开文件、seek 到起始偏移后只读本块字节，utf-8 解码再 splitlines 得到本块行，
    对每行调用 func_apply(line, *func_args)，收集所有非 None 返回值组成 list 返回（None 表示该行被 func_apply 丢弃）。
    function to apply a function to each line in a chunk

    Params :
        chunk_data : the data for this chunk
    Returns :
        list of the non-None results for this chunk
    """
    chunk_start, chunk_size, file_path, func_apply = chunk_data[:4]
    func_args = chunk_data[4:]

    # t1 = time.time()
    chunk_res = []
    with open(file_path, "rb") as f:
        f.seek(chunk_start)
        cont = f.read(chunk_size).decode(encoding='utf-8')
        lines = cont.splitlines()

        for _, line in enumerate(lines):
            ret = func_apply(line, *func_args)
            if(ret != None):
                chunk_res.append(ret)
    return chunk_res


def parallel_apply_line_by_line(
    input_file_path,
    chunk_size_factor,
    num_procs,
    skiplines,
    func_apply,
    func_args,
    fout=None
):
    """
    逐行并行处理大文件的总调度：并行度取 min(num_procs, psutil.cpu_count()) - 1（须 >= 1，Pool 在本进程之外起 worker）；
    先用 chunkify_file 按 chunk_size_factor MB（乘 1024*1024 换成字节）把输入文件切成行对齐的字节块，
    再给每块拼上 func_apply 与 func_args 组成一个任务元组，因此 worker 端是整块读取、块内逐行调用，行不会跨块；
    pool.map 按每批 num_parallel 个任务提交给 parallel_apply_line_by_line_chunk，worker 用 maxtasksperchild=1000 定期换新以防内存持续膨胀；
    结果：fout 传入文件对象时逐条 print 写入该文件（返回值保持为空 list），否则累积进 outputs 返回；
    每批结束后 del + gc.collect 并打印该批耗时，收尾调用 pool.close() 与 pool.terminate()，全过程另打印任务数、块序号与总行数。
    function to apply a supplied function line by line in parallel

    Params :
        input_file_path : path to input file
        chunk_size_factor : size of 1 chunk in MB
        num_procs : number of parallel processes to spawn, max used is num of available cores - 1
        skiplines : number of top lines to skip while processing
        func_apply : a function which expects a line and outputs None for lines we don't want processed
        func_args : arguments to function func_apply
        fout : do we want to output the processed lines to a file
    Returns :
        list of the non-None results obtained be processing each line
    """
    num_parallel = min(num_procs, psutil.cpu_count()) - 1

    jobs = chunkify_file(input_file_path, 1024 * 1024 * chunk_size_factor, skiplines)

    jobs = [list(x) + [func_apply] + func_args for x in jobs]

    print("Starting the parallel pool for {} jobs ".format(len(jobs)))

    lines_counter = 0

    pool = mp.Pool(num_parallel, maxtasksperchild=1000)  # maxtaskperchild - if not supplied some weird happend and memory blows as the processes keep on lingering

    outputs = []
    for i in range(0, len(jobs), num_parallel):
        print("Chunk start = ", i)
        t1 = time.time()
        chunk_outputs = pool.map(
            parallel_apply_line_by_line_chunk,
            jobs[i: i + num_parallel]
        )

        for i, subl in enumerate(chunk_outputs):
            for x in subl:
                if(fout != None):
                    print(x, file=fout)
                else:
                    outputs.append(x)
                lines_counter += 1
        del(chunk_outputs)
        gc.collect()
        print("All Done in time ", time.time() - t1)

    print("Total lines we have = {}".format(lines_counter))

    pool.close()
    pool.terminate()
    return outputs


def get_func_from_dir(score_dir: str) -> t.Tuple[t.Callable, str]:
    """Get function and mode from directory.
    从用户提供的打分脚本目录（或 .py 文件）动态导入入口函数：
    score_dir 以 .py 结尾时取父目录为搜索路径、文件名去后缀为模块名，否则把该目录本身当路径、模块名固定为 main；
    目标目录会被追加进 sys.path（进程级副作用，导入后不回收），再用 importlib 导入模块；
    返回 (module.main, MODE)：模块里定义了 MODE 就用它，取不到时回退 'batch'（异常被吞掉）。

    Args:
        score_dir (str): The directory path containing the function file.

    Returns:
        Tuple[Callable, str]: A tuple containing the main function and the mode.
    """
    if score_dir.endswith('.py'):
        func_dir = pathlib.Path(score_dir).parent.resolve()
        file_name = pathlib.Path(score_dir).stem
    else:
        func_dir = osp.abspath(score_dir)
        file_name = "main"

    sys.path.append(func_dir)
    module = importlib.import_module(file_name)
    try:
        mode = module.MODE
    except Exception as _:
        mode = 'batch'
    return module.main, mode


def find_images_recursive(
    directory,
    extensions=(".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
):
    """递归收集目录下的图片文件：pathlib.rglob("*") 走完整棵树，按小写后缀名是否落在 extensions 里筛选，
    返回 str 形式的文件路径列表（顺序由文件系统给出，未排序；目录不存在时列表为空）。"""
    path = Path(directory)
    return [str(file) for file in path.rglob("*") if file.suffix.lower() in extensions]
