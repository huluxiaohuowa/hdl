# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/show/plot.py
# 说明：进度条与绘图展示工具
from os import path as osp
import typing as t
from typing_extensions import Literal

import seaborn as sn
import sklearn
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

from ..path.glob import get_num_lines
from ..path.strings import splitted_strs_from_line 

cm = matplotlib.cm.get_cmap('tab20')
colors = cm.colors
LABEL = Literal[
    'training_size',
    'episode_id',
]


def accuracies_heat(y_true, y_pred, num_tasks):
    """画多任务预测的混淆热力图：y_true/y_pred 为逐样本的任务序号（两者长度必须相等，否则断言失败），
    sklearn 的 confusion_matrix 以 normalize='true' 按真实任务归一化（每行之和为 1，即该任务的预测被分到各任务的比例），行列索引都取 range(num_tasks)（num_tasks 小于最大任务序号时构造 DataFrame 因尺寸不符报错）；
    新建 10x10 英寸画布、字号 1.4、单元格按两位小数标注，只在当前 Figure 上绘制，不保存图片也不返回值。"""
    assert len(y_true) == len(y_pred)
    cm = sklearn.metrics.confusion_matrix(
        y_true, y_pred, normalize='true'
    )
    df_cm = pd.DataFrame(
        cm, range(num_tasks), range(num_tasks)
    )
    plt.figure(figsize=(10, 10))
    sn.set(font_scale=1.4) 
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='.2f')


def get_metrics_curves(
    base_dir,
    ckpts,
    num_points,
    title="Metric Curve",
    metric='accuracy',
    log_file='metrics.log',
    label: LABEL = 'training_size',
    save_dir: str = None,
    figsize=(10, 6)
):
    """把多个检查点（ckpt）的训练指标画成同一张折线图：日志路径为 base_dir/<ckpt>/<log_file>，不存在的打印 WARNING 后跳过该 ckpt；
    逐行只取「切分后恰有 3 列且第 2 列等于 metric」的记录，x 为第 1 列整数（label='training_size'，训练/数据规模）或已收点数递增序号（label='episode_id'），y 为第 3 列浮点指标值，
    而 line_id >= num_points - 1 即中断，所以 num_points 限制的是读取的日志行数而非采样点数（行数来自 wc -l，每行经 sed 子进程取出）。
    绘图按 figsize/dpi=100 新建 Figure，每个 ckpt 一条折线、颜色取 tab20 调色板下标（ckpt 多于 20 条会越界），x 轴标签为 label、y 轴为 metric、标题为 title 并开网格，图例置于图外右上；
    副作用是把 PNG 写入 save_dir（缺省 base_dir/metrics_curves.png）再 plt.show() 弹出显示。"""
    if not save_dir:
        save_dir = osp.join(base_dir, 'metrics_curves.png')
    data_dict = {}
    for ckpt in ckpts:
        log = osp.join(
            base_dir,
            ckpt,
            log_file
        )
        if not osp.exists(log):
            print(f"WARNING: no log file for {ckpt}")
            continue
        data_dict[ckpt] = []
        data_idx = 0
        for line_id in range(get_num_lines(log)):
            line = splitted_strs_from_line(log, line_id)
            if len(line) == 3 and line[1].strip() == metric:
                if label == 'episode_id':
                    x = data_idx
                elif label == 'training_size':
                    x = int(line[0].strip())
                data_dict[ckpt].append(
                    [
                        x,
                        float(line[2].strip())
                    ]
                )
                data_idx += 1
            if line_id >= num_points - 1:
                break
    plt.figure(figsize=figsize, dpi=100)
    # plt.style.use('ggplot')
    plt.title(title)
    for i, (ckpt, points) in enumerate(data_dict.items()):
        points_array = np.array(points).T
        plt.plot(points_array[0], points_array[1], label=ckpt, color=colors[i])
    lg = plt.legend(bbox_to_anchor=(1.2, 1.0), loc='upper right')
    # plt.legend(loc='lower right')
    plt.xlabel(
        label
    )
    plt.ylabel(metric)
    plt.grid(True)
    plt.savefig(
        save_dir,
        format='png', 
        bbox_extra_artists=(lg,), 
        bbox_inches='tight'
    )
    plt.show()


def get_means_vars(
    log_file: str,
    indices: t.List,
    mode: str,
    nears_each: int,
) -> t.List[t.List[int]]:
    """对指标日志做滑动窗口统计，返回 (mean_s, var_s)，两个序列长度都等于 len(indices)：窗口是中心点前后各 nears_each 个位置（含中心共 2*nears_each+1 项），取每行第 3 列的浮点值。
    mode='id' 把 indices 直接当日志的物理行号（0 基，交给 sed 逐行取），mean 为 np.mean、var_s 为 np.std（总体标准差 ddof=0）；窗口越出文件行范围时对应行取不到内容而报错。
    mode='value' 先把整份日志读成数组，按第 1 列训练规模与各 index 的绝对差 argmin 定位最近行再开窗；var_s 额外除以 sqrt(num_points)（num_points=len(indices)，与窗口宽度无关），
    因此它是缩放过的标准差、并不是窗口均值的标准误。"""
    
    mean_s, var_s = [], []
    num_points = len(indices)

    nears_lists = []
    if mode == 'id':

        for index in indices:
            nears = []
            nears.extend(list(range(
                index - nears_each, index + 1 + nears_each
            ))) 
            nears_lists.append(nears)
        
        for nears in nears_lists:
            mean_s.append(np.mean([
                float(splitted_strs_from_line(log_file, line_id)[2])
                for line_id in nears
            ]))
            var_s.append((np.std([
                float(splitted_strs_from_line(log_file, line_id)[2])
                for line_id in nears
            ])))

    elif mode == 'value':
        datas = [
            splitted_strs_from_line(log_file, line_id)
            for line_id in range(get_num_lines(log_file))
        ]
        training_sizes = [[int(data[0]) for data in datas]]
        values = np.array([float(data[2]) for data in datas])

        training_sizes = np.repeat(training_sizes, num_points, 0).T
        diffs = training_sizes - indices

        true_indices = np.argmin(np.abs(diffs), 0)

        true_indices_list = [
            list(range(
                index - nears_each, index + 1 + nears_each
            ))
            for index in true_indices
        ]
        mean_s = [
            np.mean(values[indices])
            for indices in true_indices_list
        ]
        var_s = [
            np.std(values[indices])
            for indices in true_indices_list
        ]
        var_s = np.array(var_s) / np.sqrt(num_points)
 
    return mean_s, var_s 


def get_metrics_bars(
    base_dir,
    ckpts,
    title="Metric Bars",
    training_sizes: t.List = [],
    episide_ids: t.List = [],
    nears_each: int = 5,
    pretrained_num: int = 0,
    x_diff: bool = False,
    metric='accuracy',
    log_file='metrics.log',
    label: LABEL = 'training_size',
    save_dir: str = None,
    figsize=(10, 6),
    bar_ratio=0.8,
    minimum=0.0,
    maximum=1.0
):
    """把多个 ckpt 的指标画成分组柱状图（每柱带误差棒）：label='training_size' 时 x 标签取 training_sizes 并按 mode='value' 统计，label='episode_id' 时取 episide_ids 走 mode='id'，num_points 即标签个数；
    每个 ckpt 从 base_dir/<ckpt>/<log_file> 交 get_means_vars 取 (mean_s, var_s)，日志缺失打印 WARNING 跳过该 ckpt；此处 metric 只写进 y 轴文字，日志列并不按指标名筛选（柱高恒取每行第 3 列，与 get_metrics_curves 不同）。
    分组几何：num_strategies=len(ckpts)，单柱宽 width=bar_ratio/num_strategies，x 先整体左移 (bar_ratio-width)/2 再按柱子序号右移 width*point_idx，使同一 x 位置的多个 ckpt 并排；
    yerr 直接用 var_s（'value' 模式下已被除以 sqrt(num_points)），x_diff=True 时 x 标签统一减去 pretrained_num，y 轴范围锁在 [minimum, maximum]，图例置于图外右上；
    副作用是新建 figsize/dpi=100 的 Figure、把 PNG 写入 save_dir（缺省 base_dir/metrics_bars.png）并 plt.show() 显示。"""

    if not save_dir:
        save_dir = osp.join(base_dir, 'metrics_bars.png')
    
    x_labels, num_points = [], 0
    if label == 'training_size':
        num_points = len(training_sizes)
        x_labels = training_sizes
        mode = 'value'
    elif label == 'episode_id':
        num_points = len(episide_ids)
        x_labels = episide_ids
        mode = 'id'
    
    x = np.arange(num_points)
    num_strategies = len(ckpts)
    total_width = bar_ratio
    width = total_width / num_strategies  
    x = x - (total_width - width) / 2
        
    if not save_dir:
        save_dir = osp.join(base_dir, 'metrics.png')

    data_dict = {}
    for ckpt in ckpts:

        log = osp.join(
            base_dir,
            ckpt,
            log_file
        )
        if not osp.exists(log):
            print(f"WARNING: no log file for {ckpt}")
            continue
        
        # print(x_labels)
        mean_s, var_s = get_means_vars(
            log_file=log,
            indices=x_labels,
            mode=mode,
            nears_each=nears_each
        )
        data_dict[ckpt] = (mean_s, var_s)
    
    if x_diff:
        x_labels = np.array(x_labels, dtype=np.int) - pretrained_num
 
    plt.figure(figsize=figsize, dpi=100)
    plt.title(title)
    ax = plt.gca()
    ax.set_ylim([minimum, maximum])
    
    for point_idx, (ckpt, datas) in enumerate(data_dict.items()):
        mean_s, var_s = datas
        plt.bar(
            x + width * point_idx,
            mean_s, 
            width=width,
            yerr=var_s,
            tick_label=x_labels,
            label=ckpt,
            color=colors[point_idx]
        )
    
    lg = plt.legend(bbox_to_anchor=(1.2, 1.0), loc='upper right')
    # plt.legend(loc='lower right')
    plt.xlabel(
        label
    )
    plt.ylabel(metric)
    plt.grid(True)
 
    plt.savefig(
        save_dir,
        format='png', 
        bbox_extra_artists=(lg,), 
        bbox_inches='tight'
    )
    plt.show()


