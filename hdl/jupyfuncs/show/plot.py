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


