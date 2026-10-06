# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/metric_loss/metric.py
# 说明：损失函数与评估指标
# 模块功能：评估指标（classification/regression metrics）实现与按名称取指标函数的工厂 get_metric。
import math
from typing import Callable, List, Union
from functools import partial

import numpy as np
from sklearn.metrics import (
    auc, mean_absolute_error, mean_squared_error,
    precision_recall_curve, r2_score,
    roc_auc_score, accuracy_score, log_loss, matthews_corrcoef,
    # top_k_accuracy_score
) 
import torch
import torch.nn as nn
import scipy


def prc_auc(targets: List[int], preds: List[float]) -> float:
    """
    精确率-召回率曲线下面积（PRC-AUC），输入为二值标签与正类概率。
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def bce(targets: List[int], preds: List[float]) -> float:
    """
    二分类交叉熵（binary cross entropy）：假定 preds 已经过 sigmoid，直接对概率求 BCE 并取均值。
    Computes the binary cross entropy loss.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed binary cross entropy.
    """
    # Don't use logits because the sigmoid is added in all places except training itself
    bce_func = nn.BCELoss(reduction='mean')
    loss = bce_func(target=torch.Tensor(targets), input=torch.Tensor(preds)).item()

    return loss


def rmse(targets: List[float], preds: List[float]) -> float:
    """
    均方根误差（RMSE）。
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))


def mse(targets: List[float], preds: List[float]) -> float:
    """
    均方误差（MSE）。
    Computes the mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    return mean_squared_error(targets, preds)


def accuracy(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
    准确率：preds 的每个元素是列表时按最大概率取类别（多分类），否则按 threshold 二值化（二分类）。
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.

    Alternatively, computes accuracy for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed accuracy.
    """
    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction

    return accuracy_score(targets, hard_preds)


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""
    # 决定系数（R²）：对 x、y 做一元线性回归后取相关系数的平方

    _, _, r_value, _, _ = scipy.stats.linregress(x, y)
    return r_value ** 2


def mcc(y_true, y_pred):
    """马修斯相关系数（Matthews correlation coefficient, MCC）：y_true 直接转 int，y_pred 按 >=0.5 二值化后计算。"""
    y_true = np.array(y_true).astype(int)
    # y_true = np.where(y_true == 1, 1, -1).astype(int)
    y_pred = np.array(y_pred)
    y_pred = (y_pred >= 0.5).astype(int)

    return matthews_corrcoef(y_true, y_pred)


def topk(y_true, y_pred, k=1):
    """Top-k 命中率：每行按预测概率降序取前 k 个类别索引，统计真实标签命中数占总样本数的比例。"""

    y_true = np.array(y_true).astype(int)

    y_pred = np.array(y_pred)

    sorted_pred = np.argsort(y_pred, axis=1, kind='mergesort')[:, ::-1]
    hits = (y_true == sorted_pred[:, :k].T).any(axis=0)
    num_hits = np.sum(hits)

    return num_hits / len(y_true) 


def get_metric(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    r"""
    按名称返回评估指标函数（工厂）。键为指标名，返回可调用对象 f(y_true, y_pred) -> float：
    auc/prc-auc/rmse/mse/mae/r2/acc/ce/bce 中，auc、mae、r2、ce 直接返回 sklearn 实现；
    topk/top3/top5/top10 返回带固定 k 的偏函数（partial）；未匹配的键抛 ValueError。
    Gets the metric function corresponding to a given metric name.

    Supports:

    * :code:`auc`: Area under the receiver operating characteristic curve
    * :code:`prc-auc`: Area under the precision recall curve
    * :code:`rmse`: Root mean squared error
    * :code:`mse`: Mean squared error
    * :code:`mae`: Mean absolute error
    * :code:`r2`: Coefficient of determination R\ :superscript:`2`
    * :code:`accuracy`: Accuracy (using a threshold to binarize predictions)
    * :code:`cross_entropy`: Cross entropy
    * :code:`binary_cross_entropy`: Binary cross entropy

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    if metric == 'mcc':
        return mcc

    if metric == 'rsquared':
        return rsquared

    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mse':
        return mse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    if metric == 'acc':
        return accuracy

    if metric == 'ce':
        return log_loss

    if metric == 'bce':
        return bce
    
    if metric == 'topk':
        return topk 
    
    if metric == 'top3':
        return partial(topk, k=3)
    
    if metric == 'top5':
        return partial(topk, k=5)
    
    if metric == 'top10':
        return partial(topk, k=10)

    raise ValueError(f'Metric "{metric}" not supported.')