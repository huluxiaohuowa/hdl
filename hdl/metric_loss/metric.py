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
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def bce(targets: List[int], preds: List[float]) -> float:
    """
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
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))


def mse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    return mean_squared_error(targets, preds)


def accuracy(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
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

    _, _, r_value, _, _ = scipy.stats.linregress(x, y)
    return r_value ** 2


def mcc(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    # y_true = np.where(y_true == 1, 1, -1).astype(int)
    y_pred = np.array(y_pred)
    y_pred = (y_pred >= 0.5).astype(int)

    return matthews_corrcoef(y_true, y_pred)


def topk(y_true, y_pred, k=1):

    y_true = np.array(y_true).astype(int)

    y_pred = np.array(y_pred)

    sorted_pred = np.argsort(y_pred, axis=1, kind='mergesort')[:, ::-1]
    hits = (y_true == sorted_pred[:, :k].T).any(axis=0)
    num_hits = np.sum(hits)

    return num_hits / len(y_true) 


def get_metric(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    r"""
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