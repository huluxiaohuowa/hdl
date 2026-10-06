# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/dl/uncs.py
# 说明：深度学习张量与模型辅助工具
"""UNCERTAINTY SAMPLING
不确定性采样（uncertainty sampling）度量库：对模型输出的概率数组计算四种不确定性得分（最小置信度 least confidence、
置信度间隔 margin、置信度比值 ratio、归一化信息熵 entropy），并由 get_prob_unc 按名称取用，供主动学习（active learning）挑选样本。

Uncertainty Sampling examples for Active Learning in PyTorch

It contains four Active Learning strategies:
1. Least Confidence Sampling
2. Margin of Confidence Sampling
3. Ratio of Confidence Sampling
4. Entropy-based Sampling

"""
from copy import deepcopy

import numpy as np

__all__ = [
    "get_prob_unc",
]


def least_conf_unc(prob_array: np.ndarray) -> np.ndarray:
    """Least confidence uncertainty
    最小置信度（least confidence）不确定性：用 1 减去最大概率，再乘 n/(n-1) 归一化（n 为类别数）。

    .. math::
        \phi_{L C}(x)=\left(1-P_{\theta}\left(y^{*} \mid x\right)\right) \times \frac{n}{n-1}

    Args:
        prob_array (np.array): a 1D or 2D array of probabilities
 
    Returns:
        np.ndarray: the uncertainty value(s)
    """
    # 1D 输入取全局最大概率的位置，2D 输入用（行索引, 每行最大概率列）做高级索引
    if prob_array.ndim == 1:
        indices = prob_array.argmax()
    else:
        indices = (
            np.arange(prob_array.shape[0]),
            prob_array.argmax(-1)
        )
    num_labels = prob_array.shape[-1]
    uncs = (1 - prob_array[indices]) * (num_labels / (num_labels - 1))
    return uncs


def margin_conf_unc(prob_array: np.ndarray) -> np.ndarray:
    """The margin confidence uncertainty
    置信度间隔不确定性（margin of confidence）：1 减去最大概率与次大概率之差，间隔越小越不确定。

    .. math:: 
        \phi_{M C}(x)=1-\left(P_{\theta}\left(y_{1}^{*} \mid x\right)-P_{\theta}\left(y_{2}^{*} \mid x\right)\right)

    Args:
        prob_array (np.array): a 1D or 2D probability array from an NN.  

    Returns:
        np.array: the uncertainty value(s)
    """
    probs = deepcopy(prob_array)
    probs.sort(-1)
    diffs = probs[..., -1] - probs[..., -2]
    return 1 - diffs


def ratio_conf_unc(prob_array: np.ndarray) -> np.ndarray:
    """Ratio based uncertainties
    置信度比值不确定性（ratio of confidence）：次大概率除以最大概率，比值越接近 1 越不确定。

    .. math::
            \phi_{R C}(x)=P_{\theta}\left(y_{2}^{*} \mid x\right) / P_{\theta}\left(y_{1}^{*} \mid x\right)

    Args:
        prob_array (np.array): a 1D or 2D probability array

    Returns:
        np.array: the uncertainty value(s)
    """
    probs = deepcopy(prob_array)
    probs.sort(-1)
    ratio = probs[..., -1] / probs[..., -2]
    return ratio


def entropy_unc(prob_array: np.ndarray) -> np.ndarray:
    """Entropy based uncertainty
    归一化信息熵不确定性（entropy）：以 log2 计算概率分布的熵并除以 log2(类别数)，把结果归一到 0~1。

    .. math::
        \phi_{E N T}(x)=\frac{-\Sigma_{y} P_{\theta}(y \mid x) \log _{2} P_{\theta}(y \mid x)}{\log _{2}(n)}

    Args:
        prob_array (np.array): a 1D or 2D probability array

    Returns:
        np.array: the uncertainty value(s)
    """
    num_labels = prob_array.shape[-1]
    log_probs = prob_array * np.log2(prob_array)
    
    raw_entropy = 0 - np.sum(log_probs, -1)

    normalized_entropy = raw_entropy / np.log2(num_labels)

    return normalized_entropy


# 名称到不确定性度量函数的注册表：least/margin/ratio/entropy，值均接受概率数组
unc_dict = {
    'least': least_conf_unc,
    'margin': margin_conf_unc,
    'ratio': ratio_conf_unc,
    'entropy': entropy_unc,
}


def get_prob_unc(prob_array: np.ndarray, unc: str) -> np.ndarray:
    """按 unc 名称从 unc_dict 取对应度量，计算 prob_array 的不确定性得分，供主动学习（active learning）挑选样本。"""
    return unc_dict[unc](prob_array)
