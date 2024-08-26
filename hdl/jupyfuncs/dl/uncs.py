"""UNCERTAINTY SAMPLING

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

    .. math::
        \phi_{L C}(x)=\left(1-P_{\theta}\left(y^{*} \mid x\right)\right) \times \frac{n}{n-1}

    Args:
        prob_array (np.array): a 1D or 2D array of probabilities
 
    Returns:
        np.ndarray: the uncertainty value(s)
    """
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


unc_dict = {
    'least': least_conf_unc,
    'margin': margin_conf_unc,
    'ratio': ratio_conf_unc,
    'entropy': entropy_unc,
}


def get_prob_unc(prob_array: np.ndarray, unc: str) -> np.ndarray:
    return unc_dict[unc](prob_array)
