import torch
from torch import Tensor


class BPMLLLoss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean', bias=(1, 1)):
        super(BPMLLLoss, self).__init__()
        self.bias = bias
        self.reduction = reduction
        assert len(self.bias) == 2 \
            and all(map(lambda x: isinstance(x, int) and x > 0, bias)), \
            "bias must be positive integers"

    def forward(self, c: Tensor, y: Tensor) -> Tensor:
        r"""
        compute the loss, which has the form:
        L = \sum_{i=1}^{m} \frac{1}{|Y_i| \cdot |\bar{Y}_i|} \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
        :param c: prediction tensor, size: batch_size * n_labels
        :param y: target tensor, size: batch_size * n_labels
        :return: size: scalar tensor
        """
        y = y.float()
        y_bar = -y + 1
        y_norm = torch.pow(y.sum(dim=(1,)), self.bias[0])
        y_bar_norm = torch.pow(y_bar.sum(dim=(1,)), self.bias[1])
        assert torch.all(y_norm != 0) or torch.all(y_bar_norm != 0), \
            "an instance cannot have none or all the labels"
        loss = 1 / torch.mul(y_norm, y_bar_norm) \
            * self.pairwise_sub_exp(y, y_bar, c)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'none':
            return loss  

    def pairwise_sub_exp(self, y: Tensor, y_bar: Tensor, c: Tensor) -> Tensor:
        r"""
        compute \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
        """
        truth_matrix = y.unsqueeze(2).float() @ y_bar.unsqueeze(1).float()
        exp_matrix = torch.exp(c.unsqueeze(1) - c.unsqueeze(2))
        return (torch.mul(truth_matrix, exp_matrix)).sum(dim=(1, 2))
