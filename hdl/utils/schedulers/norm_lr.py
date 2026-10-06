# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/utils/schedulers/norm_lr.py
# 说明：学习率调度器
# 模块功能：Noam 风格学习率调度——先按步线性 warmup 到 max_lr，再指数衰减到 final_lr。
from typing import List, Union
import numpy as np

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """
    学习率按优化步（step）分两段变化：前 warmup_steps 步从 init_lr 线性升到 max_lr（warmup），
    其后每步乘以 exponential_gamma 指数衰减，直到 final_lr；每个参数组有独立的轮数与学习率列表。
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        初始化调度器：各列表长度必须等于优化器参数组个数，warmup/total 步数由轮数乘每轮步数得到。
        Initializes the learning rate scheduler.
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        # warmup 步数 = warmup 轮数 × 每轮步数；linear_increment 为 warmup 阶段每步的线性增量
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        # 指数衰减因子：在 total_steps - warmup_steps 步内把 max_lr 衰减到 final_lr
        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        # 返回各参数组当前的学习率列表
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        按当前步数选取所处阶段（warmup 线性段 / 衰减段 / 超出总步数），并把结果写回优化器各参数组。
        Updates the learning rate by taking a step.
        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


def build_lr_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    n_epochs: int,
    steps_per_epoch: int,
    lr: float,
) -> _LRScheduler:
    """
    用单参数组的设置构造 NoamLR：init_lr 与 final_lr 均取 lr 的 1/10，max_lr 取 lr。
    Builds a learning rate scheduler.
    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param train_data_size: The size of the training dataset.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[warmup_epochs],
        total_epochs=[n_epochs],
        steps_per_epoch=steps_per_epoch,  # train_data_size // args.batch_size,
        init_lr=[lr / 10],
        max_lr=[lr],
        final_lr=[lr / 10]
    )
