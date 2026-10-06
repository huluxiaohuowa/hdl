# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/optims/nadam.py
# 说明：自定义优化器
# 模块功能：自定义 NAdam 优化器，把 Nesterov 动量与 Adam 的自适应矩估计结合。
import torch as torch
from torch import optim


class Nadam(optim.Adam):
    """
    NAdam：对一阶矩做偏差修正与动量延迟（momentum delay），再叠加 Nesterov 前瞻项得到更新方向。
    Adaptive moment with Nesterov gradients.

    http://cs229.stanford.edu/proj2015/054_report.pdf

    Parameters
    ----------
    params
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr
        learning rate (default: 1e-3)
    betas
        coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps
        term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay
        weight decay (L2 penalty) (default: 0)
    decay
        a decay scheme for `betas[0]`.
        Default: :math:`\\beta * (1 - 0.5 * 0.96^{\\frac{t}{250}})`
        where `t` is the training step.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 decay=lambda x, t: x * (1. - .5 * .96 ** (t / 250.))):
        """Args: params 待优化参数；lr 学习率；betas=(beta1, beta2) 一阶/二阶矩衰减系数；eps 分母稳定项；
        weight_decay L2 权重衰减；decay(beta1, t) 随步数 t 衰减 beta1 的方案，默认 beta1*(1-0.5*0.96^(t/250))。
        """
        super().__init__(params, lr, betas, eps, weight_decay)
        self.decay = decay

    def step(self, closure=None):
        """对每个参数张量累积一阶/二阶矩滑动平均，构造 Nesterov 修正后的更新方向并走一步。

        Args:
            closure: 可选的重新计算损失的闭包。

        Returns:
            closure 给出的 loss；closure 为 None 时返回 None。
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                # 首次更新前初始化步数 t、一阶矩 exp_avg、二阶矩 exp_avg_sq 与 beta1 的累积乘积
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Beta1 accumulation
                    state['beta1_cum'] = 1.

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # 按 decay 方案取当前步与下一步的 beta1，beta1_cum 为其累积乘积（偏差修正与动量延迟用）
                beta1_t = self.decay(beta1, state['step'])
                beta1_tp1 = self.decay(beta1, state['step'] + 1.)
                beta1_cum = state['beta1_cum'] * beta1_t

                # g_hat_t/m_hat_t 分别是当前梯度与动量滑动平均的偏差修正（bias correction）形式
                g_hat_t = grad / (1. - beta1_cum)
                exp_avg.mul_(beta1).add_(1. - beta1, grad)
                m_hat_t = exp_avg / (1. - beta1_cum * beta1_tp1)

                # v_hat_t 为平方梯度的偏差修正二阶矩；m_bar_t 组合出 Nesterov 前瞻动量
                exp_avg_sq.mul_(beta2).addcmul_(1. - beta2, grad, grad)
                v_hat_t = exp_avg_sq / (1. - beta2 ** state['step'])
                m_bar_t = (1. - beta1) * g_hat_t + beta1_tp1 * m_hat_t

                # 参数沿 -lr * m_bar_t / (sqrt(v_hat_t) + eps) 方向更新
                denom = v_hat_t.sqrt().add_(group['eps'])
                p.data.addcdiv_(-group['lr'], m_bar_t, denom)
                state['beta1_cum'] = beta1_cum

        return loss