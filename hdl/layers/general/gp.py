# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/layers/general/gp.py
# 说明：通用神经网络层
# 模块功能：精确高斯过程（Exact Gaussian Process, GP）回归模型层，基于 gpytorch 实现，
#           用于在输入特征上给出带预测方差的联合正态分布输出。

# import torch
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    """精确高斯过程模型：继承 gpytorch.models.ExactGP（精确 GP 后验推断基类）。
    输入 train_x [num_data, input_dim]、train_y [num_data] 与观测似然 likelihood；
    前向对任意输入返回多元正态分布 MultivariateNormal（均值 + 协方差），可直接用于预测方差估计。
    """
    def __init__(self, train_x, train_y, likelihood):
        # 基类保存训练集与似然，供精确 GP 后验（边际似然最大化）使用
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # 均值先验：常数均值，每个输出维度学习一个可训练偏置
        self.mean_module = gpytorch.means.ConstantMean()
        # 协方差先验：RBF（径向基/平方指数）核 + ScaleKernel 可学习输出尺度；
        # RBF 核使函数值平滑变化，长度尺度（lengthscale）由 gpytorch 自动学习
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        """前向：把输入映射为高斯过程的（先验/预测）联合正态分布。
        Args:
            x: [num_data, input_dim] 输入特征
        Returns:
            MultivariateNormal：loc 形状 [num_data]，covariance_matrix 形状 [num_data, num_data]
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)