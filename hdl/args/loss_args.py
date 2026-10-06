# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/args/loss_args.py
# 说明：损失与训练参数定义
from tap import Tap


class LossArgs(Tap):
    """损失参数：reduction 指定损失的归约方式（mean/sum/none），继承 tap.Tap 供命令行解析。"""
    reduction: str = 'mean'