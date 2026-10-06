# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/controllers/train/trainer_base.py
# 说明：训练流程与 Trainer 实现
# 模块功能：训练器（Trainer）抽象基类，按名称从字典构建模型与优化器、写 TensorBoard 日志、按轮次保存/加载检查点（checkpoint），并规定批量/单轮/整体训练接口
from abc import abstractmethod, ABC
from os import path as osp

import torch
from torch.utils.tensorboard import SummaryWriter

from jupyfuncs.path.glob import makedirs

from hdl.models.optim_dict import OPTIM_DICT
from hdl.models.model_dict import MODEL_DICT
from hdl.models.utils import save_model, load_model
from hdl.metric_loss.loss import get_lossfunc
from hdl.metric_loss.metric import get_metric


class TorchTrainer(ABC):
    """训练器（Trainer）抽象基类：负责训练所需对象的装配（模型、优化器、损失函数、评价指标、TensorBoard logger、设备），并把检查点（checkpoint）存取做成通用方法；具体训练循环由子类实现"""
    def __init__(
        self,
        base_dir,
        data_loader,
        test_loader,
        metrics,
        loss_func,
        model=None,
        model_name=None,
        model_init_args=None,
        ckpt_file=None,
        model_ckpt=None,
        optimizer=None,
        optimizer_name=None,
        optimizer_kwargs=None,
        device=torch.device('cpu'),
    ) -> None:
        """入参：base_dir 输出根目录（同时作为 TensorBoard 日志目录）；data_loader 训练批次迭代器；test_loader 验证/测试批次迭代器；metrics 指标名列表（如 rmse、mae）；loss_func 损失函数名；model/model_name+model_init_args 二选一地传入现成模型或按 MODEL_DICT 名称构造；ckpt_file、model_ckpt 检查点路径（仅保存于属性，加载由子类触发）；optimizer/optimizer_name+optimizer_kwargs 二选一地传入优化器或按 OPTIM_DICT 名称构造；device 计算设备（字符串会转成 torch.device）"""
        super().__init__()
        self.base_dir = base_dir
        self.data_loader = data_loader
        self.test_loader = test_loader

        if metrics is not None:
            # 指标名与指标函数按名从 hdl.metric_loss.metric 查表得到
            self.metric_names = metrics
            self.metrics = [get_metric(metric) for metric in metrics]
        if loss_func is not None:
            # 损失函数名与损失函数对象同样按名查表
            self.loss_name = loss_func
            self.loss_func = get_lossfunc(loss_func)
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        # TensorBoard 写入器，日志目录即 base_dir
        self.logger = SummaryWriter(log_dir=self.base_dir)

        # 损失记录列表：由子类在训练中逐个追加，save() 时整体写入检查点
        self.losses = []

        if model is not None:
            self.model = model
        else:
            assert model_name is not None and model_init_args is not None
            # 未直接给模型时，按模型名从 MODEL_DICT 实例化
            self.model = MODEL_DICT[model_name](**model_init_args)

        self.ckpt_file = ckpt_file
        self.model_ckpt = model_ckpt

        self.model.to(self.device)

        if optimizer is not None:
            self.optimizer = optimizer
        elif optimizer_name is not None and optimizer_kwargs is not None:
            # 参数分组：模型全部参数 + 学习率等优化器超参，再按名字从 OPTIM_DICT 取优化器类
            params = [{
                'params': self.model.parameters(),
                **optimizer_kwargs
            }]
            self.optimizer = OPTIM_DICT[optimizer_name](params)
        else:
            self.optimizer = None
        
        # 累计迭代数与轮次编号：save() 用 epoch_id 命名检查点
        self.n_iter = 0
        self.epoch_id = 0
        # 再次按 metrics 重建指标函数列表（覆盖上面的赋值）
        self.metrics = [get_metric(metric) for metric in metrics]
    
    @abstractmethod
    def load_ckpt(self, ckpt_file, train=False):
        """子类实现：从 ckpt_file 载入检查点；train=True 表示恢复训练态（含优化器与轮次）"""
        raise NotImplementedError

    @abstractmethod
    def train_a_batch(self):
        """子类实现：单批次训练（前向、损失、反向传播、优化器 step）"""
        raise NotImplementedError
    
    @abstractmethod
    def train_an_epoch(self):
        """子类实现：遍历数据集训一轮（epoch），并写日志、存检查点"""
        raise NotImplementedError
    
    @abstractmethod
    def train(self):
        """子类实现：按轮次循环调用 train_an_epoch 完成整体训练"""
        raise NotImplementedError 
    
    def save(self):
        """保存检查点（checkpoint）：在 base_dir/ckpt 下按 model_<epoch_id>.ckpt 存模型、轮次、优化器与损失历史"""
        makedirs(osp.join(self.base_dir, 'ckpt'))
        ckpt_file = osp.join(
            self.base_dir, 'ckpt',
            f'model_{self.epoch_id}.ckpt'
        )    
        
        save_model(
            model=self.model,
            save_dir=ckpt_file,
            epoch=self.epoch_id,
            optimizer=self.optimizer,
            loss=self.losses
        )
    
    def load(self, ckpt_file, train=False):
        """从 ckpt_file 恢复权重：把模型与优化器就地载入；train=True 表示按继续训练的方式加载"""
        load_model(
            save_dir=ckpt_file,
            model=self.model,
            optimizer=self.optimizer,
            train=train 
        )
    
    @abstractmethod
    def predict(self, data_loader):
        """子类实现：对给定 data_loader 逐批推理并返回拼接后的预测数组"""
        raise NotImplementedError


class IterativeTrainer(TorchTrainer):
    """迭代式（iterative）多目标任务训练器基类：在 TorchTrainer 之上额外记录 target_names（各任务/目标列名）"""
    def __init__(
        self,
        base_dir,
        data_loader,
        test_loader,
        metrics,
        target_names,
        loss_func,
        logger
    ) -> None:
        """入参在 TorchTrainer 基础上增加 target_names（任务名列表），logger 未被父类使用；注意父参是按位置传入"""
        super().__init__(
            base_dir,
            data_loader,
            test_loader,
            metrics,
            loss_func,
            logger
        )
        # 以上按位置传参：第 6 个实参 logger 落在 TorchTrainer 的形参 model 上
        self.target_names = target_names

    @abstractmethod
    def train_a_batch(self):
        """子类实现：单批次训练"""
        raise NotImplementedError
    
    @abstractmethod
    def train_an_epoch(self):
        """子类实现：遍历 data_loader 训练一轮（epoch）"""
        raise NotImplementedError
 
    @abstractmethod
    def train(self):
        """子类实现：多轮（epoch）训练总循环"""
        raise NotImplementedError
 