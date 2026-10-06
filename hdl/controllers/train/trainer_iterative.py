# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/controllers/train/trainer_iterative.py
# 说明：训练流程与 Trainer 实现
# 模块功能：多任务迭代式训练器（Trainer）实现与指纹（fingerprint）特征训练入口，含按任务过滤缺失标签、逐任务损失日志与检查点（checkpoint）保存
from os import path as osp
import typing as t

# import numpy as np
import torch
# from torch import nn
# from torch.optim import Adam
from torch import nn
# import pandas as pd

from hdl.models.utils import load_model, save_model
from hdl.models.model_dict import MODEL_DICT
from hdl.models.optim_dict import OPTIM_DICT
# from hdl.models.linear import MMIterLinear
from hdl.features.fp.features_generators import FP_BITS_DICT 
from hdl.data.dataset.fp.fp_dataset import FPDataset
from hdl.data.dataset.loaders.general import Loader
from jupyfuncs.show.pbar import tnrange, tqdm
from jupyfuncs.path.glob import makedirs
from jupyfuncs.dl.tensor import get_valid_indices
from hdl.metric_loss.loss import mtmc_loss
from hdl.controllers.train.trainer_base import IterativeTrainer


class MMIterTrainer(IterativeTrainer):
    """多任务（multi-task）迭代式训练器（Trainer）：输入为多组分子指纹（fingerprint）张量，逐任务带有效标签掩码地计算损失，每轮（epoch）存一次检查点"""
    def __init__(
        self,
        base_dir,
        data_loader,
        target_names,
        loss_func,
        missing_labels=[],
        task_weights=None,
        test_loder=None,
        metrics=None,
        model=None,
        model_name=None,
        model_init_args=None,
        ckpt_file=None,
        optimizer=None,
        optimizer_name=None,
        optimizer_kwargs=None,
        # logger=None,
        device=torch.device('cpu'),
        parallel=False,
    ):
        """入参：target_names 任务名列表；missing_labels 每个任务的缺失/无效标签取值（长度需等于 target_names）；task_weights 各任务损失权重（默认全 1）；loss_func 可为单个名字或按任务的列表；ckpt_file 非空则先恢复训练态；parallel=True 时用 nn.DataParallel 包模型；test_loder/metrics 透传给父类"""
        super().__init__(
            base_dir=base_dir,
            data_loader=data_loader,
            test_loader=test_loder,
            metrics=metrics,
            loss_func=loss_func,
            target_names=target_names,
            # logger=logger
        )
        # 校验缺失标签数量与任务数一致，缺失标签本身只存为属性
        assert len(missing_labels) == len(target_names)
        self.epoch_id = 0
        if model is not None:
            self.model = model
        else:
            assert model_name is not None and model_init_args is not None
            # 按模型名从 MODEL_DICT 构建模型
            self.model = MODEL_DICT[model_name](**model_init_args)
        self.model.to(device)
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            assert optimizer_name is not None and optimizer_kwargs is not None
            # 模型全部参数与超参（lr、weight_decay 等）打包成一个参数组，再按名从 OPTIM_DICT 取优化器
            params = [{
                'params': self.model.parameters(),
                **optimizer_kwargs
            }]
            self.optimizer = OPTIM_DICT[optimizer_name](params)
        
        if ckpt_file is not None:
            # 恢复训练态：以 train=True 载入模型权重、优化器状态与已完成的轮次号
            self.model, self.optimizer, self.epoch_id, _ = load_model(
                save_dir=ckpt_file,
                model=self.model,
                optimizer=self.optimizer,
                train=True
            )
        if self.epoch_id != 0:
            # 已有轮次则从下一轮继续训练
            self.epoch_id += 1
        
        self.metrics = metrics
        self.device = device
        self.missing_labels = missing_labels
        if parallel:
            # 多卡并行包装
            self.model = nn.DataParallel(self.model)
        
        if isinstance(loss_func, str):
            # 单一损失名：所有任务共用同一个损失函数
            self.loss_names = [loss_func] * len(target_names)
        elif isinstance(loss_func, (t.List, t.Tuple)):
            # 逐任务损失名列表
            assert len(loss_func) == len(target_names)
            self.loss_names = loss_func
        
        if task_weights is None:
            # 未给权重时各任务等权
            task_weights = [1] * len(target_names)
        self.task_weights = task_weights
    
    def train_a_batch(self, batch):
        """单批次训练：batch[0] 为各输入指纹张量列表，batch[1] 为目标张量列表，batch[-1] 为原始标签列表；返回该批次总损失张量"""
        # 清空上一批残留梯度
        self.optimizer.zero_grad()
        fps = [x.to(self.device) for x in batch[0]]
        target_tensors = [
            target_tensor.to(self.device) for target_tensor in batch[1]
        ]
        target_list = batch[-1]
        target_valid_dict = {}
        for target_name, target_labels in zip(self.target_names, target_list):

            # 计算该任务的有效样本下标（get_valid_indices 剔除标签为空值 NaN 的样本），只在这些样本上算损失
            valid_indices = get_valid_indices(labels=target_labels)
            valid_indices.to(self.device) 
                
            target_valid_dict[target_name] = valid_indices
        
        # teach=True 走迭代式（teacher forcing）前向，返回 {任务名: 预测} 字典
        y_preds = self.model(fps, target_tensors, teach=True)        

        # 按下标整理各任务的真值与预测值
        # process with y_true
        y_trues = []
        y_preds_list = []
        for target_name, target_tensor, target_labels, loss_name in zip(
            self.target_names, target_tensors, target_list, self.loss_names
        ):
            valid_indices = target_valid_dict[target_name]
            if loss_name in ['ce']:
                # 交叉熵（cross entropy）用原始整数标签，需转 long
                y_true = target_labels[valid_indices].long().to(self.device)
            elif loss_name in ['mse', 'bpmll']:
                # 回归型损失用经过变换的浮点目标张量
                y_true = target_tensor[valid_indices].to(self.device)
            y_pred = y_preds[target_name][valid_indices].to(self.device)
            y_preds_list.append(y_pred)
            y_trues.append(y_true)
        
        # print(y_preds, y_trues)
        # 多任务多分类损失（multi-task multi-class）：individual=True 时同时返回总损失与各任务损失列表
        loss, loss_list = mtmc_loss(
            y_preds=y_preds_list,
            y_trues=y_trues,
            loss_names=self.loss_names,
            individual=True,
            task_weights=self.task_weights,
            device=self.device
        )
        # 每批次把总损失与各任务损失追加写入 base_dir/loss.log（制表符分隔）
        with open(osp.join(self.base_dir, 'loss.log'), 'a') as f:
            f.write(str(loss.item()))
            f.write('\t') 
            for i_loss in loss_list:
                f.write(str(i_loss.item()))
                f.write('\t')
            f.write('\n')
            f.flush()
            
        # 反向传播并更新参数（本文件无梯度裁剪、无学习率调度器）
        loss.backward()
        self.optimizer.step()
 
        return loss
       
    def train_an_epoch(self, epoch_id):
        """一轮（epoch）训练：逐批调用 train_a_batch，结束后把模型、优化器与最后一个批次的损失存为 base_dir/ckpt/model.<epoch_id>.ckpt"""
        for batch in tqdm(self.data_loader):
            loss = self.train_a_batch(
                batch=batch
            )
        makedirs(osp.join(self.base_dir, 'ckpt'))
        self.ckpt_file = osp.join(
            self.base_dir, 'ckpt',
            f'model.{epoch_id}.ckpt'
        )
        # 检查点（checkpoint）按轮次编号命名，便于后续从任意轮续训
        save_model(
            model=self.model,
            save_dir=self.ckpt_file,
            epoch=epoch_id,
            optimizer=self.optimizer,
            loss=loss
        )
    
    def train(self, num_epochs):
        """总训练循环：从当前 epoch_id 起连续训练 num_epochs 轮，每轮调用一次 train_an_epoch"""
        for self.epoch_id in tnrange(
            self.epoch_id,
            self.epoch_id + num_epochs
        ):
            self.train_an_epoch(
                epoch_id=self.epoch_id
            )
 

class MMIterTrainerBack(IterativeTrainer):
    """逐任务训练器（Trainer）：按 target_cols 顺序一次只训练一个任务，先载入检查点并解冻该任务的分类头，再跑 num_epochs 轮；是 MMIterTrainer 的备用/回退版本"""
    def __init__(
        self,
        base_dir,
        model,
        optimizer,
        data_loader,
        target_cols,
        num_epochs,
        loss_func,
        ckpt_file,
        device,
        individual,
        logger=None
    ):
        """入参：model/optimizer 外部构建后直接注入；target_cols 任务名列表；num_epochs 每个任务训练的轮数；loss_func 单个损失名或按任务的损失名列表；individual 是否同时返回各任务独立损失；ckpt_file 起始检查点"""
        super().__init__(
            base_dir=base_dir,
            data_loader=data_loader,
            loss_func=loss_func,
            logger=logger
        )
        # 注意：此处未向 IterativeTrainer 传入 metrics/test_loader/target_names 三个必填关键字参数
        self.model = model
        self.optimizer = optimizer
        self.target_cols = target_cols
        self.num_epochs = num_epochs
        self.ckpt_file = ckpt_file
        self.device = device
        self.individual = individual

    def run(self):
        """任务级总循环：依次处理每个任务，若设置了 ckpt_file 则先恢复模型与优化器，再解冻该任务的分类头并训练 num_epochs 轮"""
        for i, task in tqdm(enumerate(self.target_cols)):
            # 任务开始前按当前 ckpt_file 载入模型与优化器；ckpt_file 会被 train_an_epoch 改写为最近一次保存的检查点（checkpoint）
            if self.ckpt_file is not None:
                self.model, self.optimizer, _, _ = load_model(
                    self.ckpt_file,
                    model=self.model,
                    optimizer=self.optimizer,
                    train=True,
                )

            self.model.freeze_classifier[i] = False
            # 把第 i 个任务分类头的 freeze_classifier 置为 False，即允许该分类头更新参数

            for epoch_id in tnrange(self.num_epochs):
                self.train_an_epoch(
                    target_ind=i,
                    target_name=task,
                    epoch_id=epoch_id
                )
    
    def train_a_batch(
        self,
        batch,
        target_ind,
        target_name,
        # epoch_id
    ):
        """单批次训练（只针对 target_ind 指定任务）：batch[-1][target_ind] 为该任务标签，取标签 >= 0 的样本为有效样本；返回该批次损失"""
        self.optimizer.zero_grad()

        y = (batch[-1][target_ind]).to(self.device)
        # 以 y >= 0 作为有效性条件，同时过滤输入 X 与标签 y
        X = [x.to(self.device)[y >= 0].float() for x in batch[0]]
        y = y[y >= 0]

        # teach=False：不使用教师强制，仅取当前任务名的预测输出
        y_preds = self.model(X, teach=False)[target_name]

        # 损失名可按任务下标取，也可以是全任务共用的单个名字
        loss_name = self.loss_func[target_ind] \
            if isinstance(self.loss_func, list) else self.loss_func

        loss = mtmc_loss(
            [y_preds],
            [y],
            loss_names=loss_name,
            individual=self.individual,
            device=self.device
        )

        # individual=True 时 mtmc_loss 返回 (总损失, 各损失分量列表)，否则只返回总损失
        if not self.individual:
            final_loss = loss
            individual_losses = []
        else:
            final_loss = loss[0]
            individual_losses = loss[1]

        # 反向传播 + 优化器更新（无梯度裁剪、无学习率调度）
        final_loss.backward()
        self.optimizer.step()

        # 逐批把该任务的总损失与分量损失追加写入 base_dir/<任务名>_loss.log
        with open(
            osp.join(self.base_dir, target_name + '_loss.log'),
            'a'
        ) as f:
            f.write(str(final_loss.item()))
            f.write('\t')
            for individual_loss in individual_losses:
                f.write(str(individual_loss))
                f.write('\t')
            f.write('\n')
        return loss

    def train_an_epoch(
        self,
        target_ind,
        target_name,
        epoch_id,
    ):
        """单轮（epoch）训练：遍历 data_loader 逐批训练该任务，结束后保存检查点为 ckpt/model.<任务名>_<轮次>.ckpt"""

        for batch in tqdm(self.data_loader):
            loss = self.train_a_batch(
                batch=batch,
                target_ind=target_ind,
                target_name=target_name
            )

        makedirs(osp.join(self.base_dir, 'ckpt'))
        self.ckpt_file = osp.join(
            self.base_dir, 'ckpt',
            f'model.{target_name}_{epoch_id}.ckpt'
        )
        save_model(
            model=self.model,
            save_dir=self.ckpt_file,
            epoch=epoch_id,
            optimizer=self.optimizer,
            loss=loss
        )


def train(
    base_dir: str,
    csv_file: str,
    splitter: str,
    model_name: str,
    # model_init_args: t.Dict,
    ckpt_file: str = None,
    smiles_cols: t.List = [],
    fp_type: str = 'morgan_count',
    num_epochs: int = 20,
    target_cols: t.List = [],
    nums_classes: t.List = [],
    missing_labels: t.List = [],
    target_transform: t.List = [],
    optimizer_name: str = 'adam',
    loss_func: str = 'ce',
    batch_size: int = 128,
    hidden_size: int = 128,
    num_hidden_layers: int = 10,
    num_workers: int = 12,
    device_id: int = 0,
    **kwargs
):
    """指纹（fingerprint）多任务训练总入口：读 CSV 建 FPDataset 与 Loader → 组装 model_init_args（任务名、类别数、指纹位数、隐藏层宽度/层数等）→ 建 MMIterTrainer → 训练 num_epochs 轮。kwargs 可传 lr、weight_decay、cpu、parallel、task_weights、dim、hard_select、iterative、num_in_feats、converters"""
    base_dir = osp.abspath(base_dir)
    makedirs(base_dir)

    # 有 CUDA 时用 cuda:<device_id>，否则退回 CPU；kwargs['cpu'] 可强制 CPU
    device = torch.device(f'cuda:{device_id}') \
        if torch.cuda.is_available() \
        else torch.device('cpu')
    if kwargs.get('cpu', False):
        device = torch.device('cpu')
    
    converters = kwargs.get('converters', {})
    # fp_type 决定指纹种类，num_classes/target_transform/missing_labels 均按列传入数据集
    dataset = FPDataset(
        csv_file=csv_file,
        splitter=splitter,
        smiles_cols=smiles_cols,
        target_cols=target_cols,
        num_classes=nums_classes,
        missing_labels=missing_labels,
        target_transform=target_transform,
        fp_type=fp_type,
        converters=converters
    )
    data_loader = Loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    model_init_args = {}
    # 模型构造参数：任务名与各任务类别数、指纹位数（由 fp_type 查 FP_BITS_DICT）、隐藏层配置
    model_init_args['nums_classes'] = nums_classes
    model_init_args['target_names'] = target_cols
    model_init_args['num_fp_bits'] = FP_BITS_DICT[fp_type]
    model_init_args['hidden_size'] = hidden_size
    model_init_args['num_hidden_layers'] = num_hidden_layers
    model_init_args['dim'] = kwargs.get('dim', -1)
    model_init_args['hard_select'] = kwargs.get('hard_select', False)
    model_init_args['iterative'] = kwargs.get('iterative', True)
    model_init_args['num_in_feats'] = kwargs.get('num_in_feats', 1024)
    
    # 组装训练器（Trainer）：优化器超参只传 lr 与 weight_decay，随后进入多轮训练
    trainer = MMIterTrainer(
        base_dir=base_dir,
        data_loader=data_loader,
        target_names=target_cols,
        loss_func=loss_func,
        missing_labels=missing_labels,
        task_weights=kwargs.get('task_weights', None),
        test_loder=None,
        metrics=None,
        model_name=model_name,
        model_init_args=model_init_args,
        ckpt_file=ckpt_file,
        optimizer_name=optimizer_name,
        optimizer_kwargs={
            'lr': kwargs.get('lr', 0.01),
            'weight_decay': kwargs.get('weight_decay', 0)
        },
        logger=None,
        device=device,
        parallel=kwargs.get('parallel', False)
    )
    trainer.train(num_epochs=num_epochs)
