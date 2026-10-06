# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/controllers/train/train_ginet.py
# 说明：训练流程与 Trainer 实现
# 模块功能：GIN 图编码器 + MLP 回归头的训练与推理流程：GINTrainer 训练器（Trainer）负责逐批训练、TensorBoard 记录与验证集评估，engine 与 predict 为流程入口
import typing as t
from os import path as osp
# from os import path as osp
from itertools import cycle
# import datetime

import torch
import numpy as np
import pandas as pd


# from jupyfuncs.glob import makedirs
from jupyfuncs.show.pbar import tnrange, tqdm
# from hdl.data.dataset.graph.gin import MoleculeDataset
from hdl.data.dataset.graph.gin import MoleculeDatasetWrapper
# from hdl.metric_loss.loss import get_lossfunc
# from hdl.models.utils import save_model
from .trainer_base import TorchTrainer


class GINTrainer(TorchTrainer):
    """分子产率回归训练器（Trainer）：在 TorchTrainer 之上实现 GIN（图同构网络）+MLP 的单批训练、每若干迭代在验证集上评估并写 TensorBoard、逐轮保存检查点（checkpoint）"""
    def __init__(
        self,
        base_dir,
        data_loader,
        test_loader,
        metrics: t.List[str] = ['rsquared', 'rmse', 'mae'],
        loss_func: str = 'mse',
        model=None,
        model_name=None,
        model_init_args=None,
        ckpt_file=None,
        model_ckpt=None,
        fix_emb=True,
        optimizer=None,
        optimizer_name=None,
        optimizer_kwargs=None,
        device=torch.device('cpu'),
        # logger=None
    ) -> None:
        """入参：metrics 默认评估 R²、RMSE、MAE；loss_func 默认均方误差（mse）；fix_emb=True 时冻结 model.gins 中各 GIN 层的预训练权重，只训练 MLP 头；其余参数透传给 TorchTrainer"""
        super().__init__(
            base_dir=base_dir,
            data_loader=data_loader,
            test_loader=test_loader,
            metrics=metrics,
            loss_func=loss_func,
            model=model,
            model_name=model_name,
            model_init_args=model_init_args,
            ckpt_file=ckpt_file,
            model_ckpt=model_ckpt,
            optimizer=optimizer,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
        )
        # self.loss_func = get_lossfunc(self.loss_func)
        # self.metrics = [get_metric(metric) for metric in metrics]
        if fix_emb:
            # 冻结各 GIN 编码器分支的参数：requires_grad=False 使其不接收梯度更新
            for gin in self.model.gins:
                for param in gin.parameters():
                    param.requires_grad = False
    
    def train_a_batch(self, data):
        """单批训练：data 前若干元素为分子图批次列表（逐个搬到 device），最后一个元素为产率标签；标签除以 100 归一化后与预测一起算 mse 损失，再反向传播并更新参数，返回损失张量"""
        self.optimizer.zero_grad()
        for i in data[: -1]:
            for j in i:
                j.to(self.device)
        y = data[-1].to(self.device)
        # 产率按 /100 缩放到 [0,1] 量纲
        y = y / 100

        y_pred = self.model(data).flatten()
        
        loss = self.loss_func(y_pred, y)

        # 反向传播 + 优化器 step（未做梯度裁剪与学习率调度）
        loss.backward()
        self.optimizer.step()
 
        return loss
    
    def load_ckpt(self):
        """委托模型自身的 load_ckpt 载入权重（不使用基类的 load）"""
        self.model.load_ckpt()
        
    def train_an_epoch(
        self,
    ):
        """一轮（epoch）训练：训练批次与验证批次用 cycle(test_loader) 配对遍历；每个迭代记录 train_loss，每 10 个迭代在验证批上算 valid_loss 与各指标（rsquared/rmse/mae）写入 TensorBoard；轮末保存检查点并把 epoch_id 加 1"""
        # 训练批与验证批配对遍历；test_loader 用 cycle 循环复用，长度不限时不会耗尽
        for i, (data, test_data) in enumerate(
            zip(
                self.data_loader,
                cycle(self.test_loader)
            )
        ):
            loss = self.train_a_batch(data)
            # 累积每迭代损失并写入 TensorBoard（横轴为累计迭代数 n_iter）
            self.losses.append(loss.item())
            self.n_iter += 1
            self.logger.add_scalar(
                'train_loss',
                loss.item(),
                global_step=self.n_iter
            )

            # 每 10 个迭代做一次验证集评估（训练集小时用 cycle 反复取验证批）
            if self.n_iter % 10 == 0:
                for i in test_data[: -1]:
                    for j in i:
                        j.to(self.device)
                y = test_data[-1].to(self.device)
                # 验证标签同样除以 100 归一化
                y = y / 100

                # 验证前向未使用 torch.no_grad()，也未调用 model.eval()
                y_pred = self.model(test_data).flatten()
                valid_loss = self.loss_func(y_pred, y)

                # 脱离计算图后转 numpy，供各指标函数使用
                y_pred = y_pred.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                
                self.logger.add_scalar(
                    'valid_loss',
                    valid_loss.item(),
                    global_step=self.n_iter
                )

                # 逐个指标（名称与函数一一对应）记录到 TensorBoard
                for metric_name, metric in zip(
                    self.metric_names,
                    self.metrics
                ):
                    self.logger.add_scalar(
                        metric_name,
                        metric(y_pred, y),
                        global_step=self.n_iter
                    )
 
        # 轮末保存检查点（checkpoint），文件名带当前轮次
        self.save() 
        self.epoch_id += 1
    
    def train(self, num_epochs):
        """总训练循环：重复 num_epochs 次 train_an_epoch（轮次编号在 train_an_epoch 内部自增）"""
        # dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        # makedirs(osp.join(self.base_dir, dir_name))

        for _ in tnrange(num_epochs):
            self.train_an_epoch()
    
    def predict(self, data_loader):
        """推理：逐批前向取扁平化预测值，转 numpy 后横向拼接成一维数组返回（不计算损失、不写日志）"""
        result_list = []
        for data in tqdm(data_loader):
            for i in data[: -1]:
                for j in i:
                    j.to(self.device)
            # print(data[0][0].x.device)
            # for param in self.model.parameters():
            #     print(param.device)
            #     break
            y_pred = self.model(data).flatten()
            result_list.append(y_pred.cpu().detach().numpy())
        results = np.hstack(result_list)
        return results
 

def engine(
    base_dir,
    data_path,
    test_data_path,
    batch_size=128,
    num_workers=64,
    model_name='GINMLPR',
    num_layers=5,
    emb_dim=300,
    feat_dim=512,
    out_dim=1,
    drop_ratio=0.0,
    pool='mean',
    ckpt_file=None,
    fix_emb: bool = False,
    device='cuda:1',
    num_epochs=300,
    optimizer_name='adam',
    lr=0.001,
    file_type: str = 'csv',
    smiles_col_names: t.List = [],
    y_col_name: str = None,  # "yield (%)",
    loss_func: str = 'mse',
    metrics: t.List[str] = ['rsquared', 'rmse', 'mae'],
):
    """训练入口：把训练/测试 CSV 各自包成 MoleculeDatasetWrapper（SMILES 转分子图批次），训练集打乱、测试集不打乱，再交给 GINTrainer 训 num_epochs 轮。model_init_args 传 GIN 层数、嵌入维度、读出维度、dropout、池化方式与 SMILES 列数"""
    model_init_args = {
        "num_layer": num_layers,
        "emb_dim": emb_dim,
        "feat_dim": feat_dim,
        "out_dim": out_dim,
        "drop_ratio": drop_ratio,
        "pool": pool,
        "ckpt_file": ckpt_file,
        "num_smiles": len(smiles_col_names),
    }
    wrapper = MoleculeDatasetWrapper(
        batch_size=batch_size,
        num_workers=num_workers,
        valid_size=0,
        data_path=data_path,
        file_type=file_type,
        smi_col_names=smiles_col_names,
        y_col_name=y_col_name
    )
    test_wrapper = MoleculeDatasetWrapper(
        batch_size=batch_size,
        num_workers=num_workers,
        valid_size=0,
        data_path=test_data_path,
        file_type=file_type,
        smi_col_names=smiles_col_names,
        y_col_name=y_col_name
    )

    # 两者都用 get_test_loader 取批次迭代器，训练侧靠 shuffle=True 打乱顺序
    data_loader = wrapper.get_test_loader(
        shuffle=True
    )
    test_loader = test_wrapper.get_test_loader(
        shuffle=False
    ) 

    trainer = GINTrainer(
        base_dir=base_dir,
        model_name=model_name,
        model_init_args=model_init_args,
        optimizer_name=optimizer_name,
        ckpt_file=ckpt_file,
        fix_emb=fix_emb,
        optimizer_kwargs={"lr": lr},
        data_loader=data_loader,
        test_loader=test_loader,
        metrics=metrics,
        loss_func=loss_func,
        device=device
    )
    
    trainer.train(num_epochs=num_epochs)


def predict(
    base_dir,
    data_path,
    batch_size=128,
    num_workers=64,
    model_name='GINMLPR',
    num_layers=5,
    emb_dim=300,
    feat_dim=512,
    out_dim=1,
    drop_ratio=0.0,
    pool='mean',
    ckpt_file=None,
    model_ckpt=None,
    device='cuda:1',
    file_type: str = 'csv',
    smiles_col_names: t.List = [],
    y_col_name: str = None,  # "yield (%)",
    metrics: t.List[str] = ['rsquared', 'rmse', 'mae'],
):
    """推理入口：建 GINTrainer（不传 test_loader）→ 用 model_ckpt 载入权重 → model.eval() 后逐批预测 → 把预测写入 base_dir/pred.csv 的 pred 列；若给了 y_col_name，则按同一列除以 100 的真值计算各指标并写 base_dir/metrics.csv"""
    model_init_args = {
        "num_layer": num_layers,
        "emb_dim": emb_dim,
        "feat_dim": feat_dim,
        "out_dim": out_dim,
        "drop_ratio": drop_ratio,
        "pool": pool,
        "ckpt_file": ckpt_file,
        "num_smiles": len(smiles_col_names),
    }
    wrapper = MoleculeDatasetWrapper(
        batch_size=batch_size,
        num_workers=num_workers,
        valid_size=0,
        data_path=data_path,
        file_type=file_type,
        smi_col_names=smiles_col_names,
        y_col_name=y_col_name
    )
    data_loader = wrapper.get_test_loader(
        shuffle=False
    )
    trainer = GINTrainer(
        base_dir=base_dir,
        model_name=model_name,
        model_init_args=model_init_args,
        model_ckpt=model_ckpt,
        data_loader=data_loader,
        test_loader=None,
        metrics=metrics,
        device=device
    )
    metric_list = trainer.metrics
    # 从 model_ckpt 恢复权重与优化器状态
    trainer.load(ckpt_file=model_ckpt)
    # 推理前置为评估模式（eval）；predict 内部未使用 torch.no_grad()
    trainer.model.eval()
    results = trainer.predict(data_loader)

    # 写预测结果：原始数据各列后追加一列 pred（模型输出，处于训练时的 /100 量纲）
    df = pd.read_csv(data_path)
    df['pred'] = results
    df.to_csv(
        osp.join(base_dir, 'pred.csv'),
        index=False
    )
    
    if y_col_name is not None:
        # 真值同样除以 100 后逐指标计算，结果一行写入 metrics.csv
        metrics_df = pd.DataFrame()
        y = df[y_col_name].array / 100
        for metric_name, metric in zip(
            metrics,
            metric_list
        ):
            metrics_df[metric_name] = np.array([metric(
                y, results
            )])
        metrics_df.to_csv(
            osp.join(
                base_dir, 'metrics.csv'
            ),
            index=False
        )
    