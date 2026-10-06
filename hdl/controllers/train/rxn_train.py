# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/controllers/train/rxn_train.py
# 说明：训练流程与 Trainer 实现
# 模块功能：反应（reaction）SMILES 模型的训练流程：单批/单轮训练函数、多轮循环 train_rxn，以及组装数据集、优化器与模型的入口 rxn_engine
from os import path as osp
import typing as t

import torch
from torch import nn

from hdl.models.rxn import build_rxn_mu
from hdl.models.utils import load_model, save_model
from hdl.data.dataset.seq.rxn_dataset import RXNCSVDataset
from hdl.data.dataset.loaders.rxn_loader import RXNLoader
from hdl.metric_loss.loss import mtmc_loss
from jupyfuncs.show.pbar import tnrange, tqdm
from jupyfuncs.path.glob import makedirs
# from hdl.optims.nadam import Nadam
from torch.optim import Adam
# from .trainer_base import TorchTrainer


def train_a_batch(
    model,
    batch_data,
    loss_func,
    optimizer,
    device,
    individual,
    **kwargs
):
    """单批次训练：batch_data[0] 为反应 SMILES 经分词后得到的输入张量列表，batch_data[1] 转置后为逐任务标签；返回 (总损失, 各任务损失列表)，individual=False 时第二项为空列表"""
    # 清空上一批残留梯度
    optimizer.zero_grad()

    X = [x.to(device) for x in batch_data[0]]
    # 标签矩阵转置成 (任务数, 批样本数) 以对齐多任务输出
    y = batch_data[1].T.to(device)

    y_preds = model(X)
    # 多任务多分类损失：按 loss_func 指定的每个任务损失加权求和
    loss = mtmc_loss(
        y_preds,
        y,
        loss_func,
        individual=individual, **kwargs
    )

    # 拆分总损失与各任务独立损失
    if not individual:
        final_loss = loss
        individual_losses = []
    else:
        final_loss = loss[0]
        individual_losses = loss[1]
        
    final_loss.backward()
    optimizer.step()

    return final_loss, individual_losses


def train_an_epoch(
    base_dir: str,
    model,
    data_loader,
    epoch_id: int,
    loss_func,
    optimizer,
    device,
    num_warm_epochs: int = 0,
    individual: bool = True,
    **kwargs
):
    """一轮（epoch）训练：epoch_id 小于 num_warm_epochs 时冻结编码器（freeze_encoder=True）做预热，之后放开；遍历 data_loader 逐批调用 train_a_batch，每批把总损失与各任务损失追加写入 base_dir/loss.log，轮末按 model.<epoch_id>.ckpt 保存检查点（checkpoint）"""
    # 预热（warmup）阶段置 model.freeze_encoder=True，具体冻结哪些参数由模型实现决定
    if epoch_id < num_warm_epochs:
        model.freeze_encoder = True
    else:
        model.freeze_encoder = False

    for batch in tqdm(data_loader):
        loss, individual_losses = train_a_batch(
            model=model,
            batch_data=batch,
            loss_func=loss_func,
            optimizer=optimizer,
            device=device,
            individual=individual,
            **kwargs
        )
        # 每批追加一行损失记录：总损失与各任务损失，制表符分隔
        with open(
            osp.join(base_dir, 'loss.log'),
            'a'
        ) as f:
            f.write(str(loss.item()))
            f.write('\t')
            for individual_loss in individual_losses:
                f.write(str(individual_loss))
                f.write('\t')
            f.write('\n')
 
    # 轮末存检查点（checkpoint）：文件名带轮次，内容含模型、优化器、轮次与最后一个批次的损失
    ckpt_file = osp.join(
        base_dir,
        f'model.{epoch_id}.ckpt'
    )
    save_model(
        model=model,
        save_dir=ckpt_file,
        epoch=epoch_id,
        optimizer=optimizer,
        loss=loss,
    ) 

 
def train_rxn(
    base_dir,
    model,
    num_epochs,
    loss_func,
    data_loader,
    optimizer,
    device,
    num_warm_epochs: int = 10,
    ckpt_file: str = None,
    individual: bool = True,
    **kwargs
):

    """多轮训练循环：给了 ckpt_file 时先按 train=True 载入模型、优化器与已完成轮次，后续 epoch_id 从该轮次继续编号；再跑 num_epochs 轮 train_an_epoch"""
    epoch = 0
    if ckpt_file is not None:

        # 断点续训：epoch 为检查点里记录的已训轮数
        model, optimizer, epoch, _ = load_model(
            ckpt_file,
            model=model,
            optimizer=optimizer,
            train=True,
            device=device,
        )
 
    # epoch_id = 续训起始轮数 + 本次循环序号
    for epoch_id in tnrange(num_epochs):

        train_an_epoch(
            base_dir=base_dir,
            model=model,
            data_loader=data_loader,
            epoch_id=epoch + epoch_id,
            loss_func=loss_func,
            optimizer=optimizer,
            num_warm_epochs=num_warm_epochs,
            device=device,
            individual=individual,
            **kwargs
        )


def rxn_engine(
    base_dir: str,
    csv_file: str,
    splitter: str,
    smiles_col: str,
    hard: bool = False,
    num_epochs: int = 20,
    target_cols: t.List = [],
    nums_classes: t.List = [],
    loss_func: str = 'ce',
    num_warm_epochs: int = 10,
    batch_size: int = 128,
    hidden_size: int = 128,
    lr: float = 0.01,
    num_hidden_layers: int = 10,
    shuffle: bool = True,
    num_workers: int = 12,
    dim=-1,
    out_act='softmax',
    device_id: int = 0,
    individual: bool = True,
    **kwargs
):

    """反应模型训练入口：建输出目录 → build_rxn_mu 构建模型与设备（多卡时 nn.DataParallel 包装）→ Adam 优化器（lr、weight_decay=0）→ RXNCSVDataset + RXNLoader 读数据 → 调 train_rxn 训 num_epochs 轮。num_warm_epochs 控制预热轮数，individual 控制是否记录各任务损失"""
    base_dir = osp.abspath(base_dir)
    makedirs(base_dir)
    # 按类别数、隐藏层配置与输出激活函数构建反应模型
    model, device = build_rxn_mu(
        nums_classes=nums_classes,
        hard=hard,
        hidden_size=hidden_size,
        nums_hidden_layers=num_hidden_layers,
        dim=dim,
        out_act=out_act,
        device_id=device_id
    )
    if torch.cuda.device_count() > 1:
        # 多 GPU 时用 DataParallel 复制模型分片计算
        model = nn.DataParallel(model)
    # 切到训练模式（与预测时的 eval 相对应）
    model.train()
    
    params = [{
        'params': model.parameters(),
        'lr': lr,
        'weight_decay': 0
    }]
    # 单一参数组：全部模型参数共用同一学习率与权重衰减
    optimizer = Adam(params)
 
    dataset = RXNCSVDataset(
        csv_file=csv_file,
        splitter=splitter,
        smiles_col=smiles_col,
        target_cols=target_cols,
    )
    # 反应专用加载器：按 batch_size 组批，shuffle 控制每轮是否打乱
    data_loader = RXNLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    train_rxn(
        base_dir=base_dir,
        model=model,
        num_epochs=num_epochs,
        loss_func=loss_func,
        data_loader=data_loader,
        optimizer=optimizer,
        device=device,
        num_warm_epochs=num_warm_epochs,
        ckpt_file=None,
        individual=individual,
        **kwargs
    )

    