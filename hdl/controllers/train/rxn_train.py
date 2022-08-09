from os import path as osp
import typing as t

import torch
from torch import nn

from hdl.models.rxn import build_rxn_mu
from hdl.models.utils import load_model, save_model
from hdl.data.dataset.seq.rxn_dataset import RXNCSVDataset
from hdl.data.dataset.loaders.rxn_loader import RXNLoader
from hdl.metric_loss.loss import mtmc_loss
from jupyfuncs.pbar import tnrange, tqdm
from jupyfuncs.glob import makedirs
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
    optimizer.zero_grad()

    X = [x.to(device) for x in batch_data[0]]
    y = batch_data[1].T.to(device)

    y_preds = model(X)
    loss = mtmc_loss(
        y_preds,
        y,
        loss_func,
        individual=individual, **kwargs
    )

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

    epoch = 0
    if ckpt_file is not None:

        model, optimizer, epoch, _ = load_model(
            ckpt_file,
            model=model,
            optimizer=optimizer,
            train=True,
            device=device,
        )
 
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

    base_dir = osp.abspath(base_dir)
    makedirs(base_dir)
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
        model = nn.DataParallel(model)
    model.train()
    
    params = [{
        'params': model.parameters(),
        'lr': lr,
        'weight_decay': 0
    }]
    optimizer = Adam(params)
 
    dataset = RXNCSVDataset(
        csv_file=csv_file,
        splitter=splitter,
        smiles_col=smiles_col,
        target_cols=target_cols,
    )
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

    