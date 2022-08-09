import typing as t
from os import path as osp
# from os import path as osp
from itertools import cycle
# import datetime

import torch
import numpy as np
import pandas as pd


# from jupyfuncs.glob import makedirs
from jupyfuncs.pbar import tnrange, tqdm
# from hdl.data.dataset.graph.gin import MoleculeDataset
from hdl.data.dataset.graph.gin import MoleculeDatasetWrapper
# from hdl.metric_loss.loss import get_lossfunc
# from hdl.models.utils import save_model
from .trainer_base import TorchTrainer


class GINTrainer(TorchTrainer):
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
            for gin in self.model.gins:
                for param in gin.parameters():
                    param.requires_grad = False
    
    def train_a_batch(self, data):
        self.optimizer.zero_grad()
        for i in data[: -1]:
            for j in i:
                j.to(self.device)
        y = data[-1].to(self.device)
        y = y / 100

        y_pred = self.model(data).flatten()
        
        loss = self.loss_func(y_pred, y)

        loss.backward()
        self.optimizer.step()
 
        return loss
    
    def load_ckpt(self):
        self.model.load_ckpt()
        
    def train_an_epoch(
        self,
    ):
        for i, (data, test_data) in enumerate(
            zip(
                self.data_loader,
                cycle(self.test_loader)
            )
        ):
            loss = self.train_a_batch(data)
            self.losses.append(loss.item())
            self.n_iter += 1
            self.logger.add_scalar(
                'train_loss',
                loss.item(),
                global_step=self.n_iter
            )

            if self.n_iter % 10 == 0:
                for i in test_data[: -1]:
                    for j in i:
                        j.to(self.device)
                y = test_data[-1].to(self.device)
                y = y / 100

                y_pred = self.model(test_data).flatten()
                valid_loss = self.loss_func(y_pred, y)

                y_pred = y_pred.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                
                self.logger.add_scalar(
                    'valid_loss',
                    valid_loss.item(),
                    global_step=self.n_iter
                )

                for metric_name, metric in zip(
                    self.metric_names,
                    self.metrics
                ):
                    self.logger.add_scalar(
                        metric_name,
                        metric(y_pred, y),
                        global_step=self.n_iter
                    )
 
        self.save() 
        self.epoch_id += 1
    
    def train(self, num_epochs):
        # dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        # makedirs(osp.join(self.base_dir, dir_name))

        for _ in tnrange(num_epochs):
            self.train_an_epoch()
    
    def predict(self, data_loader):
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
    trainer.load(ckpt_file=model_ckpt)
    trainer.model.eval()
    results = trainer.predict(data_loader)

    df = pd.read_csv(data_path)
    df['pred'] = results
    df.to_csv(
        osp.join(base_dir, 'pred.csv'),
        index=False
    )
    
    if y_col_name is not None:
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
    