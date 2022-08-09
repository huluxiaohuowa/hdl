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
from jupyfuncs.pbar import tnrange, tqdm
from jupyfuncs.glob import makedirs
from jupyfuncs.tensor import get_valid_indices
from hdl.metric_loss.loss import mtmc_loss
from hdl.controllers.train.trainer_base import IterativeTrainer


class MMIterTrainer(IterativeTrainer):
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
        super().__init__(
            base_dir=base_dir,
            data_loader=data_loader,
            test_loader=test_loder,
            metrics=metrics,
            loss_func=loss_func,
            target_names=target_names,
            # logger=logger
        )
        assert len(missing_labels) == len(target_names)
        self.epoch_id = 0
        if model is not None:
            self.model = model
        else:
            assert model_name is not None and model_init_args is not None
            self.model = MODEL_DICT[model_name](**model_init_args)
        self.model.to(device)
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            assert optimizer_name is not None and optimizer_kwargs is not None
            params = [{
                'params': self.model.parameters(),
                **optimizer_kwargs
            }]
            self.optimizer = OPTIM_DICT[optimizer_name](params)
        
        if ckpt_file is not None:
            self.model, self.optimizer, self.epoch_id, _ = load_model(
                save_dir=ckpt_file,
                model=self.model,
                optimizer=self.optimizer,
                train=True
            )
        if self.epoch_id != 0:
            self.epoch_id += 1
        
        self.metrics = metrics
        self.device = device
        self.missing_labels = missing_labels
        if parallel:
            self.model = nn.DataParallel(self.model)
        
        if isinstance(loss_func, str):
            self.loss_names = [loss_func] * len(target_names)
        elif isinstance(loss_func, (t.List, t.Tuple)):
            assert len(loss_func) == len(target_names)
            self.loss_names = loss_func
        
        if task_weights is None:
            task_weights = [1] * len(target_names)
        self.task_weights = task_weights
    
    def train_a_batch(self, batch):
        self.optimizer.zero_grad()
        fps = [x.to(self.device) for x in batch[0]]
        target_tensors = [
            target_tensor.to(self.device) for target_tensor in batch[1]
        ]
        target_list = batch[-1]
        target_valid_dict = {}
        for target_name, target_labels in zip(self.target_names, target_list):

            valid_indices = get_valid_indices(labels=target_labels)
            valid_indices.to(self.device) 
                
            target_valid_dict[target_name] = valid_indices
        
        y_preds = self.model(fps, target_tensors, teach=True)        

        # process with y_true
        y_trues = []
        y_preds_list = []
        for target_name, target_tensor, target_labels, loss_name in zip(
            self.target_names, target_tensors, target_list, self.loss_names
        ):
            valid_indices = target_valid_dict[target_name]
            if loss_name in ['ce']:
                y_true = target_labels[valid_indices].long().to(self.device)
            elif loss_name in ['mse', 'bpmll']:
                y_true = target_tensor[valid_indices].to(self.device)
            y_pred = y_preds[target_name][valid_indices].to(self.device)
            y_preds_list.append(y_pred)
            y_trues.append(y_true)
        
        # print(y_preds, y_trues)
        loss, loss_list = mtmc_loss(
            y_preds=y_preds_list,
            y_trues=y_trues,
            loss_names=self.loss_names,
            individual=True,
            task_weights=self.task_weights,
            device=self.device
        )
        with open(osp.join(self.base_dir, 'loss.log'), 'a') as f:
            f.write(str(loss.item()))
            f.write('\t') 
            for i_loss in loss_list:
                f.write(str(i_loss.item()))
                f.write('\t')
            f.write('\n')
            f.flush()
            
        loss.backward()
        self.optimizer.step()
 
        return loss
       
    def train_an_epoch(self, epoch_id):
        for batch in tqdm(self.data_loader):
            loss = self.train_a_batch(
                batch=batch
            )
        makedirs(osp.join(self.base_dir, 'ckpt'))
        self.ckpt_file = osp.join(
            self.base_dir, 'ckpt',
            f'model.{epoch_id}.ckpt'
        )
        save_model(
            model=self.model,
            save_dir=self.ckpt_file,
            epoch=epoch_id,
            optimizer=self.optimizer,
            loss=loss
        )
    
    def train(self, num_epochs):
        for self.epoch_id in tnrange(
            self.epoch_id,
            self.epoch_id + num_epochs
        ):
            self.train_an_epoch(
                epoch_id=self.epoch_id
            )
 

class MMIterTrainerBack(IterativeTrainer):
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
        super().__init__(
            base_dir=base_dir,
            data_loader=data_loader,
            loss_func=loss_func,
            logger=logger
        )
        self.model = model
        self.optimizer = optimizer
        self.target_cols = target_cols
        self.num_epochs = num_epochs
        self.ckpt_file = ckpt_file
        self.device = device
        self.individual = individual

    def run(self):
        for i, task in tqdm(enumerate(self.target_cols)):
            if self.ckpt_file is not None:
                self.model, self.optimizer, _, _ = load_model(
                    self.ckpt_file,
                    model=self.model,
                    optimizer=self.optimizer,
                    train=True,
                )

            self.model.freeze_classifier[i] = False

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
        self.optimizer.zero_grad()

        y = (batch[-1][target_ind]).to(self.device)
        X = [x.to(self.device)[y >= 0].float() for x in batch[0]]
        y = y[y >= 0]

        y_preds = self.model(X, teach=False)[target_name]

        loss_name = self.loss_func[target_ind] \
            if isinstance(self.loss_func, list) else self.loss_func

        loss = mtmc_loss(
            [y_preds],
            [y],
            loss_names=loss_name,
            individual=self.individual,
            device=self.device
        )

        if not self.individual:
            final_loss = loss
            individual_losses = []
        else:
            final_loss = loss[0]
            individual_losses = loss[1]

        final_loss.backward()
        self.optimizer.step()

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
    base_dir = osp.abspath(base_dir)
    makedirs(base_dir)

    device = torch.device(f'cuda:{device_id}') \
        if torch.cuda.is_available() \
        else torch.device('cpu')
    if kwargs.get('cpu', False):
        device = torch.device('cpu')
    
    converters = kwargs.get('converters', {})
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
    model_init_args['nums_classes'] = nums_classes
    model_init_args['target_names'] = target_cols
    model_init_args['num_fp_bits'] = FP_BITS_DICT[fp_type]
    model_init_args['hidden_size'] = hidden_size
    model_init_args['num_hidden_layers'] = num_hidden_layers
    model_init_args['dim'] = kwargs.get('dim', -1)
    model_init_args['hard_select'] = kwargs.get('hard_select', False)
    model_init_args['iterative'] = kwargs.get('iterative', True)
    model_init_args['num_in_feats'] = kwargs.get('num_in_feats', 1024)
    
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
