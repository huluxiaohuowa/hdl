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
        super().__init__()
        self.base_dir = base_dir
        self.data_loader = data_loader
        self.test_loader = test_loader

        if metrics is not None:
            self.metric_names = metrics
            self.metrics = [get_metric(metric) for metric in metrics]
        if loss_func is not None:
            self.loss_name = loss_func
            self.loss_func = get_lossfunc(loss_func)
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.logger = SummaryWriter(log_dir=self.base_dir)

        self.losses = []

        if model is not None:
            self.model = model
        else:
            assert model_name is not None and model_init_args is not None
            self.model = MODEL_DICT[model_name](**model_init_args)

        self.ckpt_file = ckpt_file
        self.model_ckpt = model_ckpt

        self.model.to(self.device)

        if optimizer is not None:
            self.optimizer = optimizer
        elif optimizer_name is not None and optimizer_kwargs is not None:
            params = [{
                'params': self.model.parameters(),
                **optimizer_kwargs
            }]
            self.optimizer = OPTIM_DICT[optimizer_name](params)
        else:
            self.optimizer = None
        
        self.n_iter = 0
        self.epoch_id = 0
        self.metrics = [get_metric(metric) for metric in metrics]
    
    @abstractmethod
    def load_ckpt(self, ckpt_file, train=False):
        raise NotImplementedError

    @abstractmethod
    def train_a_batch(self):
        raise NotImplementedError
    
    @abstractmethod
    def train_an_epoch(self):
        raise NotImplementedError
    
    @abstractmethod
    def train(self):
        raise NotImplementedError 
    
    def save(self):
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
        load_model(
            save_dir=ckpt_file,
            model=self.model,
            optimizer=self.optimizer,
            train=train 
        )
    
    @abstractmethod
    def predict(self, data_loader):
        raise NotImplementedError


class IterativeTrainer(TorchTrainer):
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
        super().__init__(
            base_dir,
            data_loader,
            test_loader,
            metrics,
            loss_func,
            logger
        )
        self.target_names = target_names

    @abstractmethod
    def train_a_batch(self):
        raise NotImplementedError
    
    @abstractmethod
    def train_an_epoch(self):
        raise NotImplementedError
 
    @abstractmethod
    def train(self):
        raise NotImplementedError
 