# import typing as t
import torch


class TorchPredictor(object):
    def __init__(
        self,
        data_loader,
        logger,
        model=None,
        reporter=None,
        device=torch.device('cpu'),
    ) -> None:
        super().__init__() 
        self.data_loader = data_loader
        self.logger = logger
        self.reporter = reporter
        self.device = device
        self.model = model.to(self.device)
    
    def predict(self, X):
        X = X.to(self.device)
        return self.collate(
            self.model(X)
        )
    
    def collate(self, data):
        return data