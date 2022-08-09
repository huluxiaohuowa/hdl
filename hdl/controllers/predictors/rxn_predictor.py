from os import path as osp

import torch
from torch import nn
import numpy as np

from .torch_predictor import TorchPredictor
from hdl.data.dataset.seq.rxn_dataset import RXNCSVDataset
from hdl.data.dataset.loaders.rxn_loader import RXNLoader
from jupyfuncs.pbar import tqdm
from hdl.models.rxn import build_rxn_mu
from hdl.models.utils import load_model


class RXNPredictor(TorchPredictor):
    def __init__(
        self,
        data_file,
        logger=None,
        smiles_col='SMILES',
        target_cols=[],
        model=None,
        reporter=None,
        device=torch.device('cpu'),
        splitter=',',
        batch_sie=128,
        num_workers=20,
    ) -> None:
        
        dataset = RXNCSVDataset(
            csv_file=data_file,
            splitter=splitter,
            smiles_col=smiles_col,
            target_cols=target_cols
        )
 
        data_loader = RXNLoader(
            dataset,
            batch_size=batch_sie,
            shuffle=False,
            num_workers=num_workers
        )

        self.target_cols = target_cols

        super().__init__(
            data_loader=data_loader,
            logger=logger,
            model=model,
            reporter=reporter,
            device=device 
        )
 
    def predict_dataset(self):
        result_list = []
        for _ in range(len(self.target_cols)):
            result_list.append([])
        self.model.eval()
        for batch in tqdm(self.data_loader):
            X = batch[0]
            X = [x.to(self.device) for x in X]
            results = self.model(X)
            for result_idx, result in enumerate(results):
                result_list[result_idx].append(result.detach().cpu().numpy())
            # result_list.append(self.model(X).detach().cpu().numpy())
        result_arr_list = []
        for result in result_list:
            result_arr_list.append(np.concatenate(result, 0))
        return result_arr_list
    
    def save_results(self, dir):
        results = self.predict_dataset()
        for result, target in zip(results, self.target_cols):
            save_file = osp.join(dir, target + '.npy')
            with open(save_file, 'wb') as f:
                np.save(f, result)
            
        
def rxn_predict(
    data_file,
    smiles_col,
    splitter,
    ckpt_file,
    save_dir,
    batch_size=128,
    target_cols=[],
    num_workers=15,
    parallel: bool = False,
    **model_kwargs,
):
    model, device = build_rxn_mu(
        **model_kwargs
    )
    model = model.eval()
    model, _, _, _ = load_model(
        ckpt_file,
        model
    )
    if parallel:
        model = nn.DataParallel(model)
    predictor = RXNPredictor(
        data_file,
        model=model,
        smiles_col=smiles_col,
        target_cols=target_cols,
        splitter=splitter,
        device=device,
        batch_sie=batch_size,
        num_workers=num_workers,
    )
    
    predictor.save_results(save_dir)

