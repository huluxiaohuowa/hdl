# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/controllers/predictors/rxn_predictor.py
# 说明：模型预测器封装
# 模块功能：反应（reaction）SMILES 模型的预测器（Predictor）与批量打分入口，把 CSV 数据集跑成 numpy 结果并按目标列存盘
from os import path as osp

import torch
from torch import nn
import numpy as np

from .torch_predictor import TorchPredictor
from hdl.data.dataset.seq.rxn_dataset import RXNCSVDataset
from hdl.data.dataset.loaders.rxn_loader import RXNLoader
from jupyfuncs.show.pbar import tqdm
from hdl.models.rxn import build_rxn_mu
from hdl.models.utils import load_model


class RXNPredictor(TorchPredictor):
    """反应模型预测器（Predictor）：在构造时把 CSV 文件包成 RXNCSVDataset 与 RXNLoader（不打乱顺序），再交给父类 TorchPredictor"""
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
        """入参：data_file 输入 CSV 路径；smiles_col 反应 SMILES 列名；target_cols 目标列名列表（决定输出顺序）；splitter SMILES 多段分隔符；batch_sie 批大小；num_workers 加载进程数；model/reporter/device/logger 透传给父类"""
        
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
        """遍历整个 data_loader 做推理：每个 batch 取 batch[0] 作为输入列表、逐个搬到 device，模型对多目标返回结果元组，按目标下标收集后 np.concatenate 成列表返回（顺序与 target_cols 一致）"""
        result_list = []
        for _ in range(len(self.target_cols)):
            result_list.append([])
        # 切换到评估模式（eval），关闭 Dropout/BN 的训练行为
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
        """调用 predict_dataset 取全部预测，按 target_cols 逐列写为 <dir>/<目标名>.npy（numpy 二进制）"""
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
    """推理流程入口：build_rxn_mu 按 model_kwargs 搭模型并取设备 → eval 后用 ckpt_file 载入权重 → 可选 nn.DataParallel 多卡包装 → RXNPredictor 批量预测并把 .npy 结果写入 save_dir"""
    model, device = build_rxn_mu(
        **model_kwargs
    )
    # 推理前置为评估模式（eval）
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

