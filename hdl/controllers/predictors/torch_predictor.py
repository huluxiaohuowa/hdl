# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/controllers/predictors/torch_predictor.py
# 说明：模型预测器封装
# 模块功能：PyTorch 预测器（Predictor）基类，封装数据加载器（DataLoader）、日志与设备迁移，提供批量推理入口
# import typing as t
import torch


class TorchPredictor(object):
    """预测器（Predictor）基类：保存 data_loader、logger、reporter 与已搬到目标设备的 model，供子类继承"""
    def __init__(
        self,
        data_loader,
        logger,
        model=None,
        reporter=None,
        device=torch.device('cpu'),
    ) -> None:
        """入参：data_loader 待推理数据的批次迭代器；logger 日志对象；model 已构建的模型（直接搬到 device）；reporter 结果上报器（此处仅保存）；device 计算设备"""
        super().__init__() 
        self.data_loader = data_loader
        self.logger = logger
        self.reporter = reporter
        self.device = device
        # 构造时即把模型迁移到目标设备，后续推理只迁移输入
        self.model = model.to(self.device)
    
    def predict(self, X):
        """对单个批次 X 做一次前向推理，返回值经 collate 整理；X 需为支持 .to() 的张量或张量容器"""
        X = X.to(self.device)
        # 基类未在此处调用 model.eval() 或 torch.no_grad()，推理模式由子类或调用方设置
        return self.collate(
            self.model(X)
        )
    
    def collate(self, data):
        """批次结果整理钩子：基类原样返回模型输出，由子类覆写做拼接/后处理"""
        return data