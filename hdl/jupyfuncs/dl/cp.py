# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/dl/cp.py
# 说明：深度学习张量与模型辅助工具
"""
Conformer Prediction for classification Task

As we only consider the predicted values as input, 
we do not differentiate between transductive and inductive conformers
"""

import numpy as np
import pandas as pd


class CpClassfier:
    """共形预测分类器（conformal prediction）：用校准集各类别概率的经验分布，把测试集的预测概率映射为校准概率。"""
    def __init__(self):
        self.cal_data = None
        self.class_num = 0

    def fit_with_data(self, cal_proba, cal_y, class_num=0):
        """
        保存校准集：按真标签把每个类别的校准概率列整理进 self.cal_data（DataFrame，列为 class_i 与 true_label）。
        :parm cal_proba: numpy array of shape [n_samples, n_classes]
                        predicted probability of calibration set
        :parm cal_y: numpy array of shape [n_samples,]
                        true label of calibration set
        """

        if class_num <= 0:
            print("Get class number for input data.")
            self.class_num = cal_proba.shape[1]
        else:
            self.class_num = class_num

        cal_df = pd.DataFrame(cal_proba)
        cal_df.columns = ["class_%d"%i for i in range(self.class_num)]
        cal_df["true_label"] = list(cal_y)
        self.cal_data = cal_df
     
    def fit_with_model(self, func):
        """预留接口：从模型直接构建校准数据，当前实现为空。"""
        #TODO: besides calibration data, we can also use our model
        pass

    def predict_with_proba(self, X_proba):
        """
        用校准集概率的升序数组做二分查找，得到每个测试概率在校准分布中的秩，作为校准后概率。
        :parm X_proba: numpy array of shape [n_samples, n_classes]
                        predicted probabilities of calibration set
        """
        cp_proba = []
        # 按真标签筛选校准集，取出各类别对应的概率列并升序排序
        class_lsts = [sorted(self.cal_data[self.cal_data["true_label"] == i]["class_%d"%i]) \
                        for i in range(self.class_num)]
        # 测试集中每个类别的预测概率列
        proba_lsts = [X_proba[:, i] for i in range(self.class_num)]
        for c_lst, p_lst in zip(class_lsts, proba_lsts):
            # 测试概率在校准分布中的秩除以校准样本数，得到该类别的校准概率
            c_proba = np.searchsorted(c_lst, p_lst, side='left')/len(c_lst)
            cp_proba.append(c_proba)

        # 按类别堆叠后转置为形状 (n_samples, n_classes) 的校准概率矩阵
        self.cp_P = np.array(cp_proba).T
        return self.cp_P
