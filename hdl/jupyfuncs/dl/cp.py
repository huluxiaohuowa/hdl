"""
Conformer Prediction for classification Task

As we only consider the predicted values as input, 
we do not differentiate between transductive and inductive conformers
"""

import numpy as np
import pandas as pd


class CpClassfier:
    def __init__(self):
        self.cal_data = None
        self.class_num = 0

    def fit_with_data(self, cal_proba, cal_y, class_num=0):
        """
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
        #TODO: besides calibration data, we can also use our model
        pass

    def predict_with_proba(self, X_proba):
        """
        :parm X_proba: numpy array of shape [n_samples, n_classes]
                        predicted probabilities of calibration set
        """
        cp_proba = []
        class_lsts = [sorted(self.cal_data[self.cal_data["true_label"] == i]["class_%d"%i]) \
                        for i in range(self.class_num)]
        proba_lsts = [X_proba[:, i] for i in range(self.class_num)]
        for c_lst, p_lst in zip(class_lsts, proba_lsts):
            c_proba = np.searchsorted(c_lst, p_lst, side='left')/len(c_lst)
            cp_proba.append(c_proba)

        self.cp_P = np.array(cp_proba).T
        return self.cp_P
