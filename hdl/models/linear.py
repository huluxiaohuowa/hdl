import typing as t

import torch
from torch import nn
from torch.nn import functional as nnfunc
import numpy as np

from hdl.layers.general.linear import (
    BNReLULinearBlock,
    BNReLULinear
)
# from hdl.ops.utils import get_activation


class MMIterLinear(nn.Module):
    _NAME = 'mumc_linear'

    def __init__(
        self,
        num_fp_bits: int,
        num_in_feats: int,
        nums_classes: t.List[int] = [3, 3],
        target_names: t.List[str] = None,
        hidden_size: int = 128,
        num_hidden_layers: int = 10,
        activation: str = 'elu',
        out_act: str = 'softmax',
        hard_select: bool = False,
        iterative: bool = True,
        **kwargs,
    ):
        super().__init__()

        if target_names is None:
            self.target_names = list(range(len(nums_classes)))
        else:
            self.target_names = target_names
        
        self.init_args = {
            'num_fp_bits': num_fp_bits,
            'num_in_feats': num_in_feats,
            'nums_classes': nums_classes,
            'target_names': target_names,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'activation': activation,
            'out_act': out_act,
            'hard_select': hard_select,
            'iterative': iterative,
            **kwargs
        }
        self.hard_select = hard_select
        self.iterative = iterative
        self._freeze_classifier = [True] * len(target_names)
        
        # self.w1 = BNReLULinear(num_fp_bits, num_in_feats, activation)
        self.w1 = nn.Linear(num_fp_bits, num_in_feats)
        # self.w2 = BNReLULinear(num_fp_bits, num_in_feats, activation)
        self.w2 = nn.Linear(num_fp_bits, num_in_feats)
        # self.w3 = BNReLULinear(num_fp_bits, num_in_feats, activation)
        self.w3 = nn.Linear(num_fp_bits, num_in_feats)

        nums_in_feats = [num_in_feats]
        if iterative:
            nums_in_feats.extend(nums_classes)
            nums_in_feats = np.cumsum(np.array(nums_in_feats, dtype=np.int))[:-1]
        else:
            nums_in_feats = nums_in_feats * len(nums_classes)
        
        if isinstance(out_act, str):
            self.out_acts = [out_act] * len(nums_classes)
        else:
            self.out_acts = out_act

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                BNReLULinearBlock(
                    num_in,
                    hidden_size,
                    num_hidden_layers,
                    hidden_size,
                    activation,
                    **kwargs
                ),
                BNReLULinear(
                    hidden_size,
                    num_out,
                    out_act,
                    **kwargs
                )
            )
            for num_in, num_out, out_act in zip(
                nums_in_feats, nums_classes, self.out_acts
            )
        ])

    @property
    def freeze_classifier(self):
        return self._freeze_classifier

    @freeze_classifier.setter
    def freeze_classifier(self, freeze: t.List = []):
        self._freeze_classifier = freeze
        self.change_classifier_grad([not f for f in freeze])

    def change_classifier_grad(self, requires_grads: t.List = []):
        for requires_grad, classifier in zip(requires_grads, self.classifiers):
            for param in classifier.parameters():
                param.requires_grad = requires_grad
    
    def forward(self, fps, target_tensors=None, teach=True):
        result_dict = {}
        fp1, fp2, fp3 = fps
        fp1 = self.w1(fp1)
        fp2 = self.w2(fp2)
        fp3 = self.w3(fp3)
        X = fp3 - (fp1 + fp2)
        if target_tensors is None:
            target_tensors = [None] * len(self.target_names)
        for classifier, target_name, target_tensor in zip(
            self.classifiers, self.target_names, target_tensors
        ):
            result = classifier(X)
            result_dict[target_name] = result
            if self.iterative:
                if teach:
                    assert target_tensors is not None
                    X = torch.cat((X, target_tensor), -1) 
                else: 
                    if not self.hard_select:
                        X = torch.cat((X, result), -1)
                    else:
                        X = torch.cat(
                            (X, nnfunc.gumbel_softmax(result, tau=1, hard=True)),
                            -1
                        )
        return result_dict
