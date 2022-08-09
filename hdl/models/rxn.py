import pkg_resources
from transformers import BertModel
import torch
# from torch import nn

from hdl.layers.general.linear import (
    MultiTaskMultiClassBlock,
    MuMcHardBlock
)
# from hdl.data.seq.rxn import rxn_model


def get_rxn_model(
    model_path: str = None
):
    if model_path is None:
        model_path = pkg_resources.resource_filename(
            "rxnfp",
            "models/transformers/bert_ft"
        )
        model = BertModel.from_pretrained(model_path)
        model = model.eval().cpu()

    return model


def build_rxn_mu(
    nums_classes,
    hard=False,
    hidden_size=128,
    nums_hidden_layers=10,
    encoder=get_rxn_model(),
    # freeze_encoder=True,
    device_id: int = 0,
    **kwargs
):
    if not hard:
        model = MultiTaskMultiClassBlock(
            encoder=encoder,
            nums_classes=nums_classes,
            hidden_size=hidden_size,
            num_hidden_layers=nums_hidden_layers,
            # freeze_encoder=freeze_encoder,
            **kwargs
        )
    else:
        model = MuMcHardBlock(
            encoder=encoder,
            nums_classes=nums_classes,
            hidden_size=hidden_size,
            num_hidden_layers=nums_hidden_layers,
            # freeze_encoder=freeze_encoder,
            **kwargs
        )
    device = torch.device(f'cuda:{device_id}') \
        if torch.cuda.is_available() \
        else torch.device('cpu')

    model = model.to(device)
    
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    return model, device