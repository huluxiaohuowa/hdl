from rxnfp.tokenization import (
    SmilesTokenizer,
    convert_reaction_to_valid_features_batch,
)
import torch
import numpy as np
import pkg_resources


__all__ = [
    'collate_rxn',
]


def collate_rxn(
    rxn_list,
    labels,
    vocab_path: str = None,
    max_len: int = 512
):
    if vocab_path is None:
        vocab_path = pkg_resources.resource_filename(
            "rxnfp",
            "models/transformers/bert_ft/vocab.txt"
        )
    tokenizer = SmilesTokenizer(
        vocab_path, max_len=max_len
    )

    feats = convert_reaction_to_valid_features_batch(
        rxn_list,
        tokenizer
    )
    X = [
        torch.tensor(feats.input_ids.astype(np.int64)),
        torch.tensor(feats.input_mask.astype(np.int64)),
        torch.tensor(feats.segment_ids.astype(np.int64))
    ]
    y = torch.LongTensor(labels)
    return X, y
