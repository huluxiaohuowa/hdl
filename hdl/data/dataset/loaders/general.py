import typing as t

import torch.utils.data as tud

from hdl.data.dataset.loaders.collate_funcs.fp import fp_collate


class Loader(tud.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 12,
        collate_fn: t.Callable = fp_collate
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )