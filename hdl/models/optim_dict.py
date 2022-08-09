from torch.optim import (
    Adadelta,
    Adam,
    SGD,
    RMSprop,
)
from xdl.optims.nadam import Nadam


OPTIM_DICT = {
    'adam': Adam,
    'adadelta': Adadelta,
    'sgd': SGD,
    'rmsprop': RMSprop,
    'nadam': Nadam,
}