from hdl.layers.general.linear import (
    MultiTaskMultiClassBlock,
    MuMcHardBlock
)
from .linear import MMIterLinear
from .chiral_gnn import GNN
from .ginet import GINet
from .ginet import GINMLPR


MODEL_DICT = {
    'rxn_trans': MultiTaskMultiClassBlock,
    'rxn_trans_hard': MuMcHardBlock,
    'mmiter_linear': MMIterLinear,
    'chiral_gnn': GNN,
    'ginet': GINet,
    'ginmlpr': GINMLPR
}