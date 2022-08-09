from .features_generators import morgan_counts_features_generator
from rxnrep.predict.fingerprint import get_rxnrep_fingerprint


def get_rxn_fp(smiles_rxns):
    return get_rxnrep_fingerprint(smiles_rxns)

