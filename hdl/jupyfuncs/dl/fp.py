from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


__all__ = [
    'get_fp',
]


def get_rdnorm_fp(smiles):
    from descriptastorus.descriptors import rdNormalizedDescriptors
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]
    arr = np.array(features)
    return arr


def get_maccs_fp(smiles):
    arr = np.zeros(167)
    try:
        mol = Chem.MolFromSmiles(smiles)
        vec = MACCSkeys.GenMACCSKeys(mol)
        bv = list(vec.GetOnBits())
        arr[bv] = 1
    except Exception as e:
        print(e)
    return arr


def get_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    bv = list(vec.GetOnBits())
    arr = np.zeros(1024)
    arr[bv] = 1
    return arr


fp_dict = {
    'rdnorm': get_rdnorm_fp,
    'maccs': get_maccs_fp,
    'morgan': get_morgan_fp
}


def get_fp(smiles, fp='maccs'):
    return fp_dict[fp](smiles)