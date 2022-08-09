from typing import Callable, List, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]


FEATURES_GENERATOR_REGISTRY = {}


FP_BITS_DICT = {
    'maccs': 167,
    'morgan': 2048,
    'morgan_count': 2048,
    'rdkit_2d_normalized': 200,
    'rdkit_2d': 200,
}


def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """
    Creates a decorator which registers a features generator in a global dictionary to enable access by name.

    :param features_generator_name: The name to use to access the features generator.
    :return: A decorator which will add a features generator to the registry using the specified name.
    """
    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    """
    Gets a registered features generator by name.

    :param features_generator_name: The name of the features generator.
    :return: The desired features generator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns a list of names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())


MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048


@register_features_generator('morgan')
def morgan_binary_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a binary Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the binary Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


@register_features_generator('morgan_count')
def morgan_counts_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a counts-based Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the counts-based Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


@register_features_generator('maccs')
def macss_features_generator(
    mol
) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    vec = MACCSkeys.GenMACCSKeys(mol)
    bv = list(vec.GetOnBits())
    arr = np.zeros(167)
    arr[bv] = 1
    return arr


try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]

        return features

    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D normalized features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]

        return features
except ImportError:
    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
        raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                          '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D features.')

    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
        """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
        raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                          '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D normalized features.')


@register_features_generator('e3fp')
def e3fp_features_generator(mol: Molecule) -> np.ndarray:
    """
    E3FP is a 3D molecular fingerprinting method inspired by Extended Connectivity FingerPrints (ECFP),

    [LINK](https://pubs.acs.org/doi/10.1021/acs.jmedchem.7b00696)
    Axen SD, Huang XP, Caceres EL, Gendelev L, Roth BL, Keiser MJ. 
    A Simple Representation Of Three-Dimensional Molecular Structure. 
    J. Med. Chem. 60 (17): 7393–7409 (2017).

    The source code: https://github.com/keiserlab/e3fp

    :param mol: A molecule(i.e., either a SMILES or an RDKit molecule).
    :return: A 1D numpy array containing the E3FP fingerprints
    """
    return NotImplemented


@register_features_generator('whales')
def whales_features_generator(mol: Molecule) -> np.ndarray:
    """
    WHALES descriptor is a Weighted Holistic Atom Localization and Entity Shape (WHALES) 
    descriptors starting from an rdkit supplier file.

    [LINK](https://www.nature.com/articles/s42004-018-0043-x)
    Francesca Grisoni, Daniel Merk, Viviana Consonni, Jan A. Hiss, 
    Sara Giani Tagliabue, Roberto Todeschini & Gisbert Schneider 
    "Scaffold hopping from natural products to synthetic mimetics by 
    holistic molecular similarity", Nature Communications Chemistry 1, 44, 2018.

    The source code: https://github.com/grisoniFr/whales_descriptors

    :param mol: A molecule(i.e., either a SMILES or an RDKit molecule).
    :return: A 2D numpy array containing the WHALES descriptors.
    """
    return NotImplemented


@register_features_generator('selfies')
def selfies_features_generator(mol) -> np.ndarray:
    """
    Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation

    A main objective is to use SELFIES as direct input into machine learning models,
    in particular in generative models, for the generation of molecular graphs
    which are syntactically and semantically valid.

    [LINK](https://iopscience.iop.org/article/10.1088/2632-2153/aba947)
    Mario Krenn et al 2020 Mach. Learn.: Sci. Technol. 1 045024

    The source code: https://github.com/aspuru-guzik-group/selfies

    :param mol: A molecule(i.e., either a SMILES or an RDKit molecule).
    :return: A 1D numpy array containing the symbols of input molecule in SELFIES style.
    """
    return NotImplemented


"""
Custom features generator template.

Note: The name you use to register the features generator is the name
you will specify on the command line when using the --features_generator <name> flag.
Ex. python train.py ... --features_generator custom ...

@register_features_generator('custom')
def custom_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    features = np.array([0, 0, 1])

    return features
"""
