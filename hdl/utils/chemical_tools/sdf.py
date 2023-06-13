from rdkit import Chem
import pandas as pd


def sdf2df(
    sdf_file,
    id_col: str = 'Molecule Name',
    target_col: str = 'Average △G (kcal/mol)'
):
    supp = Chem.SDMolSupplier(sdf_file)
    mol_dict_list = []
    for mol in supp:
        smiles = Chem.MolToSmiles(mol)
        mol_dict = mol.GetPropsAsDict()
        mol_dict['smiles'] = smiles
        mol_dict_list.append(mol_dict)
        mol_dict['y'] = mol_dict.pop(target_col)
        mol_dict['name'] = mol_dict.pop(id_col)
    df = pd.DataFrame(mol_dict_list)
    return df