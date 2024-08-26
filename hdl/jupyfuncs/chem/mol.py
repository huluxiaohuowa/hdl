# Jupyter funcs
import os
import re
import itertools
from copy import deepcopy
# from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw.IPythonConsole import addMolToView
# from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from rdkit.Chem import AllChem
from ipywidgets import (
    interact,
    # interactive,
    fixed,
)
from rdkit.Chem.rdRGroupDecomposition import (
    RGroupDecomposition,
    # RGroupDecompositionParameters,
    # RGroupMatching,
    # RGroupScore,
    # RGroupLabels,
    # RGroupCoreAlignment,
    RGroupLabelling
)
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
from IPython.display import HTML
# from rdkit import rdBase
from IPython.display import display

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem import rdqueries
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Geometry
rdDepictor.SetPreferCoordGen(True)
import pandas as pd
from PIL import Image as pilImage
from io import BytesIO
from IPython.display import SVG, Image
from ipywidgets import interact
import molvs as mv


IPythonConsole.ipython_useSVG = True
IPythonConsole.molSize = (450, 350)
params = Chem.SubstructMatchParameters()
params.aromaticMatchesConjugated = True 

__all__ = [
    'draw_mol',
    'draw_confs',
    'show_decomp',
    'get_ids_folds',
    'show_pharmacophore',
    'mol_without_indices',
    'norm_colors',
    'drawmol_with_hi',
    'draw_mols_surfs',
]


COLORS = {
    # "Tol" colormap from https://davidmathlogic.com/colorblind
    'tol': [
        (51, 34, 136),
        (17, 119, 51),
        (68, 170, 153),
        (136, 204, 238),
        (221, 204, 119),
        (204, 102, 119),
        (170, 68, 153),
        (136, 34, 85)
    ],
    # "IBM" colormap from https://davidmathlogic.com/colorblind
    'ibm': [
        (100, 143, 255),
        (120, 94, 240),
        (220, 38, 127),
        (254, 97, 0),
        (255, 176, 0)
    ],
    # Okabe_Ito colormap from https://jfly.uni-koeln.de/color/
    'okabe': [
        (230, 159, 0),
        (86, 180, 233),
        (0, 158, 115),
        (240, 228, 66),
        (0, 114, 178),
        (213, 94, 0),
        (204, 121, 167)
    ]
}


# def get_his_for_onemol(mol_sm, pat_sm):
#     atom_ids = []
#     bond_ids = []
#     m = Chem.MolFromSmiles(mol_sm)
#     pt = Chem.MolFromSmiles(pat_sm)
#     hi_id = m.GetSubstructMatches(pt, params=params)
#     if len(m.GetSubstructMatches(pt, params=params)) == 0:
#         Chem.Kekulize(m)       
#         hi_id = m.GetSubstructMatches(pt, params=params) 
#     if len(hi_id) == 0:
#         return
#     atom_ids.append(itertools.chain.from_iterable(hi_id))


# def get_match_his(mol_sms, pat_sms):
#     highlightatoms = defaultdict(list)
#     highlightbonds = defaultdict(list)
#     for i in range(len(df)):
#         try:
#             mm = df.iloc[i, 0][2:-2]
#             pm = df.iloc[i, 2]
#             m = Chem.MolFromSmiles(mm) 
#             pt = Chem.MolFromSmiles(pm)
#             hi_id = m.GetSubstructMatches(pt, params=params)
#             if len(m.GetSubstructMatches(pt, params=params)) == 0:
#                 Chem.Kekulize(m)       
#                 hi_id = m.GetSubstructMatches(pt, params=params)
#             mols.append(m)
#             hi_ids.append(hi_id)
#         except:
#             pass
#         pass


def norm_colors(colors=COLORS):
    colors = deepcopy(COLORS)
    for k, v in colors.items():
        for i, color in enumerate(v):
            colors[k][i] = tuple(y / 255 for y in color)
    return colors


def drawmol_with_hi(
    mol,
    legend,
    atom_hi_dict,
    bond_hi_dict,
    atomrads_dict,
    widthmults_dict,
    width=350,
    height=200,
):
    d2d = rdMolDraw2D.MolDraw2DCairo(width, height)
    d2d.ClearDrawing()
    d2d.DrawMoleculeWithHighlights(
        mol, legend, 
        atom_hi_dict,
        bond_hi_dict, 
        atomrads_dict,
        widthmults_dict
    )
    d2d.FinishDrawing()
    png = d2d.GetDrawingText()
    return png


def show_atom_number(mol, label='atomNote'):
    new_mol = deepcopy(mol)
    for atom in new_mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return new_mol


def moltosvg(mol, molSize=(500, 500), kekulize=True):
    mc = mol
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:', '')


def draw_mol(mol):
    return SVG(moltosvg(show_atom_number(mol)))


def drawit(m, p, confId=-1):
    mb = Chem.MolToMolBlock(m, confId=confId)
    p.removeAllModels()
    p.addModel(mb, 'sdf')
    p.setStyle({'stick': {}})
    p.setBackgroundColor('0xeeeeee')
    p.zoomTo()
    return p.show()


def draw_confs(m):
    import py3Dmol
    p = py3Dmol.view(width=500, height=500)
    return interact(drawit,
                    m=fixed(m),
                    p=fixed(p),
                    confId=(0, m.GetNumConformers() - 1))


def do_decomp(mols, cores, options):
    options.rgroupLabelling = RGroupLabelling.AtomMap
    decomp = RGroupDecomposition(cores, options)
    for mol in mols:
        decomp.Add(mol)
    decomp.Process()
    return decomp


def show_decomp(mols, cores, options, item=False):
    decomp = do_decomp(mols, cores, options)
    if item:
        rows = decomp.GetRGroupsAsRows()
        items = [
            '{}:{}'.format(
                group, Chem.MolToSmiles(row[group])
            )
            for row in rows for group in row
        ]
        return ' '.join(items)
    else:
        cols = decomp.GetRGroupsAsColumns()
        cols['mol'] = mols
        cols['input core'] = cores[0]
        df = pd.DataFrame(cols)
        PandasTools.ChangeMoleculeRendering(df)
        return HTML(df.to_html())


def get_ids_folds(id_list, num_folds, need_shuffle=False):
    if need_shuffle:
        from random import shuffle
        shuffle(id_list)
    num_ids = len(id_list)
    assert num_ids >= num_folds
    
    num_each_fold = int(num_ids / num_folds)
    
    blocks = []
    
    for i in range(num_folds):
        start = num_each_fold * i
        end = start + num_each_fold
        if end > num_ids - 1:
            end = num_ids - 1
        
        blocks.append(id_list[start: end])
    
    id_blocks = []
    for i in range(num_folds):
        id_blocks.append(
            (list(itertools.chain.from_iterable([blocks[j] for j in range(num_folds) if j != i])),
             blocks[i])
        )
        
    return id_blocks


keep = ["Donor", "Acceptor", "Aromatic", "Hydrophobe", "LumpedHydrophobe"]


def show_pharmacophore(
    sdf_path,
    keep=keep,
    fdf_dir=os.path.join(
        os.path.dirname(__file__),
        "..",
        "datasets",
        'defined_BaseFeatures.fdef'
    )
):
    template_mol = [m for m in Chem.SDMolSupplier(sdf_path)][0]
    fdef = AllChem.BuildFeatureFactory(
        fdf_dir
    )
    prob_feats = fdef.GetFeaturesForMol(template_mol)
    prob_feats = [f for f in prob_feats if f.GetFamily() in keep]
    # prob_points = [list(x.GetPos()) for x in prob_feats]

    for i, feat in enumerate(prob_feats):
        atomids = feat.GetAtomIds()
        print(
            "pharamcophore index:{0}; feature:{1}; type:{2}; atom id:{3}".format(
                i, 
                feat.GetFamily(), 
                feat.GetType(),
                atomids
            )
        )
        display(
            Draw.MolToImage(
                template_mol,
                highlightAtoms=list(atomids),
                highlightColor=[0, 1, 0],
                useSVG=True
            )
        )


def mol_without_indices( 
    mol_input: Chem.Mol, 
    remove_indices=[], 
    keep_properties=[] 
): 
     
    atom_list, bond_list, idx_map = [], [], {}  # idx_map: {old: new} 
    for atom in mol_input.GetAtoms(): 
         
        props = {} 
        for property_name in keep_properties: 
            if property_name in atom.GetPropsAsDict(): 
                props[property_name] = atom.GetPropsAsDict()[property_name] 
        symbol = atom.GetSymbol() 
         
        if symbol.startswith('*'): 
            atom_symbol = '*' 
            props['molAtomMapNumber'] = atom.GetAtomMapNum() 
        elif symbol.startswith('R'): 
            atom_symbol = '*' 
            if len(symbol) > 1: 
                atom_map_num = int(symbol[1:]) 
            else: 
                atom_map_num = atom.GetAtomMapNum() 
            props['dummyLabel'] = 'R' + str(atom_map_num) 
            props['_MolFileRLabel'] = str(atom_map_num) 
            props['molAtomMapNumber'] = atom_map_num 
             
        else: 
            atom_symbol = symbol 
        atom_list.append( 
            ( 
                atom_symbol, 
                atom.GetFormalCharge(), 
                atom.GetNumExplicitHs(), 
                props 
            ) 
        ) 
    for bond in mol_input.GetBonds(): 
        bond_list.append( 
            ( 
                bond.GetBeginAtomIdx(), 
                bond.GetEndAtomIdx(), 
                bond.GetBondType() 
            ) 
        ) 
    mol = Chem.RWMol(Chem.Mol()) 
     
    new_idx = 0 
    for atom_index, atom_info in enumerate(atom_list): 
        if atom_index not in remove_indices: 
            atom = Chem.Atom(atom_info[0]) 
            atom.SetFormalCharge(atom_info[1]) 
            atom.SetNumExplicitHs(atom_info[2]) 
             
            for property_name in atom_info[3]: 
                if isinstance(atom_info[3][property_name], str): 
                    atom.SetProp(property_name, atom_info[3][property_name]) 
                elif isinstance(atom_info[3][property_name], int): 
                    atom.SetIntProp(property_name, atom_info[3][property_name]) 
            mol.AddAtom(atom) 
            idx_map[atom_index] = new_idx 
            new_idx += 1 
    for bond_info in bond_list: 
        if ( 
            bond_info[0] not in remove_indices 
            and bond_info[1] not in remove_indices 
        ): 
            mol.AddBond( 
                idx_map[bond_info[0]], 
                idx_map[bond_info[1]], 
                bond_info[2] 
            ) 
        else: 
            one_in = False 
            if ( 
                (bond_info[0] in remove_indices) 
                and (bond_info[1] not in remove_indices) 
            ): 
                keep_index = bond_info[1] 
                # remove_index = bond_info[0] 
                one_in = True 
            elif ( 
                (bond_info[1] in remove_indices) 
                and (bond_info[0] not in remove_indices) 
            ): 
                keep_index = bond_info[0] 
                # remove_index = bond_info[1] 
                one_in = True 
            if one_in:  
                if atom_list[keep_index][0] == 'N': 
                    old_num_explicit_Hs = mol.GetAtomWithIdx( 
                        idx_map[keep_index] 
                    ).GetNumExplicitHs() 

                    mol.GetAtomWithIdx(idx_map[keep_index]).SetNumExplicitHs( 
                        old_num_explicit_Hs + 1 
                    ) 
    mol = Chem.Mol(mol) 
    return mol


def draw_mols_surfs(
    mols,
    width=400,
    height=400,
    surface=True,
    surface_opacity=0.5
):
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor('0xeeeeee')
    view.removeAllModels()
    for mol in mols:
        addMolToView(mol, view)
    if surface:
        view.addSurface(
            py3Dmol.SAS,
            {'opacity': surface_opacity}
        )
    view.zoomTo()
    return view.show()


def draw_rxn(
    rxn_smiles,
    use_smiles: bool = True,
):
    rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=use_smiles)
    d2d = Draw.MolDraw2DCairo(2000, 500)
    d2d.DrawReaction(rxn, highlightByReactant=True)
    png = d2d.GetDrawingText()
    display(Image(png))


def react(rxn_smarts, reagents):
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        # n_reactants = rxn.GetNumReactantTemplates()
        products = rxn.RunReactants([
            Chem.MolFromSmiles(smi) for smi in reagents
        ])
        return products
    except Exception as e:
        print(e)
        return []


def match_pattern(mol, patt):
    if mol:
        return mol.HasSubstructMatch(patt)
    else:
        return False


def split_rxn_smiles(smi):
    try:
        reagents1, reagents2, products = smi.split('>')
        if len(reagents2) > 0:
            reagents = '.'.join([reagents1, reagents2])
        else:
            reagents = reagents1
        return reagents, products
    except Exception as e:
        print(e)
        return '', ''


def find_mprod(rxn_smi):
    # ref: https://github.com/LiamWilbraham/uspto-analysis/blob/master/reaction-stats-uspto.ipynb
    rxn_smarts = '[C:1](=[O:2])-[OD1].[N!H0:3]>>[C:1](=[O:2])[N:3]'
    patt_acid = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    patt_amine = Chem.MolFromSmarts('[N;H3,H2,H1]')  # ammonia or primary/secondary amine
    
    products = split_rxn_smiles(rxn_smi)[1].split('.')
        
    reactants = [r for r in split_rxn_smiles(rxn_smi)[0].split('.')]
    cooh = [
        r
        for r in reactants
        if match_pattern(Chem.MolFromSmiles(r), patt_acid)
    ]
    
    cooh = [re.sub('@', '', i) for i in cooh]
    
    amine = [
        r for r in reactants
        if match_pattern(Chem.MolFromSmiles(r), patt_amine)
    ]
    amine = [re.sub('@', '', i) for i in amine]

    for perm in itertools.product(cooh, amine):
        
        cooh_i = perm[0]
        amine_i = perm[1]
        
        smarts_products = react(rxn_smarts, perm)

        for p_1 in smarts_products:
            for p_2 in products:
                p_2 = re.sub('@', '', p_2)
                patt = Chem.MolFromSmiles(p_2)  
                if Chem.MolToInchiKey(p_1[0]) == Chem.MolToInchiKey(patt):  
                    return cooh_i, amine_i, p_2
    return None


def get_largest_mol(smiles, to_smiles=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    mol_frags = rdmolops.GetMolFrags(mol, asMols=True)
    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    if to_smiles:
        return mv.standardize_smiles(Chem.MolToSmiles(largest_mol))
    return largest_mol


def standardize_tautomer(mol, max_tautomers=1000):
    params = rdMolStandardize.CleanupParameters()
    params.maxTautomers = max_tautomers
    enumerator = rdMolStandardize.TautomerEnumerator(params)
    cm = enumerator.Canonicalize(mol)
    return cm


def reorder_tautomers(m):
    enumerator = rdMolStandardize.TautomerEnumerator()
    canon = enumerator.Canonicalize(m)
    csmi = Chem.MolToSmiles(canon)
    res = [canon]
    tauts = enumerator.Enumerate(m)
    smis = [Chem.MolToSmiles(x) for x in tauts]
    stpl = sorted(
        (x, y) for x, y in zip(smis, tauts) if x!=csmi
    )
    res += [y for _, y in stpl]
    return res