# shape

import os
import subprocess
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import AllChem
from pyshapeit import AlignMol
import multiprocess as mp

from rdkit import RDLogger

from jupyfuncs.pbar import tqdm
from jupyfuncs.norm import Normalizer

# from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PyMol
from rdkit.Chem.Subshape import SubshapeBuilder, SubshapeObjects
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

lg = RDLogger.logger()
lg.setLevel(4)

__all__ = [
    'get_mols_from_smi',
    'get_aligned_mol',
    'get_aligned_sdf',
    'show_alignment',
    'pymol_running'
]


def get_mols_from_smi(probe_smifile):
    mols = []
    with open(probe_smifile) as f:
        for line in f.readlines():
            smi = line.strip()
            mol = None
            try:
                mol = Chem.MolFromSmiles(smi)
            except Exception as e:
                print(e)
            if mol:
                mols.append(mol)
    return mols


def get_aligned_mol(
    ref_mol, probe_mol, num_confs, num_cpu
):

    mol1 = ref_mol
    mol1.SetProp('_Name', 'ref')

    AllChem.EmbedMultipleConfs(
        probe_mol,
        numConfs=num_confs,
        numThreads=num_cpu
    )

    score = 0
    # conf_id = -1
    
    aligned_mol = deepcopy(probe_mol)

    for i in range(probe_mol.GetNumConformers()):

        mol2 = Chem.MolFromMolBlock(
            Chem.MolToMolBlock(probe_mol, confId=i)
        )
        mol2.SetProp('_Name', 'probe')

        sim_score = AlignMol(mol1, mol2)
        if sim_score > score:
            score = sim_score
            aligned_mol = deepcopy(mol2)
#     pbar.update(1)

    return aligned_mol


def get_aligned_mol_mp(
    config
):
    return get_aligned_mol(*config)


def gen_configs(ref_mol, mols, num_confs, num_cpu):
    configs = []
    for probe_mol in mols:
        configs.append((ref_mol, probe_mol, num_confs, num_cpu))
    return configs


def get_aligned_sdf(
    ref_sdf: str,
    probe_smifile: str,
    num_confs=150,
    num_cpu=5,
    num_workers=10,
    output_sdf=None,
    print_info=True,
    norm_mol=True
):
    ref_sdf = os.path.abspath(ref_sdf)
    ref_mol = Chem.SDMolSupplier(ref_sdf)[0]
    if not output_sdf:
        output_sdf = os.path.abspath(probe_smifile) + '.sdf'
    else:
        output_sdf = os.path.abspath(output_sdf)

    mols = get_mols_from_smi(probe_smifile)

    configs = gen_configs(
        ref_mol, mols, num_confs=num_confs, num_cpu=num_cpu
    )
    
    pool = mp.Pool(num_workers)
    aligned_mols = list(
        tqdm(
            pool.imap(get_aligned_mol_mp, configs),
            total=len(mols),
            desc='All mols'
        )
    )
    if norm_mol:
        normer = Normalizer()
    sdwriter = Chem.SDWriter(output_sdf) 
    for mol in aligned_mols:
        if norm_mol:
            mol = normer(mol)
        sdwriter.write(mol)
        sdwriter.flush()
    sdwriter.close()
    return output_sdf

    # out_aligned = output_sdf + 'ali.sdf'
    # score_file = output_sdf + 'score.csv'
    
    # command = f'shape-it -r {ref_sdf} -d {output_sdf} -o {out_aligned} -s {score_file}' 
    # out_info = subprocess.getoutput(command)
    # if print_info:
    #     print(out_info)
    
    # if norm_mol:
    #     fix_path = out_aligned + 'fix.sdf'
    #     mols = Chem.SDMolSupplier(out_aligned)
    #     sdwriter = Chem.SDWriter(fix_path)
    #     for mol in mols:
    #         mol = normer(mol)
    #         sdwriter.write(mol)
    #         sdwriter.flush()
    #     sdwriter.close()
    #     return fix_path
    # else:
    #     return out_aligned
    
    # shape-it -r ref_sdf  -d output_sdf -o out_aligned -s score_file


def show_alignment(
    ref_mol,
    probe_mol,
    gen_confs,
    num_confs=200,
    num_cpu=5,
):
    # should install pymol-open-source
    if not pymol_running():
        subprocess.Popen(['pymol', '-cKRQ'])  

    if isinstance(ref_mol, Chem.Mol):
        mol1 = ref_mol
    elif isinstance(ref_mol, str) and ref_mol.endswith('.sdf'):
        mol1 = Chem.SDMolSupplier(ref_mol)[0]
    
    if isinstance(probe_mol, Chem.Mol):
        mol2 = probe_mol
    elif isinstance(probe_mol, str) and probe_mol.endswith('.sdf'):
        mol2 = Chem.SDMolSupplier(probe_mol)[0]
    else:
        mol2 = Chem.MolFromSmiles(probe_mol)
    
    if gen_confs:
        AllChem.EmbedMultipleConfs(
            mol2,
            numConfs=num_confs,
            numThreads=num_cpu
        )

    score = 0
    for i in range(mol2.GetNumConformers()):

        probe_mol = Chem.MolFromMolBlock(
            Chem.MolToMolBlock(mol2, confId=i)
        )

        sim_score = AlignMol(mol1, probe_mol)
        if sim_score > score:
            score = sim_score
            probe = deepcopy(probe_mol)

    mol1.SetProp('_Name', 'ref')
    probe.SetProp('_Name', 'probe')

    AllChem.CanonicalizeConformer(mol1.GetConformer())
    AllChem.CanonicalizeConformer(probe.GetConformer())

    builder = SubshapeBuilder.SubshapeBuilder()
    builder.gridDims = (20., 20., 10)
    builder.gridSpacing = 0.5
    builder.winRad = 4.

    refShape = builder.GenerateSubshapeShape(mol1)
    probeShape = builder.GenerateSubshapeShape(probe)

    v = PyMol.MolViewer()

    score = AlignMol(mol1, probe)

    v.DeleteAll()

    v.ShowMol(mol1, name='ref', showOnly=False)
    SubshapeObjects.DisplaySubshape(v, refShape, 'ref_Shape')
    v.server.do('set transparency=0.5')

    v.ShowMol(probe, name='probe', showOnly=False)
    SubshapeObjects.DisplaySubshape(v, probeShape, 'prob_Shape')
    v.server.do('set transparency=0.5')

    return v.GetPNG()


def pymol_running() -> bool:
    out_info = subprocess.getoutput('ps aux | grep pymol')
    if '-cKRQ' in out_info:
        return True
    else:
        return False