# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/chem/shape.py
# 说明：化学信息学工具（RDKit / Jupyter）
# 模块功能：分子形状（molecular shape）叠合——给探针（probe）分子生成多个三维构象（conformer），
#          用 pyshapeit 的形状相似性打分逐个叠到参考（reference）分子上，取最优构象批量写出对齐后的 SDF，
#          并可在 PyMOL 中用 Subshape 体素（voxel）把形状场可视化。
# shape

import os
import subprocess
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import AllChem
# AlignMol：把一个分子的三维形状与参考分子叠合，返回形状相似性分数（float）
from pyshapeit import AlignMol
import multiprocess as mp

from rdkit import RDLogger

from jupyfuncs.pbar import tqdm
from jupyfuncs.norm import Normalizer

# from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PyMol
from rdkit.Chem.Subshape import SubshapeBuilder, SubshapeObjects
from PIL import ImageFile
# 允许读取被截断的 PNG（PyMOL 返回的截图偶尔不完整时不至于抛异常）
ImageFile.LOAD_TRUNCATED_IMAGES = True

lg = RDLogger.logger()
# RDKit 日志只保留 fatal 级，屏蔽批量构象嵌入与叠合时的刷屏告警
lg.setLevel(4)

__all__ = [
    # SMILES 读取、单个/批量形状对齐、PyMOL 可视化、PyMOL 进程检测
    'get_mols_from_smi',
    'get_aligned_mol',
    'get_aligned_sdf',
    'show_alignment',
    'pymol_running'
]


def get_mols_from_smi(probe_smifile):
    """按行读取 SMILES 文件并解析成分子列表。

    Args:
        probe_smifile: SMILES 文本文件路径，每行一个 SMILES（行首空白会被 strip）。

    Returns:
        list[Chem.Mol]：只保留解析成功的分子，失败行打印异常后跳过。
    """
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
    """把一个探针分子对齐到参考分子：生成多构象，逐个做形状叠合，返回得分最高的那个构象分子。

    Args:
        ref_mol: 参考 Chem.Mol，必须自带三维构象（形状叠合的受体）。
        probe_mol: 探针 Chem.Mol，二维分子即可，函数内部就地给它嵌入构象。
        num_confs: 探针构象（conformer）数量，越多越可能命中好构象但更慢。
        num_cpu: 构象嵌入使用的线程数。

    Returns:
        Chem.Mol：分数最高的探针构象副本（已对齐到参考系）；若探针没有任何构象，
        返回未对齐的 probe_mol 深拷贝。
    """

    mol1 = ref_mol
    # 给参考分子命名，写进 SDF/可视化时便于和 probe 区分
    mol1.SetProp('_Name', 'ref')

    # 随机坐标+距离几何（ETKDG）嵌入多个三维构象，作为候选形状
    AllChem.EmbedMultipleConfs(
        probe_mol,
        numConfs=num_confs,
        numThreads=num_cpu
    )

    score = 0
    # conf_id = -1
    
    aligned_mol = deepcopy(probe_mol)

    for i in range(probe_mol.GetNumConformers()):

        # MolToMolBlock 只导出第 i 个构象，再读回成独立分子，
        # 这样每次叠合的都是单构象分子，AlignMol 结果可直接比较
        mol2 = Chem.MolFromMolBlock(
            Chem.MolToMolBlock(probe_mol, confId=i)
        )
        mol2.SetProp('_Name', 'probe')

        # 形状相似性打分（高斯形状重叠式），分数越大表示形状越贴合参考
        sim_score = AlignMol(mol1, mol2)
        if sim_score > score:
            score = sim_score
            aligned_mol = deepcopy(mol2)
#     pbar.update(1)

    return aligned_mol


def get_aligned_mol_mp(
    config
):
    """多进程包装：config 是 (ref_mol, probe_mol, num_confs, num_cpu) 元组，解包后转调 get_aligned_mol。"""
    return get_aligned_mol(*config)


def gen_configs(ref_mol, mols, num_confs, num_cpu):
    """为每个探针分子复制同一份参考与参数，生成 get_aligned_mol 的参数元组列表（供进程池分发）。"""
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
    """批量形状对齐：读参考 SDF 的第一条记录，把 SMILES 文件里的分子逐个对齐并写成对齐后的 SDF。

    Args:
        ref_sdf: 参考分子 SDF 路径（取其第一条记录，需含三维坐标）。
        probe_smifile: 探针 SMILES 文件路径，每行一个。
        num_confs: 每个探针生成的构象数。
        num_cpu: 单分子构象嵌入线程数。
        num_workers: 进程池进程数。
        output_sdf: 输出路径；为 None 时用 "<probe_smifile 绝对路径>.sdf"。
        print_info: 当前实现未使用（相关信息打印逻辑在已停用的 shape-it 分支里）。
        norm_mol: 是否在对齐后再做一次归一化（normalization）修正电荷/官能团写法。

    Returns:
        str：实际写出的 SDF 绝对路径。
    """
    ref_sdf = os.path.abspath(ref_sdf)
    # 参考分子取 SDF 第一条记录，其坐标即形状叠合的目标系
    ref_mol = Chem.SDMolSupplier(ref_sdf)[0]
    if not output_sdf:
        output_sdf = os.path.abspath(probe_smifile) + '.sdf'
    else:
        output_sdf = os.path.abspath(output_sdf)

    mols = get_mols_from_smi(probe_smifile)

    configs = gen_configs(
        ref_mol, mols, num_confs=num_confs, num_cpu=num_cpu
    )
    
    # 进程池并行对齐各探针，imap 保证结果顺序与输入一致，tqdm 显示进度
    pool = mp.Pool(num_workers)
    aligned_mols = list(
        tqdm(
            pool.imap(get_aligned_mol_mp, configs),
            total=len(mols),
            desc='All mols'
        )
    )
    # 可选：对齐后逐分子做归一化，消除叠合过程中暴露出的分离电荷写法
    if norm_mol:
        normer = Normalizer()
    sdwriter = Chem.SDWriter(output_sdf) 
    # 边算边写：每写一条就 flush，批跑中途也能查看已完成的分子
    for mol in aligned_mols:
        if norm_mol:
            mol = normer(mol)
        sdwriter.write(mol)
        sdwriter.flush()
    sdwriter.close()
    return output_sdf

    # 以下整段为已停用的备选实现：调用外部 shape-it 命令行做叠合，并额外输出打分 CSV
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
    """在 PyMOL 里可视化参考分子与探针分子的最佳形状叠合结果。

    Args:
        ref_mol: 参考分子，Chem.Mol 或以 ".sdf" 结尾的路径（取第一条记录）。
        probe_mol: 探针分子，Chem.Mol、".sdf" 路径，否则按 SMILES 字符串解析。
        gen_confs: 是否先给探针嵌入 num_confs 个构象；为假时直接用分子已有构象。
        num_confs: 嵌入构象数量。
        num_cpu: 嵌入线程数。

    Returns:
        PyMOL 场景截图（PNG 字节串/图像对象，来自 v.GetPNG()）。
    """
    # should install pymol-open-source
    # 没有带 -cKRQ 的 PyMOL 在跑就后台拉起一个无界面实例，供 MolViewer 连接
    if not pymol_running():
        subprocess.Popen(['pymol', '-cKRQ'])  

    # 输入既可以是 RDKit 分子，也可以是 SDF 路径（取第一条记录）
    if isinstance(ref_mol, Chem.Mol):
        mol1 = ref_mol
    elif isinstance(ref_mol, str) and ref_mol.endswith('.sdf'):
        mol1 = Chem.SDMolSupplier(ref_mol)[0]
    
    # 探针比参考多一种兜底：不是 Mol 也不是 .sdf 路径时，按 SMILES 字符串解析
    if isinstance(probe_mol, Chem.Mol):
        mol2 = probe_mol
    elif isinstance(probe_mol, str) and probe_mol.endswith('.sdf'):
        mol2 = Chem.SDMolSupplier(probe_mol)[0]
    else:
        mol2 = Chem.MolFromSmiles(probe_mol)
    
    # gen_confs 为真时给探针嵌入多个构象，否则直接沿用分子自带的构象
    if gen_confs:
        AllChem.EmbedMultipleConfs(
            mol2,
            numConfs=num_confs,
            numThreads=num_cpu
        )

    # 逐个构象单独叠合，保留分数最高（严格大于当前最优）的那个
    score = 0
    for i in range(mol2.GetNumConformers()):

        # 把第 i 个构象固化成单构象分子，AlignMol 才能对单个形状做变换
        probe_mol = Chem.MolFromMolBlock(
            Chem.MolToMolBlock(mol2, confId=i)
        )

        sim_score = AlignMol(mol1, probe_mol)
        if sim_score > score:
            score = sim_score
            probe = deepcopy(probe_mol)

    mol1.SetProp('_Name', 'ref')
    probe.SetProp('_Name', 'probe')

    # 把两个构象都规范化到同一坐标系（质心平移到原点、按惯量主轴定向），
    # 这样下面生成的体素网格（voxel grid）才能覆盖同一空间做比较
    AllChem.CanonicalizeConformer(mol1.GetConformer())
    AllChem.CanonicalizeConformer(probe.GetConformer())

    # Subshape 形状场参数：网格 20x20x10 格、格间距 0.5 Å、探针特征窗半径 4.0 Å
    builder = SubshapeBuilder.SubshapeBuilder()
    builder.gridDims = (20., 20., 10)
    builder.gridSpacing = 0.5
    builder.winRad = 4.

    # 生成形状场：把原子轮廓、疏水/极性特征投影到体素上，供 PyMOL 显示为半透明曲面
    refShape = builder.GenerateSubshapeShape(mol1)
    probeShape = builder.GenerateSubshapeShape(probe)

    v = PyMol.MolViewer()

    # 用规范化后的最终构象再算一次形状相似度（该分数在当前实现里没有返回给调用方）
    score = AlignMol(mol1, probe)

    v.DeleteAll()

    # 参考与探针同屏显示，各自叠加半透明（transparency=0.5）形状面
    # 两条分子都留在场景里（showOnly=False），形状面各自设 50% 透明以便重叠观察
    v.ShowMol(mol1, name='ref', showOnly=False)
    SubshapeObjects.DisplaySubshape(v, refShape, 'ref_Shape')
    v.server.do('set transparency=0.5')

    v.ShowMol(probe, name='probe', showOnly=False)
    SubshapeObjects.DisplaySubshape(v, probeShape, 'prob_Shape')
    v.server.do('set transparency=0.5')

    # 取 PyMOL 当前场景截图
    return v.GetPNG()


def pymol_running() -> bool:
    """检测是否已有以 -cKRQ 启动的 PyMOL 进程（即本模块会拉起的无界面模式）。

    Returns:
        bool：ps 输出里出现 '-cKRQ' 即认为在运行。
    """
    out_info = subprocess.getoutput('ps aux | grep pymol')
    if '-cKRQ' in out_info:
        return True
    else:
        return False