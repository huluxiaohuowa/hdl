# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/chem/mol.py
# 说明：化学信息学工具（RDKit / Jupyter）
# 模块功能：Jupyter 下的 RDKit 分子可视化与分子操作助手——2D/3D 绘图与高亮、R 基团分解（R-group decomposition）展示、
#          反应 SMILES 拆分与酰胺偶联产物回溯、互变异构体（tautomer）标准化与排序、按原子索引删原子重建分子。
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
# 2D 坐标生成优先用 CoordGen（比 RDKit 自带算法更快、排版更规整）
rdDepictor.SetPreferCoordGen(True)
import pandas as pd
from PIL import Image as pilImage
from io import BytesIO
from IPython.display import SVG, Image
from ipywidgets import interact
# molvs 在本模块只用到 standardize_smiles，对最大片段做 SMILES 标准化
import molvs as mv


# Notebook 内联显示分子时改用 SVG 矢量图，并设定默认绘图尺寸
IPythonConsole.ipython_useSVG = True
IPythonConsole.molSize = (450, 350)
# 子结构匹配参数：允许芳香原子与共轭（conjugated）非芳香原子互相匹配，
# 这样酮/烯等共轭体系的 SMARTS 在 Kekulize 前后都能命中
params = Chem.SubstructMatchParameters()
params.aromaticMatchesConjugated = True 

__all__ = [
    # 供外部 `from ... import *` 使用的绘图/展示辅助函数
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


# 三套色盲友好（colorblind-friendly）调色板，RGB 分量为 0-255 整数，供高亮按序取色
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
    """把调色板的 0-255 整数 RGB 归一到 RDKit 绘图要求的 0-1 浮点区间。

    Args:
        colors: 形参未被使用，函数实际总是深拷贝模块级 COLORS。

    Returns:
        dict：{调色板名: [(r, g, b), ...]}，分量已除以 255。
    """
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
    """用 Cairo 后端绘制带高亮的单个分子，返回 PNG 字节串。

    Args:
        mol: 待绘制的 Chem.Mol。
        legend: 分子下方的文字标签。
        atom_hi_dict: {原子索引: (r, g, b)} 原子高亮色。
        bond_hi_dict: {键索引: (r, g, b)} 键高亮色。
        atomrads_dict: {原子索引: 半径} 原子高亮圆圈大小。
        widthmults_dict: {原子索引: 倍数} 与该原子相连键的宽度倍数。
        width / height: 画布像素尺寸。

    Returns:
        bytes：PNG 图像二进制内容（未 display，需自行交给 IPython.display.Image）。
    """
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
    """深拷贝分子并把每个原子的索引写成绘图标签属性，用于在图上标注原子序号。

    Args:
        mol: 输入 Chem.Mol（不被修改）。
        label: RDKit 绘图识别的注释属性名，默认 'atomNote'。

    Returns:
        Chem.Mol：带原子序号标签的新分子副本。
    """
    new_mol = deepcopy(mol)
    for atom in new_mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return new_mol


def moltosvg(mol, molSize=(500, 500), kekulize=True):
    """把分子渲染成 SVG 文本，并剥掉内层的 'svg:' 标签名以便 Notebook 嵌套显示。

    Args:
        mol: Chem.Mol。
        molSize: (宽, 高) 像素。
        kekulize: 形参未被使用，实际不做 Kekulize，直接画 RDKit 感知到的芳香性。

    Returns:
        str：SVG 源码字符串。
    """
    mc = mol
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:', '')


def draw_mol(mol):
    """在 Notebook 里显示分子结构图，并在每个原子旁标出原子索引。

    Returns:
        IPython.display.SVG：可直接展示的 SVG 对象。
    """
    return SVG(moltosvg(show_atom_number(mol)))


def drawit(m, p, confId=-1):
    """把分子的指定构象（conformer）以棍状模型送进已存在的 py3Dmol 视图并显示。

    Args:
        m: Chem.Mol，需含三维构象。
        p: py3Dmol view 对象（复用同一个视图，每次先清空模型）。
        confId: 构象索引，-1 表示 RDKit 默认（第一个/唯一）构象。

    Returns:
        py3Dmol 视图的 show() 返回值（JS 显示指令对象）。
    """
    mb = Chem.MolToMolBlock(m, confId=confId)
    p.removeAllModels()
    p.addModel(mb, 'sdf')
    p.setStyle({'stick': {}})
    p.setBackgroundColor('0xeeeeee')
    p.zoomTo()
    return p.show()


def draw_confs(m):
    """为分子的每个三维构象生成一个可拖动滑块的交互式 3D 查看器。

    Args:
        m: Chem.Mol，构象数决定滑块取值范围 (0, GetNumConformers()-1)。

    Returns:
        ipywidgets.interact 控件：滑块切换 confId，视图复用同一 py3Dmol view。
    """
    import py3Dmol
    p = py3Dmol.view(width=500, height=500)
    return interact(drawit,
                    m=fixed(m),
                    p=fixed(p),
                    confId=(0, m.GetNumConformers() - 1))


def do_decomp(mols, cores, options):
    """做 R 基团分解（R-group decomposition）：把一批分子相对给定母核（core）切成骨架 + 取代基（R group）。

    Args:
        mols: 待拆分的 Chem.Mol 列表。
        cores: 母核/骨架 Chem.Mol 列表（作为匹配模板）。
        options: rdRGroupDecomposition.RGroupDecompositionParameters 对象，本函数会强制改写其
                 rgroupLabelling 为 AtomMap（按原子映射号给 R 基团命名）。

    Returns:
        RGroupDecomposition 对象，已 Process()，可取 GetRGroupsAsRows()/GetRGroupsAsColumns()。
    """
    # 用原子映射号（atom map number）标记各 R 位点，使不同分子的同一取代位置列名一致
    options.rgroupLabelling = RGroupLabelling.AtomMap
    decomp = RGroupDecomposition(cores, options)
    for mol in mols:
        decomp.Add(mol)
    # 统一做骨架匹配与 R 基团切分，之后才能取行/列视图
    decomp.Process()
    return decomp


def show_decomp(mols, cores, options, item=False):
    """展示 R 基团分解（R-group decomposition）结果。

    Args:
        mols: 待拆分分子列表。
        cores: 母核列表（列展示时 'input core' 取 cores[0]）。
        options: RGroupDecompositionParameters 对象。
        item: True 则返回纯文本摘要，False（默认）返回可显示的 HTML 表格。

    Returns:
        str 或 IPython.display.HTML：文本形如 "R基团名:SMILES ..." 空格拼接；
        HTML 表格含各 R 列、分子图（mol 列）与输入母核列。
    """
    decomp = do_decomp(mols, cores, options)
    if item:
        # 逐行逐列展开成 "分组:SMILES"，便于把结果塞进一行文本/日志
        rows = decomp.GetRGroupsAsRows()
        items = [
            '{}:{}'.format(
                group, Chem.MolToSmiles(row[group])
            )
            for row in rows for group in row
        ]
        return ' '.join(items)
    else:
        # 按列取 R 基团（每列对应一个取代位点，该位点缺失的分子填 None），并附上原始分子与母核列
        cols = decomp.GetRGroupsAsColumns()
        cols['mol'] = mols
        cols['input core'] = cores[0]
        df = pd.DataFrame(cols)
        # 让 PandasTools 把 mol 列渲染成结构图而不是对象字串
        PandasTools.ChangeMoleculeRendering(df)
        return HTML(df.to_html())


def get_ids_folds(id_list, num_folds, need_shuffle=False):
    """把 id 列表切成 k 折，返回每折的 (训练 id, 验证 id) 组合（交叉验证用）。

    Args:
        id_list: id 序列（need_shuffle 为真时会被就地打乱，即原地修改传入列表）。
        num_folds: 折数 k，要求 len(id_list) >= k，否则断言失败。
        need_shuffle: 是否先随机打乱再分折。

    Returns:
        list[tuple[list, list]]：长度 k，第 i 项为 (其余折合并的训练 id, 第 i 折验证 id)。
        每折大小固定为 int(N/k)，且最后一折右边界被截到 N-1，
        所以 N 不能被 k 整除时末尾若干 id 不会进入任何一折。
    """
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
        # 第 i 块作验证集，其余块拼接成训练集
        id_blocks.append(
            (list(itertools.chain.from_iterable([blocks[j] for j in range(num_folds) if j != i])),
             blocks[i])
        )
        
    return id_blocks


# 药效团特征族（pharmacophore feature family）白名单：氢键供体/受体、芳香性、疏水基团
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
    """检测单个分子结构上的药效团特征（pharmacophore features），逐个高亮打印出来。

    Args:
        sdf_path: SDF 路径，只取第一条记录作为待分析分子。
        keep: 保留的特征族名集合，不在列表里的族（如正/负电离中心）被过滤掉。
        fdf_dir: 特征定义文件（.fdef）路径，决定各特征的 SMARTS 定义。

    Returns:
        None：结果通过 print 输出索引/族名/类型/原子 id，并用 display 逐特征显示高亮图。
    """
    template_mol = [m for m in Chem.SDMolSupplier(sdf_path)][0]
    fdef = AllChem.BuildFeatureFactory(
        fdf_dir
    )
    # 按 .fdef 定义在分子上匹配特征点（每个特征带所属族、类型与命中的原子 id）
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
    """按原子索引删原子并重建分子（保留 R 基团标记等原子属性）。

    Args:
        mol_input: 源 Chem.Mol（常为带原子映射号的母核/R 基团表达分子）。
        remove_indices: 要删除的原子索引列表，索引按源分子计。
        keep_properties: 需要从源原子拷贝到新原子上的属性名列表。

    Returns:
        Chem.Mol：重建后的新分子。两端都保留的键按重排后的新索引重建；
        只有一端保留的键被丢弃，若保留端是氮则把显式氢数加 1 以补偿失去的那根键；
        '*' 与 'R<n>' 占位原子统一写成 dummy 原子 '*'，并用 molAtomMapNumber /
        dummyLabel / _MolFileRLabel 属性记住原来的 R 标号。
    """
     
    # 先把原子信息（符号、电荷、显式氢、要保留的属性）和键信息摘成普通元组列表，
    # 后续在干净的新分子上重放，避免在源分子上就地删原子打乱索引
    atom_list, bond_list, idx_map = [], [], {}  # idx_map: {old: new} 
    for atom in mol_input.GetAtoms(): 
         
        props = {} 
        for property_name in keep_properties: 
            if property_name in atom.GetPropsAsDict(): 
                props[property_name] = atom.GetPropsAsDict()[property_name] 
        symbol = atom.GetSymbol() 
         
        # 占位原子 '*'：把原子映射号（atom map number）记进属性，重建后仍能对应回原 R 位点
        if symbol.startswith('*'): 
            atom_symbol = '*' 
            props['molAtomMapNumber'] = atom.GetAtomMapNum() 
        # 'R1'/'R2' 这类 R 基团记号：符号统一成 dummy 原子 '*'，
        # 标号取自符号尾缀（无尾缀时用原子映射号），写成 dummyLabel / _MolFileRLabel / molAtomMapNumber
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
    # 键只登记起止原子索引与键型（bond type），删原子后再判断能否保留
    for bond in mol_input.GetBonds(): 
        bond_list.append( 
            ( 
                bond.GetBeginAtomIdx(), 
                bond.GetEndAtomIdx(), 
                bond.GetBondType() 
            ) 
        ) 
    # 空的可写分子（RWMol）：保留下来的原子按顺序重放，键随后按新索引重建
    mol = Chem.RWMol(Chem.Mol()) 
     
    new_idx = 0 
    for atom_index, atom_info in enumerate(atom_list): 
        # 命中 remove_indices 的原子整条跳过，其余原子重建并登记旧→新索引
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
    # 重建键：只有两端原子都被保留的键才加回来，并把索引换成重排后的新索引
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
            # 一端被删、一端保留：这根键无法保留，先记下保留端的原子索引
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
                # 保留端是氮时把显式氢数（explicit H）加 1，抵掉随被删原子一起消失的那根键
                if atom_list[keep_index][0] == 'N': 
                    old_num_explicit_Hs = mol.GetAtomWithIdx( 
                        idx_map[keep_index] 
                    ).GetNumExplicitHs() 

                    mol.GetAtomWithIdx(idx_map[keep_index]).SetNumExplicitHs( 
                        old_num_explicit_Hs + 1 
                    ) 
    # 从可写的 RWMol 转回不可变 Chem.Mol 输出
    mol = Chem.Mol(mol) 
    return mol


def draw_mols_surfs(
    mols,
    width=400,
    height=400,
    surface=True,
    surface_opacity=0.5
):
    """把一批分子叠进同一个 py3Dmol 三维视图渲染，可选再叠加溶剂可及表面（solvent accessible surface, SAS）。

    Args:
        mols: Chem.Mol 列表，需带三维坐标（各分子按自身坐标叠在同一场景，便于比较构象/对接姿态）。
        width / height: 画布像素尺寸。
        surface: 是否叠加分子表面。
        surface_opacity: 表面不透明度（0-1 浮点，作为 opacity 传给 addSurface）。

    Returns:
        视图 show() 的返回值（Notebook 中内联渲染）。
    """
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor('0xeeeeee')
    view.removeAllModels()
    for mol in mols:
        # 用 RDKit 内置的 addMolToView 把分子逐帧写进视图，所有模型共享一个视图坐标系
        addMolToView(mol, view)
    if surface:
        # 表面是对整个场景一次性计算的，py3Dmol.SAS 即溶剂可及表面
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
    """把反应式（reaction）画成长条图片并直接在 Notebook 里 display。

    Args:
        rxn_smiles (str): 反应式字符串，形如 "A.B>C>D"。
        use_smiles (bool): True 把各侧当 SMILES 具体分子解析，False 当 SMARTS 子结构模板解析
            （两者都交给 AllChem.ReactionFromSmarts 的 useSmiles 开关）。
    """
    rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=use_smiles)
    # 2000x500 的 Cairo 光栅画布，够铺开反应物箭头产物；highlightByReactant 让产物继承来源反应物的颜色
    d2d = Draw.MolDraw2DCairo(2000, 500)
    d2d.DrawReaction(rxn, highlightByReactant=True)
    png = d2d.GetDrawingText()
    display(Image(png))


def react(rxn_smarts, reagents):
    """用反应 SMARTS 模板对反应物跑一次反应模拟。

    Args:
        rxn_smarts (str): 反应 SMARTS，形如 "A.B>>C"。
        reagents (list[str]): 反应物 SMILES 列表，顺序需与模板的反应物槽位一一对应。

    Returns:
        list[tuple[Chem.Mol, ...]]：RunReactants 的产物组（每组为一次原子映射得到的产物分子）；
        解析或反应异常时打印异常并返回空列表。
    """
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        # n_reactants = rxn.GetNumReactantTemplates()
        products = rxn.RunReactants([
            # 模板槽位是 Mol，故先把 SMILES 逐个解析
            Chem.MolFromSmiles(smi) for smi in reagents
        ])
        return products
    except Exception as e:
        print(e)
        return []


def match_pattern(mol, patt):
    """空值安全的子结构匹配：mol 为 None（SMILES 解析失败）时返回 False，否则返回 HasSubstructMatch 的布尔结果。"""
    if mol:
        return mol.HasSubstructMatch(patt)
    else:
        return False


def split_rxn_smiles(smi):
    """按两个 '>' 把反应 SMILES 拆成 "反应物>试剂>产物"，返回 (反应物串, 产物串)。
    中段试剂/催化剂非空时用 '.' 并进反应物串（即不区分反应物与试剂）；
    '>' 数量不是恰好两个时解包失败，打印异常并返回 ('', '')。"""
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
    """在成酰胺（amide）反应里反查产物来自哪一对反应物：从反应 SMILES 中筛出羧酸与胺，
    穷举两者组合用 SMARTS 模板算出理论产物，再按 InChIKey 与实际产物比对。

    Args:
        rxn_smi (str): "反应物>试剂>产物" 形式的反应 SMILES。

    Returns:
        tuple | None：首个命中返回 (羧酸 SMILES, 胺 SMILES, 产物 SMILES)，全部不匹配时返回 None。
    """
    # ref: https://github.com/LiamWilbraham/uspto-analysis/blob/master/reaction-stats-uspto.ipynb
    # 模板固定为 羧酸 + 胺 -> 酰胺：[C:1](=[O:2])-[OD1] 与 [N!H0:3] 成键，写死的反应中心映射号 1/2/3
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
    
    # 去掉 '@'（立体化学标记），比较时把反应物与产物都当作无手性版本
    cooh = [re.sub('@', '', i) for i in cooh]
    
    amine = [
        r for r in reactants
        if match_pattern(Chem.MolFromSmiles(r), patt_amine)
    ]
    amine = [re.sub('@', '', i) for i in amine]

    # 酸 x 胺 的笛卡尔积逐个试反应，p_1[0] 取该次映射的首个产物分子
    for perm in itertools.product(cooh, amine):
        
        cooh_i = perm[0]
        amine_i = perm[1]
        
        smarts_products = react(rxn_smarts, perm)

        for p_1 in smarts_products:
            for p_2 in products:
                p_2 = re.sub('@', '', p_2)
                patt = Chem.MolFromSmiles(p_2)  
                # InChIKey 相同即认为理论产物就是实际产物，据此锁定这一对反应物
                if Chem.MolToInchiKey(p_1[0]) == Chem.MolToInchiKey(patt):  
                    return cooh_i, amine_i, p_2
    return None


def get_largest_mol(smiles, to_smiles=False):
    """从可能含多个片段（盐、溶剂、共结晶物）的 SMILES 里取原子数最多的主片段。

    Args:
        smiles (str): 输入 SMILES，解析失败时返回 None。
        to_smiles (bool): True 返回经 molvs.standardize_smiles 标准化的 SMILES 字符串，False 返回 Chem.Mol。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    # asMols=True 得到按片段拆开的独立分子，default=mol 只在无片段时兜底
    mol_frags = rdmolops.GetMolFrags(mol, asMols=True)
    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    if to_smiles:
        return mv.standardize_smiles(Chem.MolToSmiles(largest_mol))
    return largest_mol


def standardize_tautomer(mol, max_tautomers=1000):
    """互变异构体（tautomer）规范化：用 rdMolStandardize 的 TautomerEnumerator 把分子统一到规范的互变异构写法。

    Args:
        mol (Chem.Mol): 输入分子。
        max_tautomers (int): 枚举上限（CleanupParameters.maxTautomers），限制候选互变异构体数量以防组合爆炸。

    Returns:
        Chem.Mol：规范化后的互变异构体。
    """
    params = rdMolStandardize.CleanupParameters()
    params.maxTautomers = max_tautomers
    enumerator = rdMolStandardize.TautomerEnumerator(params)
    cm = enumerator.Canonicalize(mol)
    return cm


def reorder_tautomers(m):
    """列出分子的全部互变异构体（tautomer）并把规范化写法排在首位。

    Args:
        m (Chem.Mol): 输入分子。

    Returns:
        list[Chem.Mol]：首项为 Canonicalize 结果，其余为枚举出且 SMILES 不同于首项的异构体，按 SMILES 字符串升序。
    """
    enumerator = rdMolStandardize.TautomerEnumerator()
    canon = enumerator.Canonicalize(m)
    csmi = Chem.MolToSmiles(canon)
    res = [canon]
    tauts = enumerator.Enumerate(m)
    smis = [Chem.MolToSmiles(x) for x in tauts]
    # Enumerate 的列表里含规范化体本身，按 SMILES 相等过滤掉以免首项重复；元组比较即以 SMILES 字符串为排序键
    stpl = sorted(
        (x, y) for x, y in zip(smis, tauts) if x!=csmi
    )
    res += [y for _, y in stpl]
    return res