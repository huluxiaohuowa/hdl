# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/features/graph/featurization.py
# 说明：分子图特征化（原子/键特征）
# 模块功能：把 SMILES 解析为分子图，按 one-hot 表构造原子特征（atom features）与键特征（bond features），并输出原子/键索引映射供图神经网络使用
from argparse import Namespace
from typing import List, Tuple, Union

from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType

import torch


# Atom feature sizes
# 参与 one-hot 的元素符号表，表外元素落到"未知"位
ATOMIC_SYMBOLS = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
# CIP 序列规则给出的原子绝对构型（R/S）
CIP_CHIRALITY = ['R', 'S']
# 原子特征（atom features）取值表：每项 one-hot 编码，长度为 len(choices)+1
ATOM_FEATURES = {
    'atomic_num': ATOMIC_SYMBOLS,
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'global_chiral_tag': CIP_CHIRALITY,
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
}
# 键特征（bond features）取值表：键级与键立体异构（E/Z）
BOND_FEATURES = {
    'bondtype':[
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'bondstereo':[
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
}
# 四面体手性标记 -> 旋向奇偶值：顺时针 +1，逆时针 -1，未指定/其他 0
CHIRALTAG_PARITY = {
    ChiralType.CHI_TETRAHEDRAL_CW: +1,
    ChiralType.CHI_TETRAHEDRAL_CCW: -1,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0,  # default
}

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic, mass and IsInRing
# 特征维度常量：ATOM_FDIM=48（含 chiral_tag 与 global_chiral_tag 两段手性特征），BOND_FDIM=13
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 3
BOND_FDIM = sum(len(choices) + 1 for choices in BOND_FEATURES.values()) + 3


def get_atom_fdim() -> int:
    """
    返回原子特征维度 ATOM_FDIM。

    Gets the dimensionality of atom features.
    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim() -> int:
    """
    返回键特征维度 BOND_FDIM。

    Gets the dimensionality of bond features.
    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value, choices: List) -> List[int]:
    """
    单热（one-hot）编码：value 在 choices 中则对应位置 1，否则末位置 1（表示未见过的取值）。

    Creates a one-hot encoding.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    # 取值不在候选表内时用 -1 索引，即命中末尾的"未知"位
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(
    atom: Chem.rdchem.Atom,
    chiral_features: bool = False,
    global_chiral_features: bool = False
) -> List[Union[bool, int, float]]:
    """
    拼接单个原子的特征向量（atom features）。

    Builds a feature vector for an atom.
    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    # 元素符号 / 总度数 / 形式电荷的 one-hot 段
    features = onek_encoding_unk(atom.GetSymbol(), ATOM_FEATURES['atomic_num']) + \
        onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
        onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    # 显式+隐式氢数、杂化方式，再接芳香性、成环标志与缩放后的原子质量（质量 x0.01）
    features += onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
        onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
        [1 if atom.GetIsAromatic() else 0] + [1 if atom.IsInRing() else 0] + \
        [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    # 可选局部手性段：RDKit ChiralTag 的 one-hot
    if chiral_features:
        features += onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
    # 可选全局手性段：读 RDKit 感知后的 CIP 码（R/S），无标记则落"未知"位
    if global_chiral_features:
        if atom.HasProp('_CIPCode'):
            features += onek_encoding_unk(atom.GetProp('_CIPCode'), ATOM_FEATURES['global_chiral_tag'])
        else:
            features += onek_encoding_unk(None, ATOM_FEATURES['global_chiral_tag'])
    return features


def parity_features(atom: Chem.rdchem.Atom) -> int:
    """
    返回原子的四面体旋向奇偶值。

    Returns the parity of an atom if it is a tetrahedral center.
    +1 if CW, -1 if CCW, and 0 if undefined/unknown
    :param atom: An RDKit atom.
    """
    return CHIRALTAG_PARITY[atom.GetChiralTag()]


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    拼接单条键的特征向量（bond features），长度为 BOND_FDIM。

    Builds a feature vector for a bond.
    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    bond_fdim = get_bond_fdim()

    if bond is None:
        # bond 为空表示自环（原子上的虚拟键）：首位 1，其余 0
        fbond = [1] + [0] * (bond_fdim - 1)
    else:
        bt = bond.GetBondType()
        # bond is not None
        # 首位 0 标记真实键，随后是键级 one-hot、键立体异构 one-hot，最后共轭与成环两个布尔位
        fbond = [0] + \
            onek_encoding_unk(bond.GetBondType(), BOND_FEATURES['bondtype']) + \
            onek_encoding_unk(bond.GetStereo(), BOND_FEATURES['bondstereo']) + \
            [(bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
    return fbond


class MolGraph:
    """
    单个分子的图结构与特征化结果：输入 SMILES 字符串，产出原子/键特征与邻接索引映射。

    A MolGraph represents the graph structure and featurization of a single molecule.
    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(
        self,
        smiles: str,
        chiral_features: bool = False,
        global_chiral_features: bool = False
        # args: Namespace
    ):
        """
        解析 SMILES 并完成分子图特征化。

        Computes the graph structure and featurization of a molecule.
        :param smiles: A smiles string.
        :param args: Arguments.
        """
        # 图缓存字段：原子/键特征表与原子-键互查索引
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.parity_atoms = []  # mapping from atom index to CW (+1), CCW (-1) or undefined tetra (0)
        self.edge_index = []  # list of tuples indicating presence of bonds
        # 手性中心的键序号列表：每个四面体原子贡献 4 条出边索引，供手性 GNN 按序聚合
        self.parity_bond_index = []

        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)

        # add chiral hydrogens
        # 找出带局部手性标记的原子，只给这些立体中心补出显式氢，保证配位数可判定
        H_ids = [a.GetIdx() for a in mol.GetAtoms() if CHIRALTAG_PARITY[a.GetChiralTag()] != 0]
        if H_ids:
            mol = Chem.AddHs(mol, onlyOnAtoms=H_ids)

        # remove stereochem label from atoms with less/more than 4 neighbors
        # 补氢后邻居数仍不等于 4 的原子无法构成四面体中心，清空其手性标记
        for i in H_ids:
            a = mol.GetAtomWithIdx(i)
            if len(a.GetNeighbors()) != 4:
                a.SetChiralTag(ChiralType.CHI_UNSPECIFIED)

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()
        
        # Get atom features
        # 逐原子生成特征向量，同时记录其四面体旋向奇偶值
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(
                atom,
                chiral_features=chiral_features,
                global_chiral_features=global_chiral_features
            ))
            self.parity_atoms.append(parity_features(atom))
        # 按原子序号重建特征列表，使其长度与 n_atoms 对齐
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        # 遍历原子对上三角，一条化学键拆成两条方向相反的有向边
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue
                    
                self.edge_index.extend([(a1, a2), (a2, a1)])

                f_bond = bond_features(bond)

                # 正、反两条边共享同一份键特征
                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)

                # Update index mappings
                # b1 = a1->a2，b2 = a2->a1，两者互为反向边
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2
        # 收集手性中心的出边索引：逆时针中心按 [1,0,2,3] 交换前两条边，使下游边序体现旋向差异
        for ai, ccw_mask in enumerate(self.parity_atoms):
            if ccw_mask == 0: continue
            nei_idx = []
            for ei, e in enumerate(self.edge_index):
                if e[0] == ai: nei_idx.append(ei)
            if ccw_mask == -1:
                nei_idx = [nei_idx[i] for i in [1, 0, 2, 3]]
            self.parity_bond_index.extend(nei_idx)


    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        返回分子图各组成部分的元组：(f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, parity_atoms)。

        Returns the components of the BatchMolGraph.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        # 注意 a_scope/b_scope（原子/键归属分子的区间）不在 __init__ 中赋值，需由外部批量构图逻辑设置
        return (
            self.f_atoms,
            self.f_bonds,
            self.a2b,
            self.b2a,
            self.b2revb,
            self.a_scope,
            self.b_scope,
            self.parity_atoms
        )

    def get_b2b(self) -> torch.LongTensor:
        """
        计算并缓存键到键（bond-to-bond）映射：每条键指向所有入边键的下标，形状 num_bonds x max_num_bonds。
        仅当 a2b/b2a/b2revb 已是 torch 张量且 b2b 属性已初始化时可用（__init__ 里存的是 list）。

        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.
        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            # 掩掉自身反向键，避免 a->b->a 的回声消息
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        计算并缓存原子到邻居原子（atom-to-atom）映射：b2a[a2b] 即由原子经入边键回到相邻原子，
        形状 num_atoms x max_num_bonds；同样依赖 a2a 属性已初始化且索引量为张量。

        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.
        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a





