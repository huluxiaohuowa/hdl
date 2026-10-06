# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/chem/norm.py
# 说明：化学信息学工具（RDKit / Jupyter）
# 模块功能：分子（molecule）归一化（normalization）——用一批反应 SMARTS（reaction SMARTS）规则修正官能团写法
#          与重组分离电荷（如硝基写成 N+(O-)=O、亚砜写成 S+(O-)），与 molvs 的 Normalizer 语义一致。
import functools
import logging

from rdkit import Chem
from rdkit.Chem import AllChem 


log = logging.getLogger(__name__)


__all__ = [
    # Normalizer: 逐条规则反复施加的执行器; Normalization: 单条规则; NORMALIZATIONS: 内置规则表
    "Normalizer",
    "Normalization",
    "NORMALIZATIONS",
]


def memoized_property(fget):
    # 中文说明：缓存型 property 装饰器——首次读取时计算一次，结果写入实例属性，之后直接返回缓存值
    """Decorator to create memoized properties."""
    # attr_name 是缓存键名，形如 _transform
    attr_name = "_{}".format(fget.__name__)

    @functools.wraps(fget)
    def fget_memoized(self):
        """property 的实际读取函数：实例上尚无 attr_name 属性时调用 fget(self) 计算一次并 setattr 写入缓存，之后直接返回 getattr 取到的缓存值。"""
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fget(self))
        return getattr(self, attr_name)

    return property(fget_memoized)


class Normalization(object):
    # 中文说明：一条归一化规则，由反应 SMARTS（reaction SMARTS）定义；
    #          关键属性 name（规则名）、transform_str（"反应物>>产物" 字符串）、transform（惰性编译的 Reaction 对象）
    """A normalization transform defined by reaction SMARTS."""

    def __init__(self, name, transform):
        """
        中文说明：只保存规则名与 SMARTS 字符串，构造时不编译反应，编译推迟到第一次访问 transform。
        :param string name: A name for this Normalization
        :param string transform: Reaction SMARTS to define the transformation.
        """
        log.debug("Initializing Normalization: %s", name)
        self.name = name
        self.transform_str = transform

    @memoized_property
    def transform(self):
        """把 SMARTS 字符串编译成 RDKit Reaction 对象，只在第一次访问时执行并缓存，避免重复解析。"""
        log.debug("Loading Normalization transform: %s", self.name)
        return AllChem.ReactionFromSmarts(self.transform_str)

    def __repr__(self):
        """返回 "Normalization('规则名', 'SMARTS')" 形式的调试字符串。"""
        return "Normalization({!r}, {!r})".format(self.name, self.transform_str)

    def __str__(self):
        """打印时只显示规则名。"""
        return self.name


# 内置归一化规则表：每条是一个 Normalization(名称, 反应SMARTS)。
# 规则按顺序施加，大致分几类：官能团价态改写（硝基/亚砜/砜/吡啶氮氧化物/叠氮/重氮/磷酸/脒基）、
# 1,3-与 1,5-分离电荷重组（把 -[A]=[N+] 之类改写为双键迁移后的中性/邻近电荷形式）、
# 1,3-与 1,5-共轭阳离子的质子化位归一、以及若干特殊写法修正（卤代锍/重氮盐/坏酰胺互变/无邻接卤素/R 组胺嘧啶互变）。
NORMALIZATIONS = [
    # 硝基/亚硝基类：把 C(=O)=O 型双键氧写法改成 N+(O-)=O，即显式分离电荷形式
    Normalization(
        "Nitro to N+(O-)=O",
        "[*:1][N,P,As,Sb:2](=[O,S,Se,Te:3])=[O,S,Se,Te:4]>>[*:1][*+1:2]([*-1:3])=[*:4]",
    ),
    # 官能团价态/电荷写法归一：砜、吡啶氮氧化物、叠氮、重氮、亚砜、磷酸、脒基、重氮盐-肼
    Normalization(
        "Sulfone to S(=O)(=O)", "[S+2:1]([O-:2])([O-:3])>>[S+0:1](=[O-0:2])(=[O-0:3])"
    ),
    Normalization("Pyridine oxide to n+O-", "[n:1]=[O:2]>>[n+:1][O-:2]"),
    Normalization(
        "Azide to N=N+=N-", "[*,H:1][N:2]=[N:3]#[N:4]>>[*,H:1][N:2]=[N+:3]=[N-:4]"
    ),
    Normalization("Diazo/azo to =N+=N-", "[*:1]=[N:2]#[N:3]>>[*:1]=[N+:2]=[N-:3]"),
    Normalization(
        "Sulfoxide to -S+(O-)-",
        "[!O:1][S+0;X3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]",
    ),
    Normalization(
        "Phosphate to P(O-)=O",
        "[O,S,Se,Te;-1:1][P+;D4:2][O,S,Se,Te;-1:3]>>[*+0:1]=[P+0;D5:2][*-1:3]",
    ),
    Normalization(
        "Amidinium to C(=NH2+)NH2",
        "[C,S;X3+1:1]([NX3:2])[NX3!H0:3]>>[*+0:1]([N:2])=[N+:3]",
    ),
    Normalization(
        "Normalize hydrazine-diazonium",
        "[CX4:1][NX3H:2]-[NX3H:3][CX4:4][NX2+:5]#[NX1:6]>>[CX4:1][NH0:2]=[NH+:3][C:4][N+0:5]=[NH:6]",
    ),
    # 1,3-分离电荷重组：双键向负电荷一侧迁移，使正负电荷落到相邻原子上
    Normalization(
        "Recombine 1,3-separated charges",
        "[N,P,As,Sb,O,S,Se,Te;-1:1]-[A:2]=[N,P,As,Sb,O,S,Se,Te;+1:3]>>[*-0:1]=[*:2]-[*+0:3]",
    ),
    Normalization(
        "Recombine 1,3-separated charges",
        "[n,o,p,s;-1:1]:[a:2]=[N,O,P,S;+1:3]>>[*-0:1]:[*:2]-[*+0:3]",
    ),
    Normalization(
        "Recombine 1,3-separated charges",
        "[N,O,P,S;-1:1]-[a:2]:[n,o,p,s;+1:3]>>[*-0:1]=[*:2]:[*+0:3]",
    ),
    # 1,5-分离电荷重组：双键沿共轭链迁移，让 +/- 电荷落到 1,5 两端并重新排布键级
    Normalization(
        "Recombine 1,5-separated charges",
        "[N,P,As,Sb,O,S,Se,Te;-1:1]-[A+0:2]=[A:3]-[A:4]=[N,P,As,Sb,O,S,Se,Te;+1:5]>>[*-0:1]=[*:2]-[*:3]=[*:4]-[*+0:5]",
    ),
    Normalization(
        "Recombine 1,5-separated charges",
        "[n,o,p,s;-1:1]:[a:2]:[a:3]:[c:4]=[N,O,P,S;+1:5]>>[*-0:1]:[*:2]:[*:3]:[c:4]-[*+0:5]",
    ),
    Normalization(
        "Recombine 1,5-separated charges",
        "[N,O,P,S;-1:1]-[c:2]:[a:3]:[a:4]:[n,o,p,s;+1:5]>>[*-0:1]=[c:2]:[*:3]:[*:4]:[*+0:5]",
    ),
    # 1,3-共轭阳离子归一：把可移动质子摆到带正电的双键氮/氧上，统一同一共轭阳离子的多种质子化写法
    Normalization(
        "Normalize 1,3 conjugated cation",
        "[N,O;+0!H0:1]-[A:2]=[N!$(*[O-]),O;+1H0:3]>>[*+1:1]=[*:2]-[*+0:3]",
    ),
    Normalization(
        "Normalize 1,3 conjugated cation",
        "[n;+0!H0:1]:[c:2]=[N!$(*[O-]),O;+1H0:3]>>[*+1:1]:[*:2]-[*+0:3]",
    ),
    Normalization(
        "Normalize 1,3 conjugated cation",
        "[N,O;+0!H0:1]-[c:2]:[n!$(*[O-]),o;+1H0:3]>>[*+1:1]=[*:2]:[*+0:3]",
    ),
    # 1,5-共轭阳离子归一：同上但迁移跨度为 1,5；后续条目覆盖芳环/芳杂环以及环内闭合（1...1）情形
    Normalization(
        "Normalize 1,5 conjugated cation",
        "[N,O;+0!H0:1]-[A:2]=[A:3]-[A:4]=[N!$(*[O-]),O;+1H0:5]>>[*+1:1]=[*:2]-[*:3]=[*:4]-[*+0:5]",
    ),
    Normalization(
        "Normalize 1,5 conjugated cation",
        "[n;+0!H0:1]:[a:2]:[a:3]:[c:4]=[N!$(*[O-]),O;+1H0:5]>>[n+1:1]:[*:2]:[*:3]:[*:4]-[*+0:5]",
    ),
    Normalization(
        "Normalize 1,5 conjugated cation",
        "[N,O;+0!H0:1]-[c:2]:[a:3]:[a:4]:[n!$(*[O-]),o;+1H0:5]>>[*+1:1]=[c:2]:[*:3]:[*:4]:[*+0:5]",
    ),
    Normalization(
        "Normalize 1,5 conjugated cation",
        "[n;+0!H0:1]1:[a:2]:[a:3]:[a:4]:[n!$(*[O-]);+1H0:5]1>>[n+1:1]1:[*:2]:[*:3]:[*:4]:[n+0:5]1",
    ),
    Normalization(
        "Normalize 1,5 conjugated cation",
        "[n;+0!H0:1]:[a:2]:[a:3]:[a:4]:[n!$(*[O-]);+1H0:5]>>[n+1:1]:[*:2]:[*:3]:[*:4]:[n+0:5]",
    ),
    # 零散价态/电荷修正：卤-氧双键、负氮与碳正并成三键、硝基、重氮盐、四价 N、三价 O/S 等
    Normalization(
        "Charge normalization",
        "[F,Cl,Br,I,At;-1:1]=[O:2]>>[*-0:1][O-:2]"),
    Normalization(
        "Charge recombination", "[N,P,As,Sb;-1:1]=[C+;v3:2]>>[*+0:1]#[C+0:2]"
    ),
    Normalization(
        "Nitro to N+(O-)=O",
        "[N;X3:1](=[O:2])=[O:3]>>[*+1:1]([*-1:2])=[*:3]"),
    Normalization(
        "Diazonium N",
        "[*:1]-[N;X2:2]#[N;X1:3]>>[*:1]-[*+1:2]#[*:3]",
    ),
    Normalization(
        "Quaternary N",
        "[N;X4;v4;+0:1]>>[*+1:1]",
    ),
    Normalization(
        "Trivalent O",
        "[*:1]=[O;X2;v3;+0:2]-[#6:3]>>[*:1]=[*+1:2]-[*:3]",
    ),
    Normalization(
        "Sulfoxide to -S+(O-)",
        "[!O:1][S+0;D3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]",
    ),
    Normalization(
        "Sulfoxide to -S+(O-) 2",
        "[!O:1][SH1+1;D3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]",
    ),
    Normalization(
        "Trivalent S",
        "[O:1]=[S;D2;+0:2]-[#6:3]>>[*:1]=[*+1:2]-[*:3]",
    ),

    # 互变异构体（tautomer）修正：把 C(OH)=N 这类错误的酰胺/内酰胺写法改回 C(=O)-NH，
    # 并把无邻接卤素写成卤素负离子、修正异常的吡啶/吡哒嗪氮氧化物
    Normalization(
        "Bad amide tautomer1",
        "[C:1]([OH1;D1:2])=;!@[NH1:3]>>[C:1](=[OH0:2])-[NH2:3]",
    ),
    Normalization(
        "Bad amide tautomer2",
        "[C:1]([OH1;D1:2])=;!@[NH0:3]>>[C:1](=[OH0:2])-[NH1:3]",
    ),
    Normalization(
        "Halogen with no neighbors", "[F,Cl,Br,I;X0;+0:1]>>[*-1:1]",
    ),
    Normalization(
        "Odd pyridine/pyridazine oxide structure",
        "[C,N;-;D2,D3:1]-[N+2;D3:2]-[O-;D1:3]>>[*-0:1]=[*+1:2]-[*-:3]",
    ),
    # 多氮杂环（嘧啶/咪唑并环类）质子分布归一：把 NH 统一落到指定位置，消除互变异构冗余写法
    Normalization(
        "qunimade2",
        "[n&H0:1][n&H1:2][n&H1,c;R2:3][c&H1,n&H1:4][c,n&H1:5](=[S,N,O:7])[n&H1:6]>>[n&H0:1][n&H1:2][n&H0,c;R2:3][c&H1,n&H0:4][c,n&H0:5]([S,N,O:7])[n&H0:6]"
    ),
    Normalization(
        "qunimade",
        "[c,n&H0,n&H1:2][n&H0,n&H1,c:3][c,n&H0,n&H1:4][c,n&H0,n&H1:5](=[S,N,O:1])>>[c,n&H0:2][n&H0,c:3][c,n&H0:4][c,n&H0:5]([S,N,O:1])"
    ),
]


class Normalizer(object):
    # 中文说明：归一化执行器——按顺序反复施加 NORMALIZATIONS 中的规则直到分子不再变化。
    #          输入 Chem.Mol，输出 Chem.Mol；多片段分子逐片段处理后拼回；实例属性 normalizations 为规则列表
    """A class for applying Normalization transforms.
    This class is typically used to apply a series of Normalization transforms to correct functional groups and
    recombine charges. Each transform is repeatedly applied until no further changes occur.
    """

    def __init__(self, normalizations=NORMALIZATIONS):
        # 中文说明：绑定要使用的规则列表，默认用内置 NORMALIZATIONS
        """Initialize a Normalizer with an optional custom list of :class:`~molvs.normalize.Normalization` transforms.
        :param normalizations: A list of  :class:`~molvs.normalize.Normalization` transforms to apply.
        :param int max_restarts: The maximum number of times to attempt to apply the series of normalizations (default
                                 200).
        """
        log.debug("Initializing Normalizer")
        self.normalizations = normalizations

    def __call__(self, mol):
        # 中文说明：实例可直接当函数调用，等价于 self.normalize(mol)
        """Calling a Normalizer instance like a function is the same as calling its normalize(mol) method."""
        return self.normalize(mol)

    def normalize(self, mol):
        # 中文说明：归一化入口——先把分子按片段拆开、逐片段轮询全部规则，再把片段拼回并 SanitizeMol 后返回
        """Apply a series of Normalization transforms to correct functional groups and recombine charges.
        A series of transforms are applied to the molecule. For each Normalization, the transform is applied repeatedly
        until no further changes occur. If any changes occurred, we go back and start from the first Normalization
        again, in case the changes mean an earlier transform is now applicable. The molecule is returned once the entire
        series of Normalizations cause no further changes or if max_restarts (default 200) is reached.
        :param mol: The molecule to normalize.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :return: The normalized fragment.
        :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        """
        log.debug("Running Normalizer")
        # Normalize each fragment separately to get around quirky RunReactants behaviour
        fragments = []
        # 拆成单片段：RunReactants 对多片段分子行为异常，故逐片段独立归一化
        for fragment in Chem.GetMolFrags(mol, asMols=True):
            fragments.append(self._normalize_fragment(fragment))
        # Join normalized fragments into a single molecule again
        outmol = fragments.pop()
        # 用 CombineMols 把处理好的片段重新合成一个无键连接的分子（仍是分离的盐/共晶体形式）
        for fragment in fragments:
            outmol = Chem.CombineMols(outmol, fragment)
        # 片段拼回后价态/环感知等派生属性失效，必须重新 sanitize 才能继续使用
        Chem.SanitizeMol(outmol)
        return outmol

    def _normalize_fragment(self, mol):
        """对单个片段做一轮规则遍历：按列表顺序依次尝试各规则，命中（有新产物）就用产物替换当前分子。"""
        for normalization in self.normalizations:
            product = self._apply_transform(mol, normalization.transform)
            if product:
                mol = product
        return mol

    def _apply_transform(self, mol, rule):
        # 中文说明：把单条规则反复施加到分子上，直到无可反应位点；返回应用后的分子，规则一次都没命中则返回 None
        # Args(中文): mol=待处理的单片段 Chem.Mol；rule=已编译的 RDKit Reaction 对象（Normalization.transform）
        # Returns(中文): Chem.Mol 或 None；多个产物时按异构 SMILES 字典序取第一个，保证结果确定
        """Repeatedly apply normalization transform to molecule until no changes occur.
        It is possible for multiple products to be produced when a rule is applied. The rule is applied repeatedly to
        each of the products, until no further changes occur or after 20 attempts. If there are multiple unique products
        after the final application, the first product (sorted alphabetically by SMILES) is chosen.
        """
        mols = [mol]
        # 迭代上限 20 轮，防止规则之间相互触发造成死循环
        for n in range(20):
            products = {}
            for mol in mols:
                for product in [x[0] for x in rule.RunReactants((mol,))]:
                    # RunReactants 返回产物元组列表，取每条反应的第 0 个产物分子
                    if Chem.SanitizeMol(product, catchErrors=True) == 0:
                        # 只保留能通过 sanitize 的产物，并以异构 SMILES 为键去重
                        products[
                            Chem.MolToSmiles(product, isomericSmiles=True)
                        ] = product
            if products:
                # 仍有产物：按 SMILES 排序后继续下一轮，把规则再施加到新产物上
                mols = [products[s] for s in sorted(products)]
            else:
                # If n == 0, the rule was not applicable and we return None
                return mols[0] if n > 0 else None