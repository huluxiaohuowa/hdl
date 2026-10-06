# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/chem/scaffold.py
# 说明：化学信息学工具（RDKit / Jupyter）
# 模块功能：用 RDKit rdScaffoldNetwork 构建骨架网络（scaffold network），
#          把一批分子的母核/骨架（scaffold，即 Murcko 类骨架）按派生关系连成网络，用于骨架演化分析。
from rdkit.Chem.Scaffolds import rdScaffoldNetwork
from rdkit import RDLogger
# 关掉 RDKit info 级日志，避免建网过程逐分子刷屏
RDLogger.DisableLog('rdApp.info')


def create_sn(
    mols,
    includeGenericScaffolds=False,
    includeGenericBondScaffolds=False,
    includeScaffoldsWithAttachments=True,
    includeScaffoldsWithoutAttachments=False,
    pruneBeforeFragmenting=True,
    keepOnlyFirstFragment=True
):
    """构建骨架网络（scaffold network）：对分子做骨架/侧链切分后，按骨架间的包含关系连成网络图。

    Args:
        mols: Chem.Mol 列表（输入分子集合）。
        includeGenericScaffolds: 是否加入泛化骨架（把原子类型统一成 * 的骨架）。
        includeGenericBondScaffolds: 是否加入泛化键骨架（键型统一为单键的骨架）。
        includeScaffoldsWithAttachments: 是否保留带侧链连接的骨架（默认开启）。
        includeScaffoldsWithoutAttachments: 是否保留去掉侧链后的纯骨架环系。
        pruneBeforeFragmenting: 切分前先剪掉非骨架片段（只留环系+连接原子）。
        keepOnlyFirstFragment: 多片段分子只取第一个片段参与建网。
        以上开关逐个写入 rdScaffoldNetwork.ScaffoldNetworkParams。

    Returns:
        rdScaffoldNetwork.ScaffoldNetwork 对象；节点是骨架 SMILES，边表示骨架派生关系，
        collectMolCounts 固定开启，因此节点带分子计数属性。
    """
    RDLogger.DisableLog('rdApp.info')
    scaffParams = rdScaffoldNetwork.ScaffoldNetworkParams()
    # 统计每个骨架被多少分子命中
    scaffParams.collectMolCounts = True
    scaffParams.includeGenericScaffolds = includeGenericScaffolds
    scaffParams.includeScaffoldsWithoutAttachments = includeScaffoldsWithoutAttachments
    scaffParams.keepOnlyFirstFragment = keepOnlyFirstFragment
    scaffParams.includeGenericBondScaffolds = includeGenericBondScaffolds
    scaffParams.includeScaffoldsWithAttachments = includeScaffoldsWithAttachments
    scaffParams.pruneBeforeFragmenting = pruneBeforeFragmenting
    # 真正执行切分与建网
    net = rdScaffoldNetwork.CreateScaffoldNetwork(mols, scaffParams)
    return net