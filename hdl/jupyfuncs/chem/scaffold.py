from rdkit.Chem.Scaffolds import rdScaffoldNetwork
from rdkit import RDLogger
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
    RDLogger.DisableLog('rdApp.info')
    scaffParams = rdScaffoldNetwork.ScaffoldNetworkParams()
    scaffParams.collectMolCounts = True
    scaffParams.includeGenericScaffolds = includeGenericScaffolds
    scaffParams.includeScaffoldsWithoutAttachments = includeScaffoldsWithoutAttachments
    scaffParams.keepOnlyFirstFragment = keepOnlyFirstFragment
    scaffParams.includeGenericBondScaffolds = includeGenericBondScaffolds
    scaffParams.includeScaffoldsWithAttachments = includeScaffoldsWithAttachments
    scaffParams.pruneBeforeFragmenting = pruneBeforeFragmenting
    net = rdScaffoldNetwork.CreateScaffoldNetwork(mols, scaffParams)
    return net