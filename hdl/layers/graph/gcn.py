from torch import nn
from torch_geometric.nn import GCNConv


class GraphConv(nn.Module):
    def __init__(self, num_features, num_out_features):
        # Init parent
        super(GraphConv, self).__init__()

        # GCN layers
        self.conv = GCNConv(num_features, num_out_features) 

    def forward(self, x, edge_index):

        hidden = self.conv(x, edge_index)
        return hidden
