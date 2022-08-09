import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from hdl.layers.graph.chiral_graph import (
    GCNConv,
    GINEConv,
    DMPNNConv,
    # get_tetra_update,
)

from hdl.layers.graph.tetra import (
    # get_tetra_update,
    TETRA_UPDATE_DICT
)


class GNN(nn.Module):
    def __init__(
        self,
        # args,
        num_node_features: int = 48,
        num_edge_features: int = 7,
        depth: int = 15,
        hidden_size: int = 128,
        dropout: float = 0.1,
        gnn_type: str = 'dmpnn',
        graph_pool: str = 'mean',
        tetra: bool = True,
        task: str = 'classification', 
        output_num: int = None,
        message: str = 'tetra_permute_concat',
        include_vars: bool = False,
    ):
        super(GNN, self).__init__()

        self.init_args = {
            "num_node_features": num_node_features,
            "num_edge_features": num_edge_features,
            "depth": depth,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "gnn_type": gnn_type,
            "graph_pool": graph_pool,
            "tetra": tetra,
            "task": task,
            "message": message,
            "include_vars": include_vars
        }

        self.depth = depth
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.graph_pool = graph_pool
        self.tetra = tetra
        self.task = task
        self.out_dim = output_num
        self.include_vars = include_vars

        if self.gnn_type == 'dmpnn':
            self.edge_init = nn.Linear(61, self.hidden_size)
            self.edge_to_node = DMPNNConv(
                hidden_size=hidden_size,
                tetra=tetra,
                message=message
            )
        else:
            self.node_init = nn.Linear(num_node_features, self.hidden_size)
            self.edge_init = nn.Linear(13, self.hidden_size)

        # layers
        self.convs = torch.nn.ModuleList()

        for _ in range(self.depth):
            if self.gnn_type == 'gin':
                self.convs.append(GINEConv(
                    hidden_size=hidden_size,
                    tetra=tetra,
                    message=message
                ))
            elif self.gnn_type == 'gcn':
                self.convs.append(GCNConv(
                    hidden_size=hidden_size,
                    tetra=tetra,
                    message=message
                ))
            elif self.gnn_type == 'dmpnn':
                self.convs.append(DMPNNConv(
                    hidden_size=hidden_size,
                    tetra=tetra,
                    message=message
                ))
            else:
                ValueError('Undefined GNN type called {}'.format(self.gnn_type))

        # graph pooling
        if self.tetra:
            self.tetra_update = TETRA_UPDATE_DICT[message](hidden_size)
            # self.tetra_update = get_tetra_update(args)

        if self.graph_pool == "sum":
            self.pool = global_add_pool
        elif self.graph_pool == "mean":
            self.pool = global_mean_pool
        elif self.graph_pool == "max":
            self.pool = global_max_pool
        elif self.graph_pool == "attn":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 2 * self.hidden_size),
                                            torch.nn.BatchNorm1d(2 * self.hidden_size),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(2 * self.hidden_size, 1)))
        elif self.graph_pool == "set2set":
            self.pool = Set2Set(self.hidden_size, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        # ffn
        self.mult = 2 if self.graph_pool == "set2set" else 1
        if self.include_vars:
            out_dim = 2
        elif self.out_dim:
            out_dim = self.out_dim
        else:
            out_dim = 1
        self.ffn = nn.Linear(self.mult * self.hidden_size, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch, parity_atoms, parity_bond_index = data.x, data.edge_index, data.edge_attr, data.batch, data.parity_atoms, data.parity_bond_index

        if self.gnn_type == 'dmpnn':
            row, col = edge_index
            edge_attr = torch.cat([x[row], edge_attr], dim=1)
            edge_attr = F.relu(self.edge_init(edge_attr))
        else:
            x = F.relu(self.node_init(x))
            edge_attr = F.relu(self.edge_init(edge_attr))

        x_list = [x]
        edge_attr_list = [edge_attr]

        # convolutions
        for layer_idx in range(self.depth):

            x_h, edge_attr_h = self.convs[layer_idx](x_list[-1], edge_index, edge_attr_list[-1], parity_atoms, parity_bond_index)
            h = edge_attr_h if self.gnn_type == 'dmpnn' else x_h

            if layer_idx == self.depth - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            if self.gnn_type == 'dmpnn':
                h += edge_attr_h
                edge_attr_list.append(h)
            else:
                h += x_h
                x_list.append(h)

        # dmpnn edge -> node aggregation
        if self.gnn_type == 'dmpnn':
            h, _ = self.edge_to_node(x_list[-1], edge_index, h, parity_atoms, parity_bond_index)

        if self.task == 'regression':
            output = torch.sigmoid(self.ffn(self.pool(h, batch)))
        elif self.task == 'classification':

            output = torch.sigmoid(self.ffn(self.pool(h, batch)))
        # mean = output[:, 0]       
        if not self.include_vars:
            return output
        else:
            mean, var = F.softplus(output[:, 1])
            return mean, var