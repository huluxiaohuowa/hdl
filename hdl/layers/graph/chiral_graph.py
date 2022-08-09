from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tetra import (
    # get_tetra_update,
    TETRA_UPDATE_DICT
)


class GCNConv(MessagePassing):
    def __init__(
        self,
        # args,
        hidden_size,
        tetra,
        message
    ):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.tetra = tetra  # bool
        if self.tetra:
            # self.tetra_update = get_tetra_update(args)
            self.tetra_update = TETRA_UPDATE_DICT[message](hidden_size)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        parity_atoms
    ):

        # no edge updates
        x = self.linear(x)

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

        if self.tetra:
            tetra_ids = parity_atoms.nonzero().squeeze(1)
            if tetra_ids.nelement() != 0:
                x_new[tetra_ids] = self.tetra_message(x, edge_index, edge_attr, tetra_ids, parity_atoms)
        x = x_new + F.relu(x)

        return self.batch_norm(x), edge_attr

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def tetra_message(self, x, edge_index, edge_attr, tetra_ids, parity_atoms):

        row, col = edge_index
        tetra_nei_ids = torch.cat([row[col == i].unsqueeze(0) for i in range(x.size(0)) if i in tetra_ids])

        # calculate pseudo tetra degree aligned with GCN method
        deg = degree(col, x.size(0), dtype=x.dtype)
        t_deg = deg[tetra_nei_ids]
        t_deg_inv_sqrt = t_deg.pow(-0.5)
        t_norm = 0.5 * t_deg_inv_sqrt.mean(dim=1)

        # switch entries for -1 rdkit labels
        ccw_mask = parity_atoms[tetra_ids] == -1
        tetra_nei_ids[ccw_mask] = tetra_nei_ids.clone()[ccw_mask][:, [1, 0, 2, 3]]

        # calculate reps
        edge_ids = torch.cat([tetra_nei_ids.view(1, -1), tetra_ids.repeat_interleave(4).unsqueeze(0)], dim=0)
        # dense_edge_attr = to_dense_adj(edge_index, batch=None, edge_attr=edge_attr).squeeze(0)
        # edge_reps = dense_edge_attr[edge_ids[0], edge_ids[1], :].view(tetra_nei_ids.size(0), 4, -1)
        attr_ids = [torch.where((a == edge_index.t()).all(dim=1))[0] for a in edge_ids.t()]
        edge_reps = edge_attr[attr_ids, :].view(tetra_nei_ids.size(0), 4, -1)
        reps = x[tetra_nei_ids] + edge_reps

        return t_norm.unsqueeze(-1) * self.tetra_update(reps)


class GINEConv(MessagePassing):
    def __init__(
        self,
        # args,
        hidden_size,
        tetra,
        message
    ):
        super(GINEConv, self).__init__(aggr="add")
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.mlp = nn.Sequential(nn.Linear(hidden_size, 2 * hidden_size),
                                 nn.BatchNorm1d(2 * hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(2 * hidden_size, hidden_size))
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.tetra = tetra
        if self.tetra:
            # self.tetra_update = get_tetra_update(args)
            self.tetra_update = TETRA_UPDATE_DICT[message](hidden_size)

    def forward(self, x, edge_index, edge_attr, parity_atoms):
        # no edge updates
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.tetra:
            tetra_ids = parity_atoms.nonzero().squeeze(1)
            if tetra_ids.nelement() != 0:
                x_new[tetra_ids] = self.tetra_message(x, edge_index, edge_attr, tetra_ids, parity_atoms)

        x = self.mlp((1 + self.eps) * x + x_new)
        return self.batch_norm(x), edge_attr

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def tetra_message(self, x, edge_index, edge_attr, tetra_ids, parity_atoms):

        row, col = edge_index
        tetra_nei_ids = torch.cat([row[col == i].unsqueeze(0) for i in range(x.size(0)) if i in tetra_ids])

        # switch entries for -1 rdkit labels
        ccw_mask = parity_atoms[tetra_ids] == -1
        tetra_nei_ids[ccw_mask] = tetra_nei_ids.clone()[ccw_mask][:, [1, 0, 2, 3]]

        # calculate reps
        edge_ids = torch.cat([tetra_nei_ids.view(1, -1), tetra_ids.repeat_interleave(4).unsqueeze(0)], dim=0)
        # dense_edge_attr = to_dense_adj(edge_index, batch=None, edge_attr=edge_attr).squeeze(0)
        # edge_reps = dense_edge_attr[edge_ids[0], edge_ids[1], :].view(tetra_nei_ids.size(0), 4, -1)
        attr_ids = [torch.where((a == edge_index.t()).all(dim=1))[0] for a in edge_ids.t()]
        edge_reps = edge_attr[attr_ids, :].view(tetra_nei_ids.size(0), 4, -1)
        reps = x[tetra_nei_ids] + edge_reps

        return self.tetra_update(reps)


class DMPNNConv(MessagePassing):
    def __init__(
        self,
        # args,
        hidden_size,
        tetra,
        message
    ):
        super(DMPNNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.ReLU())
        self.tetra = tetra
        if self.tetra:
            # self.tetra_update = get_tetra_update(args)
            self.tetra_update = TETRA_UPDATE_DICT[message](hidden_size)

    def forward(self, x, edge_index, edge_attr, parity_atoms, parity_bond_index):
        row, col = edge_index
        a_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)

        if self.tetra:
            tetra_ids = parity_atoms.nonzero().squeeze(1)
            if tetra_ids.nelement() != 0:
                a_message[tetra_ids] = self.tetra_message(x, edge_index, edge_attr, tetra_ids, parity_atoms, parity_bond_index)

        rev_message = torch.flip(edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]).view(edge_attr.size(0), -1)
        return a_message, self.mlp(a_message[row] - rev_message)

    def message(self, x_j, edge_attr):
        return F.relu(self.lin(edge_attr))

    def tetra_message(self, x, edge_index, edge_attr, tetra_ids, parity_atoms, parity_bond_index):
        edge_reps = edge_attr[parity_bond_index, :].view(parity_bond_index.size(0)//4, 4, -1)

        return self.tetra_update(edge_reps)
        # print('1')
        row, col = edge_index

        col_ids = torch.cat(
            [(col == i).nonzero() for i in tetra_ids]
        ).squeeze().unsqueeze(0)
        tetra_nei_ids = row[col_ids].reshape(-1, 4)
        
        # tetra_nei_ids = torch.cat([
        #     row[col == i].unsqueeze(0)  
        #     for i in tetra_ids
        # ])

        # print('2')
        # switch entries for -1 rdkit labels
        ccw_mask = parity_atoms[tetra_ids] == -1
        tetra_nei_ids[ccw_mask] = tetra_nei_ids.clone()[ccw_mask][:, [1, 0, 2, 3]]

        # calculate reps
        edge_ids = torch.cat([tetra_nei_ids.view(1, -1), tetra_ids.repeat_interleave(4).unsqueeze(0)], dim=0)
        # dense_edge_attr = to_dense_adj(edge_index, batch=None, edge_attr=edge_attr).squeeze(0)
        # edge_reps = dense_edge_attr[edge_ids[0], edge_ids[1], :].view(tetra_nei_ids.size(0), 4, -1)
        # edge_index_T = edge_index.t()
        # edge_ids_T = edge_ids.t()

        # attr_ids = [
        #     torch.where(
        #         (a == edge_index_T).all(dim=1)
        #     )[0]
        #     for a in edge_ids_T
        # ]
        # attr_ids = torch.cat([(edge_index_T == i).nonzero() for i in edge_ids_T])[:, 0].unique()

        edge_index_T = edge_index.t()
        edge_ids_T = edge_ids.t()        
        
        c0 = torch.cartesian_prod(
            edge_index_T[:, 0], edge_ids_T[:, 0]
        )
        c1 = torch.cartesian_prod(
            edge_index_T[:, 1], edge_ids_T[:, 1]
        )
        diff = torch.abs(c0[:, 0] - c0[:, 1]) \
            + torch.abs(c1[:, 0] - c1[:, 1])
        
        attr_ids = torch.div(
            (diff == 0).nonzero(as_tuple=True)[0],
            edge_ids.size(1),
            rounding_mode='floor'
        )

        edge_reps = edge_attr[attr_ids, :].view(tetra_nei_ids.size(0), 4, -1)

        return self.tetra_update(edge_reps)