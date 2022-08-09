import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class TetraPermuter(nn.Module):

    def __init__(
        self,
        hidden,
        # device
    ):
        super(TetraPermuter, self).__init__()

        self.W_bs = nn.ModuleList([copy.deepcopy(nn.Linear(hidden, hidden)) for _ in range(4)])
        # self.device = device
        self.drop = nn.Dropout(p=0.2)
        self.reset_parameters()
        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))

        self.tetra_perms = torch.tensor([[0, 1, 2, 3],
                                         [0, 2, 3, 1],
                                         [0, 3, 1, 2],
                                         [1, 0, 3, 2],
                                         [1, 2, 0, 3],
                                         [1, 3, 2, 0],
                                         [2, 0, 1, 3],
                                         [2, 1, 3, 0],
                                         [2, 3, 0, 1],
                                         [3, 0, 2, 1],
                                         [3, 1, 0, 2],
                                         [3, 2, 1, 0]])

    def reset_parameters(self):
        gain = 0.5
        for W_b in self.W_bs:
            nn.init.xavier_uniform_(W_b.weight, gain=gain)
            gain += 0.5

    def forward(self, x):

        nei_messages_list = [self.drop(F.tanh(l(t))) for l, t in zip(self.W_bs, torch.split(x[:, self.tetra_perms, :], 1, dim=-2))]
        nei_messages = torch.sum(self.drop(F.relu(torch.cat(nei_messages_list, dim=-2).sum(dim=-2))), dim=-2)

        return self.mlp_out(nei_messages / 3.)


class ConcatTetraPermuter(nn.Module):

    def __init__(
        self,
        hidden,
        # device
    ):
        super(ConcatTetraPermuter, self).__init__()

        self.W_bs = nn.Linear(hidden * 4, hidden)
        torch.nn.init.xavier_normal_(self.W_bs.weight, gain=1.0)
        self.hidden = hidden
        # self.device = device
        self.drop = nn.Dropout(p=0.2)
        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))

        tetra_perms = torch.tensor([
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [1, 0, 3, 2],
            [1, 2, 0, 3],
            [1, 3, 2, 0],
            [2, 0, 1, 3],
            [2, 1, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 2, 1],
            [3, 1, 0, 2],
            [3, 2, 1, 0]
        ])
        self.register_buffer('tetra_perms', tetra_perms)

    def forward(self, x):

        nei_messages = self.drop(
            F.relu(
                self.W_bs(
                    x[
                        :,
                        self.tetra_perms,
                        :
                    ].flatten(start_dim=2)
                )
            )
        )
        nei_messages_sum = nei_messages.sum(dim=-2) / 3.
        if nei_messages_sum.size(0) == 1:
            nei_messages_sum_repeat = torch.repeat_interleave(nei_messages_sum, 2, dim=0)
            return self.mlp_out(nei_messages_sum_repeat)[:1, ...]
        return self.mlp_out(nei_messages_sum)


class TetraDifferencesProduct(nn.Module):

    def __init__(
        self,
        hidden
    ):
        super(TetraDifferencesProduct, self).__init__()

        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))
        self.register_buffer('indices', torch.arange(4))

    def forward(self, x):

        # indices = torch.arange(4).to(x.device)
        message_tetra_nbs = [
            x.index_select(dim=1, index=i).squeeze(1)
            for i in self.indices
        ]
        message_tetra = torch.ones_like(message_tetra_nbs[0])

        # note: this will zero out reps for chiral centers with multiple carbon neighbors on first pass
        for i in range(4):
            for j in range(i + 1, 4):
                message_tetra = torch.mul(message_tetra, (message_tetra_nbs[i] - message_tetra_nbs[j]))
        message_tetra = torch.sign(message_tetra) * torch.pow(torch.abs(message_tetra) + 1e-6, 1 / 6)
        return self.mlp_out(message_tetra)


# def get_tetra_update(
#     hidden_size,
#     device,
#     message,
# ):

#     if message == 'tetra_permute':
#         return TetraPermuter(hidden_size, device)
#     elif message == 'tetra_permute_concat':
#         return ConcatTetraPermuter(hidden_size, device)
#     elif message == 'tetra_pd':
#         return TetraDifferencesProduct(hidden_size)
#     else:
#         raise ValueError("Invalid message type.")


TETRA_UPDATE_DICT = {
    'tetra_permute': TetraPermuter,
    'tetra_permute_concat': ConcatTetraPermuter,
    'tetra_pd': TetraDifferencesProduct
}