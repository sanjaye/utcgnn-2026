"""PyTorch Geometric R-GCN and GraphSAGE on the player–set graph (see hetero_data.py).

Forward pass for one batch of match-date graphs:
    1. Input layers: one Linear + ReLU per node type, mapping every node to `hidden` numbers.
    2. Message passing, `layers` times. Information flows player -> past_set -> player,
       and player -> target. Targets send nothing.
         rgcn: RGCNConv on the homogeneous view, one weight matrix per edge type; each
               edge type is averaged separately, then the results are summed with a
               self term.
         sage: HeteroConv with one SAGEConv per edge type, results summed per node type.
    3. Swap readout: each target exists as a real copy (h) and a swapped copy (h~),
       and logit = f(h) - f(h~), with the same small MLP f for both. Listing the teams
       the other way round exchanges h and h~, so the logit flips sign and
       P(A wins) = 1 - P(B wins) for any weights.
"""
import torch
from torch import nn
from torch_geometric.nn import HeteroConv, RGCNConv, SAGEConv

from .hetero_data import EDGE_TYPES, NODE_FEATURES, NODE_TYPES

KINDS = ('rgcn', 'sage')


class PlayerSetGNN(nn.Module):
    def __init__(self, kind='rgcn', hidden=32, layers=4, dropout=0.1):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.dropout = kind, dropout
        # Stage 1: per-node-type input layers (player 5, past_set 5, target 2 -> hidden).
        self.encoders = nn.ModuleDict({t: nn.Linear(len(NODE_FEATURES[t]), hidden) for t in NODE_TYPES})
        # Stage 2: message-passing layers.
        if kind == 'rgcn':
            self.convs = nn.ModuleList([
                RGCNConv(hidden, hidden, num_relations=len(EDGE_TYPES), aggr='mean') for _ in range(layers)])
        else:
            self.convs = nn.ModuleList([
                HeteroConv({et: SAGEConv((hidden, hidden), hidden) for et in EDGE_TYPES}, aggr='sum')
                for _ in range(layers)])
        # Stage 3: f, shared by the real and swapped copies of every target.
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def _paired_mask(self, x, role):
        """Dropout keep-mask for target rows, identical for each real copy and its swapped copy.

        Sharing the mask keeps f(h) - f(h~) exactly antisymmetric during training too.
        """
        keep = (torch.rand_like(x) >= self.dropout).to(x.dtype)
        keep[role == 1] = keep[role == 0]  # rows are aligned: k-th real <-> k-th swapped
        return keep / (1 - self.dropout)

    def _drop(self, x_dict, role):
        if not self.training or self.dropout == 0:
            return x_dict
        out = {t: nn.functional.dropout(x, self.dropout, True) for t, x in x_dict.items() if t != 'target'}
        out['target'] = x_dict['target'] * self._paired_mask(x_dict['target'], role)
        return out

    def encode(self, data):
        """Final embeddings for every node type after all message-passing layers."""
        role = data['target'].role
        x_dict = {t: torch.relu(self.encoders[t](data[t].x)) for t in NODE_TYPES}
        edge_index_dict = {et: data[et].edge_index for et in EDGE_TYPES}
        if self.kind == 'rgcn':
            # RGCNConv works on one node set with an edge-type id per edge, so stack the
            # node types and offset each edge type's indices into that combined set.
            sizes = [x_dict[t].size(0) for t in NODE_TYPES]
            offsets = dict(zip(NODE_TYPES, torch.tensor([0] + sizes[:-1]).cumsum(0).tolist()))
            edge_index = torch.cat([
                edge_index_dict[(s, r, d)] + torch.tensor([[offsets[s]], [offsets[d]]], device=role.device)
                for s, r, d in EDGE_TYPES], dim=1)
            edge_type = torch.cat([
                torch.full((edge_index_dict[et].size(1),), i, dtype=torch.long, device=role.device)
                for i, et in enumerate(EDGE_TYPES)])
        for i, conv in enumerate(self.convs):
            if self.kind == 'rgcn':
                x = conv(torch.cat([x_dict[t] for t in NODE_TYPES]), edge_index, edge_type)
                x_dict = dict(zip(NODE_TYPES, torch.relu(x).split(sizes)))
            else:
                x_dict = {t: torch.relu(x) for t, x in conv(x_dict, edge_index_dict).items()}
            if i < len(self.convs) - 1:
                x_dict = self._drop(x_dict, role)
        return x_dict

    def forward(self, data):
        """Logit of P(listed side A wins) for every target set, in graph and row order."""
        role = data['target'].role
        h = self.encode(data)['target']
        # Stage 3: f(h) - f(h~), the real copy minus its swapped copy.
        return (self.head(h[role == 0]) - self.head(h[role == 1])).squeeze(-1)
