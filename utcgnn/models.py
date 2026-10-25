"""Compact dense R-GCN and GraphSAGE for a club-sized graph; no PyG required."""
import torch
from torch import nn


class GraphModel(nn.Module):
    def __init__(self, features=5, hidden=16, kind='rgcn'):
        super().__init__()
        if kind not in ['rgcn', 'sage']:
            raise ValueError(kind)
        self.kind = kind
        self.self_layers = nn.ModuleList([nn.Linear(features,hidden),nn.Linear(hidden,hidden)])
        self.rel_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(size,hidden,bias=False) for _ in range(2 if kind=='rgcn' else 1)])
            for size in [features,hidden]])
        # Proposal decoder w^T(z_A-z_B); fix its intercept to zero for team symmetry.
        self.team_score = nn.Linear(hidden,1,bias=False)

    def encode(self, x, relations):
        if self.kind == 'sage':
            # Equal-weight unique neighbors; erase relation labels for comparison.
            adjacency = (relations.sum(0)>0).to(x.dtype)
            relations = (adjacency/adjacency.sum(1,keepdim=True).clamp(min=1)).unsqueeze(0)
        for own, transforms in zip(self.self_layers,self.rel_layers):
            x = torch.relu(own(x) + sum(layer(relations[i] @ x) for i,layer in enumerate(transforms)))
        return x

    def forward(self, x, relations, teams):
        h = self.encode(x,relations)[teams]
        # Shared decoder: permutation invariant within teams and antisymmetric across teams.
        return (self.team_score(h[:,:2].sum(1))-self.team_score(h[:,2:].sum(1))).squeeze(-1)
