"""Compact dense R-GCN and GraphSAGE for a club-sized graph; no PyG required."""
import torch
from torch import nn

# How two partners' final-layer embeddings (dimension d = hidden) become one team embedding.
#   'sum'    : z_team = z_1 + z_2                    (original; partner order cannot matter)
#   'concat' : z_team = ReLU(W [z_1 ; z_2] + b), W is d x 2d (separate weights per partner
#              slot; the ReLU lets the team embedding depend on how the partners combine)
#   'mean'   : z_team = ReLU(W (z_1 + z_2)/2 + b), W is d x d (average pooling, so partner
#              order cannot matter; nonlinear in the pooled embedding)
DECODERS = ('sum', 'concat', 'mean')


class GraphModel(nn.Module):
    def __init__(self, features=5, hidden=16, kind='rgcn', decoder='sum'):
        super().__init__()
        if kind not in ['rgcn', 'sage']:
            raise ValueError(kind)
        if decoder not in DECODERS:
            raise ValueError(decoder)
        self.kind = kind
        self.decoder = decoder
        self.self_layers = nn.ModuleList([nn.Linear(features,hidden),nn.Linear(hidden,hidden)])
        self.rel_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(size,hidden,bias=False) for _ in range(2 if kind=='rgcn' else 1)])
            for size in [features,hidden]])
        if decoder == 'concat':
            # Concat decoder only: maps the 2d concatenation of both partners back to d.
            # Created only for this decoder, so 'sum' models keep their original parameters
            # (and random initialization), and older checkpoints still load.
            self.pair = nn.Linear(2*hidden,hidden)
        elif decoder == 'mean':
            # Mean decoder only: transforms the d-dimensional average of both partners.
            self.pair = nn.Linear(hidden,hidden)
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

    def team_embedding(self, z1, z2):
        """Combine two partners' (..., d) embeddings into one (..., d) team embedding."""
        if self.decoder == 'sum':
            # Sum: symmetric in z1 and z2, so listing partners in either order gives the same team.
            return z1 + z2
        if self.decoder == 'mean':
            # Mean: average pooling is symmetric in z1 and z2, so partner order cannot matter.
            # The ReLU makes the team score nonlinear in the pooled embedding (for example,
            # diminishing returns), but the pair still enters only through z1 + z2, so it
            # cannot tell two average players from one strong and one weak player with the
            # same mean.
            return torch.relu(self.pair((z1+z2)/2))
        # Concat: [z1 ; z2] is (..., 2d), and the linear layer maps it back to (..., d).
        # z1 and z2 get different weights, so swapping partners can change the prediction.
        # The ReLU makes the team score a nonlinear function of the pair: without it, this
        # layer followed by the linear team_score would collapse to a^T z1 + c^T z2 (a
        # slot-weighted sum), unable to represent how two partners interact.
        return torch.relu(self.pair(torch.cat([z1,z2],dim=-1)))

    def forward(self, x, relations, teams):
        # teams holds player indices [A1, A2, B1, B2] per set; h is (sets, 4, d).
        h = self.encode(x,relations)[teams]
        team_a = self.team_embedding(h[:,0],h[:,1])
        team_b = self.team_embedding(h[:,2],h[:,3])
        # The same decoder scores both teams and the logits are subtracted, so for either
        # decoder P(A beats B) = 1 - P(B beats A). Partner-order invariance holds for 'sum'
        # and 'mean', not for 'concat'.
        return (self.team_score(team_a)-self.team_score(team_b)).squeeze(-1)
