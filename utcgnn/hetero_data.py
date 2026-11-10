"""Player–set heterogeneous graphs (the project proposal's architecture) for PyTorch Geometric.

One graph is built per match date t, using the same session-causal rule as the rest of
utcgnn: every set played before t is a "past_set" node with its result, and the sets
played on t are unlabeled "target" nodes to predict.

Node types and features (raw values; scaled later with training-season statistics):
    player    5  the existing pre-session history features (utcgnn.data.FEATURES)
    past_set  5  winner's games, loser's games, weeks before t, week of season, court
    target    2  week of season, court  (no score and no result)

Edge types (no edge features; per-player information is carried by the edge type):
    player -won_in->  past_set    the player was on the winning side of that set
    player -lost_in-> past_set    the player was on the losing side
    past_set -won_by->  player    reverse of won_in
    past_set -lost_by-> player    reverse of lost_in
    player -side_a-> target       the player is on side A tonight
    player -side_b-> target       the player is on side B tonight

Targets only receive messages (there are no target -> player edges), so player and
past-set embeddings never depend on tonight's matchups.

Each target set appears twice: a "real" copy (role 0) with the listed sides, and a
"swapped" copy (role 1) with side A and side B exchanged. The model scores both and
subtracts, which guarantees P(A wins) = 1 - P(B wins). Within a graph, the m real copies
come first and the m swapped copies follow in the same order, so after batching,
role == 0 and role == 1 select aligned rows.

Set number is deliberately not a feature: the order sets are played in on a court is
arbitrary. Court is a ranking band (court 1 = the four top-ranked players that day).
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .data import FEATURES, SLOTS

NODE_TYPES = ('player', 'past_set', 'target')
EDGE_TYPES = (
    ('player', 'won_in', 'past_set'),
    ('player', 'lost_in', 'past_set'),
    ('past_set', 'won_by', 'player'),
    ('past_set', 'lost_by', 'player'),
    ('player', 'side_a', 'target'),
    ('player', 'side_b', 'target'),
)
NODE_FEATURES = {
    'player': list(FEATURES),
    'past_set': ['winner_games', 'loser_games', 'weeks_before', 'week_of_season', 'court'],
    'target': ['week_of_season', 'court'],
}


def _edges(src, dst):
    return torch.tensor(np.stack([np.asarray(src), np.asarray(dst)]), dtype=torch.long).reshape(2, -1)


def build_graphs(players, matches, groups):
    """Return {split: [HeteroData per session]}, aligned one-to-one with groups[split].

    `groups` comes from utcgnn.experiment.split_sessions, so the sessions, labels,
    cold-start flags and player features are exactly those used by the other models.
    """
    lookup = {p: i for i, p in enumerate(players.player_id.tolist())}  # same order as History
    # Iterate sessions exactly as utcgnn.data.snapshots does, so row order matches Snapshot.ids.
    sessions = list(matches.groupby('session', sort=True))
    rows = pd.concat([r for _, r in sessions])

    team = np.array([[lookup[p] for p in r] for r in rows[SLOTS].to_numpy()], dtype=np.int64)
    a_won = rows.team_a_win.to_numpy().astype(bool)
    # Past sets are stored side-neutral: winners and losers instead of team A and team B.
    winners = np.where(a_won[:, None], team[:, :2], team[:, 2:])
    losers = np.where(a_won[:, None], team[:, 2:], team[:, :2])
    winner_games = np.where(a_won, rows.team_a_score, rows.team_b_score).astype(float)
    loser_games = np.where(a_won, rows.team_b_score, rows.team_a_score).astype(float)
    day = rows.session.to_numpy().astype('datetime64[D]').astype(np.int64)
    # Season start = first match date of that season. For any date in the season it is
    # at or before that date, so it uses no future information.
    season_start = rows.season.map(matches.groupby('season').session.min()).to_numpy()
    week_of_season = (day - season_start.astype('datetime64[D]').astype(np.int64)) / 7
    court = rows.court.to_numpy().astype(float)
    labels = rows.team_a_win.to_numpy().astype(np.float32)
    ids = rows.match_id.tolist()

    by_session, offset = {}, 0
    for session, session_rows in sessions:
        m, past, now = len(session_rows), slice(0, offset), slice(offset, offset + len(session_rows))
        g = HeteroData()

        # Past sets: everything strictly before this date (rows are sorted by date).
        sets = np.arange(offset)
        weeks_before = (day[now][0] - day[past]) / 7  # how long ago, from tonight's date
        g['past_set'].x = torch.tensor(np.column_stack([
            winner_games[past], loser_games[past], weeks_before, week_of_season[past], court[past]]),
            dtype=torch.float32).reshape(offset, len(NODE_FEATURES['past_set']))
        won_src, won_dst = winners[past].reshape(-1), np.repeat(sets, 2)
        lost_src, lost_dst = losers[past].reshape(-1), np.repeat(sets, 2)
        g['player', 'won_in', 'past_set'].edge_index = _edges(won_src, won_dst)
        g['player', 'lost_in', 'past_set'].edge_index = _edges(lost_src, lost_dst)
        g['past_set', 'won_by', 'player'].edge_index = _edges(won_dst, won_src)
        g['past_set', 'lost_by', 'player'].edge_index = _edges(lost_dst, lost_src)

        # Targets: tonight's sets, twice. Rows 0..m-1 are the real copies, m..2m-1 the
        # swapped copies. Features carry no score and no result.
        target_x = np.column_stack([week_of_season[now], court[now]])
        g['target'].x = torch.tensor(np.vstack([target_x, target_x]), dtype=torch.float32)
        g['target'].role = torch.cat([torch.zeros(m, dtype=torch.long), torch.ones(m, dtype=torch.long)])
        side_a, side_b = team[now, :2], team[now, 2:]
        idx = np.arange(m)
        real, swapped = np.repeat(idx, 2), np.repeat(idx + m, 2)
        # Real copy: listed side A -> side_a edges. Swapped copy: listed side B -> side_a edges.
        g['player', 'side_a', 'target'].edge_index = _edges(
            np.concatenate([side_a.reshape(-1), side_b.reshape(-1)]), np.concatenate([real, swapped]))
        g['player', 'side_b', 'target'].edge_index = _edges(
            np.concatenate([side_b.reshape(-1), side_a.reshape(-1)]), np.concatenate([real, swapped]))

        g.y = torch.tensor(labels[now])  # P(listed side A wins) target, one per real copy
        g.match_ids = ids[now.start:now.stop]
        by_session[session] = g
        offset += m

    graphs = {}
    for split, snapshots in groups.items():
        graphs[split] = []
        for snapshot in snapshots:
            g = by_session[snapshot.session]
            if g.match_ids != snapshot.ids:
                raise ValueError('Graph and snapshot set order differ')
            # Player features are the pre-session features already computed by History.
            g['player'].x = snapshot.x.clone()
            graphs[split].append(g)
    return graphs


def fit_scaling(graphs):
    """Per-node-type mean and std over the training graphs (std clamped as in utcgnn.experiment)."""
    scaling = {}
    for t in NODE_TYPES:
        x = torch.cat([g[t].x for g in graphs])
        # Guard tiny inputs (no rows, or one row whose std is undefined): no scaling then.
        mean = x.mean(0) if len(x) else torch.zeros(x.size(1))
        std = x.std(0) if len(x) > 1 else torch.ones(x.size(1))
        scaling[t] = (mean, std.clamp(min=.1))
    return scaling


def apply_scaling(graphs, scaling):
    """Scaled copies of the graphs; the inputs are left unchanged."""
    scaled = []
    for g in graphs:
        g = g.clone()
        for t, (mean, std) in scaling.items():
            g[t].x = (g[t].x - mean) / std
        scaled.append(g)
    return scaled
