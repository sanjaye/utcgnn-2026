"""Save a small, readable sample of the exact inputs the GNNs receive.

Training normally builds these inputs in memory on every run. This exports them for the
first `n` sets of one split's first match date, for both graph formulations:

    sets.csv                        the sampled sets: players, court, label, cold-start flag
    player_graph_node_features.csv  each set's four players: raw and scaled features
    player_graph_relations.csv      their row-normalized teammate/opponent neighbor weights
    player_graph_snapshot.pt        the full snapshot tensors for that date (all players)
    hetero_targets.csv              target nodes (real and swapped copies) and their side edges
    hetero_past_set_edges.csv       every won/lost edge from the sampled players to past sets
    hetero_graph.pt                 the full scaled HeteroData graph for that date

Scaling uses training-season statistics only, as in training.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import FEATURES, SLOTS
from .experiment import split_sessions

TEAM_SLOTS = ('A', 'A', 'B', 'B')


def export_sample(players, matches, output='data/sample_model_inputs', n=10, split='validation'):
    from .hetero_data import NODE_FEATURES, apply_scaling, build_graphs, fit_scaling  # needs PyG

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    _, groups, _ = split_sessions(players, matches)
    snap = groups[split][0]
    n = min(n, len(snap.ids))
    ids = players.player_id.tolist()
    date = str(snap.session.date())

    # --- Sets ---------------------------------------------------------------------------------
    rows = matches.set_index('match_id').loc[snap.ids[:n]].reset_index()
    sets = rows[['match_id', 'session', 'season', 'court', *SLOTS, 'team_a_win']].copy()
    sets['session'] = date
    sets.insert(2, 'split', split)
    sets['cold_start'] = snap.cold[:n]
    sets.to_csv(out / 'sets.csv', index=False)

    # --- Player graph -------------------------------------------------------------------------
    # Same standardization as utcgnn.experiment.train_graph.
    train_x = torch.cat([s.x for s in groups['train']])
    mean, scale = train_x.mean(0), train_x.std(0).clamp(min=.1)
    scaled = (snap.x - mean) / scale
    node_rows, relation_rows = [], []
    for k in range(n):
        match_id = snap.ids[k]
        for slot, (name, p) in enumerate(zip(SLOTS, snap.teams[k].tolist())):
            node_rows.append({'match_id': match_id, 'slot': name, 'team': TEAM_SLOTS[slot], 'player_id': ids[p],
                              **{f'raw_{f}': float(snap.x[p, j]) for j, f in enumerate(FEATURES)},
                              **{f'scaled_{f}': float(scaled[p, j]) for j, f in enumerate(FEATURES)}})
            teammate, opponent = snap.relations[0, p], snap.relations[1, p]
            for q in torch.nonzero((teammate > 0) | (opponent > 0)).flatten().tolist():
                relation_rows.append({'match_id': match_id, 'player_id': ids[p], 'neighbor_id': ids[q],
                                      'teammate_weight': round(float(teammate[q]), 6),
                                      'opponent_weight': round(float(opponent[q]), 6)})
    pd.DataFrame(node_rows).round(6).to_csv(out / 'player_graph_node_features.csv', index=False)
    pd.DataFrame(relation_rows, columns=['match_id', 'player_id', 'neighbor_id', 'teammate_weight',
                                         'opponent_weight']).to_csv(out / 'player_graph_relations.csv', index=False)
    torch.save({'session': date, 'player_ids': ids, 'features': list(FEATURES), 'x_raw': snap.x,
                'x_scaled': scaled, 'relations': snap.relations, 'relation_names': ['teammate', 'opponent'],
                'teams': snap.teams[:n], 'y': snap.y[:n], 'match_ids': snap.ids[:n]},
               out / 'player_graph_snapshot.pt')

    # --- Heterogeneous player–set graph -------------------------------------------------------
    graphs = build_graphs(players, matches, groups)
    scaling = fit_scaling(graphs['train'])
    raw = graphs[split][0]
    g = apply_scaling([raw], scaling)[0]
    m = len(raw.match_ids)  # targets 0..m-1 are real copies, m..2m-1 swapped copies

    def side(edge_type, target):
        src, dst = g['player', edge_type, 'target'].edge_index
        return ' '.join(ids[p] for p in src[dst == target].tolist())

    target_rows = []
    for k in range(n):
        for role, t in ((0, k), (1, k + m)):
            target_rows.append({'target_index': t, 'role': ['real', 'swapped'][role], 'match_id': raw.match_ids[k],
                                'side_a_players': side('side_a', t), 'side_b_players': side('side_b', t),
                                **{f'raw_{f}': float(raw['target'].x[t, j])
                                   for j, f in enumerate(NODE_FEATURES['target'])},
                                **{f'scaled_{f}': float(g['target'].x[t, j])
                                   for j, f in enumerate(NODE_FEATURES['target'])},
                                'label_team_a_win': int(raw.y[k]) if role == 0 else None})
    targets = pd.DataFrame(target_rows).round(6)
    targets['label_team_a_win'] = targets.label_team_a_win.astype('Int64')  # blank for swapped copies
    targets.to_csv(out / 'hetero_targets.csv', index=False)

    # Past-set node index i is the i-th earlier set in date order (the order build_graphs uses).
    past_ids = pd.concat([r for _, r in matches.groupby('session', sort=True)]).match_id.tolist()
    sampled = sorted({p for k in range(n) for p in snap.teams[k].tolist()})
    edge_rows = []
    for edge_type in ('won_in', 'lost_in'):
        src, dst = raw['player', edge_type, 'past_set'].edge_index
        keep = torch.isin(src, torch.tensor(sampled))
        for p, s in zip(src[keep].tolist(), dst[keep].tolist()):
            edge_rows.append({'player_id': ids[p], 'edge_type': edge_type, 'past_set_index': s,
                              'past_set_match_id': past_ids[s],
                              **{f'raw_{f}': float(raw['past_set'].x[s, j])
                                 for j, f in enumerate(NODE_FEATURES['past_set'])},
                              **{f'scaled_{f}': float(g['past_set'].x[s, j])
                                 for j, f in enumerate(NODE_FEATURES['past_set'])}})
    edges = pd.DataFrame(edge_rows).round(6).sort_values(['player_id', 'past_set_index', 'edge_type'])
    edges.to_csv(out / 'hetero_past_set_edges.csv', index=False)
    torch.save({'graph': g, 'session': date, 'player_ids': ids, 'past_set_match_ids': past_ids[:raw['past_set'].num_nodes],
                'node_features': NODE_FEATURES, 'scaling': scaling}, out / 'hetero_graph.pt')

    return {'output': str(out), 'session': date, 'split': split, 'sets': n,
            'sets_on_date': len(snap.ids), 'past_sets': int(raw['past_set'].num_nodes),
            'player_graph_relation_rows': len(relation_rows), 'hetero_past_set_edges': len(edges)}
