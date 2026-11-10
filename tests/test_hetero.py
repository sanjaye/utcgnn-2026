"""Tests for the PyG player–set graph and models. Skipped when torch_geometric is not installed."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from test_temporal import fixture
from utcgnn.experiment import split_sessions

HAS_PYG = importlib.util.find_spec('torch_geometric') is not None
if HAS_PYG:
    from torch_geometric.data import Batch

    from utcgnn.hetero_data import EDGE_TYPES, build_graphs
    from utcgnn.hetero_experiment import run_hetero
    from utcgnn.hetero_models import PlayerSetGNN


def graphs_for(players, matches):
    _, groups, _ = split_sessions(players, matches)
    graphs = build_graphs(players, matches, groups)
    return [g for split in ('train', 'validation', 'test') for g in graphs[split]]


def swap_sides(matches):
    """The same sets, listed with team A and team B exchanged."""
    m = matches.copy()
    m[['team_a_1', 'team_a_2', 'team_b_1', 'team_b_2']] = matches[['team_b_1', 'team_b_2', 'team_a_1', 'team_a_2']].to_numpy()
    m[['team_a_score', 'team_b_score']] = matches[['team_b_score', 'team_a_score']].to_numpy()
    m['team_a_win'] = 1 - matches.team_a_win
    return m


@unittest.skipUnless(HAS_PYG, 'torch_geometric not installed')
class HeteroGraphTests(unittest.TestCase):
    def test_graph_holds_only_earlier_sets_and_unlabeled_targets(self):
        p, m = fixture()
        m.loc[1, 'session'] = m.loc[0, 'session']  # two sets on the first date
        graphs = graphs_for(p, m)
        dates = sorted(m.session.unique())
        for g, date in zip(graphs, dates):
            earlier = int((m.session < date).sum())
            tonight = int((m.session == date).sum())
            self.assertEqual(g['past_set'].num_nodes, earlier)
            # Real and swapped copy of each target; 2 features, no score or result.
            self.assertEqual(tuple(g['target'].x.shape), (2 * tonight, 2))
            self.assertEqual(len(g.y), tonight)
            # Each past set: 2 winners and 2 losers, with reverse edges.
            for et in EDGE_TYPES[:4]:
                self.assertEqual(g[et].edge_index.size(1), 2 * earlier)
            # Each target copy receives exactly 2 side-A and 2 side-B edges.
            for et in EDGE_TYPES[4:]:
                self.assertEqual(g[et].edge_index.size(1), 4 * tonight)
        # Same-date sets never appear in each other's history.
        self.assertEqual(graphs[0]['past_set'].num_nodes, 0)
        self.assertEqual(len(graphs[0].y), 2)

    def test_future_results_do_not_change_past_graphs(self):
        p, m = fixture()
        before = graphs_for(p, m)
        m.loc[3:, 'team_a_win'] = 1 - m.loc[3:, 'team_a_win']
        m.loc[3:, ['team_a_score', 'team_b_score']] = m.loc[3:, ['team_b_score', 'team_a_score']].to_numpy()
        after = graphs_for(p, m)
        for g0, g1 in zip(before[:4], after[:4]):  # dates up to and including the 4th
            for t in ('player', 'past_set', 'target'):
                torch.testing.assert_close(g0[t].x, g1[t].x)
            for et in EDGE_TYPES:
                torch.testing.assert_close(g0[et].edge_index, g1[et].edge_index)
        # Past-set features are side-neutral (winner's and loser's games), so a flipped
        # result shows up in which players hold the won and lost edges.
        won = ('player', 'won_in', 'past_set')
        self.assertFalse(torch.equal(before[4][won].edge_index, after[4][won].edge_index))

    def test_swap_readout_symmetries(self):
        p, m = fixture()
        listed = Batch.from_data_list(graphs_for(p, m))
        swapped = Batch.from_data_list(graphs_for(p, swap_sides(m)))
        partners = m.copy()
        partners[['team_a_1', 'team_a_2']] = m[['team_a_2', 'team_a_1']].to_numpy()
        reordered = Batch.from_data_list(graphs_for(p, partners))
        for kind in ('rgcn', 'sage'):
            torch.manual_seed(0)
            model = PlayerSetGNN(kind, hidden=8, layers=4, dropout=0.5)
            for mode in ('eval', 'train'):  # train mode: dropout masks are shared by both copies
                getattr(model, mode)()
                torch.manual_seed(1)
                a = model(listed)
                torch.manual_seed(1)
                b = model(swapped)
                # Listing the teams the other way round negates the logit: P(A) + P(B) = 1.
                torch.testing.assert_close(a, -b)
            model.eval()
            torch.testing.assert_close(model(listed), model(reordered))  # partner order is irrelevant

    def test_run_writes_metrics_and_predictions(self):
        p, m = fixture()
        audit = {'policy': 'exclude_all_flagged', 'used_sets': len(m)}
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('rgcn', 'sage'):
                report = run_hetero(p, m, audit, tmp, kind, seeds=(1, 2), epochs=3, hidden=8, layers=2,
                                    device='cpu', log_path=Path(tmp) / 'log.md', reference_path=Path(tmp) / 'none.json')
                self.assertEqual(report['summary']['test']['log_loss']['mean'] is not None, True)
                preds = pd.read_csv(Path(tmp) / 'test_predictions.csv')
                self.assertTrue(preds[f'{kind}_seed_mean'].between(0, 1).all())
                self.assertTrue((Path(tmp) / f'{kind}_seed2.pt').exists())
                json.loads((Path(tmp) / 'metrics.json').read_text())
            self.assertIn('results:hetero_rgcn:start', (Path(tmp) / 'log.md').read_text())
            self.assertIn('results:hetero_sage:start', (Path(tmp) / 'log.md').read_text())


if __name__ == '__main__':
    unittest.main()
