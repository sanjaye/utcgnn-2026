"""Task 2 (game share): targets, metrics, standings, baselines and the score-mode GNN."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from test_temporal import fixture
from utcgnn.experiment import split_sessions
from utcgnn.score import (evaluate_score, run_score_baselines, score_targets, share_logistic, standings,
                          standings_metrics)
from utcgnn.xgboost_baseline import FEATURE_NAMES, run_xgboost_score, symmetric_predict
from xgboost import XGBRegressor

HAS_PYG = importlib.util.find_spec('torch_geometric') is not None


def toy():
    """Three sets among players 0-4: team A games a out of games."""
    return {'a': np.array([6., 3., 7.]), 'games': np.array([10., 9., 13.]),
            'share': np.array([.6, 1 / 3, 7 / 13]), 'cold': np.array([False, True, False]),
            'teams': np.array([[0, 1, 2, 3], [0, 2, 1, 4], [3, 4, 0, 1]])}


class ScoreTests(unittest.TestCase):
    def test_targets_follow_set_order(self):
        p, m = fixture()
        _, groups, _ = split_sessions(p, m)
        t = score_targets(m, groups)
        ids = [i for s in groups['test'] for i in s.ids]
        rows = m.set_index('match_id').loc[ids]
        np.testing.assert_allclose(t['test']['share'], rows.team_a_score / (rows.team_a_score + rows.team_b_score))
        self.assertEqual(t['test']['teams'].shape, (len(ids), 4))

    def test_mae_in_games(self):
        t, pred = toy(), np.array([.5, .5, .5])
        r = evaluate_score(t, pred, 5)
        expected = np.abs(pred - t['share']) * t['games']  # |0.5*10-6|, |0.5*9-3|, |0.5*13-7|
        np.testing.assert_allclose(expected, [1, 1.5, .5])
        self.assertAlmostEqual(r['mae_games'], 1.0)
        self.assertAlmostEqual(r['cold_start']['mae_games'], 1.5)
        self.assertAlmostEqual(r['established']['mae_games'], .75)

    def test_standings_sum_each_side(self):
        t = toy()
        s = standings(t, t['share'], 5, min_sets=1)
        # A perfect prediction gives every player expected = actual games.
        np.testing.assert_allclose(s.expected_games, s.actual_games)
        # Player 0: 6 (side A, set 0) + 3 (side A, set 1) + 13 - 7 (side B, set 2).
        self.assertEqual(s.actual_games[0], 15)
        np.testing.assert_array_equal(s.sets, [3, 3, 2, 2, 2])
        np.testing.assert_array_equal(s.games_played, [32, 32, 19, 23, 22])
        self.assertAlmostEqual(standings_metrics(t, t['share'], 5, min_sets=1)['per_set_spearman'], 1.0)
        # A constant prediction cannot rank per-set strength.
        self.assertIsNone(standings_metrics(t, np.full(3, .5), 5, min_sets=1)['per_set_spearman'])

    def test_share_models_are_team_symmetric(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 5))
        games = rng.integers(6, 14, 50).astype(float)
        a = np.minimum(games, rng.integers(0, 8, 50))
        lr = share_logistic(x, a, games)
        np.testing.assert_allclose(lr.predict_proba(x)[:, 1], 1 - lr.predict_proba(-x)[:, 1])
        model = XGBRegressor(objective='reg:logistic', n_estimators=10, max_depth=2, random_state=0).fit(
            rng.normal(size=(200, len(FEATURE_NAMES))), rng.uniform(size=200))
        ta, tb = rng.normal(size=(20, 15)), rng.normal(size=(20, 15))
        p = symmetric_predict(model, ta, tb)
        self.assertTrue(((p > 0) & (p < 1)).all())
        np.testing.assert_allclose(p, 1 - symmetric_predict(model, tb, ta), rtol=1e-6)

    def test_runs_write_outputs_and_log(self):
        p, m = fixture()
        audit = {'policy': 'exclude_all_flagged', 'used_sets': len(m)}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            log = tmp / 'log.md'
            run_score_baselines(p, m, audit, tmp / 'base', log_path=log)
            run_xgboost_score(p, m, audit, tmp / 'xgb', n_iter=2, cv_folds=2, log_path=log)
            for run in ('base', 'xgb'):
                report = json.loads((tmp / run / 'metrics.json').read_text())
                self.assertEqual(report['task'], 'score')
                preds = pd.read_csv(tmp / run / 'test_predictions.csv')
                self.assertTrue(preds.drop(columns=['match_id', 'session', 'cold_start']).notna().all().all())
                self.assertTrue((tmp / run / 'test_standings.csv').exists())
            text = log.read_text()
            self.assertIn('results:score_baselines:start', text)
            self.assertIn('results:score_xgboost:start', text)

    @unittest.skipUnless(HAS_PYG, 'torch_geometric not installed')
    def test_hetero_score_task(self):
        from utcgnn.hetero_experiment import run_hetero
        p, m = fixture()
        audit = {'policy': 'exclude_all_flagged', 'used_sets': len(m)}
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('rgcn', 'sage', 'gat'):
                report = run_hetero(p, m, audit, tmp, kind, seeds=(1, 2), epochs=3, hidden=8, layers=2,
                                    device='cpu', log_path=Path(tmp) / 'log.md', task='score')
                self.assertIsNotNone(report['summary']['test']['mae_games']['mean'])
                preds = pd.read_csv(Path(tmp) / 'test_predictions.csv')
                self.assertTrue(preds[f'{kind}_seed_mean'].between(0, 1).all())
                trace = pd.read_csv(Path(tmp) / f'{kind}_seed1_training.csv')
                self.assertEqual(list(trace.columns), ['epoch', 'train_games_bce', 'validation_mae_games'])
            self.assertIn('results:score_hetero_sage:start', (Path(tmp) / 'log.md').read_text())


if __name__ == '__main__':
    unittest.main()
