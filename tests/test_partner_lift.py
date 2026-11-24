import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utcgnn.partner_lift import (
    calibrated_scale, expected_results, posterior_lift, run, summarize_pairs, player_view,
)


def fixture():
    players = pd.DataFrame({'player_id': ['p1', 'p2', 'p3', 'p4'],
                            'pseudonym': ['one', 'two', 'three', 'four'],
                            'original_name': ['One', 'Two', 'Three', 'Four']})
    rows = []
    for day in range(1, 5):
        for number, ids in enumerate([
            ['p1', 'p2', 'p3', 'p4'], ['p1', 'p3', 'p2', 'p4'],
            ['p1', 'p4', 'p2', 'p3'],
        ], 1):
            rows.append(dict(match_id=f'{day}_{number}', season='2026', court=1,
                             set_number=number, timestamp=f'2026-01-{day:02}T00:00:00-08:00',
                             session=pd.Timestamp(f'2026-01-{day:02}'),
                             **dict(zip(['team_a_1', 'team_a_2', 'team_b_1', 'team_b_2'], ids)),
                             team_a_win=1, team_a_score=6, team_b_score=3,
                             validation_flags=''))
    return players, pd.DataFrame(rows)


class PartnerLiftTests(unittest.TestCase):
    def test_calibration_strength_symmetry_and_date_batching(self):
        self.assertEqual(calibrated_scale([0.4] * 99, [1] * 99), 1)
        scale = calibrated_scale([0.4] * 100, [1] * 80 + [0] * 20)
        self.assertGreater(scale, 1)
        self.assertEqual(scale, calibrated_scale([-0.4] * 100, [0] * 80 + [1] * 20))
        players, matches = fixture()
        copies = []
        for i in range(40):
            copy = matches.iloc[:3].copy()
            copy['session'] = pd.Timestamp('2026-01-01') + pd.Timedelta(days=i)
            copy['match_id'] = [f'{i}_{j}' for j in range(3)]
            copies.append(copy)
        history = pd.concat(copies, ignore_index=True)
        before = expected_results(players, history, 0)
        history.loc[history.session >= '2026-02-06', 'team_a_win'] = 0
        after = expected_results(players, history, 0)
        np.testing.assert_allclose(before.team_a_expected_win[:111], after.team_a_expected_win[:111])
        self.assertTrue(before.groupby('date').elo_scale.nunique().eq(1).all())
        self.assertEqual(before.calibration_prior_sets.iloc[3], 3)

    def test_no_same_day_or_future_leakage_and_experience_gate(self):
        players, matches = fixture()
        before = expected_results(players, matches, min_prior_sets=3)
        changed = matches.copy()
        changed.loc[changed.session >= '2026-01-02', 'team_a_win'] = 0
        after = expected_results(players, changed, min_prior_sets=3)
        np.testing.assert_allclose(before.team_a_expected_win[:6], after.team_a_expected_win[:6])
        self.assertFalse(np.allclose(before.team_a_expected_win[6:], after.team_a_expected_win[6:]))
        np.testing.assert_allclose(before.team_a_expected_win[:3], 0.5)
        self.assertFalse(before.eligible[:3].any())
        self.assertTrue(before.eligible[3:].all())
        reordered = expected_results(players, matches.iloc[::-1], min_prior_sets=3)
        pd.testing.assert_frame_equal(before, reordered)

    def test_partner_order_and_team_orientation(self):
        players, matches = fixture()
        names = players.set_index('player_id')
        baseline = summarize_pairs(expected_results(players, matches, 0), names)
        flipped = matches.copy()
        flipped[['team_a_1', 'team_a_2', 'team_b_1', 'team_b_2']] = matches[
            ['team_b_2', 'team_b_1', 'team_a_2', 'team_a_1']].to_numpy()
        flipped['team_a_win'] = 1 - matches.team_a_win
        flipped['team_a_score'] = matches.team_b_score
        flipped['team_b_score'] = matches.team_a_score
        other = summarize_pairs(expected_results(players, flipped, 0), names)
        keys = ['player_1_id', 'player_2_id']
        pd.testing.assert_frame_equal(baseline.sort_values(keys).reset_index(drop=True),
                                      other.sort_values(keys).reset_index(drop=True))
        self.assertEqual(baseline.sets.sum(), 2 * len(matches))
        self.assertAlmostEqual(baseline.wins_above_expected.sum(), 0)
        view = player_view(baseline)
        self.assertEqual(len(view), 2 * len(baseline))
        self.assertEqual(len(view.loc[(view.player_id == 'p1') & (view.partner_id == 'p2')]), 1)

    def test_posterior_shrinkage_uncertainty_and_negative_lift(self):
        small = posterior_lift([0.5], [1])
        self.assertGreater(small['adjusted_lift_pp'], 0)
        self.assertLess(small['adjusted_lift_pp'], 50)
        self.assertLess(small['lift_lower_95_pp'], 0)
        many = posterior_lift([0.5] * 100, [1] * 60 + [0] * 40)
        self.assertGreater(many['adjusted_lift_pp'], 0)
        self.assertLess(many['adjusted_lift_pp'], 10)
        self.assertLess(many['lift_upper_95_pp'] - many['lift_lower_95_pp'],
                        small['lift_upper_95_pp'] - small['lift_lower_95_pp'])
        negative = posterior_lift([0.5] * 100, [0] * 60 + [1] * 40)
        self.assertAlmostEqual(negative['adjusted_lift_pp'], -many['adjusted_lift_pp'])
        for sd in [0, -1, float('nan'), 3]:
            with self.assertRaises(ValueError):
                posterior_lift([0.5], [1], sd)

    def test_csv_pipeline_flags_date_window_and_outputs(self):
        players, matches = fixture()
        matches.loc[0, 'validation_flags'] = 'suspect'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            players.to_csv(root / 'players.csv', index=False)
            matches.drop(columns='session').to_csv(root / 'matches.csv', index=False)
            args = argparse.Namespace(players=root / 'players.csv', matches=root / 'matches.csv',
                                      min_prior_sets=0, min_pair_sets=2, prior_sd=0.75,
                                      since='2026-01-02', before='2026-01-04', output=root / 'out')
            summary = run(args)
            self.assertEqual(summary['input_audit']['excluded_sets'], 1)
            self.assertEqual(summary['eligible_sets'], 6)
            audit = pd.read_csv(root / 'out/match_expectations.csv')
            self.assertEqual(audit.date.min(), '2026-01-02')
            self.assertEqual(audit.date.max(), '2026-01-03')
            # Earlier results initialize Elo even when omitted from the report.
            self.assertTrue(audit.team_a_expected_win.ne(0.5).any())
            pairs = pd.read_csv(root / 'out/partner_lift.csv')
            self.assertEqual(pairs.sets.sum(), 12)
            self.assertEqual(json.loads((root / 'out/summary.json').read_text())['report_sets'], 6)
            report = (root / 'out/report.html').read_text(encoding='utf-8')
            self.assertIn('One', report)
            self.assertIn('Method and limits', report)


if __name__ == '__main__':
    unittest.main()
