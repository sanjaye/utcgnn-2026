import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from utcgnn.data import History
from utcgnn.score_prediction import (
    FLIPPED_INDEX,
    SCORELINES,
    fit_score_model,
    predict_match_scores,
    predict_scoreline_probabilities,
    summarize_player_points,
    write_score_workbook,
)


class ScorePredictionTests(unittest.TestCase):
    def test_allowed_scorelines_include_zero_and_mirror_every_class(self):
        self.assertEqual(len(SCORELINES), 14)
        self.assertIn((6, 0), SCORELINES)
        self.assertIn((7, 5), SCORELINES)
        self.assertIn((7, 6), SCORELINES)
        for index, (team_a, team_b) in enumerate(SCORELINES):
            self.assertEqual(SCORELINES[FLIPPED_INDEX[index]], (team_b, team_a))

    def test_score_probabilities_are_normalized_and_team_symmetric(self):
        rng = np.random.default_rng(42)
        x_values = rng.normal(size=(28, 5))
        labels = np.tile(np.arange(len(SCORELINES)), 2)
        model = fit_score_model(x_values, labels)

        probabilities = predict_scoreline_probabilities(model, x_values[:4])
        swapped = predict_scoreline_probabilities(model, -x_values[:4])

        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_allclose(probabilities, swapped[:, FLIPPED_INDEX])

    def test_predictions_report_winner_loser_points_and_preserve_input(self):
        player_ids = ['p1', 'p2', 'p3', 'p4']
        history = History(pd.DataFrame({'player_id': player_ids}))
        rng = np.random.default_rng(23)
        train_x = rng.normal(size=(len(SCORELINES) * 2, 5))
        train_y = np.tile(np.arange(len(SCORELINES)), 2)
        model = fit_score_model(train_x, train_y)
        predictions = pd.DataFrame([{
            'court': 'Court 1',
            'pairing': '1',
            'team_a_win_probability': '0.7',
            'predicted_winner': 'team_a',
            'team_a_player_1_id': 'p1',
            'team_a_player_2_id': 'p2',
            'team_b_player_1_id': 'p3',
            'team_b_player_2_id': 'p4',
        }])
        names = pd.DataFrame(
            {'original_name': player_ids, 'pseudonym': player_ids},
            index=pd.Index(player_ids, name='player_id'),
        )

        result = predict_match_scores(predictions, names, history, model)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, 'court'], 'Court 1')
        scoreline = (
            result.loc[0, 'team_a_predicted_points'],
            result.loc[0, 'team_b_predicted_points'],
        )
        self.assertIn(scoreline, SCORELINES)
        winner = result.loc[0, 'score_predicted_winner']
        self.assertEqual(winner, 'team_a')
        self.assertGreater(
            result.loc[0, 'team_a_predicted_points'],
            result.loc[0, 'team_b_predicted_points'],
        )
        self.assertEqual(
            result.loc[0, 'predicted_winner_points'],
            result.loc[0, f'{winner}_predicted_points'],
        )
        self.assertGreaterEqual(result.loc[0, 'scoreline_confidence'], 0)
        self.assertLessEqual(result.loc[0, 'scoreline_confidence'], 1)

    def test_score_prediction_respects_predicted_team_b_winner(self):
        player_ids = ['p1', 'p2', 'p3', 'p4']
        history = History(pd.DataFrame({'player_id': player_ids}))
        rng = np.random.default_rng(42)
        train_x = rng.normal(size=(len(SCORELINES) * 2, 5))
        train_y = np.tile(np.arange(len(SCORELINES)), 2)
        model = fit_score_model(train_x, train_y)
        predictions = pd.DataFrame([{
            'court': 'Court 2',
            'pairing': '2',
            'predicted_winner': 'team_b',
            **{
                f'{team}_player_{position}_id': player_id
                for team, team_players in [
                    ('team_a', ('p1', 'p2')),
                    ('team_b', ('p3', 'p4')),
                ]
                for position, player_id in enumerate(team_players, start=1)
            },
        }])
        names = pd.DataFrame(
            {'original_name': player_ids, 'pseudonym': player_ids},
            index=pd.Index(player_ids, name='player_id'),
        )

        result = predict_match_scores(predictions, names, history, model)

        self.assertEqual(result.loc[0, 'score_predicted_winner'], 'team_b')
        self.assertLess(
            result.loc[0, 'team_a_predicted_points'],
            result.loc[0, 'team_b_predicted_points'],
        )

    def test_player_totals_sum_all_three_predicted_match_points(self):
        predictions = pd.DataFrame([
            {
                'court': 'Court 1',
                'score_predicted_winner': winner,
                'team_a_predicted_points': team_a_points,
                'team_b_predicted_points': team_b_points,
                **{
                    f'{team}_player_{position}_{field}': value
                    for team, team_players in [
                        ('team_a', team_a),
                        ('team_b', team_b),
                    ]
                    for position, player_id in enumerate(team_players, start=1)
                    for field, value in [
                        ('id', player_id),
                        ('pseudonym', f'Alias {player_id}'),
                        ('actual_name', f'Actual {player_id}'),
                    ]
                },
            }
            for (team_a, team_b), (team_a_points, team_b_points, winner) in zip(
                [
                    (('p1', 'p2'), ('p3', 'p4')),
                    (('p1', 'p3'), ('p2', 'p4')),
                    (('p1', 'p4'), ('p2', 'p3')),
                ],
                [(6, 2, 'team_a'), (3, 6, 'team_b'), (7, 5, 'team_a')],
            )
        ])

        totals = summarize_player_points(predictions).set_index('player_id')

        self.assertEqual(totals.loc['p1', 'matches_played'], 3)
        self.assertEqual(totals.loc['p1', 'predicted_points_total'], 16)
        self.assertEqual(totals.loc['p2', 'predicted_points_total'], 17)
        self.assertEqual(totals.loc['p3', 'predicted_points_total'], 10)
        self.assertEqual(totals.loc['p4', 'predicted_points_total'], 15)
        self.assertEqual(totals.loc['p1', 'predicted_wins'], 2)
        self.assertEqual(totals.loc['p1', 'predicted_losses'], 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / 'scores.xlsx'
            write_score_workbook(predictions, workbook_path)
            with pd.ExcelFile(workbook_path) as workbook:
                self.assertEqual(
                    workbook.sheet_names,
                    ['Match predictions', 'Player totals'],
                )
                saved_totals = pd.read_excel(workbook, sheet_name='Player totals')
            self.assertEqual(len(saved_totals), 4)


if __name__ == '__main__':
    unittest.main()
