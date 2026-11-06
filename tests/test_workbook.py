import unittest

import openpyxl
import pandas as pd

from utcgnn.workbook import actual_matches, compare, lineup_groups, read_courts


class WorkbookTests(unittest.TestCase):
    def courts(self):
        # Each set has a different partnership; losing 6 in a 7-6 is not a win.
        values = [([6,7,5],18,2), ([6,6,7],19,2),
                  ([4,7,7],18,2), ([4,6,5],15,0)]
        return {'1': [dict(player_id=f'p{i}',name=f'N{i}',row=i+5,
                           scores=s,weekly_games=g,weekly_wins=w)
                       for i,(s,g,w) in enumerate(values)]}

    def test_read_player_table_skips_zero_blank_and_publishing_rows(self):
        wb=openpyxl.Workbook();s=wb.active
        s.cell(4,2,'Name of Player');s.cell(4,7,'Court #')
        for i in range(4):
            s.cell(i+5,2,f'N{i}');s.cell(i+5,7,1)
        s.cell(9,2,'Absent');s.cell(9,7,0)
        s.cell(10,2,'Blank court')
        s.cell(12,2,'Publishing');s.cell(12,7,2)
        courts=read_courts(s,{f'n{i}':f'p{i}' for i in range(4)})
        self.assertEqual(len(courts['1']),4)
        self.assertEqual(len(lineup_groups(courts)),3)

    def test_actuals_match_by_teams_despite_set_order_and_team_orientation(self):
        courts=self.courts()
        actual=actual_matches(courts,'2026-10-30','test.xlsx','ACTUAL')
        groups=lineup_groups(courts)
        predictions=[]
        # Pairing 2 is actual set 2, pairing 3 actual set 3 in this fixture.
        scores=[(6,4),(7,6),(5,7)]
        for row, (a,b) in zip(groups.to_dict('records'),scores):
            predictions.append(dict(court=row['court'],
                **{f'{t}_player_{i}_id': row[f'{t}_{i}'] for t in ('team_a','team_b') for i in (1,2)},
                team_a_predicted_points=a,team_b_predicted_points=b,
                predicted_winner='team_a' if a>b else 'team_b',
                team_a_win_probability=.8 if a>b else .2))
        result, report=compare(pd.DataFrame(predictions),actual.iloc[::-1])
        self.assertEqual(report['correct_winners'],3)
        self.assertEqual(report['exact_scores_correct'],3)
        self.assertEqual(report['team_score_mae'],0)
        self.assertEqual(result.actual_team_a_score.tolist(),[6,7,5])

    def test_actuals_reject_illegal_scores_and_total_mismatches(self):
        courts=self.courts();courts['1'][0]['scores']=[6,7,6]
        with self.assertRaises(ValueError):
            actual_matches(courts,'2026-10-30','test.xlsx','ACTUAL')
        courts=self.courts();courts['1'][0]['weekly_wins']=3
        with self.assertRaisesRegex(ValueError,'weekly totals'):
            actual_matches(courts,'2026-10-30','test.xlsx','ACTUAL')

    def test_actuals_reject_duplicate_partnerships(self):
        courts=self.courts()
        for p in courts['1']:
            p['scores']=[p['scores'][0]]*3
        with self.assertRaisesRegex(ValueError,'three team splits'):
            actual_matches(courts,'2026-10-30','test.xlsx','ACTUAL')

    def test_compare_rejects_missing_actual_match(self):
        with self.assertRaises(ValueError):
            compare(pd.DataFrame(),actual_matches(self.courts(),'2026-10-30','test.xlsx','ACTUAL'))


if __name__=='__main__':
    unittest.main()
