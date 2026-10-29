import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utcgnn.data import History, SLOTS, export_graph, load_data, season_splits, snapshots
from utcgnn.experiment import run_experiment
from utcgnn.inference import load_inference_context, predict_teams
from utcgnn.matchday import PAIRINGS, load_matchday_groups, load_name_registry, predict_matchday
from utcgnn.models import GraphModel


def fixture():
    players = pd.DataFrame({'player_id':['p1','p2','p3','p4','p5'],
                            'pseudonym':['one','two','three','four','five']})
    rows = []
    for i in range(6):
        day = f'{2023+i//2}-01-{1+i%2:02d}'
        rows.append({'match_id':f'm{i}','season':str(2023+i//2),
                     'timestamp':day+'T00:00:00-08:00','session':pd.Timestamp(day),
                     'court':1,'set_number':1,'team_a_1':'p1','team_a_2':'p2',
                     'team_b_1':'p3','team_b_2':'p4','team_a_win':i%2,
                     'team_a_score':6 if i%2 else 2,'team_b_score':2 if i%2 else 6,
                     'validation_flags':''})
    return players,pd.DataFrame(rows)


class TemporalTests(unittest.TestCase):
    def test_future_outcomes_do_not_change_past_inputs(self):
        p,m = fixture()
        before,_ = snapshots(p,m)
        m.loc[2:,'team_a_win'] = 1-m.loc[2:,'team_a_win']
        after,_ = snapshots(p,m)
        for i in range(3):
            torch.testing.assert_close(before[i].x,after[i].x)
            torch.testing.assert_close(before[i].relations,after[i].relations)
        self.assertFalse(torch.equal(before[3].x,after[3].x))

    def test_same_session_is_batched(self):
        p,m = fixture()
        m.loc[1,'session'] = m.loc[0,'session']
        s,_ = snapshots(p,m)
        self.assertEqual(len(s[0].y),2)
        self.assertEqual(s[0].relations.sum().item(),0)
        self.assertTrue(s[0].cold.all())
        shuffled,_ = snapshots(p,m.iloc[::-1])
        torch.testing.assert_close(s[1].x,shuffled[1].x)
        torch.testing.assert_close(s[1].relations,shuffled[1].relations)

    def test_relations_and_elo(self):
        p,m = fixture()
        h = History(p)
        h.update(m.iloc[:1])
        self.assertEqual(h.counts[0].sum(),4)
        self.assertEqual(h.counts[1].sum(),8)
        self.assertEqual(h.counts[0,0,1],1)
        self.assertEqual(h.counts[1,0,2],1)
        self.assertAlmostEqual(h.elo.sum(),7500)
        torch.testing.assert_close(h.relations().sum(2)[:,:4],torch.ones(2,4))
        self.assertEqual(h.sets[4],0)
        self.assertEqual(h.elo[4],1500)

    def test_model_team_symmetries(self):
        p,m = fixture()
        s,_ = snapshots(p,m)
        for kind in ['rgcn','sage']:
            model = GraphModel(kind=kind)
            g = s[-1]
            a = model(g.x,g.relations,g.teams)
            b = model(g.x,g.relations,g.teams[:,[1,0,3,2]])
            c = model(g.x,g.relations,g.teams[:,[2,3,0,1]])
            torch.testing.assert_close(a,b)
            torch.testing.assert_close(a,-c)

    def test_matchday_predicts_all_pairings_with_both_names(self):
        ids = [f'p{i:02}' for i in range(32)]
        players = pd.DataFrame({'player_id':ids})
        history = History(players)
        model = GraphModel()
        checkpoint = {'mean':torch.zeros(5),'scale':torch.ones(5)}
        groups = pd.DataFrame([
            {
                'court':f'Court {court+1}',
                'pairing':pairing_number,
                'team_a_1':ids[court*4+team_a[0]],
                'team_a_2':ids[court*4+team_a[1]],
                'team_b_1':ids[court*4+team_b[0]],
                'team_b_2':ids[court*4+team_b[1]],
            }
            for court in range(8)
            for pairing_number,(team_a,team_b) in enumerate(PAIRINGS,start=1)
        ])
        names = pd.DataFrame({
            'original_name':[f'Actual {player_id}' for player_id in ids],
            'pseudonym':[f'Alias {player_id}' for player_id in ids],
        },index=pd.Index(ids,name='player_id'))

        predictions = predict_matchday(
            groups,names,history,model,checkpoint,'2026-11-07'
        )

        self.assertEqual(len(predictions),24)
        self.assertEqual(predictions.court.nunique(),8)
        self.assertEqual(predictions.iloc[0].pairing,1)
        self.assertEqual(predictions.iloc[0].team_a_player_1_id,'p00')
        self.assertEqual(predictions.iloc[0].team_a_player_2_id,'p01')
        self.assertEqual(predictions.iloc[1].team_a_player_2_id,'p02')
        self.assertEqual(predictions.iloc[2].team_a_player_2_id,'p03')
        self.assertEqual(predictions.iloc[0].team_a_player_1_pseudonym,'Alias p00')
        self.assertEqual(predictions.iloc[0].team_a_player_1_actual_name,'Actual p00')
        self.assertAlmostEqual(predictions.iloc[0].team_a_win_probability,0.5)
        self.assertTrue(predictions.cold_start.all())

    def test_matchday_loader_requires_all_pairings_and_32_registered_players(self):
        ids = [f'p{i:02}' for i in range(32)]
        groups = pd.DataFrame([
            {
                'court':f'Court {court+1}',
                'pairing':pairing_number,
                'team_a_1':ids[court*4+team_a[0]],
                'team_a_2':ids[court*4+team_a[1]],
                'team_b_1':ids[court*4+team_b[0]],
                'team_b_2':ids[court*4+team_b[1]],
            }
            for court in range(8)
            for pairing_number,(team_a,team_b) in enumerate(PAIRINGS,start=1)
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'groups.csv'
            groups.to_csv(path,index=False)
            loaded = load_matchday_groups(path,ids)
            self.assertEqual(len(loaded),24)
            name_path = Path(tmp)/'players.csv'
            pd.DataFrame({
                'player_id':ids[::-1],
                'original_name':[f'Actual {player_id}' for player_id in ids[::-1]],
                'pseudonym':[f'Alias {player_id}' for player_id in ids[::-1]],
            }).to_csv(name_path,index=False)
            names = load_name_registry(name_path,ids)
            self.assertEqual(names.loc['p00'].original_name,'Actual p00')
            groups.loc[0,'pairing'] = 2
            groups.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'one row for each pairing'):
                load_matchday_groups(path,ids)

    def test_season_overlap_rejected(self):
        _,m = fixture()
        self.assertEqual(season_splits(m),['2023','2024','2025'])
        m.loc[0,'session'] = pd.Timestamp('2025-12-01')
        with self.assertRaisesRegex(ValueError,'overlap'):
            season_splits(m)

    def test_loader_and_graph_export(self):
        p,m = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p.to_excel(d/'players.xlsx',index=False)
            m.loc[0,'validation_flags'] = 'attendance_award_with_scores'
            m.to_csv(d/'matches.csv',index=False,encoding='utf-8-sig')
            players,matches,audit = load_data(d/'players.xlsx',d/'matches.csv')
            self.assertEqual(audit['excluded_sets'],1)
            self.assertEqual(len(matches),5)
            export_graph(players,matches,d/'graph')
            self.assertEqual(len(pd.read_csv(d/'graph/participation.csv')),20)
            self.assertEqual(len(pd.read_csv(d/'graph/relations.csv')),60)
            _,included,_ = load_data(d/'players.xlsx',d/'matches.csv',True)
            self.assertEqual(len(included),6)
            for field,value in [('team_a_1','unknown'),('team_a_2','p1'),
                                ('team_a_win',.5),('team_a_score',-1),
                                ('team_a_score',6),('match_id','m1')]:
                bad = m.copy()
                if isinstance(value, float):
                    bad[field] = bad[field].astype(float)
                bad.loc[0,field] = value
                bad.to_csv(d/'matches.csv',index=False)
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    load_data(d/'players.xlsx',d/'matches.csv')

    def test_training_and_saved_artifacts(self):
        p,m = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            report = run_experiment(p,m,{'policy':'exclude_all_flagged'},tmp,epochs=2)
            self.assertEqual(report['models']['rgcn']['test']['n'],2)
            self.assertEqual(report['splits']['test'],['2025'])
            saved = torch.load(Path(tmp)/'rgcn.pt',weights_only=True)
            self.assertEqual(saved['selection_end'],'2024-01-02')
            json.loads((Path(tmp)/'metrics.json').read_text())
            predictions = pd.read_csv(Path(tmp)/'test_predictions.csv')
            self.assertTrue(np.isfinite(predictions.rgcn).all())
            self.assertTrue(predictions.rgcn.between(0,1).all())
            history,model,checkpoint = load_inference_context(
                p,m,Path(tmp)/'rgcn.pt','2025-01-03',False
            )
            probability = predict_teams(history,model,checkpoint,[['p1','p2','p3','p4']])[0]
            self.assertTrue(0 <= probability <= 1)


if __name__ == '__main__':
    unittest.main()
