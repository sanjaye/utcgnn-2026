import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .data import History, export_graph, load_data, season_splits
from .experiment import run_experiment
from .models import GraphModel


def main():
    parser = argparse.ArgumentParser(description='Session-causal UTC doubles graph modeling')
    parser.add_argument('--players',default='data/morphed_players.xlsx')
    parser.add_argument('--matches',default='data/morphed_matches.csv')
    parser.add_argument('--include-flagged',action='store_true',help='Explicit data-quality sensitivity analysis')
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('profile')
    p.add_argument('--export-graph',metavar='DIRECTORY')
    p = commands.add_parser('train')
    p.add_argument('--output',default='artifacts/main')
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rolling',action='store_true',help='Also evaluate earlier expanding-window season folds')
    p = commands.add_parser('predict')
    p.add_argument('--checkpoint',default='artifacts/main/rgcn.pt')
    p.add_argument('--date',required=True,help='Session date, YYYY-MM-DD; use strictly earlier history')
    p.add_argument('--team-a',nargs=2,required=True)
    p.add_argument('--team-b',nargs=2,required=True)
    args = parser.parse_args()
    players,matches,audit = load_data(args.players,args.matches,args.include_flagged)
    if args.command == 'profile':
        if args.export_graph:
            export_graph(players,matches,args.export_graph)
        print(json.dumps(audit,indent=2))
    elif args.command == 'train':
        report = run_experiment(players,matches,audit,args.output,args.epochs,args.seed)
        print(json.dumps({name:entry['test'] for name,entry in report['models'].items()},indent=2))
        if args.rolling:
            seasons = season_splits(matches)
            for end in range(3,len(seasons)):
                subset = matches[matches.season.isin(seasons[:end])].copy()
                fold_audit = dict(audit,used_sets=len(subset),seasons=subset.groupby('season').size().to_dict())
                run_experiment(players,subset,fold_audit,Path(args.output)/f'fold_{end}',args.epochs,args.seed)
    else:
        checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if players.player_id.tolist() != checkpoint['player_ids']:
            raise ValueError('Checkpoint player registry differs from input')
        if args.include_flagged != checkpoint['include_flagged']:
            raise ValueError('Use the same flagged-row policy as training')
        date = pd.Timestamp(args.date)
        if date <= pd.Timestamp(checkpoint['selection_end']):
            raise ValueError('Prediction must be after validation/model-selection data to avoid weight leakage')
        ids = args.team_a+args.team_b
        if len(set(ids)) != 4 or not set(ids) <= set(players.player_id):
            raise ValueError('Supply four distinct registered player IDs')
        h = History(players)
        for _,rows in matches[matches.session < date].groupby('session',sort=True):
            h.update(rows)
        model = GraphModel(kind=checkpoint['kind'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        with torch.no_grad():
            t = torch.tensor([[h.lookup[p] for p in ids]])
            p = torch.sigmoid(model((h.features()-checkpoint['mean'])/checkpoint['scale'],h.relations(),t)).item()
        print(json.dumps({'team_a_win_probability':p,'date':args.date,
                          'cold_start_players':[p for p in ids if h.sets[h.lookup[p]]==0]}))


if __name__ == '__main__':
    main()
