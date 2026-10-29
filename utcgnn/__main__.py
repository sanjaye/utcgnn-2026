import argparse
import json
from pathlib import Path

from .data import export_graph, load_data, season_splits
from .experiment import run_experiment
from .inference import load_inference_context, predict_teams
from .matchday import load_matchday_groups, load_name_registry, predict_matchday


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
    p = commands.add_parser('predict-day',help='Predict all doubles pairings for eight court groups')
    p.add_argument('--checkpoint',default='artifacts/main/rgcn.pt')
    p.add_argument('--date',required=True,help='Match-day date, YYYY-MM-DD')
    p.add_argument('--groups',required=True,help='CSV with one court and four player IDs per row')
    p.add_argument('--names',default='data/players.csv',help='CSV containing player_id, original_name, pseudonym')
    p.add_argument('--output',default='artifacts/matchday_predictions.csv')
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
    elif args.command == 'predict':
        h, model, checkpoint = load_inference_context(
            players, matches, args.checkpoint, args.date, args.include_flagged
        )
        ids = args.team_a+args.team_b
        if len(set(ids)) != 4 or not set(ids) <= set(players.player_id):
            raise ValueError('Supply four distinct registered player IDs')
        p = predict_teams(h, model, checkpoint, [ids])[0]
        print(json.dumps({'team_a_win_probability':p,'date':args.date,
                          'cold_start_players':[player_id for player_id in ids
                                                if h.sets[h.lookup[player_id]]==0]}))
    else:
        groups = load_matchday_groups(args.groups, players.player_id)
        names = load_name_registry(args.names, players.player_id)
        history, model, checkpoint = load_inference_context(
            players, matches, args.checkpoint, args.date, args.include_flagged
        )
        predictions = predict_matchday(
            groups, names, history, model, checkpoint, args.date
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output, index=False)
        print(json.dumps({
            'output': str(output),
            'predictions': json.loads(predictions.to_json(orient='records')),
        }, indent=2))


if __name__ == '__main__':
    main()
