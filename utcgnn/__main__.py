import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .data import export_graph, load_data, season_splits
from .experiment import run_experiment
from .models import DECODERS
from .reporting import log_train
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
    p.add_argument('--output',default=None,
                   help='Default: artifacts/main for the sum decoder, artifacts/main_<decoder> otherwise')
    # Team decoder switch: 'sum' (original) adds partner embeddings; 'concat' concatenates
    # them and applies a linear layer back to the embedding size. See models.DECODERS.
    p.add_argument('--decoder',choices=DECODERS,default='sum')
    p.add_argument('--log',default='reports/experiments.md',help='Markdown results log to update')
    p.add_argument('--no-log',action='store_true',help='Do not update the results log')
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rolling',action='store_true',help='Also evaluate earlier expanding-window season folds')
    # Player–set graph from the project proposal, built with PyTorch Geometric (imported
    # only for this command, so the other commands do not need it installed).
    p = commands.add_parser('train-hetero',help='PyG R-GCN / GraphSAGE on the player-set graph, over several seeds')
    p.add_argument('--models',nargs='+',choices=['rgcn','sage'],default=['rgcn','sage'])
    p.add_argument('--seeds',type=int,default=5,help='Number of seeds, starting at --seed')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--dropout',type=float,default=0.1)
    p.add_argument('--device',default='auto',help="'auto' (GPU if available), 'cuda' or 'cpu'")
    p.add_argument('--output',default=None,help='Default: artifacts/hetero_<model>')
    p.add_argument('--log',default='reports/experiments.md',help='Markdown results log to update')
    p.add_argument('--no-log',action='store_true',help='Do not update the results log')
    p = commands.add_parser('xgboost',help='XGBoost baseline on the same features and splits, tuned by CV on train')
    p.add_argument('--output',default='artifacts/xgboost')
    p.add_argument('--n-iter',type=int,default=60,help='Randomized-search candidates')
    p.add_argument('--cv-folds',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--log',default='reports/experiments.md',help='Markdown results log to update')
    p.add_argument('--no-log',action='store_true',help='Do not update the results log')
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
        # Separate default folders per decoder, so an experiment never overwrites the main checkpoints.
        output = args.output or ('artifacts/main' if args.decoder == 'sum' else f'artifacts/main_{args.decoder}')
        report = run_experiment(players,matches,audit,output,args.epochs,args.seed,args.decoder)
        print(json.dumps({name:entry['test'] for name,entry in report['models'].items()},indent=2))
        if not args.no_log:
            log_train(args.log,report,output,' '.join(['python -m utcgnn',*sys.argv[1:]]),
                      datetime.now().strftime('%Y-%m-%d %H:%M'))
        if args.rolling:
            seasons = season_splits(matches)
            for end in range(3,len(seasons)):
                subset = matches[matches.season.isin(seasons[:end])].copy()
                fold_audit = dict(audit,used_sets=len(subset),seasons=subset.groupby('season').size().to_dict())
                run_experiment(players,subset,fold_audit,Path(output)/f'fold_{end}',args.epochs,args.seed,args.decoder)
    elif args.command == 'train-hetero':
        from .hetero_experiment import run_hetero
        command = ' '.join(['python -m utcgnn',*sys.argv[1:]])
        for kind in args.models:
            report = run_hetero(players,matches,audit,args.output or f'artifacts/hetero_{kind}',kind,
                                seeds=tuple(range(args.seed,args.seed+args.seeds)),epochs=args.epochs,
                                hidden=args.hidden,layers=args.layers,dropout=args.dropout,device=args.device,
                                log_path=None if args.no_log else args.log,command=command)
            print(json.dumps({'model':f'hetero_{kind}','device':report['device'],
                              'runtime_seconds':report['runtime_seconds'],
                              'selected_epochs':report['selected_epochs'],
                              **{s:{k:v['mean'] for k,v in report['summary'][s].items()}
                                 for s in ('validation','test')}},indent=2))
    elif args.command == 'xgboost':
        from .xgboost_baseline import run_xgboost
        report = run_xgboost(players,matches,audit,args.output,args.n_iter,args.cv_folds,args.seed,
                             log_path=None if args.no_log else args.log,
                             command=' '.join(['python -m utcgnn',*sys.argv[1:]]))
        print(json.dumps({'output':args.output,'best_params':report['tuning']['best_params'],
                          'cv_log_loss':report['tuning']['best_cv_log_loss'],
                          **{split:{k:report['metrics'][split][k] for k in ('accuracy','roc_auc','log_loss')}
                             for split in ('validation','test')}},indent=2))
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
