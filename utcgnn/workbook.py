"""Convert UTC weekly workbooks to CLI inputs and evaluate saved predictions."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from sklearn.metrics import brier_score_loss

from .data import SLOTS, load_data
from .experiment import metrics
from .matchday import MATCH_COLUMNS, PAIRINGS, load_matchday_groups
from .score_prediction import SCORELINES, summarize_player_points


def name_lookup(path):
    names = pd.read_csv(path, keep_default_na=False)
    lookup = {}
    for row in names.itertuples():
        variants = json.loads(getattr(row, 'name_variants', '') or '[]')
        for name in [row.original_name, *variants]:
            key = ' '.join(name.casefold().split())
            if key in lookup and lookup[key] != row.player_id:
                raise ValueError(f'Ambiguous player name: {name}')
            lookup[key] = row.player_id
    return lookup


def read_courts(sheet, lookup):
    rows = iter(sheet.values)
    for _ in range(3):
        next(rows)
    header = next(rows)
    if header[1] != 'Name of Player' or header[6] != 'Court #':
        raise ValueError(f'{sheet.title}: expected player and court headers in B4 and G4')
    courts = defaultdict(list)
    for number, row in enumerate(rows, 5):
        row = (*row, *([None] * max(0, 13 - len(row))))
        if row[1] is None or str(row[1]).strip() == '':
            break  # Do not interpret the publishing tables below the player table.
        court = row[6]
        if court is None or str(court).strip() in ('', '0') or court == 0:
            continue
        try:
            numeric = float(court)
            if not np.isfinite(numeric) or numeric < 1 or numeric != int(numeric):
                raise ValueError()
            court = str(int(numeric))
        except (ValueError, TypeError):
            raise ValueError(f'{sheet.title}!G{number}: invalid court {court}') from None
        key = ' '.join(str(row[1]).casefold().split())
        if key not in lookup:
            raise ValueError(f'{sheet.title}!B{number}: unregistered player {row[1]}')
        courts[court].append({'player_id': lookup[key], 'name': row[1],
                              'row': number, 'scores': row[8:11],
                              'weekly_games': row[11], 'weekly_wins': row[12]})
    players = [p['player_id'] for rows in courts.values() for p in rows]
    if len(players) != len(set(players)):
        raise ValueError(f'{sheet.title}: a player is assigned more than once')
    if any(len(rows) != 4 for rows in courts.values()):
        raise ValueError(f'{sheet.title}: each assigned court must contain four players')
    return dict(courts)


def lineup_groups(courts):
    records = []
    for court, rows in courts.items():
        ids = [r['player_id'] for r in rows]
        for pairing, (a, b) in enumerate(PAIRINGS, 1):
            records.append(dict(zip(MATCH_COLUMNS, [court, pairing, *[ids[i] for i in (*a, *b)]])))
    if not records:
        raise ValueError('No scheduled courts found')
    return pd.DataFrame(records)


def partition(row):
    return tuple(sorted((tuple(sorted(row[:2])), tuple(sorted(row[2:])))))


def actual_matches(courts, date, source, sheet):
    date = pd.Timestamp(date).normalize()
    records = []
    for court, players in courts.items():
        partitions = set()
        wins = defaultdict(int)
        for number in range(3):
            by_score = defaultdict(list)
            source_rows = defaultdict(list)
            for p in players:
                score = p['scores'][number]
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score) or score != int(score):
                    raise ValueError(f'{sheet}: court {court}, set {number+1}: missing/invalid score')
                by_score[int(score)].append(p['player_id'])
                source_rows[int(score)].append(p['row'])
            scores = sorted(by_score, reverse=True)
            if len(scores) != 2 or any(len(ids) != 2 for ids in by_score.values()) or tuple(scores) not in SCORELINES:
                raise ValueError(f'{sheet}: court {court}, set {number+1}: cannot reconstruct a legal doubles set')
            ids = by_score[scores[0]] + by_score[scores[1]]
            for player_id in by_score[scores[0]]:
                wins[player_id] += 1
            partitions.add(partition(ids))
            local_date = date.tz_localize('America/Los_Angeles')
            records.append(dict(zip(SLOTS, ids), match_id=f'UTC_{date.date()}_C{int(court):02}_S{number+1}',
                season=f'{date.year}-'+('Jan-Jun' if date.month <= 6 else 'Jul-Dec'),
                timestamp=local_date.isoformat(), court=court, set_number=number+1,
                team_a_win=1, team_a_score=scores[0], team_b_score=scores[1],
                source_file=str(source), source_sheet=sheet,
                source_rows=json.dumps(source_rows[scores[0]]+source_rows[scores[1]]),
                validation_flags='', time_precision='session_date', weekly_totals_verified=1))
        if len(partitions) != 3:
            raise ValueError(f'{sheet}: court {court}: sets do not cover all three team splits')
        for p in players:
            if sum(p['scores']) != p['weekly_games'] or wins[p['player_id']] != p['weekly_wins']:
                raise ValueError(f'{sheet}: row {p["row"]}: weekly totals disagree with set scores')
    return pd.DataFrame(records)


def prepare(args):
    date = pd.Timestamp(args.date).normalize()
    wb = openpyxl.load_workbook(args.workbook, data_only=True)
    lookup = name_lookup(args.names)
    groups = lineup_groups(read_courts(wb[args.sheet], lookup))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    groups.to_csv(out/'court_pairings.csv', index=False)
    load_matchday_groups(out/'court_pairings.csv', set(lookup.values()))
    history = pd.read_csv(args.matches)
    dates = pd.to_datetime(history.timestamp.str[:10])
    history = history[dates < date].copy()
    if history.empty:
        raise ValueError('No historical matches before target date')
    last = pd.to_datetime(history.timestamp.str[:10]).max()
    additions, empty_sheets = [], []
    for sheet in wb.sheetnames:
        if len(sheet) != 6 or not sheet.isdigit():
            continue
        sheet_date = pd.to_datetime(sheet, format='%m%d%y', errors='coerce')
        if pd.isna(sheet_date) or not last < sheet_date < date:
            continue
        courts = read_courts(wb[sheet], lookup)
        if courts:
            additions.append(actual_matches(courts, sheet_date, Path(args.workbook).name, sheet))
        else:
            empty_sheets.append(sheet)
    history = pd.concat([history, *additions], ignore_index=True)
    history = history.sort_values(['timestamp', 'match_id'])
    history.to_csv(out/'history.csv', index=False)
    _, accepted, audit = load_data(args.players, out/'history.csv')
    report = {'date':str(date.date()), 'workbook':str(args.workbook),'lineup_sheet':args.sheet,
        'courts':groups.court.nunique(), 'matches':len(groups),
        'history_last_date':str(accepted.session.max().date()), 'accepted_history_sets':len(accepted),
        'additional_history_sets':sum(map(len, additions)), 'empty_history_sheets':empty_sheets,
        'flagged_history_sets_excluded':audit['excluded_sets'],
        'policy':'Target-date and ACTUAL sheets excluded from all prediction inputs.'}
    (out/'preparation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def compare(predictions, actuals):
    actual_by_key = {}
    for row in actuals.to_dict('records'):
        key = (str(row['court']), partition([row[s] for s in SLOTS]))
        if key in actual_by_key:
            raise ValueError('Duplicate actual matchup')
        actual_by_key[key] = row
    records, matched = [], set()
    for row in predictions.to_dict('records'):
        ids = [row[f'{t}_player_{p}_id'] for t in ('team_a','team_b') for p in (1,2)]
        key = (str(row['court']), partition(ids))
        if key not in actual_by_key or key in matched:
            raise ValueError('Missing, duplicated, or mismatched prediction/actual matchup')
        actual = actual_by_key[key]
        aligned = set(ids[:2]) == {actual['team_a_1'],actual['team_a_2']}
        a, b = actual['team_a_score'], actual['team_b_score']
        if not aligned:
            a, b = b, a
        pa, pb = int(row['team_a_predicted_points']), int(row['team_b_predicted_points'])
        if (pa,pb) not in SCORELINES:
            raise ValueError('Prediction contains an illegal score')
        records.append(dict(row, actual_set_number=actual['set_number'],
            actual_team_a_score=a, actual_team_b_score=b, actual_team_a_win=int(a>b),
            winner_correct=(row['predicted_winner'] == ('team_a' if a>b else 'team_b')),
            exact_score_correct=(pa,pb)==(a,b), team_a_absolute_error=abs(pa-a),
            team_b_absolute_error=abs(pb-b)))
        matched.add(key)
    if matched != set(actual_by_key):
        raise ValueError('Actuals contain unmatched sets')
    result = pd.DataFrame(records)
    y=result.actual_team_a_win.to_numpy()
    probs=result.team_a_win_probability.to_numpy(dtype=float)
    summary=metrics(y,probs)
    summary.update(correct_winners=int(result.winner_correct.sum()),
        brier_score=float(brier_score_loss(y,probs)),
        exact_scores_correct=int(result.exact_score_correct.sum()),
        exact_score_accuracy=float(result.exact_score_correct.mean()),
        team_score_mae=float(result[['team_a_absolute_error','team_b_absolute_error']].to_numpy().mean()),
        team_score_rmse=float(np.sqrt(np.mean(np.concatenate([
            (result.team_a_predicted_points-result.actual_team_a_score).to_numpy()**2,
            (result.team_b_predicted_points-result.actual_team_b_score).to_numpy()**2])))),
        coin_flip_log_loss=float(np.log(2)),coin_flip_brier_score=0.25)
    return result, summary


def evaluate(args):
    wb=openpyxl.load_workbook(args.workbook,data_only=True)
    actual=actual_matches(read_courts(wb[args.sheet],name_lookup(args.names)),args.date,
                          Path(args.workbook).name,args.sheet)
    predictions=pd.read_csv(args.predictions,dtype={'court':str})
    if not pd.to_datetime(predictions.date).eq(pd.Timestamp(args.date)).all():
        raise ValueError('Prediction dates disagree with evaluation date')
    result, summary=compare(predictions,actual)
    totals=summarize_player_points(predictions)
    player_actual=defaultdict(lambda: {'actual_games':0,'actual_wins':0})
    for r in actual.to_dict('records'):
        for team in ('team_a','team_b'):
            for p in (1,2):
                rec=player_actual[r[f'{team}_{p}']]
                rec['actual_games']+=r[f'{team}_score']
                rec['actual_wins']+=int(team=='team_a')  # Actual extraction puts winner in A.
    totals['actual_games']=[player_actual[p]['actual_games'] for p in totals.player_id]
    totals['actual_wins']=[player_actual[p]['actual_wins'] for p in totals.player_id]
    totals['games_error']=totals.predicted_points_total-totals.actual_games
    totals['wins_error']=totals.predicted_wins-totals.actual_wins
    summary.update(player_total_games_mae=float(totals.games_error.abs().mean()),
                   player_total_wins_mae=float(totals.wins_error.abs().mean()),
                   date=args.date,actual_sheet=args.sheet,
                   matching='Court and unordered teams; actual set order and A/B orientation may differ.')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    actual.to_csv(out/'actual_matches.csv',index=False)
    result.to_csv(out/'match_comparison.csv',index=False)
    totals.to_csv(out/'player_comparison.csv',index=False)
    courts=result.groupby('court').agg(matches=('winner_correct','size'),
        correct_winners=('winner_correct','sum'),exact_scores=('exact_score_correct','sum'))
    courts.to_csv(out/'court_performance.csv')
    (out/'performance.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','evaluate'):
        p=sub.add_parser(command)
        p.add_argument('--workbook',required=True)
        p.add_argument('--sheet',required=True)
        p.add_argument('--date',required=True)
        p.add_argument('--names',default='data/players.csv')
        p.add_argument('--output',required=True)
        if command=='prepare':
            p.add_argument('--matches',default='data/morphed_matches.csv')
            p.add_argument('--players',default='data/morphed_players.xlsx')
        else:
            p.add_argument('--predictions',required=True)
    args=parser.parse_args()
    (prepare if args.command=='prepare' else evaluate)(args)


if __name__=='__main__':
    main()
