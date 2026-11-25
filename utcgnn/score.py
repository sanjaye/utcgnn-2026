"""Task 2 (score): predict team A's share of games in a set.

Target per set: share = team_a_score / (team_a_score + team_b_score). Swapping the teams
gives 1 - share, so every Task 1 model's antisymmetric logit carries over unchanged:
sigmoid(logit) is read as team A's expected game share instead of P(A wins). Only the
loss, model selection and metrics differ. Models are trained with the binomial
likelihood of the games (each game a Bernoulli trial with probability sigmoid(logit)),
which is BCE with the soft label `share` weighted by the set's number of games.

Metrics per split:
    mae_games   mean |predicted share - share| * games in the set (= |share_hat * n - a|)
    mae_share   the same error in share units
    rmse_games  root mean squared error in games
    cold_start / established  mae_games for sets with / without a no-history player

Standings: each player's expected points over a split are the sum, over their sets, of
their side's predicted share times the games in that set. They are compared with the
performance-only standings (actual games won over the same sets) by Spearman rho:
    total    summed over all sets (primary; attendance contributes to both rankings)
    per_set  expected vs actual share of the games in the player's sets, i.e. both totals
             divided by the games the player played (secondary; removes attendance and
             set length, so a constant prediction cannot rank anyone)
Only players with at least MIN_SETS sets in the split are ranked. Official ladder
points and attendance awards are not in the data, so this is not the official ladder.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from .data import FEATURES
from .experiment import feature_differences, split_sessions
from .reporting import log_markdown

MIN_SETS = 5
METRICS = ('mae_games', 'mae_share', 'rmse_games', 'cold_mae_games', 'total_spearman', 'per_set_spearman')
# Every Task 2 run writes a metrics.json with a 'rows' entry; the summary table reads them all.
SCORE_RUNS = ('artifacts/score_baselines', 'artifacts/xgboost_score',
              'artifacts/hetero_rgcn_score', 'artifacts/hetero_sage_score', 'artifacts/hetero_gat_score')


def score_targets(matches, groups):
    """Per split: team A games, games per set, share, team indices and cold-start flags, in set order."""
    by_id = matches.set_index('match_id')
    out = {}
    for split, g in groups.items():
        ids = [i for s in g for i in s.ids]
        a = by_id.loc[ids, 'team_a_score'].to_numpy(float)
        games = a + by_id.loc[ids, 'team_b_score'].to_numpy(float)
        out[split] = {'a': a, 'games': games, 'share': a / games, 'ids': ids,
                      'session': [str(s.session.date()) for s in g for _ in s.ids],
                      'teams': np.concatenate([s.teams.numpy() for s in g]).reshape(-1, 4),
                      'cold': np.concatenate([s.cold for s in g]).astype(bool)}
    return out


def mae_games(t, pred):
    return float(np.mean(np.abs(np.asarray(pred) - t['share']) * t['games']))


def _errors(share, games, pred):
    if len(share) == 0:
        return {'n': 0, 'mae_games': None, 'mae_share': None, 'rmse_games': None}
    err = (np.asarray(pred) - share) * games
    return {'n': len(share), 'mae_games': float(np.mean(np.abs(err))),
            'mae_share': float(np.mean(np.abs(np.asarray(pred) - share))),
            'rmse_games': float(np.sqrt(np.mean(err ** 2)))}


def standings(t, pred, n_players, min_sets=MIN_SETS):
    """Per player: sets, actual and expected games (sum over sets). Rows for every player index."""
    pred, games, a = np.asarray(pred, float), t['games'], t['a']
    # Columns follow the slots team_a_1, team_a_2, team_b_1, team_b_2.
    expected = np.column_stack([pred * games] * 2 + [(1 - pred) * games] * 2)
    actual = np.column_stack([a] * 2 + [games - a] * 2)
    ids = t['teams'].ravel()
    sets = np.bincount(ids, minlength=n_players)
    return pd.DataFrame({'sets': sets, 'games_played': np.bincount(ids, np.repeat(games, 4), n_players),
                         'actual_games': np.bincount(ids, actual.ravel(), n_players),
                         'expected_games': np.bincount(ids, expected.ravel(), n_players),
                         'ranked': sets >= min_sets})


def _spearman(x, y):
    if len(x) < 3 or np.ptp(x) < 1e-9 or np.ptp(y) < 1e-9:
        return None  # undefined, e.g. a constant prediction gives every player the same rate
    return float(spearmanr(x, y).statistic)


def standings_metrics(t, pred, n_players, min_sets=MIN_SETS):
    table = standings(t, pred, n_players, min_sets)
    r = table[table.ranked]
    return {'players': int(len(r)), 'min_sets': min_sets,
            'total_spearman': _spearman(r.expected_games.to_numpy(), r.actual_games.to_numpy()),
            'per_set_spearman': _spearman((r.expected_games / r.games_played).to_numpy(),
                                          (r.actual_games / r.games_played).to_numpy())}


def evaluate_score(t, pred, n_players):
    """Errors overall / cold-start / established, plus standings correlations, for one split."""
    pred = np.asarray(pred, float)
    result = _errors(t['share'], t['games'], pred)
    cold = t['cold']
    result['cold_start'] = _errors(t['share'][cold], t['games'][cold], pred[cold])
    result['established'] = _errors(t['share'][~cold], t['games'][~cold], pred[~cold])
    result['standings'] = standings_metrics(t, pred, n_players)
    return result


def flat(result):
    """The table metrics of one evaluate_score result."""
    return {'mae_games': result['mae_games'], 'mae_share': result['mae_share'], 'rmse_games': result['rmse_games'],
            'cold_mae_games': result['cold_start']['mae_games'],
            'total_spearman': result['standings']['total_spearman'],
            'per_set_spearman': result['standings']['per_set_spearman']}


def summarize(results):
    """Mean and standard deviation over seeds of each flat metric. results: [flat dict]."""
    out = {}
    for k in METRICS:
        values = [r[k] for r in results if r[k] is not None]
        out[k] = {'mean': float(np.mean(values)) if values else None, 'std': float(np.std(values)) if values else None}
    return out


def save_score_outputs(out, targets, preds, player_ids, standings_model=None):
    """{split}_predictions.csv (one predicted-share column per model) and {split}_standings.csv."""
    out = Path(out)
    for split, t in targets.items():
        pd.DataFrame({'match_id': t['ids'], 'session': t['session'], 'team_a_score': t['a'].astype(int),
                      'team_b_score': (t['games'] - t['a']).astype(int), 'share': t['share'],
                      'cold_start': t['cold'], **preds[split]}).to_csv(out / f'{split}_predictions.csv', index=False)
        columns = [standings_model] if standings_model else list(preds[split])
        table = None
        for name in columns:
            s = standings(t, preds[split][name], len(player_ids))
            if table is None:
                table = s[['sets', 'games_played', 'actual_games', 'ranked']].copy()
                table.insert(0, 'player_id', player_ids)
            table[f'expected_games_{name}'] = s.expected_games
        table = table[table.sets > 0].sort_values('actual_games', ascending=False)
        table.to_csv(out / f'{split}_standings.csv', index=False)


def cell(v):
    if isinstance(v, dict):
        return '—' if v['mean'] is None else f"{v['mean']:.4f} ± {v['std']:.4f}"
    return '—' if v is None else f'{v:.4f}'


def score_table(rows):
    """rows: {label: {split: {metric: value or {'mean', 'std'}}}}."""
    lines = ['| Model | Val MAE (games) | Val MAE (share) | Val ρ total | Val ρ per set '
             '| Test MAE (games) | Test MAE (share) | Test ρ total | Test ρ per set |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, r in rows.items():
        cells = [cell(r[s][m]) for s in ('validation', 'test')
                 for m in ('mae_games', 'mae_share', 'total_spearman', 'per_set_spearman')]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def score_references(splits, audit, runs=SCORE_RUNS):
    """Table rows of every Task 2 run whose metrics.json used the same data and splits."""
    rows = {}
    for run in runs:
        path = Path(run) / 'metrics.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if (report.get('task') == 'score' and report.get('splits') == splits
                and report.get('audit', {}).get('used_sets') == audit['used_sets']
                and report.get('audit', {}).get('policy') == audit['policy']):
            rows.update(report['rows'])
    return rows


def log_score(log_path, name, body, report, reference_runs=SCORE_RUNS):
    """Write this run's section, then refresh the Task 2 summary table built from every score run."""
    log_markdown(log_path, name, body)
    rows = score_references(report['splits'], report['audit'], reference_runs)
    if not rows:
        return
    ranked = report['rows'][next(iter(report['rows']))]
    log_markdown(log_path, 'score_summary', f"""
### Task 2 summary: game share and standings

Predicted share of games for team A. MAE in games is |predicted share − share| × games in the set.
ρ is Spearman correlation between expected points (predicted share × games, summed over a
player's sets) and performance-only standings (actual games won), over players with at least
{MIN_SETS} sets in the split: **total** sums over sets; **per set** divides both by the games the
player played, giving expected vs actual share of games, which removes attendance and set length.
GNN rows are mean ± standard deviation over seeds; the others are single runs. Rebuilt after every
Task 2 command from `{'`, `'.join(f'{r}/metrics.json' for r in reference_runs)}`.

{score_table(rows)}

Validation has {report['metrics_n']['validation']:,} sets and test {report['metrics_n']['test']:,};
{report['players_ranked']['validation']} and {report['players_ranked']['test']} players are ranked.
""")


def share_logistic(x, a, games):
    """Binomial logistic regression on games won, no intercept, with swapped-team augmentation.

    Each set contributes y=1 with weight a (games won by A) and y=0 with weight b, and the
    swapped copy -x the reverse, so predict_proba gives the expected share and
    share(B) = 1 - share(A) exactly.
    """
    b = games - a
    model = LogisticRegression(C=1., fit_intercept=False, max_iter=1000)
    model.fit(np.concatenate([x, x, -x, -x]), np.concatenate([np.ones_like(a), np.zeros_like(a)] * 2),
              sample_weight=np.concatenate([a, b, b, a]))
    return model


def report_base(splits, audit, targets, n_players):
    return {'task': 'score', 'splits': splits, 'audit': audit,
            'metrics_n': {s: len(t['share']) for s, t in targets.items()},
            'players_ranked': {s: standings_metrics(t, t['share'], n_players)['players'] for s, t in targets.items()}}


def run_score_baselines(players, matches, audit, output, log_path='reports/experiments.md', command=None):
    """Constant 0.5 and binomial logistic regression on the team feature differences."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    splits, groups, _ = split_sessions(players, matches)
    targets = score_targets(matches, groups)
    n_players = len(players)
    x = {s: np.concatenate([feature_differences(g) for g in groups[s]]) for s in groups}
    model = share_logistic(x['train'], targets['train']['a'], targets['train']['games'])
    preds = {s: {'constant': np.full(len(x[s]), .5), 'logistic': model.predict_proba(x[s])[:, 1]} for s in groups}
    results = {name: {s: evaluate_score(targets[s], preds[s][name], n_players) for s in groups}
               for name in ('constant', 'logistic')}
    save_score_outputs(out, targets, preds, players.player_id.tolist())
    labels = {'constant': 'Constant 0.5', 'logistic': 'Logistic regression (binomial)'}
    run_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    report = {**report_base(splits, audit, targets, n_players), 'model': 'score_baselines', 'run_at': run_at,
              'features': 'team A minus team B sums of the 5 pre-session player features',
              'coefficients': dict(zip(FEATURES, map(float, model.coef_[0]))),
              'metrics': results,
              'rows': {labels[k]: {s: flat(v[s]) for s in groups} for k, v in results.items()}}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if log_path:
        r = results['logistic']
        body = f"""
### Task 2 baselines: constant and logistic regression (`utcgnn/score.py`)

Last run: {run_at} · `{command or 'python -m utcgnn score-baselines'}` · outputs in `{out}`

- **Target:** team A's share of games, a / (a + b); same splits, history-only features and online protocol as Task 1.
- **Logistic regression:** binomial on games won (each set gives y=1 with weight a and y=0 with weight b), no intercept, swapped-team augmentation, on the team A − team B sums of the 5 pre-session player features. Fit on training seasons only.
- **Constant 0.5:** every set split evenly. Its total-standings ρ shows how much attendance (and set length) alone explains; its per-set ρ is undefined.
- **Data:** {len(targets['train']['share']):,} train, {len(targets['validation']['share']):,} validation, {len(targets['test']['share']):,} test sets.

{score_table(report['rows'])}

Logistic regression RMSE (games): validation {r['validation']['rmse_games']:.4f}, test {r['test']['rmse_games']:.4f}. Cold-start MAE (games): validation {cell(r['validation']['cold_start']['mae_games'])} (n={r['validation']['cold_start']['n']}), test {cell(r['test']['cold_start']['mae_games'])} (n={r['test']['cold_start']['n']}).
"""
        log_score(log_path, 'score_baselines', body, report)
    return report
