"""Historical partner lift relative to strictly pre-session doubles Elo."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import History, SLOTS, load_data
from .matchday import load_name_registry


def calibrated_scale(logits, outcomes):
    """Symmetric MAP scale, using only earlier eligible results (100 minimum)."""
    if len(outcomes) < 100:
        return 1.0
    scales = np.unique(np.append(np.linspace(0.25, 3., 111), 1.))
    z = scales[:, None] * np.asarray(logits)[None, :]
    loss = (np.logaddexp(0, z) - np.asarray(outcomes) * z).sum(axis=1)
    loss += 0.5 * ((scales - 1.) / 0.5) ** 2
    return float(scales[np.argmin(loss)])


def expected_results(players, matches, min_prior_sets=12):
    """Score a whole date before updating ratings or experience for that date."""
    if min_prior_sets < 0:
        raise ValueError('min_prior_sets must be nonnegative')
    history = History(players)
    records = []
    earlier_logits, earlier_wins = [], []
    for _, rows in matches.groupby('session', sort=True):
        rows = rows.sort_values('match_id')
        snapshot = history.snapshot(rows)
        scale = calibrated_scale(earlier_logits, earlier_wins)
        date_logits, date_wins = [], []
        for row, team, probability in zip(
            rows.itertuples(index=False), snapshot.teams.numpy(), snapshot.elo
        ):
            prior = int(history.sets[team].min())
            logit = float(np.log(probability / (1 - probability)))
            calibrated = float(1 / (1 + np.exp(-scale * logit)))
            record = {key: getattr(row, key) for key in ['match_id', 'court', *SLOTS,
                      'team_a_win', 'team_a_score', 'team_b_score']}
            record.update(date=row.session.date().isoformat(),
                          team_a_raw_elo_win=float(probability), elo_scale=scale,
                          calibration_prior_sets=len(earlier_wins),
                          team_a_expected_win=calibrated,
                          team_a_residual=float(row.team_a_win - calibrated),
                          minimum_prior_sets=prior, eligible=prior >= min_prior_sets)
            records.append(record)
            if prior >= min_prior_sets:
                date_logits.append(logit)
                date_wins.append(row.team_a_win)
        earlier_logits.extend(date_logits)
        earlier_wins.extend(date_wins)
        history.update(rows)
    if not records:
        raise ValueError('No accepted matches in the requested date range')
    return pd.DataFrame(records)


def posterior_lift(probabilities, wins, prior_sd=0.75):
    """Grid posterior of one pair's constant log-odds offset from fixed Elo.

    Convert the offset into percentage-point lift over that pair's actual
    schedule. Intervals condition on Elo and independent set outcomes.
    """
    if not np.isfinite(prior_sd) or not 0 < prior_sd <= 2:
        raise ValueError('prior_sd must be finite and in (0, 2]')
    p, y = np.asarray(probabilities, dtype=float), np.asarray(wins, dtype=float)
    if len(p) == 0 or p.shape != y.shape:
        raise ValueError('Need equally sized, nonempty probabilities and wins')
    if not ((p > 0) & (p < 1)).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected probabilities in (0, 1) and binary wins')
    theta = np.linspace(-max(8., 6 * prior_sd), max(8., 6 * prior_sd), 2401)
    logits = np.log(p / (1 - p))[None, :] + theta[:, None]
    log_weight = (y * logits - np.logaddexp(0, logits)).sum(axis=1)
    log_weight -= 0.5 * (theta / prior_sd) ** 2
    weight = np.exp(log_weight - log_weight.max())
    weight /= weight.sum()
    lift = (1 / (1 + np.exp(-logits)) - p).mean(axis=1) * 100
    lower, upper = np.interp([0.025, 0.975], np.cumsum(weight), lift)
    return {'adjusted_lift_pp': float(weight @ lift),
            'lift_lower_95_pp': float(lower), 'lift_upper_95_pp': float(upper)}


def summarize_pairs(audit, names, prior_sd=0.75, min_pair_sets=10):
    records = []
    for r in audit.loc[audit.eligible].itertuples(index=False):
        for ids, p, win, margin in [
            ((r.team_a_1, r.team_a_2), r.team_a_expected_win, r.team_a_win,
             r.team_a_score - r.team_b_score),
            ((r.team_b_1, r.team_b_2), 1 - r.team_a_expected_win, 1 - r.team_a_win,
             r.team_b_score - r.team_a_score),
        ]:
            a, b = sorted(ids)
            records.append(dict(player_1_id=a, player_2_id=b, date=r.date,
                                p=p, win=win, game_margin=margin))
    if not records:
        raise ValueError('No eligible sets; reduce --min-prior-sets or widen the date range')
    result = []
    for (a, b), group in pd.DataFrame(records).groupby(['player_1_id', 'player_2_id']):
        n = len(group)
        result.append(dict(
            player_1_id=a, player_1_name=names.loc[a, 'original_name'],
            player_2_id=b, player_2_name=names.loc[b, 'original_name'],
            sets=n, sessions=group.date.nunique(), wins=int(group.win.sum()),
            expected_wins=float(group.p.sum()), win_rate=float(group.win.mean()),
            expected_win_rate=float(group.p.mean()),
            wins_above_expected=float((group.win - group.p).sum()),
            raw_lift_pp=float((group.win - group.p).mean() * 100),
            average_game_margin=float(group.game_margin.mean()),
            first_date=group.date.min(), last_date=group.date.max(),
            meets_minimum_sets=n >= min_pair_sets,
            **posterior_lift(group.p, group.win, prior_sd),
        ))
    return pd.DataFrame(result).sort_values(
        ['adjusted_lift_pp', 'sets', 'player_1_id', 'player_2_id'],
        ascending=[False, False, True, True], ignore_index=True)


def player_view(pairs):
    result = []
    for row in pairs.to_dict('records'):
        metrics = {k: v for k, v in row.items() if not k.startswith('player_')}
        for a, b in [(1, 2), (2, 1)]:
            result.append(dict(player_id=row[f'player_{a}_id'],
                               player_name=row[f'player_{a}_name'],
                               partner_id=row[f'player_{b}_id'],
                               partner_name=row[f'player_{b}_name'], **metrics))
    return pd.DataFrame(result).sort_values(
        ['player_name', 'adjusted_lift_pp'], ascending=[True, False])


def baseline_diagnostics(audit):
    rows = audit.loc[audit.eligible]
    p, y = rows.team_a_expected_win.to_numpy(), rows.team_a_win.to_numpy()
    # Winner confidence avoids arbitrary team A/B orientation in calibration bins.
    confidence = np.maximum(p, 1 - p)
    correct = np.where(p >= 0.5, y, 1 - y)
    bins = []
    for low, high in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        mask = (confidence >= low) & (confidence < high)
        if mask.any():
            bins.append(dict(lower=low, upper=min(high, 1.), sets=int(mask.sum()),
                             expected=float(confidence[mask].mean()),
                             observed=float(correct[mask].mean())))
    return dict(sets=len(rows), brier=float(np.mean((p - y) ** 2)),
                raw_elo_brier=float(np.mean((rows.team_a_raw_elo_win.to_numpy() - y) ** 2)),
                brier_even_odds=0.25,
                log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
                log_loss_even_odds=float(np.log(2)), calibration=bins)


METHOD = """Expected wins use the repository's doubles Elo baseline: mean team
rating, initial player rating 1500, scale 400 and K=32 per player per set.
All sets on a date are scored before any ratings or calibration update for that date.
The expected win probability rescales Elo log-odds using earlier eligible sets
(minimum 100), with no intercept, preserving team-swap symmetry. The scale is
chosen on a 0.025 grid from 0.25 to 3 with a normal prior centered at 1,
standard deviation 0.5. Before 100 earlier eligible sets it is exactly 1.
Elo itself still updates using its original probabilities. The audit retains
both raw and calibrated probabilities, scales, and calibration sample counts.
Only sets where all four players have the required earlier experience enter
the lift report. Earlier and warm-up results still update Elo.
Raw lift = 100 × (wins − expected wins) / sets, in percentage points.
Adjusted lift uses a constant pair log-odds offset with a zero-centered normal
prior (default standard deviation 0.75), then averages its posterior probability
lift over that pair's observed opposition. Small samples move toward zero.
The 95% intervals are Bayesian credible intervals conditional on fixed calibrated Elo,
the prior and independent set outcomes. They omit rating uncertainty and
session dependence and are not adjusted for comparing many pairs.
This is a descriptive residual, not a causal chemistry effect: opponent-pair
effects, rating errors, selection and changing form can contribute.
Historical Elo can already absorb some partnership effects. Lift belongs to
the pair and is identical from either partner's perspective. It is not a
forecast for an unseen opponent or a separate estimate of each partner's lift.
No direct game-margin expectation is fitted; game margin is descriptive only.
"""


def write_html(path, pairs, summary):
    columns = [('player_1_name', 'Player'), ('player_2_name', 'Partner'),
               ('sets', 'Sets'), ('wins', 'Wins'), ('expected_wins', 'Expected wins'),
               ('win_rate', 'Win %'), ('expected_win_rate', 'Expected %'),
               ('raw_lift_pp', 'Raw lift (pp)'), ('adjusted_lift_pp', 'Adjusted lift (pp)'),
               ('lift_lower_95_pp', '95% lower (pp)'), ('lift_upper_95_pp', '95% upper (pp)')]
    body = []
    for row in pairs.to_dict('records'):
        cells = []
        for key, _ in columns:
            value = row[key]
            if key.endswith('_name'):
                shown = html.escape(value)
            elif key in ('sets', 'wins'):
                shown = str(value)
            elif key.endswith('win_rate'):
                shown = f'{value * 100:.1f}%'
            else:
                shown = f'{value:+.1f}' if key.endswith('_pp') else f'{value:.1f}'
            cells.append(f'<td>{shown}</td>')
        search = html.escape(' '.join(str(row[k]) for k in
                           ['player_1_name', 'player_2_name', 'player_1_id', 'player_2_id']).lower(), quote=True)
        body.append(f'<tr data-n="{row["sets"]}" data-search="{search}">' + ''.join(cells) + '</tr>')
    d = summary['baseline']
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Partner lift</title>
<style>body{font:16px system-ui;margin:32px;color:#172738;background:#f7f9fb}
h1{margin-bottom:8px}p{max-width:1000px;line-height:1.6}input,select{padding:9px;margin:6px}
.scroll{overflow:auto;background:white;border:1px solid #ccd5df;max-height:70vh}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:11px;text-align:right;
border-bottom:1px solid #e1e7ef;white-space:nowrap}th{position:sticky;top:0;background:#173c56;color:white}
td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}
tr:nth-child(even){background:#f0f5f9}summary{cursor:pointer;padding:14px 0}
pre{white-space:pre-wrap;font:15px system-ui;line-height:1.7;max-width:1000px}</style>
<h1>Partner lift</h1>'''
    document += f'<p>{html.escape(summary["first_date"])} through {html.escape(summary["last_date"])}. '
    document += (f'{summary["eligible_sets"]:,} eligible sets; {len(pairs):,} partnerships. '
                 f'All four players need {summary["min_prior_sets"]} earlier sets.</p>')
    document += '''<p><b>Read adjusted lift in percentage points.</b> For example, +5 pp
means five percentage points above Elo expectation over the opposition faced,
after shrinking uncertain estimates toward zero. Rows are ordered by adjusted lift.
An interval spanning zero leaves both positive and negative lift plausible.</p>'''
    qualified = pairs.loc[pairs.meets_minimum_sets]
    positive = int((qualified.lift_lower_95_pp > 0).sum())
    negative = int((qualified.lift_upper_95_pp < 0).sum())
    document += (f'<p>Of {len(qualified):,} partnerships with at least '
                 f'{summary["min_pair_sets"]} eligible sets together, '
                 f'{positive} have a 95% interval entirely above zero and '
                 f'{negative} entirely below zero. ')
    if positive == 0 and negative == 0:
        document += ('The current data leave substantial uncertainty about every '
                     'qualifying pair. Treat the ranking as a list to investigate.</p>')
    else:
        document += 'These intervals do not correct for comparing many pairs.</p>'
    document += (f'<p>Baseline Brier score: {d["brier"]:.3f} (even odds: 0.250). '
                 f'Log loss: {d["log_loss"]:.3f} (even odds: 0.693). Lower is better. '
                 f'Uncalibrated Elo Brier: {d["raw_elo_brier"]:.3f}. '
                 'Calibration uses earlier dates only. These checks do not establish '
                 'that individual pair estimates are calibrated.</p>')
    minimum = summary['min_pair_sets']
    document += ('<label>Player or ID <input id="search" placeholder="Search either partner"></label>'
                 '<label>Minimum sets <select id="minimum">')
    for n in sorted({1, 3, 5, 10, 20, minimum}):
        document += f'<option value="{n}"' + (' selected' if n == minimum else '') + f'>{n}</option>'
    document += '</select></label><p id="count" aria-live="polite"></p><div class="scroll"><table><thead><tr>'
    document += ''.join(f'<th scope="col">{label}</th>' for _, label in columns)
    document += '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'
    document += '<details><summary>Method and limits</summary><pre>' + html.escape(METHOD)
    document += f'\nPrior standard deviation used: {summary["prior_sd"]}.\n'
    document += html.escape('Sources: ' + json.dumps(summary['sources'])) + '</pre></details>'
    document += '''<script>
const search=document.querySelector('#search'), minimum=document.querySelector('#minimum');
function filter(){let n=0; for(const row of document.querySelectorAll('tbody tr')){
row.hidden=Number(row.dataset.n)<Number(minimum.value)||!row.dataset.search.includes(search.value.toLowerCase().trim());
if(!row.hidden)n++;}document.querySelector('#count').textContent=n+' partnerships shown';}
search.addEventListener('input',filter);minimum.addEventListener('change',filter);filter();
</script></html>'''
    path.write_text(document, encoding='utf-8')


def run(args):
    if args.min_pair_sets < 1:
        raise ValueError('min_pair_sets must be at least 1')
    players, matches, audit_info = load_data(args.players, args.matches)
    names = load_name_registry(args.players, players.player_id)
    before = pd.Timestamp(args.before) if args.before else None
    since = pd.Timestamp(args.since) if args.since else None
    if before is not None:
        matches = matches.loc[matches.session < before]
    audit = expected_results(players, matches, args.min_prior_sets)
    if since is not None:
        audit = audit.loc[pd.to_datetime(audit.date) >= since].copy()
    if audit.empty:
        raise ValueError('No matches in the reporting window')
    pairs = summarize_pairs(audit, names, args.prior_sd, args.min_pair_sets)
    summary = dict(sources={'players': str(args.players), 'matches': str(args.matches)},
                   input_audit=audit_info, first_date=audit.date.min(), last_date=audit.date.max(),
                   since=args.since, before=args.before, min_prior_sets=args.min_prior_sets,
                   min_pair_sets=args.min_pair_sets, prior_sd=args.prior_sd,
                   report_sets=len(audit), eligible_sets=int(audit.eligible.sum()),
                   warmup_excluded_sets=int((~audit.eligible).sum()), partnerships=len(pairs),
                   partnerships_meeting_minimum=int(pairs.meets_minimum_sets.sum()),
                   baseline=baseline_diagnostics(audit), method=METHOD)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output / 'partner_lift.csv', index=False)
    player_view(pairs).to_csv(output / 'player_partners.csv', index=False)
    audit.to_csv(output / 'match_expectations.csv', index=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_html(output / 'report.html', pairs, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--players', default='data/players.csv')
    parser.add_argument('--matches', default='data/morphed_matches.csv')
    parser.add_argument('--output', default='artifacts/partner_lift')
    parser.add_argument('--min-prior-sets', type=int, default=12)
    parser.add_argument('--min-pair-sets', type=int, default=10)
    parser.add_argument('--prior-sd', type=float, default=0.75)
    parser.add_argument('--since', help='Inclusive report start date; earlier data still initialize Elo')
    parser.add_argument('--before', help='Exclusive cutoff; later results never enter Elo or the report')
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({k: summary[k] for k in ['eligible_sets', 'partnerships',
                     'partnerships_meeting_minimum', 'baseline']}, indent=2))
    print(f'Report: {Path(args.output) / "report.html"}')


if __name__ == '__main__':
    main()
