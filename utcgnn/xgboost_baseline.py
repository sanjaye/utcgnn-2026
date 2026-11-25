"""XGBoost baseline on the GNN's pre-session player features.

Same inputs, season splits and online (prequential) protocol as `train`: each set is
described only by features computed from strictly earlier session dates.

Per set, each team is summarized by the sum, min and max of its two players' five
features (order-invariant within a team), plus the team A minus team B difference of
the sums: 35 features in total. Training rows are duplicated with teams swapped and the
label flipped, and predictions average both orientations, so P(A wins) = 1 - P(B wins).

Hyperparameters are tuned by randomized search with k-fold cross-validation on the
training seasons only. Folds are grouped by session date, so sets from one date (and
their swapped copies) never sit on both sides of a fold. The validation and test
seasons are never used for tuning or fitting.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # noqa: F401  Loaded before xgboost so its bundled OpenMP runtime serves xgboost on macOS.
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GroupKFold, ParameterSampler, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor

from .data import FEATURES
from .experiment import evaluate, save_predictions, split_sessions
from .reporting import log_markdown, metrics_table, reference_models
from .score import (evaluate_score, flat, log_score, mae_games, report_base, save_score_outputs, score_table,
                    score_targets)

AGGREGATES = ('sum', 'min', 'max')
FEATURE_NAMES = ([f'team_a_{agg}_{f}' for agg in AGGREGATES for f in FEATURES]
                 + [f'team_b_{agg}_{f}' for agg in AGGREGATES for f in FEATURES]
                 + [f'diff_sum_{f}' for f in FEATURES])
SEARCH_SPACE = {
    'n_estimators': randint(50, 601),
    'max_depth': randint(2, 7),
    'learning_rate': loguniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.5, 0.5),
    'min_child_weight': loguniform(1, 50),
    'reg_lambda': loguniform(0.1, 20),
    'gamma': uniform(0, 2),
}


def team_features(snapshot):
    """(sets, 15) arrays for team A and team B from raw pre-session player features."""
    x = snapshot.x.numpy()[snapshot.teams.numpy()]  # (sets, 4 players, 5 features)

    def summarize(team):
        return np.concatenate([team.sum(1), team.min(1), team.max(1)], axis=1)

    return summarize(x[:, :2]), summarize(x[:, 2:])


def design(team_a, team_b):
    n = len(FEATURES)
    return np.concatenate([team_a, team_b, team_a[:, :n] - team_b[:, :n]], axis=1)


def symmetric_predict(model, team_a, team_b):
    """P(A wins) for a classifier, or A's game share for a reg:logistic regressor."""
    if hasattr(model, 'predict_proba'):
        forward = model.predict_proba(design(team_a, team_b))[:, 1]
        swapped = model.predict_proba(design(team_b, team_a))[:, 1]
    else:
        forward, swapped = model.predict(design(team_a, team_b)), model.predict(design(team_b, team_a))
    return (forward + 1 - swapped) / 2


def _stack(sessions):
    pairs = [team_features(s) for s in sessions]
    return np.concatenate([a for a, _ in pairs]), np.concatenate([b for _, b in pairs])


def run_xgboost(players, matches, audit, output, n_iter=60, cv_folds=3, seed=42,
                log_path='reports/experiments.md', reference_path='reports/metrics.json', command=None):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    splits, groups, _ = split_sessions(players, matches)
    teams = {split: _stack(g) for split, g in groups.items()}
    labels = {split: np.concatenate([s.y.numpy() for s in g]) for split, g in groups.items()}
    cold = {split: np.concatenate([s.cold for s in g]) for split, g in groups.items()}

    a, b = teams['train']
    y = labels['train']
    dates = np.array([str(s.session.date()) for s in groups['train'] for _ in s.ids])
    search = RandomizedSearchCV(
        XGBClassifier(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                      n_jobs=1, random_state=seed),
        SEARCH_SPACE, n_iter=n_iter, scoring='neg_log_loss', cv=GroupKFold(n_splits=cv_folds),
        n_jobs=-1, random_state=seed, refit=True, return_train_score=True)
    search.fit(np.concatenate([design(a, b), design(b, a)]), np.concatenate([y, 1 - y]),
               groups=np.concatenate([dates, dates]))
    model = search.best_estimator_

    preds = {split: {'xgboost': symmetric_predict(model, *teams[split])} for split in groups}
    results = {split: evaluate(labels[split], preds[split]['xgboost'], cold[split]) for split in groups}
    best_params = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                   for k, v in search.best_params_.items()}
    cv_log_loss = float(-search.best_score_)
    cv_std = float(search.cv_results_['std_test_score'][search.best_index_])

    importance = pd.DataFrame({'feature': FEATURE_NAMES, 'gain': model.feature_importances_})
    importance = importance.sort_values('gain', ascending=False)
    importance.to_csv(out / 'feature_importance.csv', index=False)
    columns = ['rank_test_score', 'mean_test_score', 'std_test_score', 'mean_train_score',
               *[f'param_{k}' for k in SEARCH_SPACE]]
    pd.DataFrame(search.cv_results_)[columns].sort_values('rank_test_score').to_csv(
        out / 'cv_results.csv', index=False)
    save_predictions(out, groups, preds)
    run_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    report = {'model': 'xgboost', 'run_at': run_at, 'seed': seed, 'audit': audit, 'splits': splits,
              'features': FEATURE_NAMES,
              'tuning': {'method': 'RandomizedSearchCV', 'n_iter': n_iter,
                         'cv': f'GroupKFold({cv_folds}) by session date',
                         'scoring': 'log loss on swap-augmented training rows',
                         'best_cv_log_loss': cv_log_loss, 'cv_log_loss_std': cv_std, 'best_params': best_params},
              'metrics': results}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    if log_path:
        references, note = reference_models(reference_path, splits, audit)
        params = ', '.join(f'`{k}={v:.4g}`' if isinstance(v, float) else f'`{k}={v}`'
                           for k, v in best_params.items())
        top = ', '.join(f'`{f}`' for f in importance.feature.head(8))
        val_cold = results['validation']['cold_start']
        rounded = {k: '—' if val_cold[k] is None else f'{val_cold[k]:.4f}' for k in ('accuracy', 'log_loss')}
        body = f"""
### XGBoost baseline (`utcgnn/xgboost_baseline.py`)

Last run: {run_at} · `{command or 'python -m utcgnn xgboost'}` · outputs in `{out}`

- **Data:** {audit['used_sets']:,} sets ({len(labels['train']):,} train, {len(labels['validation']):,} validation, {len(labels['test']):,} test), same splits and online protocol as `train`.
- **Features:** 35 per set from the 5 pre-session player features: per-team sum, min and max, plus the team A − team B difference of sums. Teams are swapped in training and predictions average both orientations.
- **Tuning:** randomized search, {n_iter} candidates, {cv_folds}-fold CV on training seasons only, grouped by session date. Best CV log loss {cv_log_loss:.4f} ± {cv_std:.4f}.
- **Selected hyperparameters:** {params}
- **Top features by gain:** {top}

{metrics_table({'**XGBoost**': results, **references})}

Train: accuracy {results['train']['accuracy']:.4f}, ROC-AUC {results['train']['roc_auc']:.4f}, log loss {results['train']['log_loss']:.4f}. Validation cold-start sets (n={val_cold['n']}): accuracy {rounded['accuracy']}, log loss {rounded['log_loss']}. The test season has {results['test']['cold_start']['n']} cold-start sets.

{note} All results are from a single seed and a single test season.
"""
        log_markdown(log_path, 'xgboost', body)
    return report


def _fit_share(params, a, b, share, games, seed):
    """reg:logistic on the share, weighted by games (the binomial likelihood), swap-augmented."""
    model = XGBRegressor(objective='reg:logistic', tree_method='hist', n_jobs=-1, random_state=seed, **params)
    model.fit(np.concatenate([design(a, b), design(b, a)]), np.concatenate([share, 1 - share]),
              sample_weight=np.concatenate([games, games]))
    return model


def run_xgboost_score(players, matches, audit, output, n_iter=60, cv_folds=3, seed=42,
                      log_path='reports/experiments.md', command=None):
    """Task 2: XGBoost predicting team A's game share; tuned by MAE in games with grouped CV on train."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    splits, groups, _ = split_sessions(players, matches)
    targets = score_targets(matches, groups)
    teams = {split: _stack(g) for split, g in groups.items()}
    n_players = len(players)

    # Randomized search done by hand, so each fold is scored by MAE in games on its
    # original (unswapped) sets with symmetric predictions, exactly as reported.
    t = targets['train']
    a, b = teams['train']
    dates = np.array(t['session'])
    folds = list(GroupKFold(n_splits=cv_folds).split(a, groups=dates))
    candidates = list(ParameterSampler(SEARCH_SPACE, n_iter, random_state=seed))
    rows = []
    for i, params in enumerate(candidates):
        params = {k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in params.items()}
        scores = []
        for fit_idx, held_idx in folds:
            sub = {k: t[k][fit_idx] for k in ('share', 'games')}
            model = _fit_share(params, a[fit_idx], b[fit_idx], sub['share'], sub['games'], seed)
            held = {k: t[k][held_idx] for k in ('share', 'games')}
            scores.append(mae_games(held, symmetric_predict(model, a[held_idx], b[held_idx])))
        rows.append({'candidate': i, 'mean_mae_games': float(np.mean(scores)),
                     'std_mae_games': float(np.std(scores)), **params})
    cv = pd.DataFrame(rows).sort_values('mean_mae_games')
    cv.to_csv(out / 'cv_results.csv', index=False)
    best = cv.iloc[0]
    best_params = {k: (int(best[k]) if k in ('n_estimators', 'max_depth') else float(best[k])) for k in SEARCH_SPACE}
    model = _fit_share(best_params, a, b, t['share'], t['games'], seed)

    preds = {split: {'xgboost': symmetric_predict(model, *teams[split])} for split in groups}
    results = {split: evaluate_score(targets[split], preds[split]['xgboost'], n_players) for split in groups}
    importance = pd.DataFrame({'feature': FEATURE_NAMES, 'gain': model.feature_importances_})
    importance = importance.sort_values('gain', ascending=False)
    importance.to_csv(out / 'feature_importance.csv', index=False)
    save_score_outputs(out, targets, preds, players.player_id.tolist())
    run_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    report = {**report_base(splits, audit, targets, n_players), 'model': 'xgboost_score', 'run_at': run_at,
              'seed': seed, 'features': FEATURE_NAMES,
              'tuning': {'method': 'ParameterSampler + GroupKFold by session date', 'n_iter': n_iter, 'cv_folds': cv_folds,
                         'scoring': 'MAE in games on held-out sets, symmetric predictions',
                         'best_cv_mae_games': float(best.mean_mae_games), 'cv_mae_games_std': float(best.std_mae_games),
                         'best_params': best_params},
              'metrics': results, 'rows': {'XGBoost': {s: flat(results[s]) for s in groups}}}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    if log_path:
        params = ', '.join(f'`{k}={v:.4g}`' if isinstance(v, float) else f'`{k}={v}`' for k, v in best_params.items())
        top = ', '.join(f'`{f}`' for f in importance.feature.head(8))
        r = results
        body = f"""
### Task 2 XGBoost (`utcgnn/xgboost_baseline.py`)

Last run: {run_at} · `{command or 'python -m utcgnn xgboost --task score'}` · outputs in `{out}`

- **Target:** team A's share of games. `XGBRegressor(objective='reg:logistic')` on the share, weighted by games in the set; same 35 team features as the Task 1 XGBoost. Teams are swapped in training and predictions average both orientations, so share(B) = 1 − share(A).
- **Tuning:** {n_iter} random candidates from the Task 1 search space, {cv_folds}-fold CV on training seasons only, grouped by session date, scored by MAE in games. Best CV MAE {best.mean_mae_games:.4f} ± {best.std_mae_games:.4f} games.
- **Selected hyperparameters:** {params}
- **Top features by gain:** {top}

{score_table(report['rows'])}

Train MAE {r['train']['mae_games']:.4f} games. RMSE (games): validation {r['validation']['rmse_games']:.4f}, test {r['test']['rmse_games']:.4f}. Single seed, single test season.
"""
        log_score(log_path, 'score_xgboost', body, report)
    return report
