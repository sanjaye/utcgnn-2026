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
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBClassifier

from .data import FEATURES
from .experiment import evaluate, save_predictions, split_sessions
from .reporting import log_markdown, metrics_table, reference_models

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
    forward = model.predict_proba(design(team_a, team_b))[:, 1]
    swapped = model.predict_proba(design(team_b, team_a))[:, 1]
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
