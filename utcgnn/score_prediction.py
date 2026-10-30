"""Predict exact doubles-set scores in a separate, auditable pipeline."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import History, load_data, snapshots


SCORELINES = (
    [(6, losing_points) for losing_points in range(5)]
    + [(7, 5), (7, 6)]
    + [(losing_points, 6) for losing_points in range(5)]
    + [(5, 7), (6, 7)]
)
SCORELINE_INDEX = {scoreline: index for index, scoreline in enumerate(SCORELINES)}
FLIPPED_INDEX = [SCORELINE_INDEX[(b, a)] for a, b in SCORELINES]
PLAYER_ID_COLUMNS = [
    f'{team}_player_{position}_id'
    for team in ('team_a', 'team_b')
    for position in (1, 2)
]
OUTPUT_COLUMNS = [
    'score_predicted_winner',
    'team_a_predicted_points',
    'team_b_predicted_points',
    'predicted_winner_points',
    'predicted_loser_points',
    'scoreline_confidence',
    'expected_team_a_points',
    'expected_team_b_points',
]
PLAYER_TOTAL_COLUMNS = [
    'player_id',
    'pseudonym',
    'actual_name',
    'court',
    'matches_played',
    'predicted_wins',
    'predicted_losses',
    'predicted_points_total',
    'predicted_points_per_match',
]


def _feature_difference(features, team_ids, lookup):
    indices = [lookup[player_id] for player_id in team_ids]
    return features[indices[:2]].sum(axis=0) - features[indices[2:]].sum(axis=0)


def _scoreline_label(team_a_score, team_b_score):
    try:
        return SCORELINE_INDEX[(int(team_a_score), int(team_b_score))]
    except KeyError as error:
        raise ValueError(
            f'Unsupported historical scoreline {team_a_score}-{team_b_score}; '
            'expected a 6-point winner with 0-4, or a 7-point winner with 5-6'
        ) from error


def training_examples(players, matches, prediction_date):
    history_snapshots, _ = snapshots(players, matches)
    rows_by_session = {
        session: rows.set_index('match_id')
        for session, rows in matches.groupby('session', sort=True)
        if session < prediction_date
    }
    x_values, labels = [], []
    for snapshot in history_snapshots:
        if snapshot.session >= prediction_date:
            continue
        rows = rows_by_session[snapshot.session]
        feature_matrix = snapshot.x.numpy()
        for match_id, team in zip(snapshot.ids, snapshot.teams.numpy()):
            row = rows.loc[match_id]
            team_a_score = int(row.team_a_score)
            team_b_score = int(row.team_b_score)
            x_values.append(
                feature_matrix[team[:2]].sum(axis=0)
                - feature_matrix[team[2:]].sum(axis=0)
            )
            labels.append(_scoreline_label(team_a_score, team_b_score))
    if not labels:
        raise ValueError('No historical matches are available before the prediction date')
    return np.asarray(x_values), np.asarray(labels, dtype=np.int64)


def fit_score_model(x_values, labels):
    reverse_labels = np.asarray([FLIPPED_INDEX[label] for label in labels])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000),
    )
    model.fit(
        np.concatenate([x_values, -x_values]),
        np.concatenate([labels, reverse_labels]),
    )
    return model


def predict_scoreline_probabilities(model, x_values):
    class_model = model.named_steps['logisticregression']
    classes = class_model.classes_
    forward_observed = model.predict_proba(x_values)
    reverse_observed = model.predict_proba(-x_values)
    forward = np.zeros((len(x_values), len(SCORELINES)), dtype=float)
    reverse = np.zeros_like(forward)
    forward[:, classes] = forward_observed
    reverse[:, classes] = reverse_observed
    probabilities = (forward + reverse[:, FLIPPED_INDEX]) / 2
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _target_date(predictions, requested_date):
    if 'date' not in predictions:
        if requested_date is None:
            raise ValueError('Predictions need a date column or --date must be supplied')
        return pd.Timestamp(requested_date).normalize()
    dates = pd.to_datetime(predictions.date, errors='raise').dt.normalize()
    if dates.isna().any() or dates.nunique() != 1:
        raise ValueError('The input prediction file must contain one nonempty match date')
    inferred = dates.iloc[0]
    if requested_date is not None and pd.Timestamp(requested_date).normalize() != inferred:
        raise ValueError('--date must match the date in the input prediction file')
    return inferred


def predict_match_scores(predictions, names, history, model):
    missing = set(PLAYER_ID_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f'Prediction file is missing player ID columns: {sorted(missing)}')
    if 'predicted_winner' not in predictions:
        raise ValueError('Prediction file must include the match-level predicted_winner')
    if not predictions.predicted_winner.isin(['team_a', 'team_b']).all():
        raise ValueError('predicted_winner must be team_a or team_b for every match')
    if predictions[PLAYER_ID_COLUMNS].eq('').any().any():
        raise ValueError('Prediction file contains an empty player ID')
    all_ids = set(predictions[PLAYER_ID_COLUMNS].to_numpy().ravel())
    unknown = all_ids - set(history.lookup)
    if unknown:
        raise ValueError(f'Predictions contain unregistered player IDs: {sorted(unknown)}')
    missing_names = all_ids - set(names.index)
    if missing_names:
        raise ValueError(f'Name registry is missing player IDs: {sorted(missing_names)}')
    if set(OUTPUT_COLUMNS) & set(predictions.columns):
        raise ValueError('Input already contains score-prediction output columns')

    features = history.features().numpy()
    x_values = np.asarray([
        _feature_difference(features, row, history.lookup)
        for row in predictions[PLAYER_ID_COLUMNS].itertuples(index=False, name=None)
    ])
    probabilities = predict_scoreline_probabilities(model, x_values)
    predicted_winners = predictions.predicted_winner.to_numpy()
    scoreline_winners = np.asarray([
        'team_a' if team_a > team_b else 'team_b'
        for team_a, team_b in SCORELINES
    ])
    compatible = scoreline_winners[None, :] == predicted_winners[:, None]
    probabilities = np.where(compatible, probabilities, 0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    best_indices = probabilities.argmax(axis=1)
    result = predictions.copy()
    result['score_predicted_winner'] = predicted_winners
    result['team_a_predicted_points'] = [SCORELINES[index][0] for index in best_indices]
    result['team_b_predicted_points'] = [SCORELINES[index][1] for index in best_indices]
    result['predicted_winner_points'] = [
        max(SCORELINES[index]) for index in best_indices
    ]
    result['predicted_loser_points'] = [
        min(SCORELINES[index]) for index in best_indices
    ]
    result['scoreline_confidence'] = probabilities.max(axis=1)
    result['expected_team_a_points'] = probabilities @ np.asarray(
        [scoreline[0] for scoreline in SCORELINES]
    )
    result['expected_team_b_points'] = probabilities @ np.asarray(
        [scoreline[1] for scoreline in SCORELINES]
    )
    return result


def summarize_player_points(predictions):
    rows = []
    for row in predictions.itertuples(index=False):
        winner = row.score_predicted_winner
        for team in ('team_a', 'team_b'):
            team_points = getattr(row, f'{team}_predicted_points')
            won = team == winner
            for position in (1, 2):
                rows.append({
                    'player_id': getattr(row, f'{team}_player_{position}_id'),
                    'pseudonym': getattr(row, f'{team}_player_{position}_pseudonym'),
                    'actual_name': getattr(row, f'{team}_player_{position}_actual_name'),
                    'court': row.court,
                    'predicted_points': team_points,
                    'predicted_win': int(won),
                })
    appearances = pd.DataFrame(rows)
    if appearances.empty:
        raise ValueError('Cannot summarize player points from an empty prediction file')
    if appearances.groupby('player_id').court.nunique().gt(1).any():
        raise ValueError('A player appears on multiple courts in the match-day predictions')
    totals = appearances.groupby('player_id', sort=True).agg(
        pseudonym=('pseudonym', 'first'),
        actual_name=('actual_name', 'first'),
        court=('court', 'first'),
        matches_played=('predicted_points', 'size'),
        predicted_wins=('predicted_win', 'sum'),
        predicted_points_total=('predicted_points', 'sum'),
    ).reset_index()
    totals['predicted_losses'] = totals.matches_played - totals.predicted_wins
    totals['predicted_points_per_match'] = (
        totals.predicted_points_total / totals.matches_played
    )
    totals = totals[PLAYER_TOTAL_COLUMNS]
    invalid = totals.matches_played.ne(3)
    if invalid.any():
        raise ValueError(
            'Every scheduled player must appear in exactly three matches; '
            f'invalid players: {totals.loc[invalid, "player_id"].tolist()}'
        )
    return totals.sort_values(
        ['predicted_points_total', 'predicted_wins', 'player_id'],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def write_score_workbook(predictions, workbook_path):
    totals = summarize_player_points(predictions)
    workbook_path = Path(workbook_path)
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(workbook_path, engine='openpyxl') as writer:
        predictions.to_excel(writer, sheet_name='Match predictions', index=False)
        totals.to_excel(writer, sheet_name='Player totals', index=False)
    return totals


def run_score_prediction(
    prediction_path,
    output_path,
    report_path,
    workbook_path='artifacts/matchday_score_predictions.xlsx',
    players_path='data/morphed_players.xlsx',
    matches_path='data/morphed_matches.csv',
    names_path='data/players.csv',
    include_flagged=False,
    requested_date=None,
):
    predictions = pd.read_csv(prediction_path, dtype=str, keep_default_na=False)
    date = _target_date(predictions, requested_date)
    players, matches, audit = load_data(
        players_path, matches_path, include_flagged=include_flagged
    )
    names = pd.read_csv(names_path, dtype={'player_id': str}, keep_default_na=False)
    required_names = {'player_id', 'original_name', 'pseudonym'}
    if required_names - set(names.columns):
        raise ValueError(
            f'Name registry is missing columns: {sorted(required_names - set(names.columns))}'
        )
    if names.player_id.eq('').any() or names.player_id.duplicated().any():
        raise ValueError('Name registry must contain unique, nonempty player IDs')
    if names[['original_name', 'pseudonym']].eq('').any().any():
        raise ValueError('Name registry contains an empty actual name or pseudonym')
    names = names.set_index('player_id')

    x_train, y_train = training_examples(players, matches, date)
    model = fit_score_model(x_train, y_train)
    history = History(players)
    for _, rows in matches[matches.session < date].groupby('session', sort=True):
        history.update(rows)
    scored = predict_match_scores(predictions, names, history, model)

    output_path = Path(output_path)
    report_path = Path(report_path)
    workbook_path = Path(workbook_path)
    if output_path.resolve() == Path(prediction_path).resolve():
        raise ValueError('Score output must be separate from the input match predictions')
    output_paths = {
        Path(prediction_path).resolve(),
        output_path.resolve(),
        report_path.resolve(),
        workbook_path.resolve(),
    }
    if len(output_paths) != 4:
        raise ValueError('Input, score CSV, report JSON, and workbook need separate paths')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    player_totals = write_score_workbook(scored, workbook_path)
    report = {
        'prediction_date': str(date.date()),
        'input_predictions': str(prediction_path),
        'output_predictions': str(output_path),
        'output_workbook': str(workbook_path),
        'training_matches': len(y_train),
        'historical_data_policy': audit['policy'],
        'scoreline_classes': {
            f'{a}-{b}': int(np.count_nonzero(y_train == index))
            for index, (a, b) in enumerate(SCORELINES)
        },
        'prediction_count': len(scored),
        'player_totals_count': len(player_totals),
        'model': (
            'Symmetry-averaged multinomial logistic regression on pre-match player feature '
            'differences, conditioned on the match-level predicted winner'
        ),
        'allowed_scorelines': [f'{a}-{b}' for a, b in SCORELINES],
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Predict doubles-set winning and losing points without changing match predictions'
    )
    parser.add_argument('--predictions', default='artifacts/matchday_predictions.csv')
    parser.add_argument('--output', default='artifacts/matchday_score_predictions.csv')
    parser.add_argument('--report', default='artifacts/matchday_score_prediction_report.json')
    parser.add_argument('--workbook', default='artifacts/matchday_score_predictions.xlsx')
    parser.add_argument('--players', default='data/morphed_players.xlsx')
    parser.add_argument('--matches', default='data/morphed_matches.csv')
    parser.add_argument('--names', default='data/players.csv')
    parser.add_argument('--date', help='Optional date override; must match input predictions')
    parser.add_argument('--include-flagged', action='store_true')
    args = parser.parse_args()
    report = run_score_prediction(
        args.predictions,
        args.output,
        args.report,
        workbook_path=args.workbook,
        players_path=args.players,
        matches_path=args.matches,
        names_path=args.names,
        include_flagged=args.include_flagged,
        requested_date=args.date,
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
