"""Prepare and score all doubles pairings for a randomized match day."""

import pandas as pd

from .inference import predict_teams

MATCH_COLUMNS = ['court', 'pairing', 'team_a_1', 'team_a_2', 'team_b_1', 'team_b_2']
PAIRINGS = [
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
]


def _partition(row):
    return tuple(sorted((
        tuple(sorted((row.team_a_1, row.team_a_2))),
        tuple(sorted((row.team_b_1, row.team_b_2))),
    )))


def load_matchday_groups(path, registered_ids):
    groups = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(MATCH_COLUMNS) - set(groups.columns)
    if missing:
        raise ValueError(f'Match-day CSV is missing columns: {sorted(missing)}')
    groups = groups[MATCH_COLUMNS]
    if len(groups) != 24:
        raise ValueError(f'Match-day CSV must contain exactly 24 matchups; found {len(groups)}')
    if groups.isin(['']).any().any():
        raise ValueError('Court labels, pairing numbers, and player IDs must not be empty')
    if not groups.pairing.isin(['1', '2', '3']).all():
        raise ValueError('Pairing numbers must be 1, 2, or 3')
    if groups.duplicated(['court', 'pairing']).any():
        raise ValueError('Each court must have exactly one row for each pairing number')
    court_counts = groups.groupby('court').size()
    if len(court_counts) != 8 or not court_counts.eq(3).all():
        raise ValueError('Match-day CSV must have 8 courts with exactly 3 matches per court')
    player_columns = ['team_a_1', 'team_a_2', 'team_b_1', 'team_b_2']
    if (groups[player_columns].nunique(axis=1) != 4).any():
        raise ValueError('Each matchup must contain four distinct players')
    all_ids = set()
    for court, court_rows in groups.groupby('court', sort=False):
        court_players = set(court_rows[player_columns].to_numpy().ravel())
        if len(court_players) != 4:
            raise ValueError(f'{court} must use the same four players for all three matches')
        all_ids.update(court_players)
        expected = set()
        ordered_ids = sorted(court_players)
        for team_a_slots, team_b_slots in PAIRINGS:
            expected.add(tuple(sorted((
                tuple(sorted(ordered_ids[i] for i in team_a_slots)),
                tuple(sorted(ordered_ids[i] for i in team_b_slots)),
            ))))
        observed = {_partition(row) for row in court_rows.itertuples(index=False)}
        if observed != expected:
            raise ValueError(f'{court} must include all three unique doubles pairings')
    if len(all_ids) != 32:
        raise ValueError('The eight court groups must contain 32 distinct players')
    unknown = set(all_ids) - set(registered_ids)
    if unknown:
        raise ValueError(f'Match-day CSV contains unregistered player IDs: {sorted(unknown)}')
    return groups


def load_name_registry(path, registered_ids):
    names = pd.read_csv(path, dtype={'player_id': str}, keep_default_na=False)
    required = {'player_id', 'original_name', 'pseudonym'}
    missing = required - set(names.columns)
    if missing:
        raise ValueError(f'Name registry is missing columns: {sorted(missing)}')
    names = names[['player_id', 'original_name', 'pseudonym']]
    if names.player_id.eq('').any() or names.player_id.duplicated().any():
        raise ValueError('Name registry must contain unique, nonempty player IDs')
    if names[['original_name', 'pseudonym']].eq('').any().any():
        raise ValueError('Name registry contains an empty actual name or pseudonym')
    lookup = names.set_index('player_id')
    missing_ids = set(registered_ids) - set(lookup.index)
    if missing_ids:
        raise ValueError(f'Name registry is missing registered player IDs: {sorted(missing_ids)}')
    return lookup


def predict_matchday(groups, names, history, model, checkpoint, date):
    pairings = []
    records = []
    for row in groups.itertuples(index=False):
        team_a = [row.team_a_1, row.team_a_2]
        team_b = [row.team_b_1, row.team_b_2]
        matchup = team_a + team_b
        pairings.append(matchup)
        record = {
            'date': str(pd.Timestamp(date).date()),
            'court': row.court,
            'pairing': int(row.pairing),
            'team_a_win_probability': None,
            'team_b_win_probability': None,
            'predicted_winner': None,
            'predicted_winner_players': None,
            'cold_start': any(history.sets[history.lookup[player_id]] == 0
                              for player_id in matchup),
        }
        for team_name, team in [('team_a', team_a), ('team_b', team_b)]:
            for position, player_id in enumerate(team, start=1):
                name = names.loc[player_id]
                record[f'{team_name}_player_{position}_id'] = player_id
                record[f'{team_name}_player_{position}_pseudonym'] = name.pseudonym
                record[f'{team_name}_player_{position}_actual_name'] = name.original_name
        records.append(record)

    probabilities = predict_teams(history, model, checkpoint, pairings)

    for record, team_a_probability in zip(records, probabilities):
        team_b_probability = 1 - team_a_probability
        winner = 'team_a' if team_a_probability >= 0.5 else 'team_b'
        record['team_a_win_probability'] = team_a_probability
        record['team_b_win_probability'] = team_b_probability
        record['predicted_winner'] = winner
        record['predicted_winner_players'] = ' + '.join(
            f"{record[f'{winner}_player_{position}_pseudonym']} "
            f"({record[f'{winner}_player_{position}_actual_name']})"
            for position in (1, 2)
        )
    return pd.DataFrame(records)
