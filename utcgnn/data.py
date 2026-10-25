"""Validate source records and construct strictly historical graph snapshots."""
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SLOTS = ['team_a_1', 'team_a_2', 'team_b_1', 'team_b_2']
FEATURES = ['log_sets', 'smoothed_win_rate', 'game_share', 'elo_scaled', 'log_games_per_set']


def load_data(players_path, matches_path, include_flagged=False):
    players = pd.read_excel(players_path, dtype={'player_id': str})
    matches = pd.read_csv(matches_path, encoding='utf-8-sig')
    for frame, columns in [(players, ['player_id', 'pseudonym']),
                           (matches, ['match_id', 'season', 'timestamp', *SLOTS,
                                      'team_a_win', 'team_a_score', 'team_b_score'])]:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        if frame[columns].isna().any().any():
            raise ValueError('Required fields contain missing values')
    if players.player_id.duplicated().any() or matches.match_id.duplicated().any():
        raise ValueError('Duplicate player_id or match_id')
    known = set(players.player_id)
    if not set(matches[SLOTS].to_numpy().ravel()) <= known:
        raise ValueError('Match contains a player absent from the registry')
    if (matches[SLOTS].nunique(axis=1) != 4).any():
        raise ValueError('A doubles set must contain four distinct players')
    if not matches.team_a_win.isin([0, 1]).all():
        raise ValueError('team_a_win must be binary')
    for col in ['team_a_score', 'team_b_score']:
        matches[col] = pd.to_numeric(matches[col], errors='raise')
        if not np.isfinite(matches[col]).all() or (matches[col] < 0).any() or (matches[col] % 1 != 0).any():
            raise ValueError('Scores must be finite nonnegative integers')
    if ((matches.team_a_score == matches.team_b_score) |
        ((matches.team_a_score > matches.team_b_score).astype(int) != matches.team_a_win)).any():
        raise ValueError('Score and outcome disagree (or a tied set has a winner)')
    # Local calendar day is the only trustworthy ordering: no within-session leakage.
    matches['session'] = pd.to_datetime(matches.timestamp.str.slice(0, 10), errors='raise')
    pd.to_datetime(matches.timestamp, utc=True, errors='raise')
    if (matches.groupby('session').season.nunique() > 1).any():
        raise ValueError('One session belongs to multiple seasons')
    flags = matches.get('validation_flags', pd.Series('', index=matches.index)).fillna('').astype(str)
    flagged = flags.str.strip().ne('')
    audit = {'players': len(players), 'raw_sets': len(matches), 'flagged_sets': int(flagged.sum()),
             'flag_counts': flags[flagged].value_counts().to_dict(),
             'excluded_sets': 0 if include_flagged else int(flagged.sum()),
             'policy': 'include_flagged' if include_flagged else 'exclude_all_flagged',
             'time_resolution': 'session_date; all sets on a date predicted together'}
    matches = matches.copy() if include_flagged else matches.loc[~flagged].copy()
    matches = matches.sort_values(['session', 'match_id']).reset_index(drop=True)
    audit['used_sets'] = len(matches)
    audit['seasons'] = matches.groupby('season', sort=False).size().to_dict()
    return players.sort_values('player_id').reset_index(drop=True), matches, audit


def season_splits(matches):
    seasons = matches.groupby('season').session.agg(['min', 'max']).sort_values('min')
    if len(seasons) < 3:
        raise ValueError('At least three seasons are required for train/validation/test')
    if any(seasons['max'].iloc[i] >= seasons['min'].iloc[i + 1] for i in range(len(seasons)-1)):
        raise ValueError('Season date ranges overlap')
    return seasons.index.tolist()


@dataclass
class Snapshot:
    session: pd.Timestamp
    season: str
    x: torch.Tensor
    relations: torch.Tensor
    teams: torch.Tensor
    y: torch.Tensor
    elo: np.ndarray
    points: np.ndarray
    cold: np.ndarray
    ids: list


class History:
    def __init__(self, players):
        self.ids = players.player_id.tolist()
        self.lookup = {p: i for i, p in enumerate(self.ids)}
        n = len(players)
        self.sets = np.zeros(n)
        self.wins = np.zeros(n)
        self.games_for = np.zeros(n)
        self.games_against = np.zeros(n)
        self.elo = np.full(n, 1500.)
        self.counts = np.zeros((2, n, n), dtype=np.float32)

    def features(self):
        return torch.tensor(np.column_stack([
            np.log1p(self.sets), (self.wins+1)/(self.sets+2),
            (self.games_for+1)/(self.games_for+self.games_against+2),
            (self.elo-1500)/400, np.log1p(self.games_for/(self.sets+1))]), dtype=torch.float32)

    def relations(self):
        # Count-weighted mean over earlier encounters, separately for each type.
        return torch.tensor(self.counts / np.maximum(self.counts.sum(axis=2, keepdims=True), 1))

    def teams(self, rows):
        return np.array([[self.lookup[p] for p in row] for row in rows[SLOTS].to_numpy()], dtype=np.int64)

    def snapshot(self, rows):
        t = self.teams(rows)
        ratings = self.elo[t]
        elo = 1/(1+10**(-((ratings[:, :2].mean(1)-ratings[:, 2:].mean(1))/400)))
        # Games won are a performance-only proxy, not official ladder/attendance points.
        points = self.games_for[t][:, :2].sum(1)-self.games_for[t][:, 2:].sum(1)
        return Snapshot(rows.session.iloc[0], rows.season.iloc[0], self.features(), self.relations(),
                        torch.tensor(t), torch.tensor(rows.team_a_win.to_numpy(), dtype=torch.float32),
                        elo, points, (self.sets[t] == 0).any(1), rows.match_id.tolist())

    def update(self, rows):
        # Elo deltas use the same pre-session ratings for every result in a session.
        snap = self.snapshot(rows)
        delta = np.zeros(len(self.ids))
        for k, (t, (_, row)) in enumerate(zip(snap.teams.numpy(), rows.iterrows())):
            a, b = t[:2], t[2:]
            for team, other, won, gf, ga in [(a,b,row.team_a_win,row.team_a_score,row.team_b_score),
                                            (b,a,1-row.team_a_win,row.team_b_score,row.team_a_score)]:
                self.sets[team] += 1
                self.wins[team] += won
                self.games_for[team] += gf
                self.games_against[team] += ga
                u, v = team
                self.counts[0, u, v] += 1
                self.counts[0, v, u] += 1
                for u in team:
                    for v in other:
                        self.counts[1, u, v] += 1
            change = 32 * (row.team_a_win - snap.elo[k])
            delta[a] += change
            delta[b] -= change
        self.elo += delta


def snapshots(players, matches):
    history, result = History(players), []
    for _, rows in matches.groupby('session', sort=True):
        result.append(history.snapshot(rows))
        history.update(rows)
    return result, history


def export_graph(players, matches, directory):
    """Lossless event grouping; no score/result is a predictive edge attribute."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    players[['player_id', 'pseudonym']].to_csv(out/'players.csv', index=False)
    matches[['match_id','season','session','court','set_number','team_a_win',
             'team_a_score','team_b_score']].to_csv(out/'events.csv', index=False)
    participation, relations = [], []
    for _, row in matches.iterrows():
        ids = row[SLOTS].tolist()
        for i,p in enumerate(ids):
            participation.append((p,row.match_id,'A' if i<2 else 'B'))
        for i,j in combinations(range(4),2):
            r = 'teammate' if i//2 == j//2 else 'opponent'
            relations.extend([(ids[i],ids[j],r,row.match_id,row.session),
                              (ids[j],ids[i],r,row.match_id,row.session)])
    pd.DataFrame(participation,columns=['player_id','match_id','team']).to_csv(out/'participation.csv',index=False)
    pd.DataFrame(relations,columns=['source','target','relation','match_id','session']).to_csv(out/'relations.csv',index=False)
