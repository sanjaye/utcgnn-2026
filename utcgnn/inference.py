"""Load trained models and strictly historical state for inference."""
import pandas as pd
import torch

from .data import History
from .models import GraphModel


def load_inference_context(players, matches, checkpoint_path, date, include_flagged):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if players.player_id.tolist() != checkpoint['player_ids']:
        raise ValueError('Checkpoint player registry differs from input')
    if include_flagged != checkpoint['include_flagged']:
        raise ValueError('Use the same flagged-row policy as training')
    date = pd.Timestamp(date)
    if date <= pd.Timestamp(checkpoint['selection_end']):
        raise ValueError('Prediction must be after validation/model-selection data to avoid weight leakage')

    history = History(players)
    for _, rows in matches[matches.session < date].groupby('session', sort=True):
        history.update(rows)
    model = GraphModel(kind=checkpoint['kind'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return history, model, checkpoint


def predict_teams(history, model, checkpoint, teams):
    with torch.no_grad():
        team_indices = torch.tensor(
            [[history.lookup[player_id] for player_id in team] for team in teams],
            dtype=torch.long,
        )
        return torch.sigmoid(
            model(
                (history.features() - checkpoint['mean']) / checkpoint['scale'],
                history.relations(),
                team_indices,
            )
        ).tolist()
