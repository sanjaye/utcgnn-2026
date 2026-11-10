"""Chronological model selection, prequential evaluation and auditable artifacts."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .data import FEATURES, season_splits, snapshots
from .models import GraphModel


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0:
        return {'n': 0, 'accuracy': None, 'roc_auc': None, 'log_loss': None}
    return {'n': len(y), 'accuracy': float(accuracy_score(y, p >= .5)),
            'roc_auc': float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'log_loss': float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]))}


def feature_differences(snapshot):
    t = snapshot.teams.numpy()
    x = snapshot.x.numpy()[t]
    return x[:,:2].sum(1)-x[:,2:].sum(1)


def symmetric_classifier(x, y):
    # Swapped-team augmentation removes arbitrary A/B slot advantage.
    model = LogisticRegression(C=1.,fit_intercept=False,max_iter=1000)
    model.fit(np.concatenate([x,-x]),np.concatenate([y,1-y]))
    return model


def probabilities(model, sessions, mean, scale):
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model((s.x-mean)/scale,s.relations,s.teams)).numpy()
                               for s in sessions])


def train_graph(kind, train, validation, epochs, seed, decoder='sum'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    all_x = torch.cat([s.x for s in train])
    mean, scale = all_x.mean(0), all_x.std(0).clamp(min=.1)
    # decoder: 'sum' or 'concat' team embedding (see models.DECODERS).
    model = GraphModel(kind=kind,decoder=decoder)
    opt = torch.optim.Adam(model.parameters(),lr=.005,weight_decay=1e-3)
    best, best_state, best_epoch = float('inf'), None, 0
    trace = []
    n = sum(len(s.y) for s in train)
    vy = np.concatenate([s.y.numpy() for s in validation])
    for epoch in range(1,epochs+1):
        model.train()
        opt.zero_grad()
        total = 0.
        for s in train:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model((s.x-mean)/scale,s.relations,s.teams),s.y,reduction='sum')/n
            loss.backward()
            total += loss.item()
        opt.step()
        val_loss = metrics(vy,probabilities(model,validation,mean,scale))['log_loss']
        trace.append({'epoch':epoch,'train_bce':total,'validation_log_loss':val_loss})
        if val_loss < best:
            best, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    return model, mean, scale, best_epoch, trace


def split_sessions(players, matches):
    """Chronological season splits and their session snapshots, shared by every model."""
    seasons = season_splits(matches)
    sessions, history = snapshots(players,matches)
    splits = {'train':seasons[:-2], 'validation':[seasons[-2]], 'test':[seasons[-1]]}
    groups = {k:[s for s in sessions if s.season in v] for k,v in splits.items()}
    return splits, groups, history


def evaluate(y, p, cold):
    """Overall, cold-start and established-player metrics for one split."""
    result = metrics(y,p)
    result['cold_start'] = metrics(y[cold],p[cold])
    result['established'] = metrics(y[~cold],p[~cold])
    return result


def save_predictions(out, groups, preds):
    """Per-split CSVs with one P(team A wins) column per model in preds[split]."""
    for split,g in groups.items():
        pd.DataFrame({'match_id':[i for s in g for i in s.ids],
                      'session':[str(s.session.date()) for s in g for _ in s.ids],
                      'team_a_win':np.concatenate([s.y.numpy() for s in g]).astype(int),
                      'cold_start':np.concatenate([s.cold for s in g]),**preds[split]}).to_csv(
                          Path(out)/f'{split}_predictions.csv',index=False)


def run_experiment(players, matches, audit, output, epochs=60, seed=42, decoder='sum'):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    splits, groups, history = split_sessions(players,matches)
    labels = {k:np.concatenate([s.y.numpy() for s in g]) for k,g in groups.items()}
    masks = {k:np.concatenate([s.cold for s in g]) for k,g in groups.items()}
    train_x = np.concatenate([feature_differences(s) for s in groups['train']])
    tabular = symmetric_classifier(train_x, labels['train'])
    point_x = np.concatenate([s.points for s in groups['train']])[:,None]
    point_model = symmetric_classifier(point_x, labels['train'])
    report = {'seed':seed,'epochs_requested':epochs,'decoder':decoder,'audit':audit,'splits':splits,
              'protocol':'Frozen model weights; session-batched online history updates after all predictions on a date. '
                         'Earlier validation/test results may inform later sessions, never the same session.',
              'features':FEATURES,'models':{},
              'ranking_evaluation':{'status':'unavailable',
                 'reason':'Official standings and attendance awards were not supplied; no official Spearman/top-four claim.'}}
    preds = {split:{'elo':np.concatenate([s.elo for s in g]),
                  'tabular':tabular.predict_proba(np.concatenate([feature_differences(s) for s in g]))[:,1],
                  'historical_games':point_model.predict_proba(np.concatenate([s.points for s in g])[:,None])[:,1]}
             for split,g in groups.items()}
    for kind in ['rgcn','sage']:
        model,mean,scale,best_epoch,trace = train_graph(kind,groups['train'],groups['validation'],epochs,seed,decoder)
        for split,g in groups.items():
            preds[split][kind] = probabilities(model,g,mean,scale)
        report['models'][kind] = {'selected_epoch':best_epoch}
        pd.DataFrame(trace).to_csv(out/f'{kind}_training.csv',index=False)
        # The decoder is saved so inference rebuilds the same architecture.
        torch.save({'state_dict':model.state_dict(),'mean':mean,'scale':scale,'kind':kind,'decoder':decoder,
                    'player_ids':players.player_id.tolist(),
                    'include_flagged':audit['policy']=='include_flagged',
                    'selection_end':str(max(s.session for s in groups['validation']).date()),
                    'features':FEATURES},out/f'{kind}.pt')
        # Contextual individual strength: shared neutral partner vs neutral team,
        # using the retrospective graph through the last observed session.
        with torch.no_grad():
            h = model.encode((history.features()-mean)/scale,history.relations())
            active = torch.tensor(history.sets>0)
            neutral = h[active].mean(0).expand_as(h)

            def team_logit(z1, z2):
                return model.team_score(model.team_embedding(z1,z2)).squeeze(-1)

            # Average the player in both partner slots, since the concat decoder weights slots
            # differently. For 'sum' this equals the original score(h + n) - score(2n).
            strengths = ((team_logit(h,neutral)+team_logit(neutral,h))/2-team_logit(neutral,neutral)).numpy()
        pd.DataFrame({'player_id':history.ids,'pseudonym':players.pseudonym,
                      'observed_sets':history.sets.astype(int),'elo':history.elo,
                      'contextual_strength_logit':strengths,
                      'has_history':history.sets>0}).sort_values('contextual_strength_logit',ascending=False).to_csv(
                          out/f'{kind}_player_strength.csv',index=False)
    for split in groups:
        for name,p in preds[split].items():
            report['models'].setdefault(name,{})[split] = evaluate(labels[split],p,masks[split])
    save_predictions(out,groups,preds)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
