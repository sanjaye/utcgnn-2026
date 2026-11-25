"""Train and evaluate the player–set GNNs (hetero_models.py) over several seeds.

Uses the same season splits, labels, cold-start flags, metrics and prediction files as
`train` (via split_sessions, evaluate and save_predictions), so results are comparable.

Training is full-batch: all training-season graphs form one PyG batch, and each epoch is
one optimizer step, as in utcgnn.experiment.train_graph. The epoch with the lowest
validation log loss is kept. Validation and test graphs use each date's own history,
including earlier validation/test results (the same online protocol as `train`), while
the weights stay fixed.

task='score' (Task 2) keeps the same model: sigmoid(f(h) - f(h~)) is read as team A's
share of games instead of P(A wins), which flips to 1 - share for the other team just as
P(A wins) does. Training uses BCE with the soft label `share`, weighted by the games in
each set (the binomial likelihood of the games), and the epoch is chosen by validation
MAE in games. See utcgnn/score.py.
"""
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from .experiment import evaluate, metrics, save_predictions, split_sessions
from .hetero_data import NODE_FEATURES, apply_scaling, build_graphs, fit_scaling
from .hetero_models import PlayerSetGNN
from .reporting import log_markdown, metrics_table, reference_models
from .score import cell, evaluate_score, flat, log_score, mae_games, report_base, save_score_outputs, score_table, score_targets, summarize

SPLITS = ('train', 'validation', 'test')
LABELS = {'rgcn': 'Player–set R-GCN', 'sage': 'Player–set GraphSAGE', 'gat': 'Player–set GATv2'}
CONVS = {'rgcn': 'RGCNConv', 'sage': 'per-edge-type SAGEConv', 'gat': 'per-edge-type GATv2Conv'}
TASKS = ('win', 'score')


def resolve_device(device):
    """'auto' picks CUDA (e.g. a Colab GPU) when available, otherwise the CPU."""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def predict(model, batch):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(batch)).cpu().numpy()


def train_one(kind, batches, seed, epochs, hidden, layers, dropout, lr, weight_decay, device, score=None, heads=4):
    """Train one model with one seed; return it with its selected epoch and training trace.

    score=None trains Task 1 (BCE on the win label, epoch by validation log loss). For Task 2
    pass {'train': {'share', 'games'} tensors on the device, 'validation': score_targets row};
    the loss is games-weighted BCE on the share and the epoch is chosen by validation MAE in games.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PlayerSetGNN(kind, hidden, layers, dropout, heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train, validation = batches['train'], batches['validation']
    if score is None:
        val_y = validation.y.cpu().numpy()
        names = ('train_bce', 'validation_log_loss')

        def loss_fn(logit):
            # Mean binary cross-entropy over every training set: one full-batch step per epoch.
            return torch.nn.functional.binary_cross_entropy_with_logits(logit, train.y)

        def val_metric(p):
            return metrics(val_y, p)['log_loss']
    else:
        share, games = score['train']['share'], score['train']['games']
        names = ('train_games_bce', 'validation_mae_games')

        def loss_fn(logit):
            # Binomial log-likelihood of the games won, per game: BCE(share) weighted by games.
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, share, reduction='none')
            return (bce * games).sum() / games.sum()

        def val_metric(p):
            return mae_games(score['validation'], p)
    best, best_state, best_epoch, trace = float('inf'), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(train))
        loss.backward()
        opt.step()
        val_loss = val_metric(predict(model, validation))
        trace.append({'epoch': epoch, names[0]: loss.item(), names[1]: val_loss})
        if val_loss < best:  # model selection uses validation only, never test labels
            best, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_epoch, trace


def _heads(kind, heads):
    """Log text for the GATv2 attention setup; empty for the other models."""
    if kind != 'gat':
        return ''
    return (f' ({heads} attention heads averaged, a separate attention per edge type, a learned self term per node type,'
            ' no attention dropout)')


def _summary(per_seed):
    """Mean and standard deviation across seeds for each split and metric."""
    out = {}
    for split in SPLITS:
        out[split] = {}
        for name in ('accuracy', 'roc_auc', 'log_loss'):
            values = [r[split][name] for r in per_seed if r[split][name] is not None]
            out[split][name] = {'mean': float(np.mean(values)) if values else None,
                                'std': float(np.std(values)) if values else None}
    return out


def run_hetero(players, matches, audit, output, kind='rgcn', seeds=(42, 43, 44, 45, 46), epochs=200,
               hidden=32, layers=4, dropout=0.1, lr=0.005, weight_decay=1e-3, device='auto',
               log_path='reports/experiments.md', reference_path='reports/metrics.json', command=None, task='win', heads=4):
    if task not in TASKS:
        raise ValueError(task)
    started = time.time()
    device = resolve_device(device)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    splits, groups, _ = split_sessions(players, matches)
    labels = {s: np.concatenate([x.y.numpy() for x in g]) for s, g in groups.items()}
    cold = {s: np.concatenate([x.cold for x in g]) for s, g in groups.items()}

    # Build one graph per date, scale features with training-season statistics only,
    # then batch each split into a single graph on the target device.
    graphs = build_graphs(players, matches, groups)
    scaling = fit_scaling(graphs['train'])
    batches = {s: Batch.from_data_list(apply_scaling(graphs[s], scaling)).to(device) for s in SPLITS}
    score, targets = None, None
    if task == 'score':
        # Batches keep the split's set order, so these line up with batch.y row for row.
        targets = score_targets(matches, groups)
        score = {'validation': targets['validation'],
                 'train': {k: torch.tensor(targets['train'][k], dtype=torch.float32, device=device)
                           for k in ('share', 'games')}}

    preds = {s: {} for s in SPLITS}
    per_seed, selected = [], {}
    for seed in seeds:
        model, best_epoch, trace = train_one(kind, batches, seed, epochs, hidden, layers, dropout,
                                             lr, weight_decay, device, score, heads)
        selected[seed] = best_epoch
        result = {}
        for s in SPLITS:
            p = predict(model, batches[s])
            preds[s][f'{kind}_seed{seed}'] = p
            result[s] = (evaluate(labels[s], p, cold[s]) if task == 'win'
                         else evaluate_score(targets[s], p, len(players)))
        per_seed.append(result)
        with open(out / f'{kind}_seed{seed}_training.csv', 'w') as f:
            f.write(','.join(trace[0]) + '\n')
            f.writelines(','.join(str(v) for v in t.values()) + '\n' for t in trace)
        torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'config': {'kind': kind, 'hidden': hidden, 'layers': layers, 'dropout': dropout, 'task': task,
                               'heads': heads},
                    'scaling': {t: (m.cpu(), sd.cpu()) for t, (m, sd) in scaling.items()},
                    'node_features': NODE_FEATURES, 'player_ids': players.player_id.tolist(),
                    'include_flagged': audit['policy'] == 'include_flagged',
                    'selection_end': str(max(x.session for x in groups['validation']).date())},
                   out / f'{kind}_seed{seed}.pt')
    for s in SPLITS:  # convenience column: probability averaged over seeds
        preds[s][f'{kind}_seed_mean'] = np.mean([preds[s][f'{kind}_seed{seed}'] for seed in seeds], axis=0)
    if task == 'score':
        return _finish_score(players, audit, out, kind, seeds, epochs, hidden, layers, dropout, lr, weight_decay,
                             heads, device, splits, targets, preds, per_seed, selected, started, log_path, command)
    save_predictions(out, groups, preds)

    summary = _summary(per_seed)
    run_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    report = {'model': f'hetero_{kind}', 'run_at': run_at, 'device': str(device), 'audit': audit, 'splits': splits,
              'config': {'kind': kind, 'hidden': hidden, 'layers': layers, 'dropout': dropout, 'heads': heads, 'lr': lr,
                         'weight_decay': weight_decay, 'epochs': epochs, 'seeds': list(seeds)},
              'node_features': NODE_FEATURES, 'selected_epochs': selected,
              'summary': summary, 'per_seed': dict(zip(map(str, seeds), per_seed)),
              'runtime_seconds': round(time.time() - started, 1)}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if log_path:
        log_markdown(log_path, f'hetero_{kind}', _log_body(report, labels, out, command, reference_path))
    return report


def _other_hetero_rows(report, cell):
    """Seeded Task 1 rows of the other player–set GNNs (artifacts/hetero_<kind>), if run on the same data."""
    rows = []
    for kind in LABELS:
        path = Path(f'artifacts/hetero_{kind}/metrics.json')
        if kind == report['config']['kind'] or not path.exists():
            continue
        other = json.loads(path.read_text())
        if (other.get('splits') != report['splits'] or other.get('audit', {}).get('used_sets') != report['audit']['used_sets']
                or other.get('audit', {}).get('policy') != report['audit']['policy']):
            continue
        cells = ' | '.join(cell(other['summary'], s, n) for s in ('validation', 'test') for n in ('accuracy', 'roc_auc', 'log_loss'))
        rows.append(f"| {LABELS[kind]} | {cells} |")
    return rows


def _log_body(report, labels, out, command, reference_path):
    cfg, summary = report['config'], report['summary']

    def cell(table, split, name):
        m = table[split][name]
        return '—' if m['mean'] is None else f"{m['mean']:.4f} ± {m['std']:.4f}"

    references, note = reference_models(reference_path, report['splits'], report['audit'])
    val_cold = [r['validation']['cold_start']['log_loss'] for r in report['per_seed'].values()]
    val_cold = [v for v in val_cold if v is not None]
    n_cold = report['per_seed'][str(cfg['seeds'][0])]['validation']['cold_start']['n']
    cold_line = (f'Validation cold-start log loss (n={n_cold}): {np.mean(val_cold):.4f} ± {np.std(val_cold):.4f}.'
                 if val_cold else 'No validation cold-start sets.')
    epochs = ', '.join(str(e) for e in report['selected_epochs'].values())
    row = ' | '.join(cell(summary, s, n) for s in ('validation', 'test') for n in ('accuracy', 'roc_auc', 'log_loss'))
    others = '\n'.join(_other_hetero_rows(report, cell))
    others = '\n' + others if others else ''
    return f"""
### {LABELS[cfg['kind']]} (`utcgnn/hetero_models.py`)

Last run: {report['run_at']} · `{command or 'python -m utcgnn train-hetero'}` · {report['device']} · {report['runtime_seconds']:.0f} s · outputs in `{out}`

- **Graph:** one PyG graph per match date with player, past-set and target nodes. Past sets are linked to players by won/lost edges; targets receive side A/B edges and carry no result. Past-set features: {', '.join(report['node_features']['past_set'])}. Target features: {', '.join(report['node_features']['target'])}.
- **Model:** {cfg['layers']} {CONVS[cfg['kind']]} layers{_heads(cfg['kind'], cfg.get('heads'))}, hidden size {cfg['hidden']}, dropout {cfg['dropout']}; swap readout f(h) − f(h̃), so P(A wins) = 1 − P(B wins).
- **Training:** full batch, Adam (learning rate {cfg['lr']}, weight decay {cfg['weight_decay']}), up to {cfg['epochs']} epochs, epoch chosen by validation log loss. Seeds {', '.join(map(str, cfg['seeds']))}; selected epochs {epochs}.
- **Data:** {len(labels['train']):,} train, {len(labels['validation']):,} validation, {len(labels['test']):,} test sets.

Mean ± standard deviation over {len(cfg['seeds'])} seeds:

| Model | Val accuracy | Val ROC-AUC | Val log loss | Test accuracy | Test ROC-AUC | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **{LABELS[cfg['kind']]}** | {row} |{others}

Other player–set GNN rows come from `artifacts/hetero_<model>/metrics.json` when available. {cold_line}

Reference models (single seed):

{metrics_table(references) if references else '_None available._'}

{note}
"""


def _finish_score(players, audit, out, kind, seeds, epochs, hidden, layers, dropout, lr, weight_decay, heads, device,
                  splits, targets, preds, per_seed, selected, started, log_path, command):
    """Task 2 outputs: predictions and standings CSVs, metrics.json and the results-log section."""
    save_score_outputs(out, targets, preds, players.player_id.tolist(), standings_model=f'{kind}_seed_mean')
    summary = {s: summarize([flat(r[s]) for r in per_seed]) for s in SPLITS}
    mean_model = {s: evaluate_score(targets[s], preds[s][f'{kind}_seed_mean'], len(players)) for s in SPLITS}
    run_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    label = LABELS[kind]
    report = {**report_base(splits, audit, targets, len(players)), 'model': f'hetero_{kind}_score', 'run_at': run_at,
              'device': str(device),
              'config': {'kind': kind, 'task': 'score', 'hidden': hidden, 'layers': layers, 'dropout': dropout,
                         'heads': heads, 'lr': lr,
                         'weight_decay': weight_decay, 'epochs': epochs, 'seeds': list(seeds)},
              'node_features': NODE_FEATURES, 'selected_epochs': selected, 'summary': summary,
              'seed_mean_model': mean_model, 'per_seed': dict(zip(map(str, seeds), per_seed)),
              'rows': {label: summary}, 'runtime_seconds': round(time.time() - started, 1)}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if log_path:
        epochs_line = ', '.join(str(e) for e in selected.values())
        cold = summary['validation']['cold_mae_games']
        n_cold = per_seed[0]['validation']['cold_start']['n']
        avg = {s: flat(mean_model[s]) for s in SPLITS}
        body = f"""
### Task 2 {label} (`utcgnn/hetero_models.py`)

Last run: {run_at} · `{command or f'python -m utcgnn train-hetero --task score --models {kind}'}` · {device} · {report['runtime_seconds']:.0f} s · outputs in `{out}`

- **Graph and model:** the Task 1 player–set graph and {layers} {CONVS[kind]} layers{_heads(kind, heads)}, hidden size {hidden}, dropout {dropout}, unchanged from Task 1. sigmoid(f(h) − f(h̃)) is read as team A's share of games, so share(B) = 1 − share(A).
- **Training:** BCE with the soft label share = a / (a + b), weighted by games in the set (binomial likelihood of the games). Full batch, Adam (learning rate {lr}, weight decay {weight_decay}), up to {epochs} epochs, epoch chosen by validation MAE in games. Seeds {', '.join(map(str, seeds))}; selected epochs {epochs_line}.
- **Data:** {len(targets['train']['share']):,} train, {len(targets['validation']['share']):,} validation, {len(targets['test']['share']):,} test sets.

Mean ± standard deviation over {len(seeds)} seeds:

{score_table(report['rows'])}

Validation cold-start MAE (n={n_cold}): {cell(cold)} games. Averaging the {len(seeds)} seeds' predictions gives test MAE {cell(avg['test']['mae_games'])} games, ρ total {cell(avg['test']['total_spearman'])}, ρ per set {cell(avg['test']['per_set_spearman'])}.
"""
        log_score(log_path, f'score_hetero_{kind}', body, report)
    return report
