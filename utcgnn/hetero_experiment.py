"""Train and evaluate the player–set GNNs (hetero_models.py) over several seeds.

Uses the same season splits, labels, cold-start flags, metrics and prediction files as
`train` (via split_sessions, evaluate and save_predictions), so results are comparable.

Training is full-batch: all training-season graphs form one PyG batch, and each epoch is
one optimizer step, as in utcgnn.experiment.train_graph. The epoch with the lowest
validation log loss is kept. Validation and test graphs use each date's own history,
including earlier validation/test results (the same online protocol as `train`), while
the weights stay fixed.
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

SPLITS = ('train', 'validation', 'test')
LABELS = {'rgcn': 'Player–set R-GCN', 'sage': 'Player–set GraphSAGE'}


def resolve_device(device):
    """'auto' picks CUDA (e.g. a Colab GPU) when available, otherwise the CPU."""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def predict(model, batch):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(batch)).cpu().numpy()


def train_one(kind, batches, seed, epochs, hidden, layers, dropout, lr, weight_decay, device):
    """Train one model with one seed; return it with its selected epoch and training trace."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PlayerSetGNN(kind, hidden, layers, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train, validation = batches['train'], batches['validation']
    val_y = validation.y.cpu().numpy()
    best, best_state, best_epoch, trace = float('inf'), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        # Mean binary cross-entropy over every training set: one full-batch step per epoch.
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(train), train.y)
        loss.backward()
        opt.step()
        val_loss = metrics(val_y, predict(model, validation))['log_loss']
        trace.append({'epoch': epoch, 'train_bce': loss.item(), 'validation_log_loss': val_loss})
        if val_loss < best:  # model selection uses validation only, never test labels
            best, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_epoch, trace


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
               log_path='reports/experiments.md', reference_path='reports/metrics.json', command=None):
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

    preds = {s: {} for s in SPLITS}
    per_seed, selected = [], {}
    for seed in seeds:
        model, best_epoch, trace = train_one(kind, batches, seed, epochs, hidden, layers, dropout,
                                             lr, weight_decay, device)
        selected[seed] = best_epoch
        result = {}
        for s in SPLITS:
            p = predict(model, batches[s])
            preds[s][f'{kind}_seed{seed}'] = p
            result[s] = evaluate(labels[s], p, cold[s])
        per_seed.append(result)
        with open(out / f'{kind}_seed{seed}_training.csv', 'w') as f:
            f.write('epoch,train_bce,validation_log_loss\n')
            f.writelines(f"{t['epoch']},{t['train_bce']},{t['validation_log_loss']}\n" for t in trace)
        torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'config': {'kind': kind, 'hidden': hidden, 'layers': layers, 'dropout': dropout},
                    'scaling': {t: (m.cpu(), sd.cpu()) for t, (m, sd) in scaling.items()},
                    'node_features': NODE_FEATURES, 'player_ids': players.player_id.tolist(),
                    'include_flagged': audit['policy'] == 'include_flagged',
                    'selection_end': str(max(x.session for x in groups['validation']).date())},
                   out / f'{kind}_seed{seed}.pt')
    for s in SPLITS:  # convenience column: probability averaged over seeds
        preds[s][f'{kind}_seed_mean'] = np.mean([preds[s][f'{kind}_seed{seed}'] for seed in seeds], axis=0)
    save_predictions(out, groups, preds)

    summary = _summary(per_seed)
    run_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    report = {'model': f'hetero_{kind}', 'run_at': run_at, 'device': str(device), 'audit': audit, 'splits': splits,
              'config': {'kind': kind, 'hidden': hidden, 'layers': layers, 'dropout': dropout, 'lr': lr,
                         'weight_decay': weight_decay, 'epochs': epochs, 'seeds': list(seeds)},
              'node_features': NODE_FEATURES, 'selected_epochs': selected,
              'summary': summary, 'per_seed': dict(zip(map(str, seeds), per_seed)),
              'runtime_seconds': round(time.time() - started, 1)}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if log_path:
        log_markdown(log_path, f'hetero_{kind}', _log_body(report, labels, out, command, reference_path))
    return report


def _log_body(report, labels, out, command, reference_path):
    cfg, summary = report['config'], report['summary']

    def cell(split, name):
        m = summary[split][name]
        return '—' if m['mean'] is None else f"{m['mean']:.4f} ± {m['std']:.4f}"

    references, note = reference_models(reference_path, report['splits'], report['audit'])
    val_cold = [r['validation']['cold_start']['log_loss'] for r in report['per_seed'].values()]
    val_cold = [v for v in val_cold if v is not None]
    n_cold = report['per_seed'][str(cfg['seeds'][0])]['validation']['cold_start']['n']
    cold_line = (f'Validation cold-start log loss (n={n_cold}): {np.mean(val_cold):.4f} ± {np.std(val_cold):.4f}.'
                 if val_cold else 'No validation cold-start sets.')
    epochs = ', '.join(str(e) for e in report['selected_epochs'].values())
    row = ' | '.join(cell(s, n) for s in ('validation', 'test') for n in ('accuracy', 'roc_auc', 'log_loss'))
    return f"""
### {LABELS[cfg['kind']]} (`utcgnn/hetero_models.py`)

Last run: {report['run_at']} · `{command or 'python -m utcgnn train-hetero'}` · {report['device']} · {report['runtime_seconds']:.0f} s · outputs in `{out}`

- **Graph:** one PyG graph per match date with player, past-set and target nodes. Past sets are linked to players by won/lost edges; targets receive side A/B edges and carry no result. Past-set features: {', '.join(report['node_features']['past_set'])}. Target features: {', '.join(report['node_features']['target'])}.
- **Model:** {cfg['layers']} {'RGCNConv' if cfg['kind'] == 'rgcn' else 'per-edge-type SAGEConv'} layers, hidden size {cfg['hidden']}, dropout {cfg['dropout']}; swap readout f(h) − f(h̃), so P(A wins) = 1 − P(B wins).
- **Training:** full batch, Adam (learning rate {cfg['lr']}, weight decay {cfg['weight_decay']}), up to {cfg['epochs']} epochs, epoch chosen by validation log loss. Seeds {', '.join(map(str, cfg['seeds']))}; selected epochs {epochs}.
- **Data:** {len(labels['train']):,} train, {len(labels['validation']):,} validation, {len(labels['test']):,} test sets.

Mean ± standard deviation over {len(cfg['seeds'])} seeds:

| Model | Val accuracy | Val ROC-AUC | Val log loss | Test accuracy | Test ROC-AUC | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **{LABELS[cfg['kind']]}** | {row} |

{cold_line}

Reference models (single seed):

{metrics_table(references) if references else '_None available._'}

{note}
"""
