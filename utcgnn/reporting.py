"""Markdown experiment log: one replaceable section per experiment, compared with the main run."""
import json
from pathlib import Path

REFERENCE_LABELS = {'rgcn': 'R-GCN', 'sage': 'GraphSAGE', 'tabular': 'Tabular logistic',
                    'elo': 'Elo', 'historical_games': 'Cumulative games won'}


def reference_models(path, splits, audit):
    """Metrics of the `train` models from a metrics.json, only if it used the same data and splits."""
    path = Path(path)
    if not path.exists():
        return {}, f'{path} not found; reference models omitted.'
    report = json.loads(path.read_text())
    if (report.get('splits') != splits or report.get('audit', {}).get('used_sets') != audit['used_sets']
            or report.get('audit', {}).get('policy') != audit['policy']):
        return {}, f'{path} uses different data or splits; reference models omitted.'
    return ({REFERENCE_LABELS.get(k, k): v for k, v in report['models'].items()},
            f'Reference rows are copied from `{path}` (seed {report.get("seed")} run of `python -m utcgnn train`).')


def _fmt(value):
    return '—' if value is None else f'{value:.4f}'


def metrics_table(rows):
    """Validation and test metrics as a Markdown table. rows: {model label: {split: metrics}}."""
    lines = ['| Model | Val accuracy | Val ROC-AUC | Val log loss | Test accuracy | Test ROC-AUC | Test log loss |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, result in rows.items():
        cells = [_fmt(result[s][m]) for s in ('validation', 'test') for m in ('accuracy', 'roc_auc', 'log_loss')]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


DECODER_NOTES = {
    'sum': 'team embedding = z₁ + z₂ (partner order cannot matter)',
    'concat': 'team embedding = ReLU(Linear([z₁ ; z₂])) from 2d back to d (separate weights per partner slot)',
    'mean': 'team embedding = ReLU(Linear((z₁ + z₂) / 2)), d to d (average pooling; partner order cannot matter)',
}


def log_train(path, report, output, command, run_at, reference_path='reports/metrics.json'):
    """Log one `train` run as section train_<decoder>; non-sum decoders also show the sum-decoder GNNs."""
    decoder = report.get('decoder', 'sum')
    rows = {(f'**{REFERENCE_LABELS[k]} ({decoder} decoder)**' if k in ('rgcn', 'sage') else REFERENCE_LABELS.get(k, k)): v
            for k, v in report['models'].items()}
    note = 'Baseline rows come from this same run.'
    if decoder != 'sum':
        references, reference_note = reference_models(reference_path, report['splits'], report['audit'])
        rows.update({f'{label} (sum decoder, reference)': references[label]
                     for label in ('R-GCN', 'GraphSAGE') if label in references})
        note += ' ' + reference_note
    epochs = ', '.join(f'{REFERENCE_LABELS[k]} {report["models"][k]["selected_epoch"]}' for k in ('rgcn', 'sage'))
    body = f"""
### GNN with {decoder} decoder (`utcgnn/models.py`)

Last run: {run_at} · `{command}` · outputs in `{output}`

- **Decoder:** {DECODER_NOTES.get(decoder, decoder)}. Both teams use the same decoder and the logit is score(A) − score(B).
- **Training:** {report['epochs_requested']} epochs, seed {report['seed']}; selected epoch by validation log loss: {epochs}.

{metrics_table(rows)}

{note} All results are from a single seed and a single test season.
"""
    log_markdown(path, f'train_{decoder}', body)


def log_markdown(path, name, body):
    """Replace experiment `name`'s section in the log (appending it if new); other sections are untouched."""
    path = Path(path)
    start, end = f'<!-- results:{name}:start -->', f'<!-- results:{name}:end -->'
    block = f'{start}\n{body.strip()}\n{end}'
    text = path.read_text() if path.exists() else '# Experiments\n\n## Results log\n'
    if start in text and end in text:
        text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
    else:
        text = text.rstrip() + '\n\n' + block + '\n'
    path.write_text(text)
