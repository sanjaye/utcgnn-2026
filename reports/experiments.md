# Experiments

Alternative models are evaluated against the `train` models on the same data, season
splits, history-only features and online protocol. Each model is a module in `utcgnn/`
that reuses `split_sessions`, `evaluate` and `save_predictions` from
`utcgnn/experiment.py`, and runs as its own CLI command.

## Setup

From the repository root:

```bash
python3.12 -m venv .venv            # or: uv venv --python 3.12 .venv
.venv/bin/python -m pip install -r requirements.txt
```

On macOS, install `torch==2.9.1` without the `+cpu` suffix in `requirements.txt`;
that suffix only exists for Linux and Windows wheels.

## Running

Run `train` first, so the comparison table can include the GNN and baseline rows from
`reports/metrics.json`. Then run any experiment command:

```bash
.venv/bin/python -m utcgnn train --decoder concat
.venv/bin/python -m utcgnn train --decoder mean
.venv/bin/python -m utcgnn xgboost
.venv/bin/python -m utcgnn train-hetero          # PyG player–set graph; see "Running on Google Colab"
```

| Command | Module | What it tests | Options |
| --- | --- | --- | --- |
| `train --decoder concat` | `utcgnn/models.py` | GNN team decoder that concatenates the two partners' final embeddings (2d) and applies a linear layer back to d followed by a ReLU, in place of summing them. `--decoder sum` is the original. | `--decoder` (`sum`, `concat`, `mean`), `--output` (`artifacts/main` for sum, `artifacts/main_<decoder>` otherwise), `--log`, `--no-log`, plus the usual `train` options |
| `train --decoder mean` | `utcgnn/models.py` | GNN team decoder that averages the two partners' final embeddings, then applies a d-to-d linear layer and a ReLU. Partner order cannot matter. | Same as above |
| `train-hetero` | `utcgnn/hetero_data.py`, `utcgnn/hetero_models.py`, `utcgnn/hetero_experiment.py` | The project proposal's architecture in PyTorch Geometric: one graph per match date with player, past-set and target nodes; past sets linked to players by won/lost edges, targets by side A/B edges. R-GCN (`RGCNConv`) and GraphSAGE (one `SAGEConv` per edge type), swap readout so P(A wins) = 1 − P(B wins), reported as mean ± std over seeds | `--models` (`rgcn sage`), `--seeds` (5, starting at `--seed` 42), `--epochs` (200), `--hidden` (32), `--layers` (4), `--dropout` (0.1), `--device` (`auto`), `--output` (`artifacts/hetero_<model>`), `--log`, `--no-log` |
| `sample-inputs` | `utcgnn/sample_inputs.py` | Not an experiment: saves the preprocessed inputs of both graphs for a few sets, to inspect what the models receive. See `data/sample_model_inputs/README.md` | `--n` (10), `--split` (`validation`), `--output` (`data/sample_model_inputs`) |
| `xgboost` | `utcgnn/xgboost_baseline.py` | Gradient-boosted trees on the 5 pre-session player features, tuned by CV on the training seasons | `--n-iter` (60), `--cv-folds` (3), `--seed` (42), `--output` (`artifacts/xgboost`), `--log` (this file), `--no-log` |

Global options go before the command, as for `train`:
`python -m utcgnn --include-flagged xgboost --output artifacts/xgboost_flagged --no-log`.

Each run writes `metrics.json`, per-split prediction CSVs and model diagnostics to its
output folder, and replaces its own section in the results log below. Other sections
are left as they are.

## Running on Google Colab

`train-hetero` uses the GPU when one is available (`--device auto`). Choose a GPU
runtime in Colab (Runtime → Change runtime type), then run these cells. For a private
repository, clone with a GitHub token, for example
`https://<token>@github.com/sanjaye/utcgnn-2026.git`.

```python
!git clone https://github.com/sanjaye/utcgnn-2026.git
%cd utcgnn-2026
!git checkout nithish_branch          # the branch holding this code
!pip install -q -r requirements-colab.txt
```

```python
import torch
print(torch.__version__, 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
```

```python
!python -m utcgnn train-hetero        # both models, 5 seeds; add e.g. --seeds 10 or --epochs 300
```

```python
from IPython.display import Markdown
Markdown(open('reports/experiments.md').read())
```

Install only `requirements-colab.txt` on Colab, not `requirements.txt`: its
`torch==2.9.1+cpu` pin would replace Colab's GPU build of PyTorch. The run writes
`artifacts/hetero_rgcn/` and `artifacts/hetero_sage/` and updates this file inside the
Colab copy of the repository. Download them, or copy them to Google Drive, before the
session ends. GPU results can differ slightly from CPU results and between runs,
because some GPU operations are not bit-for-bit reproducible; the spread over seeds
shows how much that matters.

## Adding an experiment

1. Add a module `utcgnn/<name>.py` with a `run_<name>(players, matches, audit, output, ...)`
   function.
2. Get data from `split_sessions()`, score with `evaluate()`, and write per-set outputs
   with `save_predictions()` (all in `utcgnn/experiment.py`).
3. Log with `log_markdown(path, '<name>', body)` from `utcgnn/reporting.py`;
   `metrics_table()` and `reference_models()` there build the comparison table.
4. Add a subcommand in `utcgnn/__main__.py` and a row to the table above.

Tune and fit only on the training seasons. Use validation only for model selection,
and never use test labels to make choices.

## Results log

Sections below are written by the commands.

<!-- results:xgboost:start -->
### XGBoost baseline (`utcgnn/xgboost_baseline.py`)

Last run: 2026-11-09 00:28 · `python -m utcgnn xgboost` · outputs in `artifacts/xgboost`

- **Data:** 3,477 sets (2,394 train, 723 validation, 360 test), same splits and online protocol as `train`.
- **Features:** 35 per set from the 5 pre-session player features: per-team sum, min and max, plus the team A − team B difference of sums. Teams are swapped in training and predictions average both orientations.
- **Tuning:** randomized search, 60 candidates, 3-fold CV on training seasons only, grouped by session date. Best CV log loss 0.5269 ± 0.0102.
- **Selected hyperparameters:** `colsample_bytree=0.7639`, `gamma=1.986`, `learning_rate=0.01285`, `max_depth=4`, `min_child_weight=15.19`, `n_estimators=490`, `reg_lambda=1.598`, `subsample=0.8518`
- **Top features by gain:** `diff_sum_game_share`, `diff_sum_elo_scaled`, `diff_sum_smoothed_win_rate`, `diff_sum_log_games_per_set`, `team_b_min_log_sets`, `team_a_min_log_sets`, `team_a_sum_game_share`, `team_b_sum_log_sets`

| Model | Val accuracy | Val ROC-AUC | Val log loss | Test accuracy | Test ROC-AUC | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **XGBoost** | 0.7178 | 0.8079 | 0.5284 | 0.7472 | 0.8310 | 0.5081 |
| R-GCN | 0.7358 | 0.8236 | 0.5154 | 0.7417 | 0.8376 | 0.4988 |
| GraphSAGE | 0.7441 | 0.8225 | 0.5147 | 0.7667 | 0.8447 | 0.4892 |
| Elo | 0.7234 | 0.8034 | 0.5427 | 0.7278 | 0.8257 | 0.5186 |
| Tabular logistic | 0.7261 | 0.8167 | 0.5193 | 0.7417 | 0.8348 | 0.5013 |
| Cumulative games won | 0.4772 | 0.4564 | 0.7134 | 0.4944 | 0.4864 | 0.7087 |

Train: accuracy 0.7845, ROC-AUC 0.8684, log loss 0.4607. Validation cold-start sets (n=24): accuracy 0.6250, log loss 0.7978. The test season has 0 cold-start sets.

Reference rows are copied from `reports/metrics.json` (seed 42 run of `python -m utcgnn train`). All results are from a single seed and a single test season.
<!-- results:xgboost:end -->

<!-- results:train_concat:start -->
### GNN with concat decoder (`utcgnn/models.py`)

Last run: 2026-11-09 00:51 · `python -m utcgnn train --decoder concat` · outputs in `artifacts/main_concat`

- **Decoder:** team embedding = ReLU(Linear([z₁ ; z₂])) from 2d back to d (separate weights per partner slot). Both teams use the same decoder and the logit is score(A) − score(B).
- **Training:** 60 epochs, seed 42; selected epoch by validation log loss: R-GCN 32, GraphSAGE 55.

| Model | Val accuracy | Val ROC-AUC | Val log loss | Test accuracy | Test ROC-AUC | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **R-GCN (concat decoder)** | 0.7400 | 0.8218 | 0.5171 | 0.7556 | 0.8394 | 0.4986 |
| **GraphSAGE (concat decoder)** | 0.7400 | 0.8204 | 0.5168 | 0.7556 | 0.8450 | 0.4907 |
| Elo | 0.7234 | 0.8034 | 0.5427 | 0.7278 | 0.8257 | 0.5186 |
| Tabular logistic | 0.7261 | 0.8167 | 0.5193 | 0.7417 | 0.8348 | 0.5013 |
| Cumulative games won | 0.4772 | 0.4564 | 0.7134 | 0.4944 | 0.4864 | 0.7087 |
| R-GCN (sum decoder, reference) | 0.7358 | 0.8236 | 0.5154 | 0.7417 | 0.8376 | 0.4988 |
| GraphSAGE (sum decoder, reference) | 0.7441 | 0.8225 | 0.5147 | 0.7667 | 0.8447 | 0.4892 |

Baseline rows come from this same run. Reference rows are copied from `reports/metrics.json` (seed 42 run of `python -m utcgnn train`). All results are from a single seed and a single test season.
<!-- results:train_concat:end -->

<!-- results:train_mean:start -->
### GNN with mean decoder (`utcgnn/models.py`)

Last run: 2026-11-09 22:08 · `python -m utcgnn train --decoder mean` · outputs in `artifacts/main_mean`

- **Decoder:** team embedding = ReLU(Linear((z₁ + z₂) / 2)), d to d (average pooling; partner order cannot matter). Both teams use the same decoder and the logit is score(A) − score(B).
- **Training:** 60 epochs, seed 42; selected epoch by validation log loss: R-GCN 53, GraphSAGE 52.

| Model | Val accuracy | Val ROC-AUC | Val log loss | Test accuracy | Test ROC-AUC | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **R-GCN (mean decoder)** | 0.7358 | 0.8237 | 0.5142 | 0.7556 | 0.8390 | 0.4982 |
| **GraphSAGE (mean decoder)** | 0.7386 | 0.8171 | 0.5214 | 0.7417 | 0.8426 | 0.4916 |
| Elo | 0.7234 | 0.8034 | 0.5427 | 0.7278 | 0.8257 | 0.5186 |
| Tabular logistic | 0.7261 | 0.8167 | 0.5193 | 0.7417 | 0.8348 | 0.5013 |
| Cumulative games won | 0.4772 | 0.4564 | 0.7134 | 0.4944 | 0.4864 | 0.7087 |
| R-GCN (sum decoder, reference) | 0.7358 | 0.8236 | 0.5154 | 0.7417 | 0.8376 | 0.4988 |
| GraphSAGE (sum decoder, reference) | 0.7441 | 0.8225 | 0.5147 | 0.7667 | 0.8447 | 0.4892 |

Baseline rows come from this same run. Reference rows are copied from `reports/metrics.json` (seed 42 run of `python -m utcgnn train`). All results are from a single seed and a single test season.
<!-- results:train_mean:end -->

<!-- results:hetero_rgcn:start -->
### Player–set R-GCN (`utcgnn/hetero_models.py`)

Last run: 2026-11-09 23:21 · `python -m utcgnn train-hetero` · cpu · 275 s · outputs in `artifacts/hetero_rgcn`

- **Graph:** one PyG graph per match date with player, past-set and target nodes. Past sets are linked to players by won/lost edges; targets receive side A/B edges and carry no result. Past-set features: winner_games, loser_games, weeks_before, week_of_season, court. Target features: week_of_season, court.
- **Model:** 4 RGCNConv layers, hidden size 32, dropout 0.1; swap readout f(h) − f(h̃), so P(A wins) = 1 − P(B wins).
- **Training:** full batch, Adam (learning rate 0.005, weight decay 0.001), up to 200 epochs, epoch chosen by validation log loss. Seeds 42, 43, 44, 45, 46; selected epochs 35, 45, 35, 36, 31.
- **Data:** 2,394 train, 723 validation, 360 test sets.

Mean ± standard deviation over 5 seeds:

| Model | Val accuracy | Val ROC-AUC | Val log loss | Test accuracy | Test ROC-AUC | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Player–set R-GCN** | 0.7383 ± 0.0022 | 0.8333 ± 0.0011 | 0.5003 ± 0.0017 | 0.7611 ± 0.0063 | 0.8389 ± 0.0018 | 0.4991 ± 0.0027 |

Validation cold-start log loss (n=24): 0.7011 ± 0.0031.

Reference models (single seed):

| Model | Val accuracy | Val ROC-AUC | Val log loss | Test accuracy | Test ROC-AUC | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R-GCN | 0.7358 | 0.8236 | 0.5154 | 0.7417 | 0.8376 | 0.4988 |
| GraphSAGE | 0.7441 | 0.8225 | 0.5147 | 0.7667 | 0.8447 | 0.4892 |
| Elo | 0.7234 | 0.8034 | 0.5427 | 0.7278 | 0.8257 | 0.5186 |
| Tabular logistic | 0.7261 | 0.8167 | 0.5193 | 0.7417 | 0.8348 | 0.5013 |
| Cumulative games won | 0.4772 | 0.4564 | 0.7134 | 0.4944 | 0.4864 | 0.7087 |

Reference rows are copied from `reports/metrics.json` (seed 42 run of `python -m utcgnn train`).
<!-- results:hetero_rgcn:end -->
