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
# Task 2: team A's share of games (MAE in games) and standings (Spearman)
.venv/bin/python -m utcgnn score-baselines
.venv/bin/python -m utcgnn xgboost --task score
.venv/bin/python -m utcgnn train-hetero --task score
.venv/bin/python -m utcgnn train-hetero --models gat                 # GATv2 attention, Task 1
.venv/bin/python -m utcgnn train-hetero --models gat --task score    # GATv2 attention, Task 2
```

| Command | Module | What it tests | Options |
| --- | --- | --- | --- |
| `train --decoder concat` | `utcgnn/models.py` | GNN team decoder that concatenates the two partners' final embeddings (2d) and applies a linear layer back to d followed by a ReLU, in place of summing them. `--decoder sum` is the original. | `--decoder` (`sum`, `concat`, `mean`), `--output` (`artifacts/main` for sum, `artifacts/main_<decoder>` otherwise), `--log`, `--no-log`, plus the usual `train` options |
| `train --decoder mean` | `utcgnn/models.py` | GNN team decoder that averages the two partners' final embeddings, then applies a d-to-d linear layer and a ReLU. Partner order cannot matter. | Same as above |
| `train-hetero` | `utcgnn/hetero_data.py`, `utcgnn/hetero_models.py`, `utcgnn/hetero_experiment.py` | The project proposal's architecture in PyTorch Geometric: one graph per match date with player, past-set and target nodes; past sets linked to players by won/lost edges, targets by side A/B edges. R-GCN (`RGCNConv`) and GraphSAGE (one `SAGEConv` per edge type), swap readout so P(A wins) = 1 − P(B wins), reported as mean ± std over seeds | `--models` (`rgcn sage`; also `gat`: one `GATv2Conv` per edge type with a learned self term), `--heads` (4, for `gat`), `--task` (`win`, or `score` for Task 2), `--seeds` (5, starting at `--seed` 42), `--epochs` (200), `--hidden` (32), `--layers` (4), `--dropout` (0.1), `--device` (`auto`), `--output` (`artifacts/hetero_<model>`), `--log`, `--no-log` |
| `sample-inputs` | `utcgnn/sample_inputs.py` | Not an experiment: saves the preprocessed inputs of both graphs for a few sets, to inspect what the models receive. See `data/sample_model_inputs/README.md` | `--n` (10), `--split` (`validation`), `--output` (`data/sample_model_inputs`) |
| `xgboost` | `utcgnn/xgboost_baseline.py` | Gradient-boosted trees on the 5 pre-session player features, tuned by CV on the training seasons | `--n-iter` (60), `--cv-folds` (3), `--seed` (42), `--output` (`artifacts/xgboost`), `--log` (this file), `--no-log` |
| `score-baselines` | `utcgnn/score.py` | Task 2: team A's share of games with a constant 0.5 and a binomial logistic regression (games-weighted, no intercept, swapped teams) on the team feature differences. Also writes per-player expected vs actual standings | `--output` (`artifacts/score_baselines`), `--log`, `--no-log` |
| `xgboost --task score` | `utcgnn/xgboost_baseline.py` | Task 2 XGBoost: `reg:logistic` on the share, games-weighted, same 35 features and search space, tuned by MAE in games with date-grouped CV on train | `--task` (`win`), `--output` (`artifacts/xgboost_score`), plus the `xgboost` options |
| `train-hetero --task score` | `utcgnn/hetero_experiment.py` | Task 2 with the unchanged player–set R-GCN and GraphSAGE: games-weighted BCE on the share, epoch by validation MAE in games | `--task` (`win`), `--output` (`artifacts/hetero_<model>_score`), plus the `train-hetero` options |

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
| Player–set GraphSAGE | 0.7364 ± 0.0050 | 0.8336 ± 0.0015 | 0.4986 ± 0.0024 | 0.7606 ± 0.0054 | 0.8400 ± 0.0028 | 0.4966 ± 0.0033 |
| Player–set GATv2 | 0.7364 ± 0.0033 | 0.8336 ± 0.0010 | 0.4987 ± 0.0008 | 0.7661 ± 0.0021 | 0.8402 ± 0.0020 | 0.4964 ± 0.0017 |

Other player–set GNN rows come from `artifacts/hetero_<model>/metrics.json` when available. Validation cold-start log loss (n=24): 0.7011 ± 0.0031.

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

<!-- results:hetero_sage:start -->
### Player–set GraphSAGE (`utcgnn/hetero_models.py`)

Last run: 2026-11-09 23:24 · `python -m utcgnn train-hetero` · cpu · 181 s · outputs in `artifacts/hetero_sage`

- **Graph:** one PyG graph per match date with player, past-set and target nodes. Past sets are linked to players by won/lost edges; targets receive side A/B edges and carry no result. Past-set features: winner_games, loser_games, weeks_before, week_of_season, court. Target features: week_of_season, court.
- **Model:** 4 per-edge-type SAGEConv layers, hidden size 32, dropout 0.1; swap readout f(h) − f(h̃), so P(A wins) = 1 − P(B wins).
- **Training:** full batch, Adam (learning rate 0.005, weight decay 0.001), up to 200 epochs, epoch chosen by validation log loss. Seeds 42, 43, 44, 45, 46; selected epochs 20, 36, 31, 34, 40.
- **Data:** 2,394 train, 723 validation, 360 test sets.

Mean ± standard deviation over 5 seeds:

| Model | Val accuracy | Val ROC-AUC | Val log loss | Test accuracy | Test ROC-AUC | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Player–set GraphSAGE** | 0.7364 ± 0.0050 | 0.8336 ± 0.0015 | 0.4986 ± 0.0024 | 0.7606 ± 0.0054 | 0.8400 ± 0.0028 | 0.4966 ± 0.0033 |
| Player–set R-GCN | 0.7383 ± 0.0022 | 0.8333 ± 0.0011 | 0.5003 ± 0.0017 | 0.7611 ± 0.0063 | 0.8389 ± 0.0018 | 0.4991 ± 0.0027 |
| Player–set GATv2 | 0.7364 ± 0.0033 | 0.8336 ± 0.0010 | 0.4987 ± 0.0008 | 0.7661 ± 0.0021 | 0.8402 ± 0.0020 | 0.4964 ± 0.0017 |

Other player–set GNN rows come from `artifacts/hetero_<model>/metrics.json` when available. Validation cold-start log loss (n=24): 0.7108 ± 0.0128.

Reference models (single seed):

| Model | Val accuracy | Val ROC-AUC | Val log loss | Test accuracy | Test ROC-AUC | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R-GCN | 0.7358 | 0.8236 | 0.5154 | 0.7417 | 0.8376 | 0.4988 |
| GraphSAGE | 0.7441 | 0.8225 | 0.5147 | 0.7667 | 0.8447 | 0.4892 |
| Elo | 0.7234 | 0.8034 | 0.5427 | 0.7278 | 0.8257 | 0.5186 |
| Tabular logistic | 0.7261 | 0.8167 | 0.5193 | 0.7417 | 0.8348 | 0.5013 |
| Cumulative games won | 0.4772 | 0.4564 | 0.7134 | 0.4944 | 0.4864 | 0.7087 |

Reference rows are copied from `reports/metrics.json` (seed 42 run of `python -m utcgnn train`).
<!-- results:hetero_sage:end -->

<!-- results:hetero_gat:start -->
### Player–set GATv2 (`utcgnn/hetero_models.py`)

Last run: 2026-11-25 17:35 · `python -m utcgnn train-hetero --models gat` · cpu · 1388 s · outputs in `artifacts/hetero_gat`

- **Graph:** one PyG graph per match date with player, past-set and target nodes. Past sets are linked to players by won/lost edges; targets receive side A/B edges and carry no result. Past-set features: winner_games, loser_games, weeks_before, week_of_season, court. Target features: week_of_season, court.
- **Model:** 4 per-edge-type GATv2Conv layers (4 attention heads averaged, a separate attention per edge type, a learned self term per node type, no attention dropout), hidden size 32, dropout 0.1; swap readout f(h) − f(h̃), so P(A wins) = 1 − P(B wins).
- **Training:** full batch, Adam (learning rate 0.005, weight decay 0.001), up to 200 epochs, epoch chosen by validation log loss. Seeds 42, 43, 44, 45, 46; selected epochs 48, 57, 38, 77, 84.
- **Data:** 2,394 train, 723 validation, 360 test sets.

Mean ± standard deviation over 5 seeds:

| Model | Val accuracy | Val ROC-AUC | Val log loss | Test accuracy | Test ROC-AUC | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Player–set GATv2** | 0.7364 ± 0.0033 | 0.8336 ± 0.0010 | 0.4987 ± 0.0008 | 0.7661 ± 0.0021 | 0.8402 ± 0.0020 | 0.4964 ± 0.0017 |
| Player–set R-GCN | 0.7383 ± 0.0022 | 0.8333 ± 0.0011 | 0.5003 ± 0.0017 | 0.7611 ± 0.0063 | 0.8389 ± 0.0018 | 0.4991 ± 0.0027 |
| Player–set GraphSAGE | 0.7364 ± 0.0050 | 0.8336 ± 0.0015 | 0.4986 ± 0.0024 | 0.7606 ± 0.0054 | 0.8400 ± 0.0028 | 0.4966 ± 0.0033 |

Other player–set GNN rows come from `artifacts/hetero_<model>/metrics.json` when available. Validation cold-start log loss (n=24): 0.7213 ± 0.0207.

Reference models (single seed):

| Model | Val accuracy | Val ROC-AUC | Val log loss | Test accuracy | Test ROC-AUC | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R-GCN | 0.7358 | 0.8236 | 0.5154 | 0.7417 | 0.8376 | 0.4988 |
| GraphSAGE | 0.7441 | 0.8225 | 0.5147 | 0.7667 | 0.8447 | 0.4892 |
| Elo | 0.7234 | 0.8034 | 0.5427 | 0.7278 | 0.8257 | 0.5186 |
| Tabular logistic | 0.7261 | 0.8167 | 0.5193 | 0.7417 | 0.8348 | 0.5013 |
| Cumulative games won | 0.4772 | 0.4564 | 0.7134 | 0.4944 | 0.4864 | 0.7087 |

Reference rows are copied from `reports/metrics.json` (seed 42 run of `python -m utcgnn train`).
<!-- results:hetero_gat:end -->

<!-- results:score_summary:start -->
### Task 2 summary: game share and standings

Predicted share of games for team A. MAE in games is |predicted share − share| × games in the set.
ρ is Spearman correlation between expected points (predicted share × games, summed over a
player's sets) and performance-only standings (actual games won), over players with at least
5 sets in the split: **total** sums over sets; **per set** divides both by the games the
player played, giving expected vs actual share of games, which removes attendance and set length.
GNN rows are mean ± standard deviation over seeds; the others are single runs. Rebuilt after every
Task 2 command from `artifacts/score_baselines/metrics.json`, `artifacts/xgboost_score/metrics.json`, `artifacts/hetero_rgcn_score/metrics.json`, `artifacts/hetero_sage_score/metrics.json`, `artifacts/hetero_gat_score/metrics.json`.

| Model | Val MAE (games) | Val MAE (share) | Val ρ total | Val ρ per set | Test MAE (games) | Test MAE (share) | Test ρ total | Test ρ per set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant 0.5 | 1.6452 | 0.2065 | 0.9591 | — | 1.5764 | 0.1955 | 0.9422 | — |
| Logistic regression (binomial) | 1.2654 | 0.1521 | 0.9812 | 0.8600 | 1.1901 | 0.1397 | 0.9904 | 0.8386 |
| XGBoost | 1.2899 | 0.1556 | 0.9794 | 0.8396 | 1.2050 | 0.1426 | 0.9902 | 0.8452 |
| Player–set R-GCN | 1.2340 ± 0.0011 | 0.1470 ± 0.0006 | 0.9763 ± 0.0006 | 0.8754 ± 0.0045 | 1.1813 ± 0.0052 | 0.1378 ± 0.0004 | 0.9833 ± 0.0010 | 0.8330 ± 0.0047 |
| Player–set GraphSAGE | 1.2344 ± 0.0044 | 0.1468 ± 0.0008 | 0.9757 ± 0.0011 | 0.8667 ± 0.0047 | 1.1879 ± 0.0026 | 0.1379 ± 0.0004 | 0.9828 ± 0.0011 | 0.8407 ± 0.0034 |
| Player–set GATv2 | 1.2412 ± 0.0034 | 0.1484 ± 0.0005 | 0.9780 ± 0.0006 | 0.8659 ± 0.0063 | 1.1905 ± 0.0046 | 0.1387 ± 0.0003 | 0.9862 ± 0.0010 | 0.8450 ± 0.0051 |

Validation has 723 sets and test 360;
68 and 56 players are ranked.
<!-- results:score_summary:end -->

<!-- results:score_baselines:start -->
### Task 2 baselines: constant and logistic regression (`utcgnn/score.py`)

Last run: 2026-11-24 23:27 · `python -m utcgnn score-baselines` · outputs in `artifacts/score_baselines`

- **Target:** team A's share of games, a / (a + b); same splits, history-only features and online protocol as Task 1.
- **Logistic regression:** binomial on games won (each set gives y=1 with weight a and y=0 with weight b), no intercept, swapped-team augmentation, on the team A − team B sums of the 5 pre-session player features. Fit on training seasons only.
- **Constant 0.5:** every set split evenly. Its total-standings ρ shows how much attendance (and set length) alone explains; its per-set ρ is undefined.
- **Data:** 2,394 train, 723 validation, 360 test sets.

| Model | Val MAE (games) | Val MAE (share) | Val ρ total | Val ρ per set | Test MAE (games) | Test MAE (share) | Test ρ total | Test ρ per set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant 0.5 | 1.6452 | 0.2065 | 0.9591 | — | 1.5764 | 0.1955 | 0.9422 | — |
| Logistic regression (binomial) | 1.2654 | 0.1521 | 0.9812 | 0.8600 | 1.1901 | 0.1397 | 0.9904 | 0.8386 |

Logistic regression RMSE (games): validation 1.5138, test 1.4355. Cold-start MAE (games): validation 1.6896 (n=24), test — (n=0).
<!-- results:score_baselines:end -->

<!-- results:score_xgboost:start -->
### Task 2 XGBoost (`utcgnn/xgboost_baseline.py`)

Last run: 2026-11-24 23:27 · `python -m utcgnn xgboost --task score` · outputs in `artifacts/xgboost_score`

- **Target:** team A's share of games. `XGBRegressor(objective='reg:logistic')` on the share, weighted by games in the set; same 35 team features as the Task 1 XGBoost. Teams are swapped in training and predictions average both orientations, so share(B) = 1 − share(A).
- **Tuning:** 60 random candidates from the Task 1 search space, 3-fold CV on training seasons only, grouped by session date, scored by MAE in games. Best CV MAE 1.3223 ± 0.0242 games.
- **Selected hyperparameters:** `n_estimators=303`, `max_depth=5`, `learning_rate=0.01331`, `subsample=0.6406`, `colsample_bytree=0.8899`, `min_child_weight=33.62`, `reg_lambda=0.105`, `gamma=1.284`
- **Top features by gain:** `diff_sum_game_share`, `diff_sum_elo_scaled`, `diff_sum_log_games_per_set`, `diff_sum_smoothed_win_rate`, `team_a_min_log_sets`, `team_b_sum_log_sets`, `team_b_min_log_sets`, `team_b_max_log_games_per_set`

| Model | Val MAE (games) | Val MAE (share) | Val ρ total | Val ρ per set | Test MAE (games) | Test MAE (share) | Test ρ total | Test ρ per set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 1.2899 | 0.1556 | 0.9794 | 0.8396 | 1.2050 | 0.1426 | 0.9902 | 0.8452 |

Train MAE 1.1763 games. RMSE (games): validation 1.5226, test 1.4271. Single seed, single test season.
<!-- results:score_xgboost:end -->

<!-- results:score_hetero_rgcn:start -->
### Task 2 Player–set R-GCN (`utcgnn/hetero_models.py`)

Last run: 2026-11-24 23:32 · `python -m utcgnn train-hetero --task score` · cpu · 271 s · outputs in `artifacts/hetero_rgcn_score`

- **Graph and model:** the Task 1 player–set graph and 4 RGCNConv layers, hidden size 32, dropout 0.1, unchanged. sigmoid(f(h) − f(h̃)) is read as team A's share of games, so share(B) = 1 − share(A).
- **Training:** BCE with the soft label share = a / (a + b), weighted by games in the set (binomial likelihood of the games). Full batch, Adam (learning rate 0.005, weight decay 0.001), up to 200 epochs, epoch chosen by validation MAE in games. Seeds 42, 43, 44, 45, 46; selected epochs 85, 102, 83, 121, 74.
- **Data:** 2,394 train, 723 validation, 360 test sets.

Mean ± standard deviation over 5 seeds:

| Model | Val MAE (games) | Val MAE (share) | Val ρ total | Val ρ per set | Test MAE (games) | Test MAE (share) | Test ρ total | Test ρ per set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Player–set R-GCN | 1.2340 ± 0.0011 | 0.1470 ± 0.0006 | 0.9763 ± 0.0006 | 0.8754 ± 0.0045 | 1.1813 ± 0.0052 | 0.1378 ± 0.0004 | 0.9833 ± 0.0010 | 0.8330 ± 0.0047 |

Validation cold-start MAE (n=24): 1.6615 ± 0.0262 games. Averaging the 5 seeds' predictions gives test MAE 1.1797 games, ρ total 0.9838, ρ per set 0.8342.
<!-- results:score_hetero_rgcn:end -->

<!-- results:score_hetero_sage:start -->
### Task 2 Player–set GraphSAGE (`utcgnn/hetero_models.py`)

Last run: 2026-11-24 23:35 · `python -m utcgnn train-hetero --task score` · cpu · 177 s · outputs in `artifacts/hetero_sage_score`

- **Graph and model:** the Task 1 player–set graph and 4 per-edge-type SAGEConv layers, hidden size 32, dropout 0.1, unchanged. sigmoid(f(h) − f(h̃)) is read as team A's share of games, so share(B) = 1 − share(A).
- **Training:** BCE with the soft label share = a / (a + b), weighted by games in the set (binomial likelihood of the games). Full batch, Adam (learning rate 0.005, weight decay 0.001), up to 200 epochs, epoch chosen by validation MAE in games. Seeds 42, 43, 44, 45, 46; selected epochs 60, 48, 42, 59, 62.
- **Data:** 2,394 train, 723 validation, 360 test sets.

Mean ± standard deviation over 5 seeds:

| Model | Val MAE (games) | Val MAE (share) | Val ρ total | Val ρ per set | Test MAE (games) | Test MAE (share) | Test ρ total | Test ρ per set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Player–set GraphSAGE | 1.2344 ± 0.0044 | 0.1468 ± 0.0008 | 0.9757 ± 0.0011 | 0.8667 ± 0.0047 | 1.1879 ± 0.0026 | 0.1379 ± 0.0004 | 0.9828 ± 0.0011 | 0.8407 ± 0.0034 |

Validation cold-start MAE (n=24): 1.6562 ± 0.0222 games. Averaging the 5 seeds' predictions gives test MAE 1.1861 games, ρ total 0.9836, ρ per set 0.8428.
<!-- results:score_hetero_sage:end -->

<!-- results:score_hetero_gat:start -->
### Task 2 Player–set GATv2 (`utcgnn/hetero_models.py`)

Last run: 2026-11-25 17:57 · `python -m utcgnn train-hetero --models gat --task score` · cpu · 1347 s · outputs in `artifacts/hetero_gat_score`

- **Graph and model:** the Task 1 player–set graph and 4 per-edge-type GATv2Conv layers (4 attention heads averaged, a separate attention per edge type, a learned self term per node type, no attention dropout), hidden size 32, dropout 0.1, unchanged from Task 1. sigmoid(f(h) − f(h̃)) is read as team A's share of games, so share(B) = 1 − share(A).
- **Training:** BCE with the soft label share = a / (a + b), weighted by games in the set (binomial likelihood of the games). Full batch, Adam (learning rate 0.005, weight decay 0.001), up to 200 epochs, epoch chosen by validation MAE in games. Seeds 42, 43, 44, 45, 46; selected epochs 67, 89, 56, 165, 49.
- **Data:** 2,394 train, 723 validation, 360 test sets.

Mean ± standard deviation over 5 seeds:

| Model | Val MAE (games) | Val MAE (share) | Val ρ total | Val ρ per set | Test MAE (games) | Test MAE (share) | Test ρ total | Test ρ per set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Player–set GATv2 | 1.2412 ± 0.0034 | 0.1484 ± 0.0005 | 0.9780 ± 0.0006 | 0.8659 ± 0.0063 | 1.1905 ± 0.0046 | 0.1387 ± 0.0003 | 0.9862 ± 0.0010 | 0.8450 ± 0.0051 |

Validation cold-start MAE (n=24): 1.6416 ± 0.0084 games. Averaging the 5 seeds' predictions gives test MAE 1.1892 games, ρ total 0.9857, ρ per set 0.8448.
<!-- results:score_hetero_gat:end -->
