# UTC temporal doubles graph

Python implementation of the supplied **Player Strength and Doubles Outcome
Prediction in a Tennis Ladder Using Graph Neural Networks** project proposal.
This is a project proposal, not an experimental result or a complete published
architecture specification. The implementation makes the choices below explicit.

## Run

Python 3.12, CPU. In this environment the interpreter is
`/workspace/utcgnn-venv/bin/python`. To reproduce elsewhere:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On macOS, install `torch==2.9.1` without the `+cpu` suffix (that suffix only exists
for Linux and Windows wheels). `xgboost` needs an OpenMP runtime there; torch ships
one and is imported first, so no separate `brew install libomp` is needed.

Put the two inputs under `data/` (already done locally). From the repository:

```bash
/workspace/utcgnn-venv/bin/python -m utcgnn profile --export-graph artifacts/graph
/workspace/utcgnn-venv/bin/python -m unittest discover -s tests -v
/workspace/utcgnn-venv/bin/python -m utcgnn train --epochs 60
# Optional: additional expanding-window tests, starting with three seasons.
/workspace/utcgnn-venv/bin/python -m utcgnn train --epochs 60 --rolling --output artifacts/rolling
# Concatenation team decoder instead of summing partners (writes artifacts/main_concat).
/workspace/utcgnn-venv/bin/python -m utcgnn train --decoder concat
# Average-pooling team decoder with a ReLU layer (writes artifacts/main_mean).
/workspace/utcgnn-venv/bin/python -m utcgnn train --decoder mean
# Proposal's player–set graph in PyTorch Geometric (R-GCN and GraphSAGE, 5 seeds);
# runs on a GPU when available. Colab steps are in reports/experiments.md.
/workspace/utcgnn-venv/bin/python -m utcgnn train-hetero
# XGBoost baseline on the same features and splits; see reports/experiments.md.
/workspace/utcgnn-venv/bin/python -m utcgnn xgboost
/workspace/utcgnn-venv/bin/python -m utcgnn predict --date 2026-10-17 \
  --team-a P0002 P0074 --team-b P0009 P0063
# Predict the 24 explicit matches (three pairings per court).
/workspace/utcgnn-venv/bin/python -m utcgnn predict-day --date 2026-11-07 \
  --groups data/matchday_groups.csv
```

Use `.venv/bin/python` instead of the environment path when installed elsewhere.
Global input options precede the subcommand, e.g.
`python -m utcgnn --matches alternate.csv --include-flagged train`.
The optional flagged-row sensitivity run should use its own output directory.

For `predict-day`, create a CSV with the columns
`court,pairing,team_a_1,team_a_2,team_b_1,team_b_2`, one row per match. Use
registered player IDs; the file must contain eight courts, three matches per
court, and 32 distinct players. Each court's three rows must cover all possible
two-versus-two pairings for its four players. Predictions are written to
`artifacts/matchday_predictions.csv` by default, with both team win probabilities
and each player's pseudonym and actual name from `data/players.csv`. Override
the name registry with `--names` or the output path with `--output`. As with
single-match prediction, the match date must be after the model-selection
period, and only earlier match history is used.

## Graph and model

Player nodes are joined to set-event nodes by four team-labeled participation
edges. Each event projects into **4 directed teammate edges** and **8 directed
opponent edges**. Graph exports retain event IDs and dates so four-player
groupings can be reconstructed. Labels and scores in `events.csv` are targets
and retrospective observations; never use them as target-event input features.

The primary encoder is a two-layer R-GCN, with one transformation per relation
and a self transformation, 16 hidden units and ReLU activations. Each relation
aggregates a count-weighted historical neighbor mean. It uses the player-player
projection; event nodes are preserved in the data model/export, not a separate
learned event encoder. All previous sessions contribute with no time decay.
The GraphSAGE comparison uses unique, untyped neighbors and mean aggregation.
Dense PyTorch operations are sufficient for 113 players; PyG is unnecessary.

Pre-session node features: log sets played, Laplace-smoothed set win rate,
smoothed games share, centered Elo, and log games won per set. Normalization
statistics are fit only on training snapshots. Registry metadata, source counts,
end-of-season totals, pseudonyms and future outcomes are excluded. There are no
learned player-ID embeddings. Players with no history get the same defined prior
(Elo 1500, win/game rates 0.5, zero sets/games), enabling registered cold starts.

By default, team embeddings sum the two player embeddings. The proposal's linear decoder
uses `wᵀ(z_A - z_B)`, fixing its intercept to zero. This enforces
partner-order invariance and P(A beats B) = 1 - P(B beats A). `train --decoder concat`
instead concatenates the two partners' final embeddings (2d), applies a linear
layer back to d, then a ReLU. The ReLU makes the team embedding nonlinear in the pair,
so it can represent interactions between partners. Swapping teams still gives
P(B beats A) = 1 - P(A beats B), but the two partner slots get separate weights, so
partner order can change the prediction. `train --decoder mean` averages the two
partners' embeddings, then applies a d-to-d linear layer and a ReLU. Partner order
cannot matter, but the pair enters only through its average. Checkpoints record
their decoder. Training uses BCE,
Adam (0.005), weight decay 0.001, and full training-session gradient accumulation.
Seed 42 and one CPU thread make the run reproducible on this software stack.
Select the epoch by validation log loss; test labels never select weights.

## Temporal protocol and baselines

Default split: 2023-Jul-Dec through 2025-Jul-Dec train, 2026-Jan-Jun validation,
2026-Jul-Dec test. Seasons are ordered by actual dates and overlapping seasons
are rejected. Because every record has `session_date` precision, all sets on a
date use **strictly earlier dates**, including all courts and set numbers. No
within-day ordering is inferred. Elo updates for a day use pre-day ratings.

Evaluation is **online/prequential**: weights stay fixed, but observed results
from an earlier validation/test session update history for later sessions.
This is not a frozen-history prediction of an entire unseen season. New-date
inference rejects dates at or before model-selection data. It also excludes
all input matches on the requested date, even when their outcomes are available.

Baselines share these splits and histories: doubles Elo (mean team rating,
K=32 per player per set), logistic regression on team feature differences, and
a calibrated cumulative-games-won difference. Logistic baselines are fit only
on training data with swapped-team augmentation. The games baseline is a
performance proxy, **not the official ladder-points baseline**: neither official
points nor separate attendance awards are supplied.

The `train-hetero` command implements the project proposal's graph with PyTorch
Geometric (`utcgnn/hetero_data.py`, `hetero_models.py`, `hetero_experiment.py`). Each
match date gets its own graph. Every earlier set is a node with the winner's and loser's
games, weeks before that date, week of season and court (a ranking band: court 1 holds
the top four players). Players link to these sets by won/lost edges in both directions.
The date's own sets are target nodes with only week of season and court, receiving
side A/B edges from their players and sending nothing. Set number is not used, since
play order on a court is arbitrary. Four layers of `RGCNConv` (one weight matrix per
edge type), or one `SAGEConv` per edge type combined with `HeteroConv`, produce the
target embedding. Each target also exists as a copy with the sides swapped, and the
logit is `f(h) - f(h_swapped)`, so P(A beats B) = 1 - P(B beats A) and partner order
cannot matter. Training is full-batch with validation-selected epochs, over five seeds
by default; `--device auto` uses a GPU when available. Handedness, availability,
self-rating and weather from the proposal are not in the data.

The separate `xgboost` command adds a gradient-boosted tree baseline on the same
pre-session player features, splits and online protocol. Each team is summarized by
the sum, min and max of its players' features, plus the team difference of sums.
Hyperparameters are tuned by randomized search with 3-fold CV on the training seasons
only, grouped by session date. Training uses swapped-team augmentation, and
predictions average both team orders.

All rows with nonempty `validation_flags` are excluded by default (41/3,518
in the supplied data). This deliberately conservative policy also excludes
potentially usable sets with weekly aggregate discrepancies. Structural errors
(unknown/duplicate players, duplicate IDs, missing fields, invalid scores/labels)
fail validation before exclusion; `--include-flagged` never bypasses them.

## Outputs and limits

`artifacts/main/metrics.json` records accuracy, ROC-AUC and log loss for train,
validation and test, plus cold-start and established-player subsets. A set is
cold-start if any participant has no earlier accepted set. Single-class AUC
and empty-subset metrics are null. Per-split prediction CSVs support auditing.
Training traces, R-GCN/GraphSAGE checkpoints and player-strength CSVs are saved.
`artifacts/graph/` contains player, event, participation and typed-edge CSVs.
`train --decoder concat` writes the same files to `artifacts/main_concat/`, so it
never replaces the main checkpoints. `xgboost` writes metrics, per-split predictions,
CV results and feature importance to `artifacts/xgboost/`. `train-hetero` writes one
checkpoint and training trace per seed, plus metrics (per seed and mean ± std) and
predictions, to `artifacts/hetero_rgcn/` and `artifacts/hetero_sage/`.

Model inputs are built in memory on each run. `python -m utcgnn sample-inputs` saves a
readable sample, the inputs for 10 sets of the first validation date for both graphs
(raw and scaled node features, relation weights, past-set edges and the full `.pt`
tensors), to `data/sample_model_inputs/`. Its README describes each file;
`--n`, `--split` and `--output` change the sample.

`train`, `train-hetero` and `xgboost` also update `reports/experiments.md`, a results log with one
section per experiment compared against the main run (`--no-log` skips this). That
file also explains how to run the experiments and add new ones.

Player-strength logits compare each player plus the mean active-player partner
against a team of two mean active-player embeddings. This is an explicit
neutral-partner heuristic, not a fully opponent-adjusted latent skill estimate.
Rankings use history through the final observed session and are retrospective;
they are not inputs to the historical predictions. `has_history` identifies
registry-only players whose rankings should not be treated as established skill.

Official standings and separate attendance awards are absent. Official ranking
Spearman correlation, top-four overlap, and ladder-points comparison are therefore
unavailable; `metrics.json` states this instead of inventing targets. The small,
observational ladder dataset does not establish that a GNN beats the baselines.
Repeat seeds and optional rolling folds before drawing scientific conclusions.

Inputs, exports, checkpoints and per-player/per-set artifacts stay ignored by
Git. No Git commit or remote push is performed automatically. Aggregate results
in `reports/` are reproducible run evidence.

References: Schlichtkrull et al. (2018), *Modeling Relational Data with Graph
Convolutional Networks*; Hamilton et al. (2017), *Inductive Representation
Learning on Large Graphs*. The supplied DOCX is used as the project design
reference; its prose is not treated as operational instructions.
