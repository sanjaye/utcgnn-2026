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

Put the two inputs under `data/` (already done locally). From the repository:

```bash
/workspace/utcgnn-venv/bin/python -m utcgnn profile --export-graph artifacts/graph
/workspace/utcgnn-venv/bin/python -m unittest discover -s tests -v
/workspace/utcgnn-venv/bin/python -m utcgnn train --epochs 60
# Optional: additional expanding-window tests, starting with three seasons.
/workspace/utcgnn-venv/bin/python -m utcgnn train --epochs 60 --rolling --output artifacts/rolling
/workspace/utcgnn-venv/bin/python -m utcgnn predict --date 2026-10-17 \
  --team-a P0002 P0074 --team-b P0009 P0063
# Predict all explicit matches (three pairings per court).
/workspace/utcgnn-venv/bin/python -m utcgnn predict-day --date 2026-11-07 \
  --groups data/matchday_groups.csv
# Add a separate exact-score prediction file without changing match predictions.
/workspace/utcgnn-venv/bin/python -m utcgnn.score_prediction
```

Use `.venv/bin/python` instead of the environment path when installed elsewhere.
Global input options precede the subcommand, e.g.
`python -m utcgnn --matches alternate.csv --include-flagged train`.
The optional flagged-row sensitivity run should use its own output directory.

For `predict-day`, create a CSV with the columns
`court,pairing,team_a_1,team_a_2,team_b_1,team_b_2`, one row per match. Players
are divided into as many four-player courts as possible; up to three players
may remain unassigned. The file must contain three rows per court, with each
court's rows covering all possible two-versus-two pairings for its four players.
Predictions are written to
`artifacts/matchday_predictions.csv` by default, with both team win probabilities
and each player's pseudonym and actual name from `data/players.csv`. Override
the name registry with `--names` or the output path with `--output`. Any
unassigned players and their names are written to
`artifacts/matchday_unassigned_players.csv` by default. As with single-match
prediction, the match date must be after the model-selection period, and only
earlier match history is used.

For exact set scores, run `python -m utcgnn.score_prediction`. It reads
`artifacts/matchday_predictions.csv`, uses matches strictly before that file's
date to train a separate symmetry-aware scoreline classifier, and writes
`artifacts/matchday_score_predictions.csv` plus a training summary JSON. This
score output is separate from the win-probability file. The companion
`artifacts/matchday_score_predictions.xlsx` workbook has a `Match predictions`
tab and a `Player totals` tab. Player totals sum that player's team's predicted
points across their three pairing matches and include predicted wins/losses.
Scorelines are learned
from historical results and currently allow 6-0 through 6-4 and 7-5 or 7-6,
including 6-0 because it occurs in the source results. The output includes a
most-likely score, its probability, expected team points, and winner/loser
points.

## Weekly workbook predictions and actual-results comparison

Use the reusable `utcgnn.workbook` commands to prepare inputs and evaluate
results. No date-specific Python script is needed. The workbook layout has
headers on row 4, player names in column B, court numbers in G, individual
set scores in I:K, and weekly games/sets won in L:M. The player table ends
at its first blank name. Blank courts and court 0 are excluded. Each assigned
court must contain four registered players.

The following PowerShell commands reproduce the October 30 run on this Windows
checkout. Substitute your Python interpreter on other systems.

```powershell
# 1. Convert 103026 to the existing predict-day CSV format and prepare history.
.venv312/Scripts/python.exe -m utcgnn.workbook prepare --workbook data/UTC_Current_OCT30.xlsx --sheet 103026 --date 2026-10-30 --output artifacts/oct30_evaluation

# 2. Use the existing trained R-GCN to predict winners.
.venv312/Scripts/python.exe -m utcgnn --matches artifacts/oct30_evaluation/history.csv predict-day --date 2026-10-30 --groups artifacts/oct30_evaluation/court_pairings.csv --output artifacts/oct30_evaluation/match_predictions.csv --unassigned-output artifacts/oct30_evaluation/unassigned_players.csv

# 3. Fit the score classifier on the same earlier history and predict legal scores.
.venv312/Scripts/python.exe -m utcgnn.score_prediction --predictions artifacts/oct30_evaluation/match_predictions.csv --matches artifacts/oct30_evaluation/history.csv --output artifacts/oct30_evaluation/score_predictions.csv --report artifacts/oct30_evaluation/score_report.json --workbook artifacts/oct30_evaluation/score_predictions.xlsx

# 4. Only after prediction, compare against ACTUAL_103026.
.venv312/Scripts/python.exe -m utcgnn.workbook evaluate --workbook data/UTC_Current_OCT30.xlsx --sheet ACTUAL_103026 --date 2026-10-30 --predictions artifacts/oct30_evaluation/score_predictions.csv --output artifacts/oct30_evaluation
```

`prepare` starts with `data/morphed_matches.csv` (override with `--matches`),
removes target-date/future rows, and adds completed date-named workbook tabs
strictly after the latest existing history date and before the prediction date.
It ignores `ACTUAL_*` tabs. For this run it adds 21 October 17 sets; `102326`
has no assigned courts. The result contains 3,498 accepted historical sets
through October 17, with the existing 41 flagged rows still excluded by the
prediction commands. It does not replace dates already covered by the base
history. Existing history files and the source workbook are unchanged.

The R-GCN checkpoint is reused without retraining. The separate score model is
fitted on earlier history and conditioned on the R-GCN winner. Legal scores are
6-0 through 6-4, 7-5 and 7-6, in either team orientation. `pairing` enumerates
the three team splits; it does not assert the actual order of play.

`evaluate` reconstructs actual teams from the two players sharing each set's
score, checks legal scores and weekly totals, and matches by court and the two
unordered teams. This handles different set ordering, partner ordering, and
reversed team A/B orientation. Missing/duplicate matchups or inconsistent
results raise an error rather than silently dropping sets. Actual results
never enter the prediction history or score-model fitting for this date.

Outputs under `artifacts/oct30_evaluation/` include `court_pairings.csv`,
`history.csv`, `match_predictions.csv`, `score_predictions.csv`, the score
workbook, `actual_matches.csv`, `match_comparison.csv`, `player_comparison.csv`,
`court_performance.csv`, and `performance.json`. The match comparison retains
probabilities, predicted and actual scores, correctness, and score errors.
The player comparison includes predicted/actual games and wins, excluding
attendance and sit-out awards.

The October 30 run predicted **19/24 winners (79.17%)**, **6/24 exact scores
(25.00%)**, and had **1.2708 games mean absolute error per team per set**.
Log loss was **0.4208**, Brier score **0.1407**, and ROC-AUC **0.8672**.
These describe one session, not evidence of general superiority. See
[the aggregate results](reports/oct30_results.md) for court breakdowns and
metric definitions. For another week, change the workbook, lineup sheet,
actual sheet, date, and output directory in these commands.

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

Team embeddings sum the two player embeddings. The proposal's linear decoder
uses `wᵀ(z_A - z_B)`, fixing its intercept to zero. This enforces
partner-order invariance and P(A beats B) = 1 - P(B beats A). Training uses BCE,
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
