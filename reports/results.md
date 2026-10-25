# Supplied-data run

Default seed 42, 60 epochs; best validation log loss selects the epoch.

3,518 source sets, 41 flagged sets excluded, 3,477 used. Train: 2,394 sets; validation: 723 sets; test: 360 sets.

| Model | Test accuracy | Test ROC-AUC | Test log loss | Selected epoch |
| --- | ---: | ---: | ---: | ---: |
| rgcn | 0.7417 | 0.8376 | 0.4988 | 55 |
| sage | 0.7667 | 0.8447 | 0.4892 | 52 |
| elo | 0.7278 | 0.8257 | 0.5186 | — |
| tabular | 0.7417 | 0.8348 | 0.5013 | — |
| historical_games | 0.4944 | 0.4864 | 0.7087 | — |

These are online, session-causal test predictions: prior test-session results update history, while learned weights stay fixed. All results come from one seed and one newest-season test; they do not establish a statistically reliable GNN advantage.

The test has no cold-start sets. Validation/train cold-start metrics remain in metrics.json. Official standings and attendance awards were not provided, so official Spearman/top-four metrics are unavailable.

Local artifacts under artifacts/main include checkpoints, traces, per-set predictions and retrospective contextual player-strength rankings. artifacts/graph contains the event graph and typed projections. Inputs and these detailed artifacts are ignored by Git.
