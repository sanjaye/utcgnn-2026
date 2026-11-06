# October 30, 2026 predictions versus actual results

The input lineups came from `UTC_Current_OCT30.xlsx`, tab `103026`. Actuals
were read separately from `ACTUAL_103026` after the prediction files were saved.
All 8 courts, 32 players, and 24 sets were matched by court and team membership,
allowing different set order and A/B orientation. All actual and predicted
scores were valid tennis set scores. Source weekly totals also reconciled.

The existing R-GCN checkpoint was reused without retraining; its model-selection
period ends June 27, 2026. Historical state used 3,498 accepted sets through
October 17, including 21 sets imported from `101726`. Tab `102326` has no
completed court results. The separate score classifier was fitted using the
same earlier history and conditioned on the R-GCN predicted winner. No October
30 outcomes or standings were used as prediction inputs.

| Measure | Result |
| --- | ---: |
| Correct winners | 19 / 24 |
| Winner accuracy | 79.17% |
| Incorrect winners | 5 / 24 |
| Exact team-aligned score matches | 6 / 24 (25.00%) |
| Team score mean absolute error | 1.2708 games |
| Team score root mean squared error | 1.8985 games |
| Winner log loss | 0.4208 |
| Winner Brier score | 0.1407 |
| Winner ROC-AUC | 0.8672 |
| Player total games mean absolute error | 2.6875 games |
| Player total wins mean absolute error | 0.5000 sets |

Team score errors average over 48 team scores (two per set); an exact score
requires both team scores to match, including the winning team. Player errors
average over 32 players and cover their three played sets, excluding attendance
and sit-out awards. Exact-score confidence in the prediction file is conditional
on the predicted winner; it is not an unconditional exact-score probability.

| Court | Correct winners | Exact scores |
| --- | ---: | ---: |
| 1 | 3 / 3 | 0 / 3 |
| 2 | 2 / 3 | 0 / 3 |
| 3 | 3 / 3 | 2 / 3 |
| 4 | 2 / 3 | 1 / 3 |
| 5 | 2 / 3 | 0 / 3 |
| 6 | 3 / 3 | 2 / 3 |
| 7 | 1 / 3 | 0 / 3 |
| 8 | 3 / 3 | 1 / 3 |

Winner predictions performed better on this session than assigning 50% to each
team: log loss 0.4208 versus 0.6931, and Brier score 0.1407 versus 0.2500
(lower is better). Exact-score accuracy was substantially lower than winner
accuracy. Court 7 accounted for two of the five winner errors. This is a small,
single-session evaluation with shared players; it does not establish long-run
accuracy, calibration, or an advantage over trained Elo/logistic baselines.

Reproduce with the four commands in the README. Detailed predictions, team-aligned
actuals, player comparisons, per-court counts, and machine-readable metrics are
saved under `artifacts/oct30_evaluation/`.
