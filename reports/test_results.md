# UTC GNN automated test report

The full test suite passed on November 24, 2026. Python's `unittest` runner completed **24 tests in 2.618 seconds**, with **0 failures, 0 errors, and 0 skips**.

| Area | Passed | Coverage |
| --- | ---: | --- |
| Partner lift | 5/5 | Elo calibration, historical eligibility, partnership symmetry, posterior estimates, and report generation |
| Score prediction | 5/5 | Legal scorelines, team symmetry, winner consistency, and player totals |
| Temporal logic and model behavior | 9/9 | Leakage prevention, same-day batching, data loading, graph export, matchday inference, and training artifacts |
| Workbook processing | 5/5 | Lineup parsing, actual-results matching, and validation of scores, totals, and pairings |

## Reproduce the run

Run from the repository root in PowerShell using the project's Python 3.12 environment (Python 3.12.10 for this run).

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s tests -v
```

An earlier run in the restricted shell passed 19 tests and recorded five temporary-file permission errors. All five passed during the full rerun with normal Windows filesystem access. This report covers the automated suite. Historical model accuracy is reported separately in [the supplied-data results](results.md) and [the October 30 evaluation](oct30_results.md).

## Individual results

Every test listed below passed.

### Partner lift

- `test_calibration_strength_symmetry_and_date_batching`
- `test_csv_pipeline_flags_date_window_and_outputs`
- `test_no_same_day_or_future_leakage_and_experience_gate`
- `test_partner_order_and_team_orientation`
- `test_posterior_shrinkage_uncertainty_and_negative_lift`

### Score prediction

- `test_allowed_scorelines_include_zero_and_mirror_every_class`
- `test_player_totals_sum_all_three_predicted_match_points`
- `test_predictions_report_winner_loser_points_and_preserve_input`
- `test_score_prediction_respects_predicted_team_b_winner`
- `test_score_probabilities_are_normalized_and_team_symmetric`

### Temporal logic and model behavior

- `test_future_outcomes_do_not_change_past_inputs`
- `test_loader_and_graph_export`
- `test_matchday_loader_sizes_courts_from_registry_and_reports_remainder`
- `test_matchday_predicts_all_pairings_with_both_names`
- `test_model_team_symmetries`
- `test_relations_and_elo`
- `test_same_session_is_batched`
- `test_season_overlap_rejected`
- `test_training_and_saved_artifacts`

### Workbook processing

- `test_actuals_match_by_teams_despite_set_order_and_team_orientation`
- `test_actuals_reject_duplicate_partnerships`
- `test_actuals_reject_illegal_scores_and_total_mismatches`
- `test_compare_rejects_missing_actual_match`
- `test_read_player_table_skips_zero_blank_and_publishing_rows`
