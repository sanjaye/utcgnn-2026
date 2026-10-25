Place `morphed_players.xlsx` and `morphed_matches.csv` here. The provided copies
are installed locally and ignored by Git. Source data and generated artifacts
are excluded from commits; the repository contains code and aggregate results.

Player schema: unique `player_id`, `pseudonym`. `source_count` and `source_files`
are provenance only and never predictive features.

Match schema: unique `match_id`, `season`, ISO `timestamp`, `court`, `set_number`,
`team_a_1`, `team_a_2`, `team_b_1`, `team_b_2` (registered IDs), binary
`team_a_win`, nonnegative integer `team_a_score`, `team_b_score`.
Optional `validation_flags` controls default exclusion. Other source columns
are retained in the input but never model inputs. Scores/outcomes update history
only after the entire local calendar session. Dates come from the local date
portion of timestamp, preserving UTC's session day across timezone offsets.
