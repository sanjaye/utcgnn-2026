# Sample model inputs

The preprocessed inputs the GNNs receive, for the first 10 sets of the first validation
date (2026-01-10; 30 sets were played that day). Training builds these in memory on every
run; this folder is a readable snapshot of them. Regenerate with:

```
python -m utcgnn sample-inputs                       # defaults: --n 10 --split validation
python -m utcgnn sample-inputs --n 5 --split test --output data/sample_test_inputs
```

All inputs use only sets from strictly earlier dates. "Scaled" values are standardized with
training-season means and standard deviations (std clamped at 0.1), exactly as in training.

## The sets

`sets.csv`: one row per sampled set. Players by slot, court, the label `team_a_win`
(1 if team A won; used only as the training/evaluation target, never as an input), and
`cold_start` (a player has no earlier sets).

## Player graph (original dense R-GCN / GraphSAGE)

| File | Contents |
|---|---|
| `player_graph_node_features.csv` | The four players of each set: `raw_*` and `scaled_*` versions of the five features (log sets played, smoothed win rate, game share, Elo/400 centered at 1500, log games per set). |
| `player_graph_relations.csv` | For those players, every neighbor with a nonzero row-normalized weight. `teammate_weight` and `opponent_weight` are identical, because every pair meets twice as often as opponents as teammates. |
| `player_graph_snapshot.pt` | Full tensors for the date: `x_raw`, `x_scaled` (113 players × 5), `relations` (2 × 113 × 113), `teams` and `y` for the 10 sets. Load with `torch.load(path)`. |

## Heterogeneous player–set graph (PyG)

| File | Contents |
|---|---|
| `hetero_targets.csv` | Target nodes for the 10 sets: a `real` copy and a `swapped` copy (sides exchanged), their side-A/side-B players, and raw/scaled features (week of season, court). The label is on the real copy only. Target indices m..2m−1 are the swapped copies, with m = 30 sets that day. |
| `hetero_past_set_edges.csv` | Every `won_in` / `lost_in` edge from the sampled sets' players to earlier sets, with that past set's match ID and raw/scaled features (winner games, loser games, weeks before the date, week of season, court). Each edge also exists in reverse (`won_by`, `lost_by`). |
| `hetero_graph.pt` | The complete scaled `HeteroData` the model receives for the date (113 players, 2,394 past sets, 60 targets, 6 edge types), plus the scaling statistics and node-index → ID maps. Load with `torch.load(path, weights_only=False)`; needs `torch_geometric`. |

The data are the morphed (pseudonymized) records in `data/morphed_*`.
