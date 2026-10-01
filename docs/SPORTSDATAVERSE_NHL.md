# SportsDataverse NHL goal-scorer pipeline

## Purpose

Build a reproducible pregame skater-game dataset for NHL goal-scorer research using the SportsDataverse/fastRhockey data ecosystem.

## Inputs

The adapter loads:

- NHL play-by-play;
- player boxscores;
- shot events and expected goals when available;
- on-ice player IDs used to derive historical deployment overlap.

Install the optional integration with:

```bash
uv sync --extra sports
```

Build a dataset with:

```bash
uv run python scripts/build_goal_scorer_dataset.py \
  --seasons 2025 \
  --recent-games 10 \
  --min-history-games 3 \
  --output workspace/nhl_goal_scorer_features.csv
```

## Pregame-only rule

Every rolling predictor is shifted by one match before its rolling window is calculated. Current-game goals are labels only and never predictors.

The Linemate Boost follows the same rule. On-ice overlap observed during match N is stored as an observation, but only becomes a feature for match N+1. Current-game deployment is never used to predict the same game.

Key generated fields include:

- `recent_goals_*`;
- `recent_shots_per60_*`;
- `recent_ixg_per60_*`;
- `recent_pp_shot_share_*`;
- `recent_toi_avg_*`;
- `top_linemate_event_share`;
- `top_linemate_recent_ixg_per60`;
- `linemate_boost_score`;
- `label_goal`;
- `history_ready`.

## Validation

`tests/test_sportsdataverse_features.py` explicitly verifies that current-match goals and current-match linemate deployment do not leak into pregame features.

`.github/workflows/sportsdataverse-smoke.yml` performs a real SportsDataverse load and creates a dataset artifact. The first live smoke run on the 2025 NHL season produced 55,913 player-game rows, including 52,827 history-ready rows and 7,530 positive goal labels.

## Economic claims

This data pipeline does not create ROI, CLV or betting-profit evidence by itself. The existing project rule remains: economic claims require genuine historical market prices valid at the prediction timestamp.
