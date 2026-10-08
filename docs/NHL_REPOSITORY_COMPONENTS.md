# Shooting context and locked-pick evaluation

## Source audit and integration decision — 2026-10-09

| Source | Useful idea | Decision |
| --- | --- | --- |
| [kylewish19/wshl-x-nhl-model](https://github.com/kylewish19/wshl-x-nhl-model) | Separate probability and price cohorts; diagnose markets with Brier, log loss and ROI | Implement independent evaluation of caller-supplied locked picks. Do not import its probabilities or selection thresholds. |
| [higgiAPPMASTER/NHL-SHOTS](https://github.com/higgiAPPMASTER/NHL-SHOTS) | Recent shooting volume; venue/opponent splits; PP usage; sample-size-aware hit-rate diagnostics | Implement strictly pregame shooting features. Do not treat the README's 80% historical hit-rate filters as calibrated probabilities. |

Code is independently implemented here; no upstream source is copied or vendored.
Inspected commits: WSHL `ee1d497b7dfc7e193b6fa6e6996a492c980b5601`;
NHL-SHOTS `5c54444dbe9c72de8df1766bf7a3e8b548f73cf5`.
Neither inspected repository supplied an explicit license in its tracked tree at review.
The WSHL repository's published clean priced record through October 6 is 31–61,
−30.5785 units / −33.24% ROI. Those results are repository claims, not independently
verified match outcomes. Its October 8 experimental selection layer quarantines
anytime-goal bets. Useful workflow ideas do not establish predictive superiority.

NHL-SHOTS is an application using FastAPI, NHL endpoints, The Odds API and optional
browser/provider access. Its `main.py` mixes roughly 15,000 lines of server logic,
data processing and embedded UI. The inspected shooting utilities include recent
venue/matchup hit rates, a Wilson lower bound, PP usage and heuristically adjusted
count projections. No independently reproduced out-of-sample benchmark was found
in the small tracked tree. We have not executed its web app or any paid provider.

## Shooting features

`data/shot_context.py` builds `shot_context_*` columns for one player-game row.
It is enabled by default in `build_goal_scorer_dataset`; callers can explicitly
disable it with `include_shot_context=False`. Existing training scripts do not
automatically select these additional columns or promote a new predictive model.

Required history columns: `game_id`, `player_id`, `game_date`, `shots_on_goal`.
Optional: `goals`, `toi_seconds`, `pp_toi_seconds`, `is_home` (boolean/0/1),
`opponent_abbrev`. Venue/opponent/PP fields must be normalized from a verified
source before calling the builder. Absent context is flagged unavailable and
the corresponding measurements stay missing; it is not inferred from later data.

Features include:

- Last 5 and last 10 games: mean SOG, empirical 1+/2+/3+/4+ rates, and sample counts.
- Same venue and same opponent (same venue when available), capped at 10 games.
- A Wilson lower interval endpoint alongside each rate, making thin samples visible.
- Shots per 60 and average observed TOI when supplied.
- Goal-hit frequency and observed PP usage/time when supplied, with observed counts.

Only history dated strictly before the candidate's UTC game date is used; all
same-day observations are excluded conservatively. This is not an exact intraday
availability model. Row IDs must be unique and shots nonmissing integer counts.
Historical rates and Wilson intervals are diagnostics, not tomorrow's calibrated
probabilities. Serial dependence, opponent selection and role changes remain.
The parent dataset builder now rejects missing goals or SOG instead of filling
unobserved outcomes with zero. Input adapters must supply actual observed counts.
Observed PP time does not establish a confirmed future PP1 role. No goalie-based
boost or arbitrary coefficient has been added to the goal model.

For a future slate, pass separate candidate rows:

```python
from nhl_goal_probability_research_team.data import build_shot_context_features
features = build_shot_context_features(history, candidates)
```

Use only named pregame feature columns in training. The base table retains outcome
columns for labels; never feed all numeric columns blindly into a model. Existing
linemate and rolling features are separate and were not fully audited by this change.

## Locked-pick ledger

Required CSV columns:
`pick_id,cohort,market,probability,predicted_at,starts_at,result`.
Result is one of `WIN,LOSS,PUSH,VOID,PENDING`. Use timezone-qualified timestamps.
Use separate `cohort` values for probability-ranked and price-selected picks;
the same pick may occur in both cohorts, once per cohort.

Optional prices: `decimal_odds,odds_at,odds_source`. When a price is present,
its timestamp and source are mandatory; quotes later than the prediction lock
are rejected. Predictions at or after puck drop are rejected, rather than folded
into a prospective record. Odds must come from genuine market observations.

```bash
PYTHONPATH=src python scripts/evaluate_locked_picks.py \
  --picks workspace/locked_picks.csv --output reports/market_summary.csv
```

Outputs are separated by cohort/market: counts, hit rate, expected wins, Brier,
log loss and (only with valid price metadata) flat one-unit profit and ROI.
PUSH/VOID/PENDING do not count as wins or losses or contribute to ROI's risked-unit
denominator. Unpriced settled picks still enter probability diagnostics, but never
economic calculations. No settled observations yields missing metrics, not 0%.
Timestamp/source metadata cannot prove authenticity; caller-supplied fixtures or
synthetic data remain smoke tests. No CLV is claimed without closing prices.

## Validation and operational scope

Unit tests cover future-result mutation invariance, same-day exclusion, unsorted
history, missing context, thin samples, duplicate IDs, unavailable prices,
late lock/quotes and push/void accounting. All new fixture data is synthetic and
is not betting evidence. No user API key, paid API quota, scheduler or deployment
is used. These additions prepare data and diagnostics; predictive and economic
validation still require real chronological holdouts and genuine priced picks.

Executed locally on Python 3.12: `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v`
— **18 tests passed**. A separate CLI smoke test with explicitly synthetic unpriced
input passed and left economic metrics missing. `git diff --check` passed.
