# Upcoming NHL forecast protocol

The `Upcoming NHL experimental forecasts` workflow produces a GitHub-readable
Markdown report and CSV/JSON downloads for games in the next 48 hours. Sources
are public NHL regular-season schedules, completed boxscores and current rosters.
No bookmaker odds, API credits, Telegram secrets or paid model calls are used.
No website deployment is required to read the report from a phone.

## Model

The independent logistic goal model consumes the shooting-context module added
from the previous repository audit. Recent 5/10-game shooting/goals/TOI, venue
and opponent histories enter explicitly. A simple recent-history logistic model
is the comparison baseline. Both models are sigmoid-calibrated on a separate
chronological sample. Features entirely unavailable in training are omitted.
The NHL boxscore often lacks PP TOI: the report states that absence and does not
invent PP1 assignments or power-play time. No xG source is present in this path.

At most 600 completed games from a bounded 420-day schedule history are loaded.
Only dates strictly earlier than the prediction date enter historical collection.
All per-player features use strictly earlier dates, including venue/opponent
splits. Missing observed shots or goals fail the import instead of becoming zero.
More than 5% failed boxscores blocks forecasting; smaller omissions are reported.
The archived cache contains only completed boxscores; schedules/rosters refresh.

## Validation

After requiring five earlier player games, distinct game dates are split 60% /
20% / 20% for training / calibration / test. Entire dates and therefore whole
games stay within one partition. Model coefficients/imputation use training only;
probability calibration uses the second partition only. The final test is not
used to select hyperparameters or choose between models. The prespecified enriched
candidate is still explicitly experimental if its Brier is worse than baseline.

The exact train/calibration models used for test metrics produce the live report;
there is no silent refit on the test set. New completed games can enter context
features at inference time, while model parameters remain frozen for that run.
One chronological holdout is not a robust multi-season walk-forward evaluation.
The report includes Brier, log loss, mean probability, observed frequency, sample
sizes, date ranges, used/unavailable features and failed historical imports.

## Forecast interpretation

Only future regular-season games enter the next-48-hour slate. Players are drawn
from current NHL rosters; that does not confirm game-day availability. Predictions
estimate a goal conditional on participation. Players with fewer than five observed
historical games receive no probability. Displayed top-ten rows per game are pure
probability rankings, without any asserted edge or staking recommendation.
The full CSV retains all eligible roster players and missing-history statuses.
The displayed `Tirs/match, 10 derniers` is an observed historical average, not a
separate forecast. The theoretical decimal price is `1 / probability`, with no
bookmaker margin or availability assumption. ROI/CLV are not computed.

The report also lists games whose scheduled start has passed on the current
Europe/Paris calendar day, including after UTC midnight. Their NHL status and
observed score are displayed separately. On request, late whole-game scorer
estimates are also shown using only strictly earlier game dates and the frozen
model. No current-game shots, goals, scores or remaining time enter these estimates.
They are not remaining-game/live probabilities or timestamp-locked predictions.
The CSV's `prediction_kind` explicitly separates `pregame` from
`retrospective_history_only`; late rows must never enter pregame betting backtests.
The schedule is refreshed after training and roster retrieval. Games starting
during calculation are removed from forecasts; prediction timestamps reflect
actual calculation completion rather than the earlier workflow start time.

## Refresh and failures

The first code push starts a run. Further runs use the Actions `Run workflow`
button; no new recurring schedule has been installed. Generated reports are
committed to `reports/upcoming_predictions.md`, `.csv` and validation JSON.
Every report is timestamped in Europe/Paris and also appears in the Actions run
summary/artifact. A failed calculation publishes a dated failure status and clears
the CSV rather than presenting the previous forecast as current. This new workflow
does not change or invoke the existing US-odds/Telegram workflow.

Local command: `PYTHONPATH=src python scripts/upcoming_nhl_predictions.py`.
Dependencies are pandas, NumPy and scikit-learn; it imports no CrewAI or AI provider.
Tests use labelled synthetic fixtures exclusively; passing them is a code check,
not a backtest or a profitability result.

Local verification: 25 unit tests passed, including late-estimate invariance to current/future results, Paris-day/UTC-midnight coverage and a synthetic full training /
calibration / test / future-slate rendering exercise, leakage-invariance tests and
strict missing-stat checks. The workflow YAML parses and `git diff --check` passes.
Historical real-data metrics are populated only by a successful NHL-source run.
