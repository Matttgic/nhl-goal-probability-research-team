# Agent collaboration protocol

This repository is an evidence-first NHL goal-probability research project.

## Non-negotiable rules

- Never fabricate historical odds, player stats, results, source availability, API coverage or model metrics.
- Historical features must use only information available before puck drop.
- Use temporal/walk-forward validation; never random-split time-dependent betting data without justification.
- Treat synthetic/fixture data only as a pipeline smoke test, never as betting evidence.
- Compute ROI, CLV, drawdown or Kelly results only from genuine timestamp-valid historical market prices.
- Keep rejected/failed experiments visible; do not cherry-pick winners.
- Prefer reproducible scripts and saved outputs over prose claims.
- Keep secrets out of source control.

## Roles

- Researcher: verify sources, coverage, terms and access.
- Data Engineer: ingest, validate and join datasets safely.
- ML Scientist: train calibrated probabilistic models.
- Critic: look for leakage, bias, overfitting and unsupported claims.
- Backtester: run walk-forward evaluation and economic tests when valid odds exist.
- Developer: turn validated research into maintainable software.
- Synthesizer: reconcile evidence and disagreements.
- Manager: delegate, demand rework and block unsupported conclusions.

When agents disagree, prefer executed code, verified primary sources and out-of-sample evidence over confidence or consensus.
