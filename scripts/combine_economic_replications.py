from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEV = Path(os.getenv("META_DEV_FOLDS", "cached/dev/workspace/walkforward_economic_us/fold_economic_metrics.csv"))
REP = Path(os.getenv("META_REP_FOLDS", "cached/rep/workspace/walkforward_economic_us_replication/fold_economic_metrics.csv"))
OUT = Path("workspace/economic_replication_meta")
REPORT = Path("reports/economic_replication_meta.md")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)


def temporal_fold_bootstrap(df: pd.DataFrame, reps: int = 20000) -> tuple[float, float]:
    if len(df) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(42)
    bets = df["bets"].to_numpy(float)
    net = df["net_units"].to_numpy(float)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, len(df), len(df))
        b = bets[idx].sum()
        vals.append(float(net[idx].sum() / b) if b > 0 else math.nan)
    vals = np.asarray([x for x in vals if np.isfinite(x)])
    if len(vals) == 0:
        return math.nan, math.nan
    return tuple(map(float, np.quantile(vals, [0.025, 0.975])))


def main() -> None:
    if not DEV.exists() or not REP.exists():
        raise RuntimeError(f"Missing fold files: dev={DEV.exists()} rep={REP.exists()}")
    dev = pd.read_csv(DEV).copy(); dev["period"] = "2025-26"
    rep = pd.read_csv(REP).copy(); rep["period"] = "2024-25 replication"
    both = pd.concat([rep, dev], ignore_index=True)

    required = {"fold", "bets", "net_units", "roi", "priced_games", "joined_rows"}
    missing = required - set(both.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    bets = int(both.bets.sum())
    net = float(both.net_units.sum())
    weighted_roi = float(net / bets) if bets else math.nan
    lo, hi = temporal_fold_bootstrap(both)
    positive_folds = int((both.roi > 0).sum())
    negative_folds = int((both.roi < 0).sum())
    zero_folds = int((both.roi == 0).sum())

    rep_bets = int(rep.bets.sum()); rep_net = float(rep.net_units.sum())
    rep_roi = float(rep_net / rep_bets) if rep_bets else math.nan
    dev_bets = int(dev.bets.sum()); dev_net = float(dev.net_units.sum())
    dev_roi = float(dev_net / dev_bets) if dev_bets else math.nan

    summary = {
        "primary_rule": "EV >= 5%, decimal odds <= 8.00, flat 1 unit",
        "folds_total": int(len(both)),
        "positive_folds": positive_folds,
        "negative_folds": negative_folds,
        "zero_folds": zero_folds,
        "bets_total": bets,
        "net_units_total": net,
        "weighted_roi": weighted_roi,
        "temporal_fold_bootstrap_roi_ci95": [lo, hi],
        "development_period": {"folds": int(len(dev)), "bets": dev_bets, "net_units": dev_net, "roi": dev_roi},
        "independent_replication": {"folds": int(len(rep)), "bets": rep_bets, "net_units": rep_net, "roi": rep_roi},
        "replication_primary_success": bool(rep_roi > 0) if np.isfinite(rep_roi) else False,
        "note": "The confidence interval resamples whole monthly folds, preserving temporal clustering at a coarse level. It is separate from within-period game-cluster bootstrap intervals.",
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    both.to_csv(OUT / "all_fold_metrics.csv", index=False)

    lines = [
        "# Economic replication meta-summary",
        "",
        "The primary rule was frozen before the independent 2024-25 replication: **EV >= 5%, decimal odds <= 8.00, flat 1 unit**.",
        "",
        f"- Independent replication ROI: **{rep_roi:.2%}** ({rep_net:+.2f}u on {rep_bets} bets)",
        f"- Development-period ROI: **{dev_roi:.2%}** ({dev_net:+.2f}u on {dev_bets} bets)",
        f"- Combined weighted ROI: **{weighted_roi:.2%}** ({net:+.2f}u on {bets} bets)",
        f"- Positive monthly folds: **{positive_folds}/{len(both)}**",
        f"- Temporal-fold bootstrap 95% ROI interval: **[{lo:.2%}, {hi:.2%}]**",
        "",
        "| Period | Fold | Priced games | Bets | Net units | ROI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in both.itertuples(index=False):
        lines.append(f"| {r.period} | {r.fold} | {int(r.priced_games)} | {int(r.bets)} | {float(r.net_units):+.2f} | {float(r.roi):.2%} |")
    lines += [
        "",
        "## Interpretation rule",
        "",
        "A positive replication is supportive, not sufficient by itself to claim durable profitability. France-specific profitability remains unverified until timestamp-valid French bookmaker prices are available and tested independently.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
