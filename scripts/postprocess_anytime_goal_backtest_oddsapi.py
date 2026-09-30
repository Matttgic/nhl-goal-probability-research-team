from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

V2 = Path("cached/v2/workspace/real_nhl_v2/test_predictions.csv")
ENTRY = Path("cached/odds/workspace/historical_anytime_goal_oddsapi/entry_raw_prices.csv")
CLOSE = Path("cached/odds/workspace/historical_anytime_goal_oddsapi/close_raw_prices.csv")
OUT = Path("workspace/historical_anytime_goal_oddsapi_postprocess")
REPORT = Path("reports/historical_anytime_goal_oddsapi_postprocess.md")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

PRIMARY_EV = 0.05
MAX_ODDS = 8.0


def norm(value: str | None) -> str:
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(toks)


def match_key(value: str | None) -> str:
    toks = norm(value).split()
    return f"{toks[0][0]}:{toks[-1]}" if len(toks) >= 2 else ""


def max_dd(profits: pd.Series) -> float:
    if profits.empty:
        return math.nan
    curve = profits.cumsum()
    peak = curve.cummax().clip(lower=0)
    return float((peak - curve).max())


def bootstrap_roi(picks: pd.DataFrame, reps: int = 4000) -> tuple[float, float]:
    if picks.empty or picks.game_id.nunique() < 2:
        return math.nan, math.nan
    groups = [g.profit_unit.to_numpy(float) for _, g in picks.groupby("game_id")]
    rng = np.random.default_rng(42)
    rois = []
    for _ in range(reps):
        vals = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        rois.append(float(vals.mean()))
    return tuple(float(x) for x in np.quantile(rois, [0.025, 0.975]))


def aggregate(prices: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    prices = prices[prices.timestamp_valid].copy()
    prices["match_key"] = prices.odds_player_name.map(match_key)
    counts = (prices[["nhl_game_id", "match_key", "odds_player_name"]]
              .drop_duplicates().groupby(["nhl_game_id", "match_key"]).size())
    ambiguous = set(counts[counts > 1].index)
    prices = prices[~prices.apply(lambda r: (int(r.nhl_game_id), r.match_key) in ambiguous, axis=1)].copy()
    rows = []
    for (gid, key), g in prices.groupby(["nhl_game_id", "match_key"]):
        best = g.loc[g.decimal_odds.idxmax()]
        no_vig = g.no_vig_prob.dropna()
        rows.append({
            "nhl_game_id": int(gid),
            "match_key": key,
            f"{label}_odds_player_name": str(best.odds_player_name),
            f"{label}_best_decimal_odds": float(best.decimal_odds),
            f"{label}_best_bookmaker": str(best.bookmaker_key),
            f"{label}_book_count": int(g.bookmaker_key.nunique()),
            f"{label}_median_raw_prob": float(g.raw_implied_prob.median()),
            f"{label}_median_no_vig_prob": float(no_vig.median()) if not no_vig.empty else math.nan,
        })
    return pd.DataFrame(rows), len(ambiguous)


def strategy(df: pd.DataFrame, threshold: float) -> dict:
    p = df[(df.model_ev >= threshold) & (df.decimal_odds <= MAX_ODDS)].copy()
    p = p.sort_values(["game_date", "game_id", "player_name"])
    if p.empty:
        return {"threshold": threshold, "bets": 0, "wins": 0, "net_units": 0.0, "roi": math.nan}
    lo, hi = bootstrap_roi(p)
    return {
        "threshold": threshold,
        "bets": int(len(p)),
        "wins": int(p.target_goal.sum()),
        "hit_rate": float(p.target_goal.mean()),
        "avg_odds": float(p.decimal_odds.mean()),
        "net_units": float(p.profit_unit.sum()),
        "roi": float(p.profit_unit.mean()),
        "max_drawdown_units": max_dd(p.profit_unit),
        "roi_ci95_low": lo,
        "roi_ci95_high": hi,
        "avg_log_price_clv": float(p.clv_log_price.dropna().mean()) if p.clv_log_price.notna().any() else math.nan,
    }


def main() -> None:
    for path in (V2, ENTRY, CLOSE):
        if not path.exists():
            raise RuntimeError(f"Missing cached input: {path}")

    test = pd.read_csv(V2)
    entry_raw = pd.read_csv(ENTRY)
    close_raw = pd.read_csv(CLOSE)

    test["match_key"] = test.player_name.map(match_key)
    test_counts = test.groupby(["game_id", "match_key"])["player_id"].nunique()
    ambiguous_test = set(test_counts[test_counts > 1].index)
    test = test[~test.apply(lambda r: (int(r.game_id), r.match_key) in ambiguous_test, axis=1)].copy()

    entry, ambiguous_entry = aggregate(entry_raw, "entry")
    close, ambiguous_close = aggregate(close_raw, "close")

    merged = test.merge(entry, left_on=["game_id", "match_key"], right_on=["nhl_game_id", "match_key"], how="inner")
    if merged.empty:
        raise RuntimeError("No player joins after initial/surname matching")
    merged = merged.merge(close, on=["nhl_game_id", "match_key"], how="left")

    merged["model_prob"] = merged.p_logistic
    merged["decimal_odds"] = merged.entry_best_decimal_odds
    merged["market_prob"] = merged.entry_median_no_vig_prob.where(
        merged.entry_median_no_vig_prob.notna(), merged.entry_median_raw_prob
    )
    merged["model_ev"] = merged.model_prob * merged.decimal_odds - 1.0
    merged["profit_unit"] = np.where(merged.target_goal == 1, merged.decimal_odds - 1.0, -1.0)
    merged["clv_log_price"] = np.where(
        merged.close_best_decimal_odds.notna(),
        np.log(merged.decimal_odds / merged.close_best_decimal_odds),
        np.nan,
    )

    priced_games = set(entry.nhl_game_id.unique())
    eligible_rows = int(pd.read_csv(V2).query("game_id in @priced_games").shape[0])
    coverage = float(len(merged) / eligible_rows) if eligible_rows else math.nan

    primary = strategy(merged, PRIMARY_EV)
    sensitivity = [strategy(merged, t) for t in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]]
    picks = merged[(merged.model_ev >= PRIMARY_EV) & (merged.decimal_odds <= MAX_ODDS)].copy()

    y = merged.target_goal.to_numpy(int)
    mp = np.clip(merged.model_prob.to_numpy(float), 1e-6, 1 - 1e-6)
    bp = np.clip(merged.market_prob.to_numpy(float), 1e-6, 1 - 1e-6)
    scores = {
        "model_brier": float(brier_score_loss(y, mp)),
        "model_log_loss": float(log_loss(y, mp, labels=[0, 1])),
        "market_brier_raw_implied": float(brier_score_loss(y, bp)),
        "market_log_loss_raw_implied": float(log_loss(y, bp, labels=[0, 1])),
    }

    summary = {
        "source_odds_run": 36763766175,
        "source_v2_run": 36753181078,
        "historical_api_credits_already_used": 2726,
        "additional_api_credits_used": 0,
        "priced_games": int(entry.nhl_game_id.nunique()),
        "eligible_v2_rows_on_priced_games": eligible_rows,
        "joined_rows": int(len(merged)),
        "join_coverage": coverage,
        "ambiguous_test_keys_excluded": len(ambiguous_test),
        "ambiguous_entry_keys_excluded": ambiguous_entry,
        "ambiguous_close_keys_excluded": ambiguous_close,
        "close_price_coverage": float(merged.close_best_decimal_odds.notna().mean()),
        "primary_rule": {"min_ev": PRIMARY_EV, "max_decimal_odds": MAX_ODDS},
        "primary": primary,
        "sensitivity": sensitivity,
        "probability_scores": scores,
    }

    merged.sort_values(["game_date", "game_id", "player_name"]).to_csv(OUT / "joined_player_prices.csv", index=False)
    picks.sort_values(["game_date", "game_id", "player_name"]).to_csv(OUT / "primary_strategy_picks.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "sensitivity.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Historical NHL anytime-goal backtest — cached Odds API post-process",
        "",
        "This run reuses the historical odds already bought in run 36763766175 and makes zero additional Odds API requests.",
        "",
        "## Matching fix",
        "The NHL artifact uses abbreviated names (for example `N. Kucherov`) while The Odds API uses full names (`Nikita Kucherov`). Exact normalized names therefore produced zero joins. This run uses first initial + surname inside the same game and excludes ambiguous collisions instead of guessing.",
        "",
        f"- Priced games: **{summary['priced_games']}**",
        f"- Eligible V2 player-game rows: **{eligible_rows}**",
        f"- Joined rows: **{len(merged)} ({coverage:.1%})**",
        f"- Close-price coverage: **{summary['close_price_coverage']:.1%}**",
        "",
        "## Primary rule",
        f"EV >= **{PRIMARY_EV:.0%}**, decimal odds <= **{MAX_ODDS:.2f}**, flat stake 1 unit.",
        "",
        f"- Bets: **{primary['bets']}**",
        f"- Wins: **{primary['wins']}**",
        f"- Hit rate: **{primary['hit_rate']:.2%}**",
        f"- Average odds: **{primary['avg_odds']:.3f}**",
        f"- Net units: **{primary['net_units']:+.2f}**",
        f"- ROI: **{primary['roi']:.2%}**",
        f"- Max drawdown: **{primary['max_drawdown_units']:.2f} units**",
        f"- 95% cluster-bootstrap ROI interval: **[{primary['roi_ci95_low']:.2%}, {primary['roi_ci95_high']:.2%}]**",
        f"- Mean log-price CLV: **{primary['avg_log_price_clv']:.4f}**",
        "",
        "The ROI interval crosses zero, so this sample does not establish a statistically reliable positive betting edge.",
        "",
        "## Probability scoring",
        f"- Model Brier: **{scores['model_brier']:.6f}**",
        f"- Market raw-implied Brier: **{scores['market_brier_raw_implied']:.6f}**",
        f"- Model Log Loss: **{scores['model_log_loss']:.6f}**",
        f"- Market raw-implied Log Loss: **{scores['market_log_loss_raw_implied']:.6f}**",
        "",
        "Market probabilities are raw implied medians where no paired No price exists, so the market benchmark is not fully de-vigged.",
        "",
        "## Sensitivity — diagnostic only",
        "| EV threshold | Bets | Net units | ROI | 95% ROI interval |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s in sensitivity:
        ci = f"[{s['roi_ci95_low']:.1%}, {s['roi_ci95_high']:.1%}]" if s.get("roi_ci95_low") is not None and np.isfinite(s.get("roi_ci95_low", math.nan)) else "n/a"
        lines.append(f"| {s['threshold']:.0%} | {s['bets']} | {s['net_units']:+.2f} | {s['roi']:.2%} | {ci} |")
    lines += [
        "",
        "The 5% threshold remains the primary rule; sensitivity rows are not used to choose a post-hoc winner.",
        "",
        "## Limitations",
        "- 69 of the 80 V2 test games had mapped historical events in the source run.",
        "- The V2 holdout is narrow and comes from the last-500-game experiment.",
        "- Ambiguous initial/surname collisions are excluded.",
        "- More seasons and walk-forward windows are required before production use.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
