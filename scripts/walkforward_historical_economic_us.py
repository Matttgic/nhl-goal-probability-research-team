from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

import scripts.real_nhl_player_goal_v2 as v2
from scripts.historical_anytime_goal_backtest_oddsapi import (
    OddsAPIClient,
    discover_event,
    extract_yes_prices,
    historical_event_odds,
)

PRED_PATH = Path(os.getenv("WF_PRED_PATH", "cached/walkforward/workspace/real_nhl_walkforward/walkforward_predictions_sampled_games.csv"))
OUT = Path("workspace/walkforward_economic_us")
REPORT = Path("reports/walkforward_economic_us.md")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

EV_THRESHOLD = 0.05
MAX_ODDS = 8.0
GAMES_PER_FOLD = int(os.getenv("WF_ODDS_GAMES_PER_FOLD", "40"))

TEAM_NAMES = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres",
    "CAR": "Carolina Hurricanes", "CBJ": "Columbus Blue Jackets", "CGY": "Calgary Flames",
    "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche", "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers", "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NJD": "New Jersey Devils", "NSH": "Nashville Predators", "NYI": "New York Islanders",
    "NYR": "New York Rangers", "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SEA": "Seattle Kraken", "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets", "WSH": "Washington Capitals",
}


def norm(s: str | None) -> str:
    x = unicodedata.normalize("NFKD", str(s or ""))
    x = "".join(c for c in x if not unicodedata.combining(c)).lower()
    x = re.sub(r"[^a-z0-9 ]+", " ", x)
    toks = [t for t in x.split() if t not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(toks)


def match_key(s: str | None) -> str:
    toks = norm(s).split()
    return f"{toks[0][0]}:{toks[-1]}" if len(toks) >= 2 else ""


def select_games(pred: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for fold, g in pred.groupby("fold", sort=True):
        games = g[["game_id", "game_date"]].drop_duplicates().sort_values(["game_date", "game_id"])
        if len(games) > GAMES_PER_FOLD:
            idx = np.unique(np.linspace(0, len(games) - 1, GAMES_PER_FOLD, dtype=int))
            games = games.iloc[idx]
        selected.append(games.assign(fold=fold))
    return pd.concat(selected, ignore_index=True)


def attach_game_meta(pred: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in selected.itertuples(index=False):
        g = pred[pred.game_id == row.game_id]
        home = g[g.home == 1]
        away = g[g.home == 0]
        if home.empty or away.empty:
            continue
        landing = v2.get_json(f"gamecenter/{int(row.game_id)}/landing")
        start = landing.get("startTimeUTC")
        if not start:
            continue
        rows.append({
            "fold": row.fold,
            "nhl_game_id": int(row.game_id),
            "home": str(home.team.iloc[0]),
            "away": str(away.team.iloc[0]),
            "start_dt": pd.to_datetime(start, utc=True),
        })
    return pd.DataFrame(rows)


def aggregate_entry(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    raw = raw[raw.timestamp_valid.astype(bool)].copy()
    raw["match_key"] = raw.odds_player_name.map(match_key)
    raw = raw[raw.match_key != ""]
    counts = raw[["nhl_game_id", "match_key", "odds_player_name"]].drop_duplicates().groupby(["nhl_game_id", "match_key"]).size()
    ambiguous = set(counts[counts > 1].index)
    raw = raw[~raw.apply(lambda r: (int(r.nhl_game_id), r.match_key) in ambiguous, axis=1)].copy()
    rows = []
    for (gid, key), g in raw.groupby(["nhl_game_id", "match_key"]):
        best = g.loc[g.decimal_odds.idxmax()]
        nv = g.no_vig_prob.dropna()
        rows.append({
            "nhl_game_id": int(gid), "match_key": key,
            "decimal_odds": float(best.decimal_odds),
            "best_book": str(best.bookmaker_key),
            "market_prob": float(nv.median()) if not nv.empty else float(g.raw_implied_prob.median()),
        })
    return pd.DataFrame(rows), len(ambiguous)


def max_dd(p: pd.Series) -> float:
    if p.empty:
        return math.nan
    c = p.cumsum(); peak = c.cummax().clip(lower=0)
    return float((peak - c).max())


def bootstrap_roi(picks: pd.DataFrame, reps: int = 4000) -> tuple[float, float]:
    if picks.empty or picks.game_id.nunique() < 2:
        return math.nan, math.nan
    groups = [g.profit_unit.to_numpy(float) for _, g in picks.groupby("game_id")]
    rng = np.random.default_rng(42)
    vals = []
    for _ in range(reps):
        x = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(float(x.mean()))
    return tuple(map(float, np.quantile(vals, [0.025, 0.975])))


def stats(df: pd.DataFrame) -> dict:
    picks = df[(df.model_ev >= EV_THRESHOLD) & (df.decimal_odds <= MAX_ODDS)].copy()
    if picks.empty:
        return {"bets": 0, "wins": 0, "net_units": 0.0, "roi": None}
    lo, hi = bootstrap_roi(picks)
    return {
        "bets": int(len(picks)), "wins": int(picks.target_goal.sum()),
        "hit_rate": float(picks.target_goal.mean()), "avg_odds": float(picks.decimal_odds.mean()),
        "net_units": float(picks.profit_unit.sum()), "roi": float(picks.profit_unit.mean()),
        "max_drawdown_units": max_dd(picks.profit_unit), "roi_ci95_low": lo, "roi_ci95_high": hi,
    }


def main() -> None:
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ODDS_API_KEY is required")
    pred = pd.read_csv(PRED_PATH)
    selected = select_games(pred)
    meta = attach_game_meta(pred, selected)
    print(f"walk-forward economic sample games={len(meta)} folds={meta.fold.nunique()}")

    client = OddsAPIClient(key)
    raw_parts = []
    map_rows = []
    errors = []
    for i, row in enumerate(meta.itertuples(index=False), start=1):
        class R: pass
        r = R(); r.start_dt = row.start_dt; r.home = row.home; r.away = row.away; r.nhl_game_id = row.nhl_game_id
        try:
            client.assert_credit_reserve(expected_next_cost=11)
            event = discover_event(client, r)
            if not event:
                errors.append({"game_id": row.nhl_game_id, "reason": "event_not_found"})
                continue
            payload = historical_event_odds(client, str(event["odds_event_id"]), row.start_dt - timedelta(minutes=60))
            part = extract_yes_prices(payload, row.nhl_game_id, "entry", row.start_dt)
            if not part.empty:
                raw_parts.append(part)
                map_rows.append({"game_id": row.nhl_game_id, "fold": row.fold})
        except Exception as exc:
            errors.append({"game_id": row.nhl_game_id, "reason": repr(exc)})
        if i % 20 == 0 or i == len(meta):
            print(f"US WF odds {i}/{len(meta)} priced_parts={len(raw_parts)} calls={client.calls} remaining={client.credits_remaining}")

    if not raw_parts:
        raise RuntimeError("No historical US player-goal prices returned")
    raw = pd.concat(raw_parts, ignore_index=True)
    entry, ambiguous_odds = aggregate_entry(raw)

    pred["match_key"] = pred.player_name.map(match_key)
    test_counts = pred.groupby(["game_id", "match_key"])["player_id"].nunique()
    ambiguous_test = set(test_counts[test_counts > 1].index)
    pred = pred[~pred.apply(lambda r: (int(r.game_id), r.match_key) in ambiguous_test, axis=1)].copy()
    pred = pred.merge(pd.DataFrame(map_rows).drop_duplicates(), on=["game_id", "fold"], how="inner")
    merged = pred.merge(entry, left_on=["game_id", "match_key"], right_on=["nhl_game_id", "match_key"], how="inner")
    if merged.empty:
        raise RuntimeError("No player joins")

    merged["model_prob"] = merged.p_logistic
    merged["model_ev"] = merged.model_prob * merged.decimal_odds - 1.0
    merged["profit_unit"] = np.where(merged.target_goal == 1, merged.decimal_odds - 1.0, -1.0)

    fold_stats = []
    for fold, g in merged.groupby("fold", sort=True):
        s = stats(g); s["fold"] = fold; s["priced_games"] = int(g.game_id.nunique()); s["joined_rows"] = int(len(g))
        s["model_brier"] = float(brier_score_loss(g.target_goal, g.model_prob))
        s["market_brier"] = float(brier_score_loss(g.target_goal, g.market_prob))
        s["model_log_loss"] = float(log_loss(g.target_goal, np.clip(g.model_prob, 1e-6, 1-1e-6), labels=[0,1]))
        s["market_log_loss"] = float(log_loss(g.target_goal, np.clip(g.market_prob, 1e-6, 1-1e-6), labels=[0,1]))
        fold_stats.append(s)

    overall = stats(merged)
    overall.update({
        "priced_games": int(merged.game_id.nunique()), "joined_rows": int(len(merged)),
        "model_brier": float(brier_score_loss(merged.target_goal, merged.model_prob)),
        "market_brier": float(brier_score_loss(merged.target_goal, merged.market_prob)),
        "model_log_loss": float(log_loss(merged.target_goal, np.clip(merged.model_prob,1e-6,1-1e-6), labels=[0,1])),
        "market_log_loss": float(log_loss(merged.target_goal, np.clip(merged.market_prob,1e-6,1-1e-6), labels=[0,1])),
    })
    summary = {
        "primary_rule": {"ev_min": EV_THRESHOLD, "max_decimal_odds": MAX_ODDS},
        "sample_games_requested": int(len(meta)), "priced_games": int(merged.game_id.nunique()),
        "joined_rows": int(len(merged)), "ambiguous_test_keys": len(ambiguous_test),
        "ambiguous_odds_keys": ambiguous_odds, "api_calls": client.calls,
        "credits_used_header": client.credits_used_header, "credits_remaining": client.credits_remaining,
        "folds": fold_stats, "overall": overall, "errors": len(errors),
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(fold_stats).to_csv(OUT / "fold_economic_metrics.csv", index=False)
    if errors:
        (OUT / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    lines = [
        "# Walk-forward economic validation — US historical NHL anytime goal",
        "",
        "Primary rule was fixed before this run: EV >= 5%, odds <= 8.00, flat 1u. This is a market-stability research check, not evidence for French-book availability.",
        "",
        "| Fold | Priced games | Joined rows | Bets | Net u | ROI | 95% ROI CI | Model Brier | Market Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in fold_stats:
        ci = f"[{s['roi_ci95_low']:.1%}, {s['roi_ci95_high']:.1%}]" if s.get("roi_ci95_low") is not None else "n/a"
        roi = f"{s['roi']:.2%}" if s.get("roi") is not None else "n/a"
        lines.append(f"| {s['fold']} | {s['priced_games']} | {s['joined_rows']} | {s['bets']} | {s['net_units']:+.2f} | {roi} | {ci} | {s['model_brier']:.6f} | {s['market_brier']:.6f} |")
    lines += ["", "## Overall", "", f"```json\n{json.dumps(overall, indent=2)}\n```", "", f"Credits remaining after run: **{client.credits_remaining}**."]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
