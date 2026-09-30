from __future__ import annotations

import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import brier_score_loss, log_loss

import scripts.real_nhl_player_goal_v2 as v2
from scripts.historical_anytime_goal_backtest_bdl import (
    PRIMARY_EV_THRESHOLD,
    PRIMARY_MAX_DECIMAL_ODDS,
    build_frozen_v2_test_predictions,
    normalise_name,
    strategy_stats,
    test_game_keys,
)

BASE = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"
MARKET = "player_goal_scorer_anytime"
OUT = Path("workspace/historical_anytime_goal_oddsapi")
REPORT = Path("reports/historical_anytime_goal_oddsapi.md")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

PRIMARY_MINUTES_BEFORE = int(os.getenv("ODDS_PRIMARY_MINUTES_BEFORE", "60"))
CLOSE_MINUTES_BEFORE = int(os.getenv("ODDS_CLOSE_MINUTES_BEFORE", "10"))
REGIONS = os.getenv("ODDS_REGIONS", "us").strip() or "us"
MAX_TEST_GAMES = int(os.getenv("ODDS_MAX_TEST_GAMES", "0"))
CREDIT_RESERVE = int(os.getenv("ODDS_CREDIT_RESERVE", "1000"))

TEAM_NAMES = {
    "ANA": "Anaheim Ducks",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CAR": "Carolina Hurricanes",
    "CBJ": "Columbus Blue Jackets",
    "CGY": "Calgary Flames",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens",
    "NJD": "New Jersey Devils",
    "NSH": "Nashville Predators",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets",
    "WSH": "Washington Capitals",
}


def iso_z(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


class OddsAPIClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("ODDS_API_KEY is required")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "nhl-goal-probability-research-team/0.7"})
        self.calls = 0
        self.credits_used_header: int | None = None
        self.credits_remaining: int | None = None
        self.last_cost: int | None = None
        self.min_interval = float(os.getenv("ODDS_MIN_INTERVAL_SECONDS", "0.20"))
        self.last_request = 0.0

    def get(self, path: str, params: dict | None = None, tries: int = 6) -> dict:
        params = dict(params or {})
        params["apiKey"] = self.api_key
        url = f"{BASE}/{path.lstrip('/')}"
        last = None
        for attempt in range(tries):
            wait = self.min_interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.session.get(url, params=params, timeout=45)
                self.last_request = time.monotonic()
                self.calls += 1
                for key, attr in [
                    ("x-requests-used", "credits_used_header"),
                    ("x-requests-remaining", "credits_remaining"),
                    ("x-requests-last", "last_cost"),
                ]:
                    value = r.headers.get(key)
                    if value is not None:
                        try:
                            setattr(self, attr, int(float(value)))
                        except ValueError:
                            pass
                if r.status_code == 429:
                    time.sleep(float(r.headers.get("Retry-After", 3.0)))
                    continue
                if r.status_code >= 500:
                    time.sleep(min(20.0, 1.5 * (attempt + 1)))
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                last = exc
                time.sleep(min(20.0, 1.5 * (attempt + 1)))
        raise RuntimeError(f"The Odds API request failed {url}: {last}")

    def assert_credit_reserve(self, expected_next_cost: int = 10) -> None:
        if self.credits_remaining is None:
            return
        if self.credits_remaining - expected_next_cost < CREDIT_RESERVE:
            raise RuntimeError(
                f"Stopping to preserve quota: remaining={self.credits_remaining}, "
                f"reserve={CREDIT_RESERVE}, expected_next_cost={expected_next_cost}"
            )


def attach_start_times(games: pd.DataFrame) -> pd.DataFrame:
    starts = []
    errors = []
    for row in games.itertuples(index=False):
        payload = None
        try:
            payload = v2.get_json(f"gamecenter/{int(row.nhl_game_id)}/landing")
        except Exception as exc:
            try:
                payload = v2.get_json(f"gamecenter/{int(row.nhl_game_id)}/boxscore")
            except Exception as exc2:
                errors.append({"game_id": int(row.nhl_game_id), "error": f"{exc!r}; fallback={exc2!r}"})
                continue
        start = (payload or {}).get("startTimeUTC")
        if not start:
            errors.append({"game_id": int(row.nhl_game_id), "error": "missing startTimeUTC"})
            continue
        starts.append({"nhl_game_id": int(row.nhl_game_id), "start_time_utc": start})
    if errors:
        (OUT / "nhl_start_time_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    out = games.merge(pd.DataFrame(starts), on="nhl_game_id", how="left")
    out["start_dt"] = pd.to_datetime(out["start_time_utc"], utc=True, errors="coerce")
    return out[out["start_dt"].notna()].copy()


def find_event(payload: dict, home_abbrev: str, away_abbrev: str) -> dict | None:
    home = TEAM_NAMES.get(str(home_abbrev))
    away = TEAM_NAMES.get(str(away_abbrev))
    if not home or not away:
        return None
    for event in payload.get("data", []) or []:
        if event.get("home_team") == home and event.get("away_team") == away:
            return event
    return None


def discover_event(client: OddsAPIClient, row) -> dict | None:
    close_dt = row.start_dt - timedelta(minutes=CLOSE_MINUTES_BEFORE)
    payload = client.get(
        f"historical/sports/{SPORT}/events",
        params={"date": iso_z(close_dt), "dateFormat": "iso"},
    )
    event = find_event(payload, row.home, row.away)
    if event is None:
        return None
    return {
        "nhl_game_id": int(row.nhl_game_id),
        "odds_event_id": event.get("id"),
        "odds_home": event.get("home_team"),
        "odds_away": event.get("away_team"),
        "odds_commence_time": event.get("commence_time"),
        "discovery_snapshot": payload.get("timestamp"),
    }


def historical_event_odds(client: OddsAPIClient, event_id: str, at: pd.Timestamp) -> dict:
    client.assert_credit_reserve(expected_next_cost=10 * max(1, len(REGIONS.split(","))))
    return client.get(
        f"historical/sports/{SPORT}/events/{event_id}/odds",
        params={
            "date": iso_z(at),
            "dateFormat": "iso",
            "oddsFormat": "decimal",
            "regions": REGIONS,
            "markets": MARKET,
        },
    )


def extract_yes_prices(payload: dict, nhl_game_id: int, label: str, start_dt: pd.Timestamp) -> pd.DataFrame:
    event = payload.get("data") or {}
    snapshot = pd.to_datetime(payload.get("timestamp"), utc=True, errors="coerce")
    rows: list[dict] = []
    for book in event.get("bookmakers", []) or []:
        book_key = str(book.get("key") or "")
        book_title = str(book.get("title") or book_key)
        for market in book.get("markets", []) or []:
            if market.get("key") != MARKET:
                continue
            last_update = market.get("last_update") or book.get("last_update")
            outcomes = market.get("outcomes", []) or []
            by_player: dict[str, dict[str, float]] = {}
            names: dict[str, str] = {}
            for outcome in outcomes:
                player_name = str(outcome.get("description") or "").strip()
                side = str(outcome.get("name") or "").strip().lower()
                if not player_name or side not in {"yes", "no"}:
                    continue
                try:
                    price = float(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(price) or price <= 1:
                    continue
                key = normalise_name(player_name)
                names[key] = player_name
                by_player.setdefault(key, {})[side] = price
            for key, sides in by_player.items():
                yes = sides.get("yes")
                if yes is None:
                    continue
                no = sides.get("no")
                raw_prob = 1.0 / yes
                fair_prob = math.nan
                if no is not None and no > 1:
                    q_yes, q_no = 1.0 / yes, 1.0 / no
                    if q_yes + q_no > 0:
                        fair_prob = q_yes / (q_yes + q_no)
                rows.append({
                    "nhl_game_id": int(nhl_game_id),
                    "name_key": key,
                    "odds_player_name": names[key],
                    "snapshot_label": label,
                    "snapshot_timestamp": payload.get("timestamp"),
                    "bookmaker_key": book_key,
                    "bookmaker_title": book_title,
                    "market_last_update": last_update,
                    "decimal_odds": yes,
                    "no_decimal_odds": no,
                    "raw_implied_prob": raw_prob,
                    "no_vig_prob": fair_prob,
                    "timestamp_valid": bool(pd.notna(snapshot) and snapshot < start_dt),
                })
    return pd.DataFrame(rows)


def aggregate_snapshot(prices: pd.DataFrame, label: str) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["nhl_game_id", "name_key"])
    prices = prices[prices["timestamp_valid"]].copy()
    if prices.empty:
        return pd.DataFrame(columns=["nhl_game_id", "name_key"])
    rows = []
    for (gid, key), g in prices.groupby(["nhl_game_id", "name_key"]):
        best = g.loc[g["decimal_odds"].idxmax()]
        no_vig = g["no_vig_prob"].dropna()
        rows.append({
            "nhl_game_id": int(gid),
            "name_key": key,
            f"{label}_best_decimal_odds": float(best["decimal_odds"]),
            f"{label}_best_bookmaker": str(best["bookmaker_key"]),
            f"{label}_book_count": int(g["bookmaker_key"].nunique()),
            f"{label}_median_raw_prob": float(g["raw_implied_prob"].median()),
            f"{label}_median_no_vig_prob": float(no_vig.median()) if not no_vig.empty else math.nan,
            f"{label}_snapshot_timestamp": str(best["snapshot_timestamp"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ODDS_API_KEY is missing. Add it as a GitHub Actions repository secret; never commit the key.")

    print("Building frozen leakage-safe V2 test predictions...")
    test = build_frozen_v2_test_predictions()
    games = attach_start_times(test_game_keys(test))
    games = games.sort_values(["start_dt", "nhl_game_id"]).reset_index(drop=True)
    if MAX_TEST_GAMES > 0:
        games = games.tail(MAX_TEST_GAMES).copy()
        test = test[test["game_id"].isin(games["nhl_game_id"])].copy()
    print(f"V2 test rows={len(test)} games={len(games)} regions={REGIONS}")

    client = OddsAPIClient(key)
    discovered = []
    discovery_errors = []
    for i, row in enumerate(games.itertuples(index=False), start=1):
        try:
            event = discover_event(client, row)
            if event:
                discovered.append(event)
            else:
                discovery_errors.append({"nhl_game_id": int(row.nhl_game_id), "reason": "event_not_found"})
        except Exception as exc:
            discovery_errors.append({"nhl_game_id": int(row.nhl_game_id), "reason": repr(exc)})
        if i % 10 == 0 or i == len(games):
            print(f"event discovery {i}/{len(games)} matched={len(discovered)} calls={client.calls} remaining={client.credits_remaining}")

    event_map = pd.DataFrame(discovered)
    if event_map.empty:
        raise RuntimeError("No NHL games mapped to The Odds API historical events")
    if discovery_errors:
        (OUT / "event_discovery_errors.json").write_text(json.dumps(discovery_errors, indent=2), encoding="utf-8")
    event_map.to_csv(OUT / "event_map.csv", index=False)

    mapped = games.merge(event_map, on="nhl_game_id", how="inner")
    primary_parts: list[pd.DataFrame] = []
    close_parts: list[pd.DataFrame] = []
    odds_errors = []
    for i, row in enumerate(mapped.itertuples(index=False), start=1):
        primary_at = row.start_dt - timedelta(minutes=PRIMARY_MINUTES_BEFORE)
        close_at = row.start_dt - timedelta(minutes=CLOSE_MINUTES_BEFORE)
        try:
            payload = historical_event_odds(client, str(row.odds_event_id), primary_at)
            part = extract_yes_prices(payload, int(row.nhl_game_id), "entry", row.start_dt)
            if not part.empty:
                primary_parts.append(part)
        except Exception as exc:
            odds_errors.append({"nhl_game_id": int(row.nhl_game_id), "snapshot": "entry", "error": repr(exc)})
        try:
            payload = historical_event_odds(client, str(row.odds_event_id), close_at)
            part = extract_yes_prices(payload, int(row.nhl_game_id), "close", row.start_dt)
            if not part.empty:
                close_parts.append(part)
        except Exception as exc:
            odds_errors.append({"nhl_game_id": int(row.nhl_game_id), "snapshot": "close", "error": repr(exc)})
        if i % 10 == 0 or i == len(mapped):
            print(
                f"historical props {i}/{len(mapped)} entry_parts={len(primary_parts)} close_parts={len(close_parts)} "
                f"calls={client.calls} used={client.credits_used_header} remaining={client.credits_remaining}"
            )

    if odds_errors:
        (OUT / "odds_errors.json").write_text(json.dumps(odds_errors, indent=2), encoding="utf-8")
    entry_prices = pd.concat(primary_parts, ignore_index=True) if primary_parts else pd.DataFrame()
    close_prices = pd.concat(close_parts, ignore_index=True) if close_parts else pd.DataFrame()
    if entry_prices.empty:
        raise RuntimeError("No historical anytime-goal prices returned at the entry snapshot")
    entry_prices.to_csv(OUT / "entry_raw_prices.csv", index=False)
    if not close_prices.empty:
        close_prices.to_csv(OUT / "close_raw_prices.csv", index=False)

    entry = aggregate_snapshot(entry_prices, "entry")
    close = aggregate_snapshot(close_prices, "close")
    merged = test.merge(
        entry,
        left_on=["game_id", "name_key"],
        right_on=["nhl_game_id", "name_key"],
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("Odds were returned, but no game/player joins succeeded")
    if not close.empty:
        merged = merged.merge(close, on=["nhl_game_id", "name_key"], how="left")

    merged["decimal_odds"] = merged["entry_best_decimal_odds"]
    merged["raw_implied_prob"] = merged["entry_median_raw_prob"]
    merged["market_prob"] = merged["entry_median_no_vig_prob"].where(
        merged["entry_median_no_vig_prob"].notna(), merged["entry_median_raw_prob"]
    )
    merged["model_edge_prob"] = merged["model_prob"] - merged["market_prob"]
    merged["model_ev"] = merged["model_prob"] * merged["decimal_odds"] - 1.0
    merged["profit_unit"] = np.where(
        merged["target_goal"] == 1,
        merged["decimal_odds"] - 1.0,
        -1.0,
    )
    if "close_best_decimal_odds" in merged:
        merged["clv_log_price"] = np.where(
            merged["close_best_decimal_odds"].notna(),
            np.log(merged["decimal_odds"] / merged["close_best_decimal_odds"]),
            np.nan,
        )
        merged["clv_implied_prob"] = np.where(
            merged["close_median_raw_prob"].notna(),
            merged["close_median_raw_prob"] - merged["entry_median_raw_prob"],
            np.nan,
        )

    merged = merged.sort_values(["game_date", "game_id", "player_name"]).reset_index(drop=True)
    merged.to_csv(OUT / "joined_player_prices.csv", index=False)

    primary = strategy_stats(merged, PRIMARY_EV_THRESHOLD, PRIMARY_MAX_DECIMAL_ODDS)
    sensitivity = [strategy_stats(merged, t, PRIMARY_MAX_DECIMAL_ODDS) for t in [0.0, 0.03, 0.05, 0.08, 0.10]]
    picks = merged[(merged["model_ev"] >= PRIMARY_EV_THRESHOLD) & (merged["decimal_odds"] <= PRIMARY_MAX_DECIMAL_ODDS)].copy()
    picks.to_csv(OUT / "primary_strategy_picks.csv", index=False)

    model_brier = float(brier_score_loss(merged["target_goal"], merged["model_prob"]))
    model_log = float(log_loss(merged["target_goal"], merged["model_prob"], labels=[0, 1]))
    market_p = np.clip(merged["market_prob"].to_numpy(float), 1e-6, 1 - 1e-6)
    market_brier = float(brier_score_loss(merged["target_goal"], market_p))
    market_log = float(log_loss(merged["target_goal"], market_p, labels=[0, 1]))

    clv_coverage = int(picks["close_best_decimal_odds"].notna().sum()) if "close_best_decimal_odds" in picks else 0
    mean_clv_log = float(picks["clv_log_price"].dropna().mean()) if "clv_log_price" in picks and picks["clv_log_price"].notna().any() else math.nan
    mean_clv_prob = float(picks["clv_implied_prob"].dropna().mean()) if "clv_implied_prob" in picks and picks["clv_implied_prob"].notna().any() else math.nan

    summary = {
        "provider": "The Odds API",
        "sport": SPORT,
        "market": MARKET,
        "regions": REGIONS,
        "entry_minutes_before": PRIMARY_MINUTES_BEFORE,
        "near_close_minutes_before": CLOSE_MINUTES_BEFORE,
        "v2_test_games": int(games["nhl_game_id"].nunique()),
        "mapped_games": int(mapped["nhl_game_id"].nunique()),
        "joined_player_rows": int(len(merged)),
        "joined_games": int(merged["game_id"].nunique()),
        "api_calls": client.calls,
        "credits_used_header": client.credits_used_header,
        "credits_remaining": client.credits_remaining,
        "model_brier_on_joined": model_brier,
        "market_brier_on_joined": market_brier,
        "model_log_loss_on_joined": model_log,
        "market_log_loss_on_joined": market_log,
        "primary_strategy": primary,
        "primary_clv_rows": clv_coverage,
        "primary_mean_log_price_clv": mean_clv_log,
        "primary_mean_implied_prob_clv": mean_clv_prob,
        "sensitivity": sensitivity,
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    lines = [
        "# Historical NHL anytime-goal backtest — The Odds API",
        "",
        "This freezes the previously tested V2 logistic model before market prices are examined.",
        "Historical player-prop prices are queried at fixed pregame timestamps. No synthetic or fixed odds are used.",
        "",
        f"- Regions: **{REGIONS}**",
        f"- Entry snapshot target: **{PRIMARY_MINUTES_BEFORE} minutes before puck drop**",
        f"- Near-close snapshot target: **{CLOSE_MINUTES_BEFORE} minutes before puck drop**",
        f"- V2 test games considered: **{summary['v2_test_games']}**",
        f"- Games mapped to The Odds API: **{summary['mapped_games']}**",
        f"- Joined player-price rows: **{summary['joined_player_rows']}** across **{summary['joined_games']}** games",
        f"- API calls made: **{client.calls}**",
        f"- API credits remaining after run: **{client.credits_remaining}**",
        "",
        "## Probability benchmark on rows with historical odds",
        "",
        "| Source | Brier | Log Loss |",
        "|---|---:|---:|",
        f"| Frozen V2 model | {model_brier:.6f} | {model_log:.6f} |",
        f"| Market implied probability | {market_brier:.6f} | {market_log:.6f} |",
        "",
        "The market probability uses the median no-vig Yes probability when both Yes and No are available at a bookmaker; otherwise it uses median raw implied probability.",
        "",
        "## Primary strategy fixed before results",
        "",
        f"Rule: model EV >= **{PRIMARY_EV_THRESHOLD:.0%}**, best listed entry price <= **{PRIMARY_MAX_DECIMAL_ODDS:.2f}**, flat stake **1 unit**.",
        "",
        f"- Bets: **{primary['bets']}**",
        f"- Wins: **{primary['wins']}**",
        f"- Hit rate: **{primary['hit_rate']:.2%}**" if np.isfinite(primary["hit_rate"]) else "- Hit rate: n/a",
        f"- Average odds: **{primary['avg_odds']:.3f}**" if np.isfinite(primary["avg_odds"]) else "- Average odds: n/a",
        f"- Net units: **{primary['net_units']:.2f}**",
        f"- ROI: **{primary['roi']:.2%}**" if np.isfinite(primary["roi"]) else "- ROI: n/a",
        f"- Maximum drawdown: **{primary['max_drawdown_units']:.2f} units**" if np.isfinite(primary["max_drawdown_units"]) else "- Maximum drawdown: n/a",
        f"- Cluster-bootstrap ROI 95% interval: **[{primary['roi_ci95_low']:.2%}, {primary['roi_ci95_high']:.2%}]**" if np.isfinite(primary["roi_ci95_low"]) else "- ROI confidence interval: n/a",
        f"- Picks with near-close comparison: **{clv_coverage}**",
        f"- Mean log price CLV: **{mean_clv_log:.4f}**" if np.isfinite(mean_clv_log) else "- Mean log price CLV: n/a",
        f"- Mean change in market implied probability: **{mean_clv_prob:.3%}**" if np.isfinite(mean_clv_prob) else "- Mean change in market implied probability: n/a",
        "",
        "## Sensitivity analysis (not a post-hoc strategy selection)",
        "",
        "| EV threshold | Bets | Net units | ROI |",
        "|---:|---:|---:|---:|",
    ]
    for s in sensitivity:
        roi = f"{s['roi']:.2%}" if np.isfinite(s["roi"]) else "n/a"
        lines.append(f"| {s['ev_threshold']:.0%} | {s['bets']} | {s['net_units']:.2f} | {roi} |")
    lines += [
        "",
        "## Scope limitation",
        "",
        "The default `us` region is a historical market-efficiency benchmark, not proof that the same prices were legally available to a bettor in France. French-market profitability requires timestamp-valid prices from bookmakers actually available in France.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
