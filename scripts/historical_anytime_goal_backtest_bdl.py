from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import scripts.real_nhl_player_goal_v2 as v2

BDL_BASE = "https://api.balldontlie.io/nhl/v1"
OUT = Path("workspace/historical_anytime_goal_bdl")
REPORT = Path("reports/historical_anytime_goal_bdl.md")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "home",
    "position_D",
    "prior_games",
    "days_since_prev",
    "career_goal_rate",
    "target_goal_avg_3",
    "target_goal_avg_5",
    "target_goal_avg_10",
    "goals_avg_3",
    "goals_avg_5",
    "goals_avg_10",
    "points_avg_3",
    "points_avg_5",
    "points_avg_10",
    "shots_avg_3",
    "shots_avg_5",
    "shots_avg_10",
    "toi_avg_3",
    "toi_avg_5",
    "toi_avg_10",
    "shots_trend_5v10",
    "toi_trend_5v10",
    "goal_form_5v10",
    "team_rest_days",
    "team_goals_for_5",
    "team_goals_for_10",
    "team_shots_for_5",
    "team_shots_for_10",
    "team_goals_allowed_5",
    "team_goals_allowed_10",
    "team_shots_allowed_5",
    "team_shots_allowed_10",
    "opp_rest_days",
    "opp_goals_for_5",
    "opp_goals_for_10",
    "opp_shots_for_5",
    "opp_shots_for_10",
    "opp_goals_allowed_5",
    "opp_goals_allowed_10",
    "opp_shots_allowed_5",
    "opp_shots_allowed_10",
]

PRIMARY_EV_THRESHOLD = 0.05
PRIMARY_MAX_DECIMAL_ODDS = 8.0


def normalise_name(value: str | None) -> str:
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    tokens = [t for t in s.split() if t not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(tokens)


def american_to_decimal(value) -> float:
    try:
        a = float(value)
    except (TypeError, ValueError):
        return math.nan
    if a == 0:
        return math.nan
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / abs(a))


class BDLClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("BALLDONTLIE_API_KEY is required")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": api_key,
            "User-Agent": "nhl-goal-probability-research-team/0.6",
        })
        self.min_interval = float(os.getenv("BDL_MIN_INTERVAL_SECONDS", "12.3"))
        self.last_request = 0.0
        self.calls = 0

    def get(self, path: str, params=None, tries: int = 7) -> dict:
        url = f"{BDL_BASE}/{path.lstrip('/')}"
        last = None
        for attempt in range(tries):
            wait = self.min_interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.session.get(url, params=params, timeout=45)
                self.last_request = time.monotonic()
                self.calls += 1
                if r.status_code == 429:
                    retry = float(r.headers.get("Retry-After", max(13.0, self.min_interval)))
                    time.sleep(retry)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                last = exc
                time.sleep(min(30.0, 2.0 * (attempt + 1)))
        raise RuntimeError(f"BDL request failed {url}: {last}")


def build_frozen_v2_test_predictions() -> pd.DataFrame:
    df = v2.build_dataset().replace([np.inf, -np.inf], np.nan).copy()
    dates = sorted(df["game_date"].dt.normalize().unique())
    if len(dates) < 30:
        raise RuntimeError(f"Not enough dates for frozen V2 split: {len(dates)}")
    train_cut = dates[int(len(dates) * 0.70)]
    val_cut = dates[int(len(dates) * 0.85)]
    train = df[df["game_date"] < train_cut].copy()
    test = df[df["game_date"] >= val_cut].copy()

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2500, C=0.7)),
    ])
    model.fit(train[FEATURES], train["target_goal"])
    test["model_prob"] = np.clip(model.predict_proba(test[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    test["game_date_str"] = test["game_date"].dt.strftime("%Y-%m-%d")
    test["name_key"] = test["player_name"].map(normalise_name)
    return test[
        [
            "game_date", "game_date_str", "game_id", "player_id", "player_name", "name_key",
            "team", "opponent", "home", "target_goal", "model_prob",
        ]
    ].copy()


def test_game_keys(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, g in test.groupby("game_id"):
        home = g[g["home"] == 1]
        away = g[g["home"] == 0]
        if home.empty or away.empty:
            continue
        rows.append({
            "nhl_game_id": int(gid),
            "game_date": str(g["game_date_str"].iloc[0]),
            "home": str(home["team"].iloc[0]),
            "away": str(away["team"].iloc[0]),
        })
    return pd.DataFrame(rows)


def repeated_params(name: str, values: Iterable) -> list[tuple[str, str]]:
    return [(name, str(v)) for v in values]


def fetch_bdl_games(client: BDLClient, dates: list[str]) -> pd.DataFrame:
    base = repeated_params("dates[]", dates) + [("per_page", "100")]
    data: list[dict] = []
    cursor = None
    while True:
        params = list(base)
        if cursor is not None:
            params.append(("cursor", str(cursor)))
        payload = client.get("games", params=params)
        data.extend(payload.get("data", []) or [])
        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
    rows = []
    for x in data:
        home = x.get("home_team") or {}
        away = x.get("away_team") or {}
        rows.append({
            "bdl_game_id": x.get("id"),
            "game_date": x.get("game_date"),
            "start_time_utc": x.get("start_time_utc"),
            "home": home.get("tricode"),
            "away": away.get("tricode"),
            "status_state": x.get("status_state"),
        })
    return pd.DataFrame(rows)


def fetch_opening_props(client: BDLClient, game_id: int) -> list[dict]:
    payload = client.get(
        "odds/player_props/opening",
        params={"game_id": int(game_id), "prop_type": "anytime_goal"},
    )
    out = []
    for x in payload.get("data", []) or []:
        market = x.get("market") or {}
        if x.get("prop_type") != "anytime_goal" or market.get("type") != "milestone":
            continue
        dec = american_to_decimal(market.get("odds"))
        if not np.isfinite(dec) or dec <= 1:
            continue
        out.append({
            "bdl_game_id": int(game_id),
            "bdl_player_id": x.get("player_id"),
            "vendor": x.get("vendor"),
            "american_odds": market.get("odds"),
            "decimal_odds": dec,
            "opened_at": x.get("opened_at"),
        })
    return out


def fetch_player_names(client: BDLClient, ids: list[int]) -> dict[int, str]:
    names: dict[int, str] = {}
    ids = sorted(set(int(x) for x in ids if pd.notna(x)))
    for start in range(0, len(ids), 75):
        chunk = ids[start:start + 75]
        base = repeated_params("player_ids[]", chunk) + [("per_page", "100")]
        cursor = None
        while True:
            params = list(base)
            if cursor is not None:
                params.append(("cursor", str(cursor)))
            payload = client.get("players", params=params)
            for p in payload.get("data", []) or []:
                if p.get("id") is not None:
                    names[int(p["id"])] = str(p.get("full_name") or "")
            cursor = (payload.get("meta") or {}).get("next_cursor")
            if not cursor:
                break
    return names


def max_drawdown(profits: pd.Series) -> float:
    if profits.empty:
        return math.nan
    curve = profits.cumsum()
    peak = curve.cummax().clip(lower=0)
    return float((peak - curve).max())


def cluster_bootstrap_roi(picks: pd.DataFrame, reps: int = 1200) -> tuple[float, float]:
    if picks.empty or picks["game_id"].nunique() < 2:
        return math.nan, math.nan
    groups = [g["profit_unit"].to_numpy(float) for _, g in picks.groupby("game_id")]
    rng = np.random.default_rng(42)
    rois = []
    for _ in range(reps):
        sampled = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        vals = np.concatenate(sampled)
        rois.append(float(vals.mean()))
    lo, hi = np.quantile(rois, [0.025, 0.975])
    return float(lo), float(hi)


def strategy_stats(df: pd.DataFrame, ev_threshold: float, max_odds: float = PRIMARY_MAX_DECIMAL_ODDS) -> dict:
    picks = df[(df["model_ev"] >= ev_threshold) & (df["decimal_odds"] <= max_odds)].copy()
    picks = picks.sort_values(["game_date", "game_id", "player_name"])
    if picks.empty:
        return {
            "ev_threshold": ev_threshold, "max_decimal_odds": max_odds, "bets": 0,
            "wins": 0, "hit_rate": math.nan, "avg_odds": math.nan,
            "net_units": 0.0, "roi": math.nan, "max_drawdown_units": math.nan,
            "roi_ci95_low": math.nan, "roi_ci95_high": math.nan,
        }
    lo, hi = cluster_bootstrap_roi(picks)
    return {
        "ev_threshold": ev_threshold,
        "max_decimal_odds": max_odds,
        "bets": int(len(picks)),
        "wins": int(picks["target_goal"].sum()),
        "hit_rate": float(picks["target_goal"].mean()),
        "avg_odds": float(picks["decimal_odds"].mean()),
        "net_units": float(picks["profit_unit"].sum()),
        "roi": float(picks["profit_unit"].mean()),
        "max_drawdown_units": max_drawdown(picks["profit_unit"]),
        "roi_ci95_low": lo,
        "roi_ci95_high": hi,
    }


def main() -> None:
    key = os.getenv("BALLDONTLIE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "BALLDONTLIE_API_KEY is missing. Add it as a GitHub Actions secret; never commit API keys."
        )

    print("Building frozen leakage-safe V2 test predictions...")
    test = build_frozen_v2_test_predictions()
    games = test_game_keys(test)
    dates = sorted(games["game_date"].unique().tolist())
    print(f"V2 test rows={len(test)} games={len(games)} dates={len(dates)}")

    client = BDLClient(key)
    bdl_games = fetch_bdl_games(client, dates)
    mapped = games.merge(bdl_games, on=["game_date", "home", "away"], how="left")
    mapped_ok = mapped[mapped["bdl_game_id"].notna()].copy()
    print(f"BDL mapped games={len(mapped_ok)}/{len(games)}")

    prop_rows: list[dict] = []
    prop_errors: list[dict] = []
    for i, row in enumerate(mapped_ok.itertuples(index=False), start=1):
        try:
            props = fetch_opening_props(client, int(row.bdl_game_id))
            for p in props:
                p["nhl_game_id"] = int(row.nhl_game_id)
                p["game_date"] = row.game_date
                p["start_time_utc"] = row.start_time_utc
            prop_rows.extend(props)
        except Exception as exc:
            prop_errors.append({"bdl_game_id": int(row.bdl_game_id), "error": repr(exc)})
        if i % 10 == 0 or i == len(mapped_ok):
            print(f"opening props {i}/{len(mapped_ok)} rows={len(prop_rows)} errors={len(prop_errors)} api_calls={client.calls}")

    props = pd.DataFrame(prop_rows)
    if props.empty:
        raise RuntimeError("No historical opening anytime-goal props were returned")

    player_names = fetch_player_names(client, props["bdl_player_id"].dropna().astype(int).tolist())
    props["player_name_bdl"] = props["bdl_player_id"].map(player_names)
    props["name_key"] = props["player_name_bdl"].map(normalise_name)
    props["opened_at_dt"] = pd.to_datetime(props["opened_at"], utc=True, errors="coerce")
    props["start_time_dt"] = pd.to_datetime(props["start_time_utc"], utc=True, errors="coerce")
    props["timestamp_valid"] = (
        props["opened_at_dt"].notna()
        & props["start_time_dt"].notna()
        & (props["opened_at_dt"] < props["start_time_dt"])
    )
    props = props[props["timestamp_valid"]].copy()

    merged = test.merge(
        props,
        left_on=["game_id", "name_key"],
        right_on=["nhl_game_id", "name_key"],
        how="inner",
        suffixes=("", "_odds"),
    )
    if merged.empty:
        raise RuntimeError("Historical odds returned, but no player-name/game joins succeeded")

    merged["raw_implied_prob"] = 1.0 / merged["decimal_odds"]
    merged["model_edge_prob"] = merged["model_prob"] - merged["raw_implied_prob"]
    merged["model_ev"] = merged["model_prob"] * merged["decimal_odds"] - 1.0

    # Best available opening price across the BDL-supported US books. This is a market benchmark,
    # not a claim that these sportsbooks are legally accessible to a bettor in France.
    best_idx = merged.groupby(["game_id", "player_id"])["decimal_odds"].idxmax()
    best = merged.loc[best_idx].copy().sort_values(["game_date", "game_id", "player_name"])
    best["profit_unit"] = np.where(best["target_goal"] == 1, best["decimal_odds"] - 1.0, -1.0)

    primary = best[(best["model_ev"] >= PRIMARY_EV_THRESHOLD) & (best["decimal_odds"] <= PRIMARY_MAX_DECIMAL_ODDS)].copy()
    sensitivity = [strategy_stats(best, x) for x in [0.00, 0.03, 0.05, 0.08, 0.10]]
    primary_stats = strategy_stats(best, PRIMARY_EV_THRESHOLD)

    quoted_y = best["target_goal"].to_numpy(int)
    model_p = np.clip(best["model_prob"].to_numpy(float), 1e-6, 1 - 1e-6)
    market_p = np.clip(best["raw_implied_prob"].to_numpy(float), 1e-6, 1 - 1e-6)

    summary = {
        "provider": "BALLDONTLIE NHL opening player props",
        "market": "anytime_goal",
        "price_type": "opening",
        "v2_test_rows": int(len(test)),
        "v2_test_games": int(test["game_id"].nunique()),
        "bdl_mapped_games": int(len(mapped_ok)),
        "games_with_valid_props": int(best["game_id"].nunique()),
        "all_vendor_quote_rows_joined": int(len(merged)),
        "unique_player_games_with_best_quote": int(len(best)),
        "quoted_player_row_coverage": float(len(best) / max(1, len(test))),
        "bdl_api_calls": int(client.calls),
        "prop_fetch_errors": int(len(prop_errors)),
        "quoted_subset_scoring": {
            "model_brier": float(brier_score_loss(quoted_y, model_p)),
            "raw_best_price_implied_brier": float(brier_score_loss(quoted_y, market_p)),
            "model_log_loss": float(log_loss(quoted_y, model_p, labels=[0, 1])),
            "raw_best_price_implied_log_loss": float(log_loss(quoted_y, market_p, labels=[0, 1])),
        },
        "primary_rule_predeclared": {
            "description": "Best opening price; model EV >= 5%; decimal odds <= 8.00; flat 1 unit",
            **primary_stats,
        },
        "threshold_sensitivity_exploratory": sensitivity,
        "limitations": [
            "BDL opening props are a US-book market benchmark, not evidence of availability on French-regulated sportsbooks.",
            "The one-sided anytime-goal milestone price embeds bookmaker margin; raw implied probability is not de-vigged fair probability.",
            "This backtest uses opening prices only, not closing-line value.",
            "Only players who appear in the NHL historical player-game dataset are evaluated; sportsbook void/scratch rules are not reconstructed separately.",
            "No strategy threshold is to be selected after seeing this test set; the 5% EV / 8.00 odds-cap rule is the primary predeclared rule.",
        ],
    }

    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    best.to_csv(OUT / "best_opening_quotes.csv", index=False)
    primary.to_csv(OUT / "primary_picks.csv", index=False)
    if prop_errors:
        (OUT / "prop_fetch_errors.json").write_text(json.dumps(prop_errors, indent=2), encoding="utf-8")

    lines = [
        "# Historical NHL anytime-goal economic backtest — BALLDONTLIE opening prices",
        "",
        "This report keeps the V2 logistic probability model frozen and evaluates it only against timestamp-valid historical opening anytime-goal prices.",
        "",
        f"- V2 test games: **{summary['v2_test_games']}**",
        f"- BDL mapped games: **{summary['bdl_mapped_games']}**",
        f"- Games with joined valid props: **{summary['games_with_valid_props']}**",
        f"- Player-games with a best opening quote: **{summary['unique_player_games_with_best_quote']}**",
        f"- Quoted player-row coverage: **{summary['quoted_player_row_coverage']:.1%}**",
        "",
        "## Same quoted subset: probability scoring",
        "",
        f"- Model Brier: **{summary['quoted_subset_scoring']['model_brier']:.6f}**",
        f"- Raw best-price implied Brier: **{summary['quoted_subset_scoring']['raw_best_price_implied_brier']:.6f}**",
        f"- Model Log Loss: **{summary['quoted_subset_scoring']['model_log_loss']:.6f}**",
        f"- Raw best-price implied Log Loss: **{summary['quoted_subset_scoring']['raw_best_price_implied_log_loss']:.6f}**",
        "",
        "The market implied probability is raw (1 / decimal odds), not de-vigged, because anytime-goal milestone pricing is one-sided in this feed.",
        "",
        "## Primary predeclared economic rule",
        "",
        "Best available opening price across supported BDL vendors; model EV >= 5%; decimal odds <= 8.00; flat 1-unit stake.",
        "",
        f"- Bets: **{primary_stats['bets']}**",
        f"- Wins: **{primary_stats['wins']}**",
        f"- Hit rate: **{primary_stats['hit_rate']:.2%}**" if np.isfinite(primary_stats['hit_rate']) else "- Hit rate: n/a",
        f"- Average odds: **{primary_stats['avg_odds']:.3f}**" if np.isfinite(primary_stats['avg_odds']) else "- Average odds: n/a",
        f"- Net units: **{primary_stats['net_units']:.2f}**",
        f"- ROI: **{primary_stats['roi']:.2%}**" if np.isfinite(primary_stats['roi']) else "- ROI: n/a",
        f"- Max drawdown: **{primary_stats['max_drawdown_units']:.2f} units**" if np.isfinite(primary_stats['max_drawdown_units']) else "- Max drawdown: n/a",
        f"- Cluster-bootstrap ROI 95% interval: **[{primary_stats['roi_ci95_low']:.2%}, {primary_stats['roi_ci95_high']:.2%}]**" if np.isfinite(primary_stats['roi_ci95_low']) else "- ROI uncertainty interval: n/a",
        "",
        "## Exploratory threshold sensitivity",
        "",
        "| EV threshold | Bets | Net units | ROI | Max DD |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s in sensitivity:
        roi = f"{s['roi']:.2%}" if np.isfinite(s['roi']) else "n/a"
        dd = f"{s['max_drawdown_units']:.2f}" if np.isfinite(s['max_drawdown_units']) else "n/a"
        lines.append(f"| {s['ev_threshold']:.0%} | {s['bets']} | {s['net_units']:.2f} | {roi} | {dd} |")
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- The 5% EV / 8.00 cap rule is the primary rule fixed before looking at these economic results; the other thresholds are sensitivity checks only.",
        "- These vendors are a US market benchmark. This report does not claim that their prices are directly wagerable in France.",
        "- A positive ROI on one held-out period is not proof of durable profitability. A negative ROI is also informative and must remain visible.",
        "- Opening prices do not provide CLV. A separate timestamp-snapshot source is needed for robust closing-line analysis.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
