from __future__ import annotations

import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = "https://api-web.nhle.com/v1"
OUT = Path("workspace/real_nhl")
REPORT = Path("reports/real_nhl_smoke.md")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "nhl-goal-probability-research-team/0.2"})


def get_json(path: str, tries: int = 4) -> dict:
    url = f"{BASE}/{path.lstrip('/')}"
    last = None
    for attempt in range(tries):
        try:
            r = SESSION.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed {url}: {last}")


def toi_minutes(value) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    if ":" in s:
        mm, ss = s.split(":", 1)
        return float(mm) + float(ss) / 60.0
    try:
        return float(s)
    except ValueError:
        return math.nan


def collect_game_ids(max_games: int = 160) -> list[tuple[str, int]]:
    # Use a completed regular-season window from 2025-26 so the test remains reproducible.
    start = date(2026, 1, 5)
    end = date(2026, 4, 16)
    found: dict[int, str] = {}
    cursor = start
    while cursor <= end:
        payload = get_json(f"schedule/{cursor.isoformat()}")
        for day in payload.get("gameWeek", []):
            game_date = day.get("date")
            for game in day.get("games", []):
                if int(game.get("gameType", 0)) != 2:
                    continue
                gid = game.get("id")
                if gid and game_date:
                    found[int(gid)] = str(game_date)
        cursor += timedelta(days=7)
    items = sorted(((d, gid) for gid, d in found.items()))
    return items[-max_games:]


def skater_rows(game_date: str, game_id: int, box: dict) -> list[dict]:
    rows: list[dict] = []
    pstats = box.get("playerByGameStats") or {}
    away = box.get("awayTeam") or {}
    home = box.get("homeTeam") or {}
    team_meta = {
        "awayTeam": (away.get("abbrev"), home.get("abbrev"), 0),
        "homeTeam": (home.get("abbrev"), away.get("abbrev"), 1),
    }
    for side, (team, opponent, is_home) in team_meta.items():
        section = pstats.get(side) or {}
        for group in ("forwards", "defense"):
            for p in section.get(group, []) or []:
                name_obj = p.get("name") or {}
                rows.append(
                    {
                        "game_date": game_date,
                        "game_id": game_id,
                        "player_id": p.get("playerId"),
                        "player_name": name_obj.get("default") if isinstance(name_obj, dict) else str(name_obj),
                        "position": p.get("position") or ("D" if group == "defense" else "F"),
                        "team": team,
                        "opponent": opponent,
                        "home": is_home,
                        "goals": float(p.get("goals", 0) or 0),
                        "shots": float(p.get("sog", p.get("shots", 0)) or 0),
                        "toi": toi_minutes(p.get("toi")),
                    }
                )
    return rows


def build_dataset() -> pd.DataFrame:
    games = collect_game_ids()
    all_rows: list[dict] = []
    errors: list[dict] = []
    for i, (game_date, gid) in enumerate(games, start=1):
        try:
            box = get_json(f"gamecenter/{gid}/boxscore")
            all_rows.extend(skater_rows(game_date, gid, box))
        except Exception as exc:
            errors.append({"game_id": gid, "game_date": game_date, "error": str(exc)})
        if i % 25 == 0:
            print(f"fetched {i}/{len(games)} games; rows={len(all_rows)}; errors={len(errors)}")

    if errors:
        (OUT / "fetch_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No player rows were returned from the NHL API")
    df = df.dropna(subset=["player_id", "game_date"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["target_goal"] = (df["goals"] >= 1).astype(int)
    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    g = df.groupby("player_id", group_keys=False)
    df["prior_games"] = g.cumcount()
    for col in ["target_goal", "shots", "toi"]:
        for window in [5, 10]:
            df[f"{col}_avg_{window}"] = g[col].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )
    df["days_since_prev"] = g["game_date"].diff().dt.days.clip(lower=0, upper=14)
    df = df[df["prior_games"] >= 3].copy()
    df["days_since_prev"] = df["days_since_prev"].fillna(7)
    return df


def calibration_table(y_true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    tmp = pd.DataFrame({"y": y_true, "p": pred})
    tmp["bin"] = pd.cut(tmp["p"], bins=np.linspace(0, 1, 11), include_lowest=True)
    out = tmp.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), actual_rate=("y", "mean"))
    return out.reset_index()


def train_and_report(df: pd.DataFrame) -> None:
    unique_dates = sorted(df["game_date"].dt.normalize().unique())
    if len(unique_dates) < 12:
        raise RuntimeError(f"Not enough distinct dates after feature construction: {len(unique_dates)}")
    train_cut = unique_dates[int(len(unique_dates) * 0.70)]
    val_cut = unique_dates[int(len(unique_dates) * 0.85)]

    train = df[df["game_date"] < train_cut].copy()
    val = df[(df["game_date"] >= train_cut) & (df["game_date"] < val_cut)].copy()
    test = df[df["game_date"] >= val_cut].copy()

    numeric = [
        "home",
        "prior_games",
        "target_goal_avg_5",
        "target_goal_avg_10",
        "shots_avg_5",
        "shots_avg_10",
        "toi_avg_5",
        "toi_avg_10",
        "days_since_prev",
    ]
    categorical = ["position"]
    needed = numeric + categorical
    train = train.dropna(subset=needed)
    val = val.dropna(subset=needed)
    test = test.dropna(subset=needed)

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1500, C=1.0)),
        ]
    )
    model.fit(train[needed], train["target_goal"])

    pred_val = model.predict_proba(val[needed])[:, 1]
    pred_test = model.predict_proba(test[needed])[:, 1]
    y_test = test["target_goal"].to_numpy()

    train_base = float(train["target_goal"].mean())
    base_pred = np.full(len(test), train_base)
    metrics = {
        "rows_total": int(len(df)),
        "players": int(df["player_id"].nunique()),
        "games": int(df["game_id"].nunique()),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_base_rate": train_base,
        "validation_brier": float(brier_score_loss(val["target_goal"], pred_val)),
        "test_brier": float(brier_score_loss(y_test, pred_test)),
        "test_log_loss": float(log_loss(y_test, pred_test, labels=[0, 1])),
        "baseline_test_brier": float(brier_score_loss(y_test, base_pred)),
        "baseline_test_log_loss": float(log_loss(y_test, base_pred, labels=[0, 1])),
        "test_roc_auc": float(roc_auc_score(y_test, pred_test)) if len(np.unique(y_test)) == 2 else math.nan,
        "train_end_exclusive": str(pd.Timestamp(train_cut).date()),
        "validation_end_exclusive": str(pd.Timestamp(val_cut).date()),
    }

    preds = test[["game_date", "game_id", "player_id", "player_name", "team", "opponent", "target_goal"]].copy()
    preds["predicted_goal_probability"] = pred_test
    preds.to_csv(OUT / "test_predictions.csv", index=False)
    df.to_csv(OUT / "player_game_features.csv", index=False)
    cal = calibration_table(y_test, pred_test)
    cal.to_csv(OUT / "calibration.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    verdict = "BEATS BASE RATE" if metrics["test_brier"] < metrics["baseline_test_brier"] else "DOES NOT BEAT BASE RATE"
    report = f"""# Real NHL player-goal smoke test\n\n## Status\n\n**{verdict} on test-set Brier score.** This is a research smoke test, not a betting model.\n\n## Data\n\n- Source: NHL public web API (`api-web.nhle.com`)\n- Real completed 2025-26 regular-season boxscores only\n- Player-game rows after requiring at least 3 prior appearances: **{metrics['rows_total']}**\n- Players: **{metrics['players']}**\n- Games represented: **{metrics['games']}**\n- Train / validation / test rows: **{metrics['train_rows']} / {metrics['validation_rows']} / {metrics['test_rows']}**\n- Training goal base rate: **{metrics['train_base_rate']:.3%}**\n\n## Leakage controls\n\nEvery rolling feature is shifted by one game before the rolling average. The current game's goals, shots and TOI are therefore excluded from its features. Splits are chronological by game date.\n\n## Metrics\n\n| Metric | Model | Train-base-rate baseline |\n|---|---:|---:|\n| Test Brier | {metrics['test_brier']:.6f} | {metrics['baseline_test_brier']:.6f} |\n| Test log loss | {metrics['test_log_loss']:.6f} | {metrics['baseline_test_log_loss']:.6f} |\n| Test ROC-AUC | {metrics['test_roc_auc']:.6f} | n/a |\n| Validation Brier | {metrics['validation_brier']:.6f} | n/a |\n\nTrain ends before **{metrics['train_end_exclusive']}**; validation ends before **{metrics['validation_end_exclusive']}**.\n\n## What this does not validate\n\nNo historical anytime-goal bookmaker prices are included, so **ROI, EV, CLV, Kelly sizing and profitability are NOT validated**. The feature set is intentionally small and currently omits expected line/PP role, goalie quality, opponent defense, injuries, xG and market information.\n\n## Calibration\n\n{cal.to_markdown(index=False)}\n\n## Artifacts\n\n- `workspace/real_nhl/player_game_features.csv`\n- `workspace/real_nhl/test_predictions.csv`\n- `workspace/real_nhl/calibration.csv`\n- `workspace/real_nhl/metrics.json`\n"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"report={REPORT}")


def main() -> int:
    df = build_dataset()
    train_and_report(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
