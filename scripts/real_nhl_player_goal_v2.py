from __future__ import annotations

import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = "https://api-web.nhle.com/v1"
OUT = Path("workspace/real_nhl_v2")
REPORT = Path("reports/real_nhl_v2.md")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "nhl-goal-probability-research-team/0.3"})


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
            time.sleep(1.25 * (attempt + 1))
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


def collect_game_ids(max_games: int = 500) -> list[tuple[str, int]]:
    start = date(2025, 10, 7)
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
                        "assists": float(p.get("assists", 0) or 0),
                        "points": float((p.get("goals", 0) or 0) + (p.get("assists", 0) or 0)),
                        "shots": float(p.get("sog", p.get("shots", 0)) or 0),
                        "toi": toi_minutes(p.get("toi")),
                    }
                )
    return rows


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date", "game_id"]).copy()
    g = df.groupby("player_id", group_keys=False)
    df["target_goal"] = (df["goals"] >= 1).astype(int)
    df["prior_games"] = g.cumcount()
    df["days_since_prev"] = g["game_date"].diff().dt.days.clip(lower=0, upper=14).fillna(7)

    for col in ["target_goal", "goals", "points", "shots", "toi"]:
        for window in [3, 5, 10]:
            df[f"{col}_avg_{window}"] = g[col].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )

    df["career_goal_rate"] = g["target_goal"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["shots_trend_5v10"] = df["shots_avg_5"] - df["shots_avg_10"]
    df["toi_trend_5v10"] = df["toi_avg_5"] - df["toi_avg_10"]
    df["goal_form_5v10"] = df["target_goal_avg_5"] - df["target_goal_avg_10"]
    df["position_D"] = (df["position"] == "D").astype(int)
    return df


def add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    tg = (
        df.groupby(["game_id", "game_date", "team", "opponent", "home"], as_index=False)
        .agg(team_goals=("goals", "sum"), team_shots=("shots", "sum"))
        .sort_values(["team", "game_date", "game_id"])
    )

    opp_actual = tg[["game_id", "team", "team_goals", "team_shots"]].rename(
        columns={
            "team": "opponent",
            "team_goals": "opp_goals_actual",
            "team_shots": "opp_shots_actual",
        }
    )
    tg = tg.merge(opp_actual, on=["game_id", "opponent"], how="left")
    team_g = tg.groupby("team", group_keys=False)
    tg["team_rest_days"] = team_g["game_date"].diff().dt.days.clip(lower=0, upper=14).fillna(7)

    for source, stem in [
        ("team_goals", "goals_for"),
        ("team_shots", "shots_for"),
        ("opp_goals_actual", "goals_allowed"),
        ("opp_shots_actual", "shots_allowed"),
    ]:
        for window in [5, 10]:
            tg[f"team_{stem}_{window}"] = team_g[source].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )

    own_cols = [
        "game_id",
        "team",
        "team_rest_days",
        "team_goals_for_5",
        "team_goals_for_10",
        "team_shots_for_5",
        "team_shots_for_10",
        "team_goals_allowed_5",
        "team_goals_allowed_10",
        "team_shots_allowed_5",
        "team_shots_allowed_10",
    ]
    df = df.merge(tg[own_cols], on=["game_id", "team"], how="left")

    opp_cols = own_cols.copy()
    opp = tg[opp_cols].rename(
        columns={
            "team": "opponent",
            "team_rest_days": "opp_rest_days",
            "team_goals_for_5": "opp_goals_for_5",
            "team_goals_for_10": "opp_goals_for_10",
            "team_shots_for_5": "opp_shots_for_5",
            "team_shots_for_10": "opp_shots_for_10",
            "team_goals_allowed_5": "opp_goals_allowed_5",
            "team_goals_allowed_10": "opp_goals_allowed_10",
            "team_shots_allowed_5": "opp_shots_allowed_5",
            "team_shots_allowed_10": "opp_shots_allowed_10",
        }
    )
    df = df.merge(opp, on=["game_id", "opponent"], how="left")
    return df


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
        if i % 50 == 0:
            print(f"fetched {i}/{len(games)} games; rows={len(all_rows)}; errors={len(errors)}")

    if errors:
        (OUT / "fetch_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No player rows returned from NHL API")
    df = df.dropna(subset=["player_id", "game_date", "team", "opponent"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = add_player_features(df)
    df = add_team_features(df)
    df = df[df["prior_games"] >= 5].copy()
    return df


def expected_calibration_error(y_true: np.ndarray, pred: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.digitize(pred, edges[1:-1], right=False)
    total = len(y_true)
    ece = 0.0
    for b in range(bins):
        mask = ids == b
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(float(pred[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def calibration_table(y_true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    tmp = pd.DataFrame({"y": y_true, "p": pred})
    tmp["bin"] = pd.cut(tmp["p"], bins=np.linspace(0, 1, 11), include_lowest=True)
    return (
        tmp.groupby("bin", observed=True)
        .agg(n=("y", "size"), mean_pred=("p", "mean"), actual_rate=("y", "mean"))
        .reset_index()
    )


def score_model(name: str, model, x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    p_val = np.clip(model.predict_proba(x_val)[:, 1], 1e-6, 1 - 1e-6)
    p_test = np.clip(model.predict_proba(x_test)[:, 1], 1e-6, 1 - 1e-6)
    return {
        "name": name,
        "model": model,
        "p_val": p_val,
        "p_test": p_test,
        "validation_brier": float(brier_score_loss(y_val, p_val)),
        "validation_log_loss": float(log_loss(y_val, p_val, labels=[0, 1])),
        "test_brier": float(brier_score_loss(y_test, p_test)),
        "test_log_loss": float(log_loss(y_test, p_test, labels=[0, 1])),
        "test_auc": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) == 2 else math.nan,
        "test_ece": expected_calibration_error(np.asarray(y_test), p_test),
    }


def train_and_report(df: pd.DataFrame) -> None:
    feature_cols = [
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

    df = df.replace([np.inf, -np.inf], np.nan).copy()
    unique_dates = sorted(df["game_date"].dt.normalize().unique())
    if len(unique_dates) < 30:
        raise RuntimeError(f"Not enough distinct dates: {len(unique_dates)}")
    train_cut = unique_dates[int(len(unique_dates) * 0.70)]
    val_cut = unique_dates[int(len(unique_dates) * 0.85)]

    train = df[df["game_date"] < train_cut].copy()
    val = df[(df["game_date"] >= train_cut) & (df["game_date"] < val_cut)].copy()
    test = df[df["game_date"] >= val_cut].copy()

    x_train, y_train = train[feature_cols], train["target_goal"]
    x_val, y_val = val[feature_cols], val["target_goal"]
    x_test, y_test = test[feature_cols], test["target_goal"]

    models = {
        "logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2500, C=0.7)),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.035,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=5,
                        reg_lambda=2.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=2,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    lgb.LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.03,
                        num_leaves=15,
                        min_child_samples=35,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        random_state=42,
                        n_jobs=2,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
    }

    results = [
        score_model(name, model, x_train, y_train, x_val, y_val, x_test, y_test)
        for name, model in models.items()
    ]
    results = sorted(results, key=lambda r: r["validation_brier"])
    best = results[0]

    train_base = float(y_train.mean())
    baseline = np.full(len(test), train_base)
    baseline_metrics = {
        "test_brier": float(brier_score_loss(y_test, baseline)),
        "test_log_loss": float(log_loss(y_test, baseline, labels=[0, 1])),
    }

    summary = {
        "rows_total": int(len(df)),
        "players": int(df["player_id"].nunique()),
        "games": int(df["game_id"].nunique()),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_base_rate": train_base,
        "best_model_by_validation_brier": best["name"],
        "train_end_exclusive": str(pd.Timestamp(train_cut).date()),
        "validation_end_exclusive": str(pd.Timestamp(val_cut).date()),
        "baseline": baseline_metrics,
        "models": [
            {k: v for k, v in r.items() if k not in {"model", "p_val", "p_test"}}
            for r in results
        ],
    }

    pred_cols = [
        "game_date",
        "game_id",
        "player_id",
        "player_name",
        "team",
        "opponent",
        "target_goal",
    ]
    preds = test[pred_cols].copy()
    for r in results:
        preds[f"p_{r['name']}"] = r["p_test"]
    preds.to_csv(OUT / "test_predictions.csv", index=False)
    df.to_csv(OUT / "player_game_features.csv", index=False)
    cal = calibration_table(y_test.to_numpy(), best["p_test"])
    cal.to_csv(OUT / "calibration_best.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for r in results:
        rows.append(
            {
                "model": r["name"],
                "validation_brier": r["validation_brier"],
                "test_brier": r["test_brier"],
                "test_log_loss": r["test_log_loss"],
                "test_auc": r["test_auc"],
                "test_ece": r["test_ece"],
            }
        )
    table = pd.DataFrame(rows)

    improvement_brier = 1 - best["test_brier"] / baseline_metrics["test_brier"]
    improvement_log = 1 - best["test_log_loss"] / baseline_metrics["test_log_loss"]

    report = f"""# Real NHL player-goal model V2\n\n## Status\n\nBest validation model: **{best['name']}**.\n\nThis is still a research probability model, not a validated betting strategy.\n\n## Data\n\n- Source: NHL public web API (`api-web.nhle.com`)\n- Completed 2025-26 regular-season boxscores\n- Player-game rows after 5 prior appearances: **{len(df)}**\n- Players: **{df['player_id'].nunique()}**\n- Games: **{df['game_id'].nunique()}**\n- Train / validation / test rows: **{len(train)} / {len(val)} / {len(test)}**\n- Training scorer base rate: **{train_base:.3%}**\n\n## Leakage controls\n\n- Every player rolling statistic is shifted by one game before use.\n- Team and opponent rolling offense/defense statistics are also shifted by one game.\n- Current-game goals, shots, points and TOI are never used as predictors for that same game.\n- Splits are chronological by game date.\n\n## Added V2 pregame features\n\n- player goal rate / points / shots / TOI over 3, 5 and 10 games\n- long-run player scoring rate\n- short-vs-long form trends\n- player rest\n- team recent goals/shots for and against\n- opponent recent goals/shots for and against\n- team/opponent rest\n- home/away and position\n\n## Model comparison\n\n{table.to_markdown(index=False)}\n\nBaseline test Brier: **{baseline_metrics['test_brier']:.6f}**  \nBaseline test log loss: **{baseline_metrics['test_log_loss']:.6f}**\n\nBest-model Brier improvement vs base-rate baseline: **{improvement_brier:.2%}**  \nBest-model log-loss improvement vs base-rate baseline: **{improvement_log:.2%}**\n\nTrain ends before **{pd.Timestamp(train_cut).date()}**; validation ends before **{pd.Timestamp(val_cut).date()}**.\n\n## Calibration of selected model\n\n{cal.to_markdown(index=False)}\n\n## Still missing before betting validation\n\nThe public boxscore-only pipeline does not yet provide reliable historical pregame PP1/PP2 assignment, expected line combinations, confirmed starting goalie quality, injury status, xG, or timestamp-valid historical anytime-goal bookmaker odds. ROI, EV, CLV and Kelly sizing therefore remain **not validated**.\n\n## Artifacts\n\n- `workspace/real_nhl_v2/player_game_features.csv`\n- `workspace/real_nhl_v2/test_predictions.csv`\n- `workspace/real_nhl_v2/calibration_best.csv`\n- `workspace/real_nhl_v2/metrics.json`\n"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report={REPORT}")


def main() -> int:
    df = build_dataset()
    train_and_report(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
