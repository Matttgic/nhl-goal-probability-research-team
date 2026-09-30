from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import scripts.real_nhl_player_goal_v2 as v2

OUT = Path("workspace/real_nhl_walkforward_replication")
REPORT = Path("reports/real_nhl_walkforward_replication.md")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "home", "position_D", "prior_games", "days_since_prev", "career_goal_rate",
    "target_goal_avg_3", "target_goal_avg_5", "target_goal_avg_10",
    "goals_avg_3", "goals_avg_5", "goals_avg_10", "points_avg_3", "points_avg_5", "points_avg_10",
    "shots_avg_3", "shots_avg_5", "shots_avg_10", "toi_avg_3", "toi_avg_5", "toi_avg_10",
    "shots_trend_5v10", "toi_trend_5v10", "goal_form_5v10",
    "team_rest_days", "team_goals_for_5", "team_goals_for_10", "team_shots_for_5", "team_shots_for_10",
    "team_goals_allowed_5", "team_goals_allowed_10", "team_shots_allowed_5", "team_shots_allowed_10",
    "opp_rest_days", "opp_goals_for_5", "opp_goals_for_10", "opp_shots_for_5", "opp_shots_for_10",
    "opp_goals_allowed_5", "opp_goals_allowed_10", "opp_shots_allowed_5", "opp_shots_allowed_10",
]

FOLDS = [
    ("2024-12", pd.Timestamp("2024-12-01"), pd.Timestamp("2025-01-01")),
    ("2025-01", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")),
    ("2025-02", pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-01")),
    ("2025-03", pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-01")),
    ("2025-04", pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-18")),
]
TARGET_GAMES_PER_FOLD = 60


def collect_games(start: date, end: date) -> list[tuple[str, int]]:
    found: dict[int, str] = {}
    cursor = start
    while cursor <= end:
        payload = v2.get_json(f"schedule/{cursor.isoformat()}")
        for day in payload.get("gameWeek", []) or []:
            gd = day.get("date")
            for game in day.get("games", []) or []:
                if int(game.get("gameType", 0) or 0) != 2:
                    continue
                gid = game.get("id")
                if gid and gd:
                    found[int(gid)] = str(gd)
        cursor += timedelta(days=7)
    return sorted((gd, gid) for gid, gd in found.items())


def build_dataset() -> pd.DataFrame:
    spans = [
        (date(2023, 10, 10), date(2024, 4, 19)),
        (date(2024, 10, 4), date(2025, 4, 18)),
    ]
    game_map: dict[int, str] = {}
    for start, end in spans:
        for gd, gid in collect_games(start, end):
            game_map[gid] = gd
    games = sorted((gd, gid) for gid, gd in game_map.items())

    rows: list[dict] = []
    errors: list[dict] = []
    for i, (gd, gid) in enumerate(games, start=1):
        try:
            rows.extend(v2.skater_rows(gd, gid, v2.get_json(f"gamecenter/{gid}/boxscore")))
        except Exception as exc:
            errors.append({"game_id": gid, "game_date": gd, "error": repr(exc)})
        if i % 100 == 0 or i == len(games):
            print(f"replication fetch {i}/{len(games)} rows={len(rows)} errors={len(errors)}")
    if errors:
        (OUT / "fetch_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No NHL rows fetched")
    df = df.dropna(subset=["player_id", "game_date", "team", "opponent"]).copy()
    df["game_date"] = pd.to_datetime(df.game_date)
    df = v2.add_player_features(df)
    df = v2.add_team_features(df)
    return df[df.prior_games >= 5].replace([np.inf, -np.inf], np.nan).copy()


def make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2500, C=0.7)),
    ])


def sampled_game_ids(test: pd.DataFrame, n: int) -> list[int]:
    games = test[["game_id", "game_date"]].drop_duplicates().sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if len(games) <= n:
        return games.game_id.astype(int).tolist()
    return games.iloc[np.unique(np.linspace(0, len(games) - 1, n, dtype=int))].game_id.astype(int).tolist()


def main() -> None:
    df = build_dataset()
    all_preds, sampled_preds, metrics = [], [], []
    for label, start, end in FOLDS:
        train = df[df.game_date < start].copy()
        test = df[(df.game_date >= start) & (df.game_date < end)].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"Empty replication fold {label}: train={len(train)} test={len(test)}")
        m = make_model(); m.fit(train[FEATURES], train.target_goal)
        p = np.clip(m.predict_proba(test[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
        test["p_logistic"] = p; test["fold"] = label
        ids = sampled_game_ids(test, TARGET_GAMES_PER_FOLD)
        sampled_preds.append(test[test.game_id.isin(ids)].copy())
        all_preds.append(test)
        y = test.target_goal.to_numpy(int)
        base_rate = float(train.target_goal.mean()); base = np.full(len(test), base_rate)
        item = {
            "fold": label, "train_rows": int(len(train)), "test_rows": int(len(test)),
            "test_games": int(test.game_id.nunique()), "sample_games": int(len(ids)),
            "brier": float(brier_score_loss(y, p)), "baseline_brier": float(brier_score_loss(y, base)),
            "log_loss": float(log_loss(y, p, labels=[0,1])), "baseline_log_loss": float(log_loss(y, base, labels=[0,1])),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        }
        metrics.append(item); print(json.dumps(item))

    preds = pd.concat(all_preds, ignore_index=True)
    sampled = pd.concat(sampled_preds, ignore_index=True)
    keep = ["fold","game_date","game_id","player_id","player_name","team","opponent","home","target_goal","p_logistic"]
    preds[keep].to_csv(OUT / "walkforward_predictions_all.csv", index=False)
    sampled[keep].to_csv(OUT / "walkforward_predictions_sampled_games.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "fold_metrics.csv", index=False)
    summary = {
        "replication_period": "2024-12 through 2025-04",
        "folds": metrics,
        "evaluation_rows": int(len(preds)), "evaluation_games": int(preds.game_id.nunique()),
        "sampled_games_for_odds": int(sampled.game_id.nunique()),
        "overall_brier": float(brier_score_loss(preds.target_goal, preds.p_logistic)),
        "overall_log_loss": float(log_loss(preds.target_goal, preds.p_logistic, labels=[0,1])),
        "overall_auc": float(roc_auc_score(preds.target_goal, preds.p_logistic)),
        "rule_frozen_before_results": True,
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Independent 2024-25 replication of V2 logistic", "", "Model specification and economic rule were frozen before this replication run.", "", "| Fold | Test games | Brier | Baseline Brier | Log Loss | AUC |", "|---|---:|---:|---:|---:|---:|"]
    for x in metrics:
        lines.append(f"| {x['fold']} | {x['test_games']} | {x['brier']:.6f} | {x['baseline_brier']:.6f} | {x['log_loss']:.6f} | {x['auc']:.4f} |")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
