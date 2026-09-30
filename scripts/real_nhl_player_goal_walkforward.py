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

OUT = Path("workspace/real_nhl_walkforward")
REPORT = Path("reports/real_nhl_walkforward.md")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "home", "position_D", "prior_games", "days_since_prev", "career_goal_rate",
    "target_goal_avg_3", "target_goal_avg_5", "target_goal_avg_10",
    "goals_avg_3", "goals_avg_5", "goals_avg_10",
    "points_avg_3", "points_avg_5", "points_avg_10",
    "shots_avg_3", "shots_avg_5", "shots_avg_10",
    "toi_avg_3", "toi_avg_5", "toi_avg_10",
    "shots_trend_5v10", "toi_trend_5v10", "goal_form_5v10",
    "team_rest_days", "team_goals_for_5", "team_goals_for_10",
    "team_shots_for_5", "team_shots_for_10", "team_goals_allowed_5",
    "team_goals_allowed_10", "team_shots_allowed_5", "team_shots_allowed_10",
    "opp_rest_days", "opp_goals_for_5", "opp_goals_for_10",
    "opp_shots_for_5", "opp_shots_for_10", "opp_goals_allowed_5",
    "opp_goals_allowed_10", "opp_shots_allowed_5", "opp_shots_allowed_10",
]

FOLDS = [
    ("2025-12", pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01")),
    ("2026-01", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")),
    ("2026-02", pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-01")),
    ("2026-03", pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-01")),
    ("2026-04", pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-17")),
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
        (date(2024, 10, 4), date(2025, 4, 18)),
        (date(2025, 10, 7), date(2026, 4, 17)),
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
            box = v2.get_json(f"gamecenter/{gid}/boxscore")
            rows.extend(v2.skater_rows(gd, gid, box))
        except Exception as exc:
            errors.append({"game_id": gid, "game_date": gd, "error": repr(exc)})
        if i % 100 == 0 or i == len(games):
            print(f"walkforward fetch {i}/{len(games)} rows={len(rows)} errors={len(errors)}")

    if errors:
        (OUT / "fetch_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No NHL rows fetched")
    df = df.dropna(subset=["player_id", "game_date", "team", "opponent"]).copy()
    df["game_date"] = pd.to_datetime(df.game_date)
    df = v2.add_player_features(df)
    df = v2.add_team_features(df)
    df = df[df.prior_games >= 5].replace([np.inf, -np.inf], np.nan).copy()
    return df


def model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2500, C=0.7)),
    ])


def deterministic_game_sample(fold_df: pd.DataFrame, n: int) -> list[int]:
    games = (
        fold_df[["game_id", "game_date"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )
    if len(games) <= n:
        return games.game_id.astype(int).tolist()
    idx = np.linspace(0, len(games) - 1, n, dtype=int)
    idx = np.unique(idx)
    return games.iloc[idx].game_id.astype(int).tolist()


def main() -> None:
    df = build_dataset()
    df.to_csv(OUT / "features_two_seasons.csv", index=False)

    all_preds = []
    sampled_preds = []
    fold_metrics = []

    for label, start, end in FOLDS:
        train = df[df.game_date < start].copy()
        test = df[(df.game_date >= start) & (df.game_date < end)].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"Empty fold {label}: train={len(train)} test={len(test)}")

        clf = model()
        clf.fit(train[FEATURES], train.target_goal)
        p = np.clip(clf.predict_proba(test[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
        test["p_logistic"] = p
        test["fold"] = label
        sample_games = deterministic_game_sample(test, TARGET_GAMES_PER_FOLD)
        sampled = test[test.game_id.isin(sample_games)].copy()

        y = test.target_goal.to_numpy(int)
        base_p = float(train.target_goal.mean())
        base = np.full(len(test), base_p)
        metrics = {
            "fold": label,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_games": int(test.game_id.nunique()),
            "sample_games": int(len(sample_games)),
            "train_end_exclusive": str(start.date()),
            "test_end_exclusive": str(end.date()),
            "train_base_rate": base_p,
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "baseline_brier": float(brier_score_loss(y, base)),
            "baseline_log_loss": float(log_loss(y, base, labels=[0, 1])),
        }
        fold_metrics.append(metrics)
        all_preds.append(test)
        sampled_preds.append(sampled)
        print(json.dumps(metrics))

    preds = pd.concat(all_preds, ignore_index=True)
    sampled = pd.concat(sampled_preds, ignore_index=True)

    keep = [
        "fold", "game_date", "game_id", "player_id", "player_name", "team", "opponent", "home",
        "target_goal", "p_logistic",
    ]
    preds[keep].to_csv(OUT / "walkforward_predictions_all.csv", index=False)
    sampled[keep].to_csv(OUT / "walkforward_predictions_sampled_games.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUT / "fold_metrics.csv", index=False)

    overall_y = preds.target_goal.to_numpy(int)
    overall_p = preds.p_logistic.to_numpy(float)
    summary = {
        "features_rows": int(len(df)),
        "players": int(df.player_id.nunique()),
        "games": int(df.game_id.nunique()),
        "folds": fold_metrics,
        "evaluation_rows": int(len(preds)),
        "evaluation_games": int(preds.game_id.nunique()),
        "sampled_games_for_odds": int(sampled.game_id.nunique()),
        "sampled_rows_for_odds": int(len(sampled)),
        "overall_brier": float(brier_score_loss(overall_y, overall_p)),
        "overall_log_loss": float(log_loss(overall_y, overall_p, labels=[0, 1])),
        "overall_auc": float(roc_auc_score(overall_y, overall_p)),
        "sampling_rule": f"chronologically even deterministic sample up to {TARGET_GAMES_PER_FOLD} games per fold; no outcome-based selection",
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = [
        "# NHL player-goal V2 logistic — multi-period walk-forward",
        "",
        "The same fixed V2 logistic specification is refit before each calendar fold using only earlier games. Player/team rolling features are shifted before the current game.",
        "",
        f"Two-season feature rows: **{len(df)}** across **{df.game_id.nunique()}** games.",
        f"Odds-validation sample: **{sampled.game_id.nunique()} games**, selected deterministically by chronology rather than results.",
        "",
        "| Fold | Train rows | Test games | Brier | Baseline Brier | Log Loss | AUC | Odds sample games |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in fold_metrics:
        auc = "n/a" if m["auc"] is None else f"{m['auc']:.4f}"
        rows.append(
            f"| {m['fold']} | {m['train_rows']} | {m['test_games']} | {m['brier']:.6f} | "
            f"{m['baseline_brier']:.6f} | {m['log_loss']:.6f} | {auc} | {m['sample_games']} |"
        )
    rows += [
        "",
        f"Overall Brier: **{summary['overall_brier']:.6f}**",
        f"Overall Log Loss: **{summary['overall_log_loss']:.6f}**",
        f"Overall AUC: **{summary['overall_auc']:.4f}**",
        "",
        "This stage measures probability quality only. Economic validation remains separate and must use timestamp-valid historical prices.",
    ]
    REPORT.write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
