from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
V2_PATH = ROOT / "real_nhl_player_goal_v2.py"
spec = importlib.util.spec_from_file_location("nhl_v2", V2_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not import V2 module")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

OUT = Path("workspace/real_nhl_v3")
REPORT = Path("reports/real_nhl_v3.md")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)


def fnum(obj: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in obj and obj.get(key) is not None:
            try:
                return float(obj.get(key))
            except (TypeError, ValueError):
                pass
    return float(default)


def goalie_summary(section: dict) -> dict:
    goalies = section.get("goalies", []) or []
    saves = 0.0
    shots_against = 0.0
    goals_against = 0.0
    usable = 0
    for g in goalies:
        s = fnum(g, "saves")
        sa = fnum(g, "shotsAgainst", "shotsAgainstTotal")
        ga = fnum(g, "goalsAgainst")
        if sa <= 0 and (s > 0 or ga > 0):
            sa = s + ga
        if sa > 0:
            usable += 1
        saves += s
        shots_against += sa
        goals_against += ga
    sv = saves / shots_against if shots_against > 0 else math.nan
    return {
        "goalie_saves_actual": saves,
        "goalie_shots_against_actual": shots_against,
        "goalie_goals_against_actual": goals_against,
        "goalie_save_pct_actual": sv,
        "goalie_fields_present": int(usable > 0),
    }


def enhanced_skater_rows(game_date: str, game_id: int, box: dict) -> list[dict]:
    rows: list[dict] = []
    pstats = box.get("playerByGameStats") or {}
    away = box.get("awayTeam") or {}
    home = box.get("homeTeam") or {}
    team_meta = {
        "awayTeam": (away.get("abbrev"), home.get("abbrev"), 0),
        "homeTeam": (home.get("abbrev"), away.get("abbrev"), 1),
    }

    goalie_meta = {side: goalie_summary(pstats.get(side) or {}) for side in team_meta}

    for side, (team, opponent, is_home) in team_meta.items():
        section = pstats.get(side) or {}
        gmeta = goalie_meta[side]
        for group in ("forwards", "defense"):
            for p in section.get(group, []) or []:
                name_obj = p.get("name") or {}
                goals = fnum(p, "goals")
                assists = fnum(p, "assists")
                pp_goals_present = int("powerPlayGoals" in p or "ppGoals" in p)
                pp_points_present = int("powerPlayPoints" in p or "ppPoints" in p)
                pp_goals = fnum(p, "powerPlayGoals", "ppGoals")
                pp_points = fnum(p, "powerPlayPoints", "ppPoints")
                if pp_points_present == 0 and pp_goals_present:
                    pp_points = pp_goals
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
                        "goals": goals,
                        "assists": assists,
                        "points": goals + assists,
                        "shots": fnum(p, "sog", "shots"),
                        "toi": v2.toi_minutes(p.get("toi")),
                        "pp_goals": pp_goals,
                        "pp_points": pp_points,
                        "pp_fields_present": int(pp_goals_present or pp_points_present),
                        "plus_minus": fnum(p, "plusMinus"),
                        "hits": fnum(p, "hits"),
                        "blocked": fnum(p, "blockedShots", "blocked"),
                        **gmeta,
                    }
                )
    return rows


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date", "game_id"]).copy()
    df["target_goal"] = (df["goals"] >= 1).astype(int)

    # Actual-game quantities are used only as lag sources below; none enter feature_cols directly.
    safe_toi = df["toi"].where(df["toi"] > 0)
    df["goals_per60_actual"] = (60.0 * df["goals"] / safe_toi).clip(0, 12)
    df["points_per60_actual"] = (60.0 * df["points"] / safe_toi).clip(0, 20)
    df["shots_per60_actual"] = (60.0 * df["shots"] / safe_toi).clip(0, 30)
    df["shoot_pct_actual"] = np.where(df["shots"] > 0, df["goals"] / df["shots"], np.nan)

    totals = df.groupby(["game_id", "team"], as_index=False).agg(
        team_skater_shots=("shots", "sum"),
        team_skater_toi=("toi", "sum"),
    )
    df = df.merge(totals, on=["game_id", "team"], how="left")
    df["shot_share_actual"] = np.where(df["team_skater_shots"] > 0, df["shots"] / df["team_skater_shots"], 0.0)
    df["toi_share_actual"] = np.where(df["team_skater_toi"] > 0, df["toi"] / df["team_skater_toi"], 0.0)

    g = df.groupby("player_id", group_keys=False)
    df["prior_games"] = g.cumcount()
    df["days_since_prev"] = g["game_date"].diff().dt.days.clip(lower=0, upper=14).fillna(7)

    rolling_sources = [
        "target_goal", "goals", "points", "shots", "toi", "pp_goals", "pp_points",
        "goals_per60_actual", "points_per60_actual", "shots_per60_actual",
        "shoot_pct_actual", "shot_share_actual", "toi_share_actual",
    ]
    for col in rolling_sources:
        for window in [3, 5, 10]:
            df[f"{col}_avg_{window}"] = g[col].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )

    df["career_goal_rate"] = g["target_goal"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    prior_goals = g["goals"].cumsum() - df["goals"]
    prior_shots = g["shots"].cumsum() - df["shots"]
    df["career_shoot_pct"] = np.where(prior_shots > 0, prior_goals / prior_shots, np.nan)

    df["shots_trend_5v10"] = df["shots_avg_5"] - df["shots_avg_10"]
    df["toi_trend_5v10"] = df["toi_avg_5"] - df["toi_avg_10"]
    df["goal_form_5v10"] = df["target_goal_avg_5"] - df["target_goal_avg_10"]
    df["shots60_trend_5v10"] = df["shots_per60_actual_avg_5"] - df["shots_per60_actual_avg_10"]
    df["shot_share_trend_5v10"] = df["shot_share_actual_avg_5"] - df["shot_share_actual_avg_10"]
    df["position_D"] = (df["position"] == "D").astype(int)

    # Pregame role proxies based only on the player's own lagged history.
    df["high_toi_role"] = np.where(
        df["position_D"].eq(1),
        (df["toi_avg_5"] >= 20.0).astype(float),
        (df["toi_avg_5"] >= 17.0).astype(float),
    )
    df["high_shot_role"] = (df["shots_avg_5"] >= 3.0).astype(float)
    df["recent_pp_role_proxy"] = (df["pp_points_avg_10"].fillna(0) > 0.10).astype(float)
    return df


def add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    tg = (
        df.groupby(["game_id", "game_date", "team", "opponent", "home"], as_index=False)
        .agg(
            team_goals=("goals", "sum"),
            team_shots=("shots", "sum"),
            team_pp_goals=("pp_goals", "sum"),
            team_goalie_save_pct_actual=("goalie_save_pct_actual", "first"),
        )
        .sort_values(["team", "game_date", "game_id"])
    )

    opp_actual = tg[["game_id", "team", "team_goals", "team_shots", "team_pp_goals"]].rename(
        columns={
            "team": "opponent",
            "team_goals": "opp_goals_actual",
            "team_shots": "opp_shots_actual",
            "team_pp_goals": "opp_pp_goals_actual",
        }
    )
    tg = tg.merge(opp_actual, on=["game_id", "opponent"], how="left")
    team_g = tg.groupby("team", group_keys=False)
    tg["team_rest_days"] = team_g["game_date"].diff().dt.days.clip(lower=0, upper=14).fillna(7)

    sources = [
        ("team_goals", "goals_for"),
        ("team_shots", "shots_for"),
        ("opp_goals_actual", "goals_allowed"),
        ("opp_shots_actual", "shots_allowed"),
        ("team_pp_goals", "pp_goals_for"),
        ("opp_pp_goals_actual", "pp_goals_allowed"),
        ("team_goalie_save_pct_actual", "goalie_save_pct"),
    ]
    for source, stem in sources:
        for window in [5, 10]:
            tg[f"team_{stem}_{window}"] = team_g[source].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )

    own_cols = ["game_id", "team", "team_rest_days"] + [
        f"team_{stem}_{w}" for _, stem in sources for w in [5, 10]
    ]
    df = df.merge(tg[own_cols], on=["game_id", "team"], how="left")

    rename = {"team": "opponent", "team_rest_days": "opp_rest_days"}
    for _, stem in sources:
        for w in [5, 10]:
            rename[f"team_{stem}_{w}"] = f"opp_{stem}_{w}"
    opp = tg[own_cols].rename(columns=rename)
    df = df.merge(opp, on=["game_id", "opponent"], how="left")
    return df


def build_dataset() -> pd.DataFrame:
    games = v2.collect_game_ids(max_games=500)
    all_rows: list[dict] = []
    errors: list[dict] = []
    for i, (game_date, gid) in enumerate(games, start=1):
        try:
            box = v2.get_json(f"gamecenter/{gid}/boxscore")
            all_rows.extend(enhanced_skater_rows(game_date, gid, box))
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


def make_models() -> dict:
    return {
        "logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=0.6)),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(
                n_estimators=350, max_depth=3, learning_rate=0.03,
                subsample=0.85, colsample_bytree=0.85, min_child_weight=6,
                reg_lambda=2.5, objective="binary:logistic", eval_metric="logloss",
                random_state=42, n_jobs=2,
            )),
        ]),
        "lightgbm": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", lgb.LGBMClassifier(
                n_estimators=350, learning_rate=0.025, num_leaves=15,
                min_child_samples=40, subsample=0.9, colsample_bytree=0.85,
                reg_lambda=2.0, random_state=42, n_jobs=2, verbose=-1,
            )),
        ]),
    }


def evaluate_model(name: str, model, train, val, test, feature_cols: list[str]) -> dict:
    x_train, y_train = train[feature_cols], train["target_goal"]
    x_val, y_val = val[feature_cols], val["target_goal"]
    x_test, y_test = test[feature_cols], test["target_goal"]
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
        "test_ece": v2.expected_calibration_error(np.asarray(y_test), p_test),
    }


def train_and_report(df: pd.DataFrame) -> None:
    v2_features = [
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
    extra_features = [
        "career_shoot_pct",
        "goals_per60_actual_avg_3", "goals_per60_actual_avg_5", "goals_per60_actual_avg_10",
        "points_per60_actual_avg_5", "points_per60_actual_avg_10",
        "shots_per60_actual_avg_3", "shots_per60_actual_avg_5", "shots_per60_actual_avg_10",
        "shoot_pct_actual_avg_5", "shoot_pct_actual_avg_10",
        "shot_share_actual_avg_3", "shot_share_actual_avg_5", "shot_share_actual_avg_10",
        "toi_share_actual_avg_5", "toi_share_actual_avg_10",
        "shots60_trend_5v10", "shot_share_trend_5v10",
        "pp_goals_avg_5", "pp_goals_avg_10", "pp_points_avg_5", "pp_points_avg_10",
        "high_toi_role", "high_shot_role", "recent_pp_role_proxy",
        "team_pp_goals_for_5", "team_pp_goals_for_10",
        "team_pp_goals_allowed_5", "team_pp_goals_allowed_10",
        "team_goalie_save_pct_5", "team_goalie_save_pct_10",
        "opp_pp_goals_for_5", "opp_pp_goals_for_10",
        "opp_pp_goals_allowed_5", "opp_pp_goals_allowed_10",
        "opp_goalie_save_pct_5", "opp_goalie_save_pct_10",
    ]
    full_features = v2_features + extra_features

    df = df.replace([np.inf, -np.inf], np.nan).copy()
    unique_dates = sorted(df["game_date"].dt.normalize().unique())
    train_cut = unique_dates[int(len(unique_dates) * 0.70)]
    val_cut = unique_dates[int(len(unique_dates) * 0.85)]
    train = df[df["game_date"] < train_cut].copy()
    val = df[(df["game_date"] >= train_cut) & (df["game_date"] < val_cut)].copy()
    test = df[df["game_date"] >= val_cut].copy()

    # Same-data ablation: V2-style logistic vs V3 full logistic.
    ablation_v2 = evaluate_model("logistic_v2_features", make_models()["logistic"], train, val, test, v2_features)

    results = []
    for name, model in make_models().items():
        results.append(evaluate_model(name, model, train, val, test, full_features))
    best = min(results, key=lambda r: r["validation_brier"])

    y_test = test["target_goal"].to_numpy()
    base_rate = float(train["target_goal"].mean())
    base_pred = np.full(len(test), base_rate)
    baseline = {
        "test_brier": float(brier_score_loss(y_test, base_pred)),
        "test_log_loss": float(log_loss(y_test, base_pred, labels=[0, 1])),
    }

    coverage = {
        "pp_field_row_coverage": float(df["pp_fields_present"].mean()),
        "goalie_field_row_coverage": float(df["goalie_fields_present"].mean()),
        "pp_nonzero_rate": float((df["pp_points"] > 0).mean()),
        "goalie_save_pct_nonmissing": float(df["goalie_save_pct_actual"].notna().mean()),
    }

    cal = v2.calibration_table(y_test, best["p_test"])
    cal.to_csv(OUT / "calibration_best.csv", index=False)
    preds = test[["game_date", "game_id", "player_id", "player_name", "team", "opponent", "target_goal"]].copy()
    preds["predicted_goal_probability"] = best["p_test"]
    preds.to_csv(OUT / "test_predictions_best.csv", index=False)
    df.to_csv(OUT / "player_game_features_v3.csv", index=False)

    serial_results = [{k: v for k, v in r.items() if k not in {"model", "p_val", "p_test"}} for r in results]
    metrics = {
        "rows_total": int(len(df)),
        "players": int(df["player_id"].nunique()),
        "games": int(df["game_id"].nunique()),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_base_rate": base_rate,
        "best_model_by_validation_brier": best["name"],
        "train_end_exclusive": str(pd.Timestamp(train_cut).date()),
        "validation_end_exclusive": str(pd.Timestamp(val_cut).date()),
        "baseline": baseline,
        "v2_feature_ablation": {k: v for k, v in ablation_v2.items() if k not in {"model", "p_val", "p_test"}},
        "coverage": coverage,
        "models": serial_results,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rows = []
    rows.append({"model": "baseline", "val_brier": math.nan, "test_brier": baseline["test_brier"], "log_loss": baseline["test_log_loss"], "auc": math.nan, "ece": math.nan})
    rows.append({"model": "logistic_v2_features", "val_brier": ablation_v2["validation_brier"], "test_brier": ablation_v2["test_brier"], "log_loss": ablation_v2["test_log_loss"], "auc": ablation_v2["test_auc"], "ece": ablation_v2["test_ece"]})
    for r in results:
        rows.append({"model": r["name"], "val_brier": r["validation_brier"], "test_brier": r["test_brier"], "log_loss": r["test_log_loss"], "auc": r["test_auc"], "ece": r["test_ece"]})
    table = pd.DataFrame(rows)

    v3_gain = (ablation_v2["test_brier"] - best["test_brier"]) / ablation_v2["test_brier"] if ablation_v2["test_brier"] else math.nan
    base_gain = (baseline["test_brier"] - best["test_brier"]) / baseline["test_brier"] if baseline["test_brier"] else math.nan
    report = f"""# Real NHL player-goal V3\n\n## Status\n\nLeakage-safe feature expansion on real NHL data. Best model selected only by validation Brier: **{best['name']}**.\n\n## Dataset\n\n- Rows: **{len(df)}**\n- Players: **{df['player_id'].nunique()}**\n- Games represented after history requirement: **{df['game_id'].nunique()}**\n- Train / validation / test: **{len(train)} / {len(val)} / {len(test)}**\n- Train base goal rate: **{base_rate:.3%}**\n\n## New V3 pregame features\n\n- Player goals/60, points/60, shots/60 rolling history\n- Rolling shooting percentage and career shooting percentage\n- Player shot share and TOI share as role proxies\n- Historical PP goals/points when the NHL boxscore exposes those fields\n- Lagged team PP scoring/allowance context\n- Lagged team goaltending save percentage; opponent value is a team-goaltending proxy, **not a confirmed-starter feature**\n- High-TOI/high-shot/PP-production role proxies built only from prior games\n\n## Data coverage audit\n\n- PP field row coverage: **{coverage['pp_field_row_coverage']:.2%}**\n- Rows with non-zero PP points: **{coverage['pp_nonzero_rate']:.2%}**\n- Goalie-field row coverage: **{coverage['goalie_field_row_coverage']:.2%}**\n- Non-missing goalie save pct: **{coverage['goalie_save_pct_nonmissing']:.2%}**\n\n## Results\n\n{table.to_markdown(index=False, floatfmt='.6f')}\n\nBest V3 Brier improvement vs constant baseline: **{base_gain:.2%}**.\nBest V3 Brier improvement vs the same-data V2-feature logistic ablation: **{v3_gain:.2%}**.\n\n## Leakage controls\n\nEvery performance, rate, PP, team, opponent and goaltending feature is shifted before rolling aggregation. Current-game goals, shots, TOI, PP output and goalie result never enter the current row's model features. Chronological train/validation/test splits are preserved.\n\n## Limitations before betting validation\n\n- This still does **not** contain timestamped historical anytime-goal odds, so EV/ROI/CLV are not validated.\n- Opponent goaltending is team-level historical form, not the actual pregame confirmed starter.\n- PP production is not the same as PP1/PP2 deployment or PP TOI.\n- Exact forward lines, injuries/scratches and xG are not yet timestamped pregame inputs.\n- Role proxies do not prove historical market availability for every skater.\n\n## Next gate\n\nOnly after the predictive pipeline remains stable should timestamp-valid historical scorer prices be joined for economic backtesting.\n"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"report={REPORT}")


def main() -> int:
    df = build_dataset()
    train_and_report(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
