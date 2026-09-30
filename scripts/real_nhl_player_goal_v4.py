from __future__ import annotations

import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

NHL_BASE = "https://api-web.nhle.com/v1"
MP_BASE = "https://moneypuck.com/moneypuck/playerData/games/20252026"
OUT = Path("workspace/real_nhl_v4")
REPORT = Path("reports/real_nhl_v4.md")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "nhl-goal-probability-research-team/0.4"}


def get_json(url: str, tries: int = 4) -> dict:
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Failed {url}: {last}")


def get_csv(url: str, tries: int = 4) -> pd.DataFrame:
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as exc:
            last = exc
            time.sleep(0.8 * (attempt + 1))
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
    found: dict[int, str] = {}
    cursor = date(2025, 10, 7)
    end = date(2026, 4, 16)
    while cursor <= end:
        payload = get_json(f"{NHL_BASE}/schedule/{cursor.isoformat()}")
        for day in payload.get("gameWeek", []):
            gd = day.get("date")
            for game in day.get("games", []):
                if int(game.get("gameType", 0)) == 2 and game.get("id") and gd:
                    found[int(game["id"])] = str(gd)
        cursor += timedelta(days=7)
    return sorted(((d, gid) for gid, d in found.items()))[-max_games:]


def nhl_rows(game_date: str, game_id: int) -> list[dict]:
    box = get_json(f"{NHL_BASE}/gamecenter/{game_id}/boxscore")
    pstats = box.get("playerByGameStats") or {}
    away = box.get("awayTeam") or {}
    home = box.get("homeTeam") or {}
    rows: list[dict] = []
    for side, team, opp, home_flag in [
        ("awayTeam", away.get("abbrev"), home.get("abbrev"), 0),
        ("homeTeam", home.get("abbrev"), away.get("abbrev"), 1),
    ]:
        sec = pstats.get(side) or {}
        for group in ("forwards", "defense"):
            for p in sec.get(group, []) or []:
                name = p.get("name") or {}
                rows.append({
                    "game_date": game_date,
                    "game_id": game_id,
                    "player_id": p.get("playerId"),
                    "player_name": name.get("default") if isinstance(name, dict) else str(name),
                    "position": p.get("position") or ("D" if group == "defense" else "F"),
                    "team": team,
                    "opponent": opp,
                    "home": home_flag,
                    "goals": float(p.get("goals", 0) or 0),
                    "assists": float(p.get("assists", 0) or 0),
                    "points": float((p.get("goals", 0) or 0) + (p.get("assists", 0) or 0)),
                    "shots": float(p.get("sog", p.get("shots", 0)) or 0),
                    "toi": toi_minutes(p.get("toi")),
                })
    return rows


def moneypuck_rows(game_id: int) -> pd.DataFrame:
    df = get_csv(f"{MP_BASE}/{game_id}.csv")
    needed = {
        "gameId", "playerId", "team", "position", "situation", "I_F_xGoals",
        "I_F_highDangerxGoals", "I_F_shotsOnGoal", "I_F_iceTime",
        "OnIce_F_xGoals", "OnIce_A_xGoals",
    }
    missing = sorted(needed - set(df.columns))
    if missing:
        raise RuntimeError(f"MoneyPuck columns missing for {game_id}: {missing}")
    df = df[df["position"].isin(["C", "L", "R", "D"])].copy()
    numeric = [
        "I_F_xGoals", "I_F_highDangerxGoals", "I_F_shotsOnGoal", "I_F_iceTime",
        "OnIce_F_xGoals", "OnIce_A_xGoals",
    ]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    pieces = []
    spec = {
        "all": "all",
        "5on5": "ev",
        "5on4": "pp",
    }
    for situation, stem in spec.items():
        part = df[df["situation"] == situation][["gameId", "playerId", "team"] + numeric].copy()
        part = part.rename(columns={
            "gameId": "game_id", "playerId": "player_id",
            "I_F_xGoals": f"mp_xg_{stem}",
            "I_F_highDangerxGoals": f"mp_hd_xg_{stem}",
            "I_F_shotsOnGoal": f"mp_sog_{stem}",
            "I_F_iceTime": f"mp_toi_sec_{stem}",
            "OnIce_F_xGoals": f"mp_onice_xgf_{stem}",
            "OnIce_A_xGoals": f"mp_onice_xga_{stem}",
        })
        pieces.append(part)

    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p.drop(columns=["team"]), on=["game_id", "player_id"], how="outer")
    out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    return out


def fetch_game(pair: tuple[str, int]) -> tuple[list[dict], pd.DataFrame | None, dict | None]:
    gd, gid = pair
    try:
        nr = nhl_rows(gd, gid)
    except Exception as exc:
        return [], None, {"game_id": gid, "source": "nhl", "error": repr(exc)}
    try:
        mp = moneypuck_rows(gid)
        return nr, mp, None
    except Exception as exc:
        return nr, None, {"game_id": gid, "source": "moneypuck", "error": repr(exc)}


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date", "game_id"]).copy()
    g = df.groupby("player_id", group_keys=False)
    df["target_goal"] = (df["goals"] >= 1).astype(int)
    df["prior_games"] = g.cumcount()
    df["days_since_prev"] = g["game_date"].diff().dt.days.clip(0, 14).fillna(7)
    for col in ["target_goal", "goals", "points", "shots", "toi"]:
        for w in [3, 5, 10]:
            df[f"{col}_avg_{w}"] = g[col].transform(lambda s, ww=w: s.shift(1).rolling(ww, min_periods=1).mean())
    df["career_goal_rate"] = g["target_goal"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["shots_trend_5v10"] = df["shots_avg_5"] - df["shots_avg_10"]
    df["toi_trend_5v10"] = df["toi_avg_5"] - df["toi_avg_10"]
    df["goal_form_5v10"] = df["target_goal_avg_5"] - df["target_goal_avg_10"]
    df["position_D"] = (df["position"] == "D").astype(int)
    return df


def add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    tg = df.groupby(["game_id", "game_date", "team", "opponent"], as_index=False).agg(
        team_goals=("goals", "sum"), team_shots=("shots", "sum")
    ).sort_values(["team", "game_date", "game_id"])
    opp = tg[["game_id", "team", "team_goals", "team_shots"]].rename(columns={
        "team": "opponent", "team_goals": "opp_goals_actual", "team_shots": "opp_shots_actual"
    })
    tg = tg.merge(opp, on=["game_id", "opponent"], how="left")
    gg = tg.groupby("team", group_keys=False)
    tg["team_rest_days"] = gg["game_date"].diff().dt.days.clip(0, 14).fillna(7)
    for src, stem in [("team_goals", "goals_for"), ("team_shots", "shots_for"),
                      ("opp_goals_actual", "goals_allowed"), ("opp_shots_actual", "shots_allowed")]:
        for w in [5, 10]:
            tg[f"team_{stem}_{w}"] = gg[src].transform(lambda s, ww=w: s.shift(1).rolling(ww, min_periods=1).mean())
    own = ["game_id", "team", "team_rest_days"] + [c for c in tg.columns if c.startswith("team_") and c != "team_goals" and c != "team_shots" and c != "team_rest_days"]
    own = list(dict.fromkeys(own))
    df = df.merge(tg[own], on=["game_id", "team"], how="left")
    rename = {"team": "opponent", "team_rest_days": "opp_rest_days"}
    for c in own:
        if c.startswith("team_") and c != "team_rest_days":
            rename[c] = "opp_" + c[len("team_"):]
    odf = tg[own].rename(columns=rename)
    return df.merge(odf, on=["game_id", "opponent"], how="left")


def add_moneypuck_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date", "game_id"]).copy()
    raw = [
        "mp_xg_all", "mp_hd_xg_all", "mp_sog_all", "mp_toi_sec_all",
        "mp_xg_ev", "mp_hd_xg_ev", "mp_sog_ev", "mp_toi_sec_ev",
        "mp_xg_pp", "mp_hd_xg_pp", "mp_sog_pp", "mp_toi_sec_pp",
        "mp_onice_xgf_ev", "mp_onice_xga_ev",
    ]
    for c in raw:
        if c not in df:
            df[c] = np.nan
    for stem, xg, toi in [
        ("mp_xg60_all", "mp_xg_all", "mp_toi_sec_all"),
        ("mp_xg60_ev", "mp_xg_ev", "mp_toi_sec_ev"),
        ("mp_xg60_pp", "mp_xg_pp", "mp_toi_sec_pp"),
    ]:
        denom = pd.to_numeric(df[toi], errors="coerce")
        df[stem] = np.where(denom > 0, pd.to_numeric(df[xg], errors="coerce") * 3600.0 / denom, np.nan)
    df["mp_pp_toi_share"] = np.where(df["mp_toi_sec_all"] > 0, df["mp_toi_sec_pp"] / df["mp_toi_sec_all"], np.nan)
    mp_sources = raw + ["mp_xg60_all", "mp_xg60_ev", "mp_xg60_pp", "mp_pp_toi_share"]
    g = df.groupby("player_id", group_keys=False)
    for c in mp_sources:
        for w in [5, 10]:
            df[f"{c}_avg_{w}"] = g[c].transform(lambda s, ww=w: s.shift(1).rolling(ww, min_periods=1).mean())

    # Pregame PP/role proxies: ranks use only lagged historical averages.
    df["pp_role_rank"] = df.groupby(["game_id", "team"])["mp_toi_sec_pp_avg_5"].rank(method="min", ascending=False)
    df["pp_top5_proxy"] = (df["pp_role_rank"] <= 5).astype(float)
    df["toi_role_rank"] = df.groupby(["game_id", "team"])["mp_toi_sec_all_avg_5"].rank(method="min", ascending=False)

    # Team/opponent xG context, also shifted before use.
    current = df.groupby(["game_id", "game_date", "team", "opponent"], as_index=False).agg(
        team_mp_xg=("mp_xg_all", "sum"), team_mp_pp_xg=("mp_xg_pp", "sum")
    ).sort_values(["team", "game_date", "game_id"])
    other = current[["game_id", "team", "team_mp_xg", "team_mp_pp_xg"]].rename(columns={
        "team": "opponent", "team_mp_xg": "opp_mp_xg_actual", "team_mp_pp_xg": "opp_mp_pp_xg_actual"
    })
    current = current.merge(other, on=["game_id", "opponent"], how="left")
    tg = current.groupby("team", group_keys=False)
    for src, dst in [
        ("team_mp_xg", "team_mp_xg_for"), ("team_mp_pp_xg", "team_mp_pp_xg_for"),
        ("opp_mp_xg_actual", "team_mp_xg_allowed"), ("opp_mp_pp_xg_actual", "team_mp_pp_xg_allowed"),
    ]:
        for w in [5, 10]:
            current[f"{dst}_{w}"] = tg[src].transform(lambda s, ww=w: s.shift(1).rolling(ww, min_periods=1).mean())
    keep = ["game_id", "team"] + [c for c in current.columns if c.startswith("team_mp_") and c not in ["team_mp_xg", "team_mp_pp_xg"]]
    df = df.merge(current[keep], on=["game_id", "team"], how="left")
    opp_keep = current[keep].rename(columns={"team": "opponent", **{c: "opp_" + c[len("team_"):] for c in keep if c.startswith("team_")}})
    return df.merge(opp_keep, on=["game_id", "opponent"], how="left")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    ids = np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1])
    ans = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            ans += m.mean() * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(ans)


def score(name: str, model, xtr, ytr, xv, yv, xt, yt) -> dict:
    model.fit(xtr, ytr)
    pv = np.clip(model.predict_proba(xv)[:, 1], 1e-6, 1 - 1e-6)
    pt = np.clip(model.predict_proba(xt)[:, 1], 1e-6, 1 - 1e-6)
    return {
        "name": name,
        "validation_brier": float(brier_score_loss(yv, pv)),
        "validation_log_loss": float(log_loss(yv, pv, labels=[0, 1])),
        "test_brier": float(brier_score_loss(yt, pt)),
        "test_log_loss": float(log_loss(yt, pt, labels=[0, 1])),
        "test_auc": float(roc_auc_score(yt, pt)),
        "test_ece": ece(np.asarray(yt), pt),
    }


def main() -> None:
    games = collect_game_ids(500)
    nhl_all: list[dict] = []
    mp_parts: list[pd.DataFrame] = []
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch_game, pair): pair for pair in games}
        for i, fut in enumerate(as_completed(futs), start=1):
            nr, mp, err = fut.result()
            nhl_all.extend(nr)
            if mp is not None:
                mp_parts.append(mp)
            if err:
                errors.append(err)
            if i % 50 == 0:
                print(f"fetched {i}/{len(games)} games; nhl_rows={len(nhl_all)}; mp_games={len(mp_parts)}; errors={len(errors)}")

    if errors:
        (OUT / "fetch_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    df = pd.DataFrame(nhl_all)
    if df.empty:
        raise RuntimeError("No NHL rows")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    mp = pd.concat(mp_parts, ignore_index=True) if mp_parts else pd.DataFrame(columns=["game_id", "player_id"])
    mp = mp.drop_duplicates(["game_id", "player_id"])
    df = df.merge(mp, on=["game_id", "player_id"], how="left", suffixes=("", "_mp"))
    if "team_mp" in df:
        df = df.drop(columns=["team_mp"])

    df = add_basic_features(df)
    df = add_team_features(df)
    df = add_moneypuck_features(df)
    df = df[df["prior_games"] >= 5].replace([np.inf, -np.inf], np.nan).copy()

    control = [
        "home", "position_D", "prior_games", "days_since_prev", "career_goal_rate",
        "target_goal_avg_3", "target_goal_avg_5", "target_goal_avg_10",
        "goals_avg_3", "goals_avg_5", "goals_avg_10",
        "points_avg_3", "points_avg_5", "points_avg_10",
        "shots_avg_3", "shots_avg_5", "shots_avg_10",
        "toi_avg_3", "toi_avg_5", "toi_avg_10",
        "shots_trend_5v10", "toi_trend_5v10", "goal_form_5v10",
        "team_rest_days", "team_goals_for_5", "team_goals_for_10", "team_shots_for_5", "team_shots_for_10",
        "team_goals_allowed_5", "team_goals_allowed_10", "team_shots_allowed_5", "team_shots_allowed_10",
        "opp_rest_days", "opp_goals_for_5", "opp_goals_for_10", "opp_shots_for_5", "opp_shots_for_10",
        "opp_goals_allowed_5", "opp_goals_allowed_10", "opp_shots_allowed_5", "opp_shots_allowed_10",
    ]
    augmented = control + [
        c for c in df.columns if (
            (c.startswith("mp_") and (c.endswith("_avg_5") or c.endswith("_avg_10")))
            or c in ["pp_role_rank", "pp_top5_proxy", "toi_role_rank"]
            or c.startswith("team_mp_") or c.startswith("opp_mp_")
        )
    ]
    augmented = list(dict.fromkeys(augmented))

    dates = sorted(df["game_date"].dt.normalize().unique())
    train_cut = dates[int(len(dates) * 0.70)]
    val_cut = dates[int(len(dates) * 0.85)]
    tr = df[df["game_date"] < train_cut]
    va = df[(df["game_date"] >= train_cut) & (df["game_date"] < val_cut)]
    te = df[df["game_date"] >= val_cut]

    ytr, yv, yt = tr["target_goal"], va["target_goal"], te["target_goal"]
    logistic = lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, C=0.7))])
    models = [
        score("logistic_v2_control", logistic(), tr[control], ytr, va[control], yv, te[control], yt),
        score("logistic_v4", logistic(), tr[augmented], ytr, va[augmented], yv, te[augmented], yt),
        score("xgboost_v4", Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", xgb.XGBClassifier(n_estimators=350, max_depth=3, learning_rate=0.03, subsample=0.85, colsample_bytree=0.8, min_child_weight=6, reg_lambda=2.5, objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=2))]), tr[augmented], ytr, va[augmented], yv, te[augmented], yt),
        score("lightgbm_v4", Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", lgb.LGBMClassifier(n_estimators=350, learning_rate=0.025, num_leaves=15, min_child_samples=40, reg_lambda=2.0, verbosity=-1, random_state=42, n_jobs=2))]), tr[augmented], ytr, va[augmented], yv, te[augmented], yt),
    ]
    best_aug = min([m for m in models if m["name"] != "logistic_v2_control"], key=lambda m: m["validation_brier"])
    baseline_p = np.full(len(te), float(ytr.mean()))
    summary = {
        "rows_total": len(df), "players": int(df["player_id"].nunique()), "games": int(df["game_id"].nunique()),
        "moneypuck_game_coverage": len(mp_parts) / max(1, len(games)),
        "moneypuck_player_row_coverage": float(df["mp_xg_all"].notna().mean()),
        "train_rows": len(tr), "validation_rows": len(va), "test_rows": len(te),
        "train_base_rate": float(ytr.mean()), "best_augmented_by_validation_brier": best_aug["name"],
        "baseline": {"test_brier": float(brier_score_loss(yt, baseline_p)), "test_log_loss": float(log_loss(yt, baseline_p, labels=[0, 1]))},
        "feature_counts": {"control": len(control), "augmented": len(augmented)},
        "models": models,
    }
    print(json.dumps(summary, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df[["game_date", "game_id", "player_id", "player_name", "team", "target_goal"] + [c for c in augmented if c not in control]].to_csv(OUT / "v4_features.csv", index=False)

    lines = [
        "# Real NHL player-goal V4",
        "",
        "Leakage-safe comparison using NHL boxscores plus lagged MoneyPuck xG/PP/on-ice features. MoneyPuck is credited as the source of the xG/player situation data.",
        "",
        f"- Rows: **{len(df)}**",
        f"- Games: **{df['game_id'].nunique()}**",
        f"- MoneyPuck game coverage: **{summary['moneypuck_game_coverage']:.1%}**",
        f"- MoneyPuck player-row coverage: **{summary['moneypuck_player_row_coverage']:.1%}**",
        f"- Selected augmented model by validation Brier: **{best_aug['name']}**",
        "",
        "| Model | Val Brier | Test Brier | Test Log Loss | Test AUC | Test ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in models:
        lines.append(f"| {m['name']} | {m['validation_brier']:.6f} | {m['test_brier']:.6f} | {m['test_log_loss']:.6f} | {m['test_auc']:.6f} | {m['test_ece']:.6f} |")
    lines += ["", "Economic ROI is intentionally not reported because genuine timestamp-valid historical anytime-goal odds are not yet integrated."]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
