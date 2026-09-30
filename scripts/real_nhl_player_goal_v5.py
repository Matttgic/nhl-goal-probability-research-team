from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import scripts.real_nhl_player_goal_v4 as v4

SHIFT_URL = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={}"
OUT = Path("workspace/real_nhl_v5")
REPORT = Path("reports/real_nhl_v5.md")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)


def clock_sec(value: str | None) -> int | None:
    if not value or ":" not in str(value):
        return None
    try:
        mm, ss = str(value).split(":", 1)
        return int(mm) * 60 + int(ss)
    except Exception:
        return None


def fetch_shifts(game_id: int, tries: int = 4) -> list[dict]:
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(SHIFT_URL.format(game_id), headers=v4.HEADERS, timeout=30)
            r.raise_for_status()
            return (r.json() or {}).get("data", []) or []
        except Exception as exc:
            last = exc
            import time
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"shift fetch failed {game_id}: {last}")


def overlap_seconds(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    i = j = total = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            total += e - s
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def shift_game_features(game_id: int, rows: list[dict]) -> list[dict]:
    by_team: dict[str, dict[int, list[tuple[int, int]]]] = {}
    for r in rows:
        pid = r.get("playerId")
        team = r.get("teamAbbrev")
        period = r.get("period")
        start = clock_sec(r.get("startTime"))
        end = clock_sec(r.get("endTime"))
        if not pid or not team or not period or start is None or end is None or end <= start:
            continue
        # Period offsets are sufficient for regulation + OT and avoid cross-period overlap.
        base = (int(period) - 1) * 1200
        by_team.setdefault(str(team), {}).setdefault(int(pid), []).append((base + start, base + end))

    out: list[dict] = []
    for team, players in by_team.items():
        for pid in players:
            players[pid].sort()
        pids = sorted(players)
        for pid in pids:
            intervals = players[pid]
            toi = sum(max(0, e - s) for s, e in intervals)
            overlaps: list[tuple[int, int]] = []
            for other in pids:
                if other == pid:
                    continue
                sec = overlap_seconds(intervals, players[other])
                if sec > 0:
                    overlaps.append((sec, other))
            overlaps.sort(reverse=True)
            top1_sec, top1_id = overlaps[0] if overlaps else (0, None)
            top2_sec, top2_id = overlaps[1] if len(overlaps) > 1 else (0, None)
            total_partner_sec = sum(sec for sec, _ in overlaps)
            out.append({
                "game_id": game_id,
                "player_id": pid,
                "shift_team": team,
                "shift_toi_sec": float(toi),
                "coice_top1_sec": float(top1_sec),
                "coice_top2_sec": float(top2_sec),
                "coice_top1_share": float(top1_sec / toi) if toi else math.nan,
                "coice_top2_share": float(top2_sec / toi) if toi else math.nan,
                "coice_top2_concentration": float((top1_sec + top2_sec) / total_partner_sec) if total_partner_sec else math.nan,
                "partner_count_120": float(sum(1 for sec, _ in overlaps if sec >= 120)),
                "top_partner1": top1_id,
                "top_partner2": top2_id,
            })
    return out


def fetch_bundle(pair: tuple[str, int]):
    gd, gid = pair
    nr, mp, base_err = v4.fetch_game(pair)
    shift_rows = []
    shift_err = None
    try:
        shift_rows = fetch_shifts(gid)
    except Exception as exc:
        shift_err = {"game_id": gid, "source": "shifts", "error": repr(exc)}
    feats = shift_game_features(gid, shift_rows) if shift_rows else []
    errs = [e for e in [base_err, shift_err] if e]
    return nr, mp, feats, errs


def _partner_set(a, b) -> set[int]:
    vals = set()
    for x in (a, b):
        if pd.notna(x):
            vals.add(int(x))
    return vals


def add_shift_history_features(df: pd.DataFrame, shift_df: pd.DataFrame) -> pd.DataFrame:
    if shift_df.empty:
        for c in ["shift_toi_sec", "coice_top1_share", "coice_top2_share", "coice_top2_concentration", "partner_count_120", "top_partner1", "top_partner2"]:
            df[c] = np.nan
    else:
        shift_df = shift_df.drop_duplicates(["game_id", "player_id"])
        df = df.merge(shift_df, on=["game_id", "player_id"], how="left")

    df = df.sort_values(["player_id", "game_date", "game_id"]).copy()
    g = df.groupby("player_id", group_keys=False)
    retrospective = [
        "shift_toi_sec", "coice_top1_sec", "coice_top2_sec", "coice_top1_share",
        "coice_top2_share", "coice_top2_concentration", "partner_count_120",
    ]
    for c in retrospective:
        if c not in df:
            df[c] = np.nan
        for w in [3, 5, 10]:
            df[f"{c}_avg_{w}"] = g[c].transform(lambda s, ww=w: s.shift(1).rolling(ww, min_periods=1).mean())

    p1 = g["top_partner1"].shift(1)
    p2 = g["top_partner2"].shift(1)
    q1 = g["top_partner1"].shift(2)
    q2 = g["top_partner2"].shift(2)
    r1 = g["top_partner1"].shift(3)

    jacc = []
    for a, b, c, d in zip(p1, p2, q1, q2):
        s1, s2 = _partner_set(a, b), _partner_set(c, d)
        jacc.append(float(len(s1 & s2) / len(s1 | s2)) if (s1 or s2) else math.nan)
    df["line_jaccard_prev"] = jacc
    df["same_top_partner_prev2"] = np.where(p1.notna() & q1.notna(), (p1 == q1).astype(float), np.nan)
    df["same_top_partner_prev3"] = np.where(p1.notna() & r1.notna(), (p1 == r1).astype(float), np.nan)
    df["historical_shift_games"] = g["shift_toi_sec"].transform(lambda s: s.shift(1).expanding(min_periods=1).count())
    return df


def main() -> None:
    games = v4.collect_game_ids(500)
    nhl_all: list[dict] = []
    mp_parts: list[pd.DataFrame] = []
    shift_parts: list[dict] = []
    errors: list[dict] = []

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch_bundle, pair): pair for pair in games}
        for i, fut in enumerate(as_completed(futs), start=1):
            nr, mp, sf, errs = fut.result()
            nhl_all.extend(nr)
            shift_parts.extend(sf)
            if mp is not None:
                mp_parts.append(mp)
            errors.extend(errs)
            if i % 50 == 0:
                print(f"fetched {i}/{len(games)}; nhl_rows={len(nhl_all)}; mp_games={len(mp_parts)}; shift_rows={len(shift_parts)}; errors={len(errors)}")

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

    df = v4.add_basic_features(df)
    df = v4.add_team_features(df)
    df = v4.add_moneypuck_features(df)
    shift_df = pd.DataFrame(shift_parts)
    df = add_shift_history_features(df, shift_df)
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
    v4_features = control + [
        c for c in df.columns if (
            (c.startswith("mp_") and (c.endswith("_avg_5") or c.endswith("_avg_10")))
            or c in ["pp_role_rank", "pp_top5_proxy", "toi_role_rank"]
            or c.startswith("team_mp_") or c.startswith("opp_mp_")
        )
    ]
    v4_features = list(dict.fromkeys(v4_features))
    shift_features = [
        c for c in df.columns if (
            (c.startswith("shift_toi_sec_avg_") or c.startswith("coice_top1_sec_avg_") or c.startswith("coice_top2_sec_avg_")
             or c.startswith("coice_top1_share_avg_") or c.startswith("coice_top2_share_avg_")
             or c.startswith("coice_top2_concentration_avg_") or c.startswith("partner_count_120_avg_"))
        )
    ] + ["line_jaccard_prev", "same_top_partner_prev2", "same_top_partner_prev3", "historical_shift_games"]
    v5_features = list(dict.fromkeys(v4_features + shift_features))

    dates = sorted(df["game_date"].dt.normalize().unique())
    train_cut = dates[int(len(dates) * 0.70)]
    val_cut = dates[int(len(dates) * 0.85)]
    tr = df[df["game_date"] < train_cut]
    va = df[(df["game_date"] >= train_cut) & (df["game_date"] < val_cut)]
    te = df[df["game_date"] >= val_cut]
    ytr, yv, yt = tr["target_goal"], va["target_goal"], te["target_goal"]

    logistic = lambda: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, C=0.7)),
    ])
    xgb_model = lambda: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", xgb.XGBClassifier(n_estimators=350, max_depth=3, learning_rate=0.03, subsample=0.85, colsample_bytree=0.8, min_child_weight=6, reg_lambda=2.5, objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=2)),
    ])
    lgb_model = lambda: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", lgb.LGBMClassifier(n_estimators=350, learning_rate=0.025, num_leaves=15, min_child_samples=40, reg_lambda=2.0, verbosity=-1, random_state=42, n_jobs=2)),
    ])

    models = [
        v4.score("logistic_v2_control", logistic(), tr[control], ytr, va[control], yv, te[control], yt),
        v4.score("xgboost_v4_control", xgb_model(), tr[v4_features], ytr, va[v4_features], yv, te[v4_features], yt),
        v4.score("logistic_v5", logistic(), tr[v5_features], ytr, va[v5_features], yv, te[v5_features], yt),
        v4.score("xgboost_v5", xgb_model(), tr[v5_features], ytr, va[v5_features], yv, te[v5_features], yt),
        v4.score("lightgbm_v5", lgb_model(), tr[v5_features], ytr, va[v5_features], yv, te[v5_features], yt),
    ]
    best_v5 = min([m for m in models if m["name"].endswith("_v5")], key=lambda m: m["validation_brier"])
    summary = {
        "rows_total": len(df),
        "players": int(df["player_id"].nunique()),
        "games": int(df["game_id"].nunique()),
        "moneypuck_game_coverage": len(mp_parts) / max(1, len(games)),
        "shift_game_coverage": float(shift_df["game_id"].nunique() / max(1, len(games))) if not shift_df.empty else 0.0,
        "shift_player_row_coverage": float(df["shift_toi_sec"].notna().mean()),
        "train_rows": len(tr), "validation_rows": len(va), "test_rows": len(te),
        "feature_counts": {"v2": len(control), "v4": len(v4_features), "v5": len(v5_features), "shift_added": len(shift_features)},
        "best_v5_by_validation_brier": best_v5["name"],
        "models": models,
    }
    print(json.dumps(summary, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df[["game_date", "game_id", "player_id", "player_name", "team", "target_goal"] + shift_features].to_csv(OUT / "v5_shift_features.csv", index=False)

    lines = [
        "# Real NHL player-goal V5 — historical line stability",
        "",
        "V5 adds leakage-safe historical teammate/co-ice stability reconstructed from NHL shift charts. Shift information from the game being predicted is never used: all overlap/partner features are lagged to prior games before modeling.",
        "",
        f"- Rows: **{len(df)}**",
        f"- Games: **{df['game_id'].nunique()}**",
        f"- Shift game coverage: **{summary['shift_game_coverage']:.1%}**",
        f"- Shift player-row coverage: **{summary['shift_player_row_coverage']:.1%}**",
        f"- Added shift/continuity features: **{len(shift_features)}**",
        f"- Selected V5 model by validation Brier: **{best_v5['name']}**",
        "",
        "| Model | Val Brier | Test Brier | Test Log Loss | Test AUC | Test ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in models:
        lines.append(f"| {m['name']} | {m['validation_brier']:.6f} | {m['test_brier']:.6f} | {m['test_log_loss']:.6f} | {m['test_auc']:.6f} | {m['test_ece']:.6f} |")
    lines += [
        "",
        "## Leakage controls",
        "- Co-ice and top-partner measurements are reconstructed retrospectively per completed game.",
        "- Every numeric shift feature is shifted by one game before rolling averages.",
        "- Partner continuity for a target game compares only the player's two preceding completed games.",
        "- Model selection uses validation Brier only; the final test is untouched for selection.",
        "",
        "Economic ROI is intentionally not reported because genuine timestamp-valid historical anytime-goal odds are not yet integrated.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
