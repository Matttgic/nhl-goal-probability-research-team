"""Pregame shooting diagnostics. Empirical hit rates are not model probabilities."""
from __future__ import annotations

from bisect import bisect_left
from math import isfinite, sqrt

import numpy as np
import pandas as pd


def wilson_lower(hits: int, sample: int, z: float = 1.96) -> float:
    """Lower endpoint of the two-sided 95% Wilson interval by default."""
    if sample < 0 or hits < 0 or hits > sample or not isfinite(z) or z <= 0:
        raise ValueError("Invalid hit count, sample size or confidence parameter")
    if sample == 0:
        return float("nan")
    p = hits / sample
    return (p + z*z/(2*sample) - z*sqrt(p*(1-p)/sample + z*z/(4*sample*sample))) / (1 + z*z/sample)


def _prepare(frame: pd.DataFrame, *, history: bool) -> pd.DataFrame:
    required = {"game_id", "player_id", "game_date"}
    if history:
        required.add("shots_on_goal")
    if required - set(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    out = frame.copy()
    if out[["game_id", "player_id"]].isna().any().any():
        raise ValueError("Game and player IDs cannot be missing")
    if out.duplicated(["game_id", "player_id"]).any():
        raise ValueError("Duplicate player-game observations")
    out["_day"] = pd.to_datetime(out["game_date"], utc=True, errors="raise").dt.normalize()
    if out["_day"].isna().any():
        raise ValueError("Game dates cannot be missing")
    if "is_home" in out:
        if not out["is_home"].dropna().isin([True, False, 0, 1]).all():
            raise ValueError("is_home must be boolean, 0/1 or missing")
    if history:
        for col in ("shots_on_goal", "goals", "toi_seconds", "pp_toi_seconds"):
            if col not in out:
                continue
            values = pd.to_numeric(out[col], errors="raise")
            present = values.dropna()
            if not np.isfinite(present).all() or (present < 0).any():
                raise ValueError(f"Invalid negative/non-finite values in {col}")
            if col == "shots_on_goal" and values.isna().any():
                raise ValueError("Missing shots must not be interpreted as zero")
            if col in ("shots_on_goal", "goals") and (present % 1 != 0).any():
                raise ValueError(f"{col} must contain integer counts")
            out[col] = values
    return out


def _profile(rows: pd.DataFrame, prefix: str) -> dict:
    n = len(rows)
    result = {f"{prefix}_games": n, f"{prefix}_sog_avg": float(rows["shots_on_goal"].mean())}
    for threshold in (1, 2, 3, 4):
        hits = int((rows["shots_on_goal"] >= threshold).sum())
        result[f"{prefix}_sog_{threshold}plus_rate"] = hits/n if n else np.nan
        result[f"{prefix}_sog_{threshold}plus_wilson_lower"] = wilson_lower(hits, n)
    if "toi_seconds" in rows:
        complete = rows["toi_seconds"].notna().all() and n > 0
        toi = float(rows["toi_seconds"].sum()) if complete else np.nan
        result[f"{prefix}_toi_avg_seconds"] = toi/n if n else np.nan
        result[f"{prefix}_sog_per60"] = float(rows["shots_on_goal"].sum())*3600/toi if toi > 0 else np.nan
    if "goals" in rows:
        valid = rows["goals"].dropna()
        result[f"{prefix}_goal_observed_games"] = len(valid)
        result[f"{prefix}_goal_hit_rate"] = float((valid > 0).mean()) if len(valid) else np.nan
    if "pp_toi_seconds" in rows:
        valid = rows["pp_toi_seconds"].dropna()
        result[f"{prefix}_pp_observed_games"] = len(valid)
        result[f"{prefix}_pp_usage_share"] = float((valid > 0).mean()) if len(valid) else np.nan
        result[f"{prefix}_pp_toi_avg_seconds"] = float(valid.mean()) if len(valid) else np.nan
    return result


def build_shot_context_features(history: pd.DataFrame, candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return features for candidate IDs, using only games on strictly earlier UTC dates.

    Required history: game_id, player_id, game_date, shots_on_goal.
    Optional: goals, toi_seconds, pp_toi_seconds, is_home, opponent_abbrev.
    Candidates default to history, for training-table construction. The conservative
    date cutoff excludes every same-day game, even when exact puck-drop times exist.
    No current-game performance, future roster, odds or live API is consulted.
    """
    source = _prepare(history, history=True).sort_values(["_day", "game_id"])
    targets = _prepare(history if candidates is None else candidates, history=False)
    by_player = {pid: (group.reset_index(drop=True), group["_day"].tolist())
                 for pid, group in source.groupby("player_id", sort=False)}
    rows = []
    for target in targets.to_dict("records"):
        group, dates = by_player.get(target["player_id"], (source.iloc[:0], []))
        prior = group.iloc[:bisect_left(dates, target["_day"])]
        row = {"game_id": target["game_id"], "player_id": target["player_id"],
               "shot_context_history_games": len(prior)}
        for window in (5, 10):
            row.update(_profile(prior.tail(window), f"shot_context_l{window}"))
        venue = target.get("is_home")
        venue_known = venue is not None and not pd.isna(venue) and "is_home" in prior
        same_venue = prior[prior["is_home"].eq(venue)] if venue_known else prior.iloc[:0]
        row["shot_context_venue_available"] = bool(venue_known)
        row.update(_profile(same_venue.tail(10), "shot_context_venue_l10"))
        opponent = target.get("opponent_abbrev")
        opponent_known = opponent is not None and not pd.isna(opponent) and str(opponent).strip() != "" and "opponent_abbrev" in prior
        versus = prior[prior["opponent_abbrev"].eq(opponent)] if opponent_known else prior.iloc[:0]
        if venue_known:
            versus = versus[versus["is_home"].eq(venue)]
        row["shot_context_opponent_available"] = bool(opponent_known)
        row.update(_profile(versus.tail(10), "shot_context_matchup_l10"))
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["game_id", "player_id"])
