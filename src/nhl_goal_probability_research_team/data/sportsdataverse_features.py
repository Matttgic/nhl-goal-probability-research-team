from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

SHOT_EVENTS = {"SHOT", "GOAL", "MISSED_SHOT", "MISSED SHOT"}


def load_sportsdataverse_frames(seasons: int | Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load NHL play-by-play and player boxscores from SportsDataverse releases.

    The loader intentionally uses the Python companion of fastRhockey because it
    reads the same SportsDataverse release store while fitting this Python project.
    """
    try:
        from sportsdataverse.nhl import load_nhl_pbp, load_nhl_player_boxscore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "SportsDataverse is not installed. Install the project with the sports extra: "
            "`uv sync --extra sports`."
        ) from exc

    pbp = load_nhl_pbp(seasons=seasons, return_as_pandas=True)
    box = load_nhl_player_boxscore(seasons=seasons, return_as_pandas=True)
    if pbp is None or len(pbp) == 0:
        raise ValueError("SportsDataverse returned no NHL play-by-play rows for the requested season(s).")
    if box is None or len(box) == 0:
        raise ValueError("SportsDataverse returned no NHL player boxscore rows for the requested season(s).")
    return pd.DataFrame(pbp), pd.DataFrame(box)


def _toi_to_seconds(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return 0.0
    minutes, seconds = text.split(":", 1)
    try:
        return float(minutes) * 60.0 + float(seconds)
    except ValueError:
        return 0.0


def _rolling_sum_shifted(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=min_periods).sum()


def _rolling_mean_shifted(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def _is_power_play(row: pd.Series) -> bool:
    strength = str(row.get("strength_state", "") or row.get("strength_code", "")).lower()
    if "pp" in strength or "power" in strength:
        return True
    try:
        home = int(row.get("home_skaters", 0) or 0)
        away = int(row.get("away_skaters", 0) or 0)
    except (TypeError, ValueError):
        return False
    event_side = str(row.get("event_team_type", "")).lower()
    if event_side == "home":
        return home > away
    if event_side == "away":
        return away > home
    event_team = str(row.get("event_team_abbr", ""))
    if event_team and event_team == str(row.get("home_abbr", "")):
        return home > away
    if event_team and event_team == str(row.get("away_abbr", "")):
        return away > home
    return False


def _aggregate_shot_events(pbp: pd.DataFrame) -> pd.DataFrame:
    work = pbp.copy()
    event = work.get("event_type", work.get("event", pd.Series(index=work.index, dtype="object")))
    work["_event"] = event.astype(str).str.upper().str.strip()
    shots = work[work["_event"].isin(SHOT_EVENTS)].copy()
    if shots.empty:
        return pd.DataFrame(columns=["game_id", "player_id", "pbp_shots", "pbp_goals", "ixg", "pp_shots", "pp_ixg", "avg_shot_distance"])

    shots["player_id"] = pd.to_numeric(shots.get("event_player_1_id"), errors="coerce")
    shots = shots.dropna(subset=["game_id", "player_id"])
    shots["player_id"] = shots["player_id"].astype("int64")
    shots["pbp_goals"] = (shots["_event"] == "GOAL").astype(int)
    shots["ixg"] = pd.to_numeric(shots.get("xg", 0.0), errors="coerce").fillna(0.0)
    shots["_is_pp"] = shots.apply(_is_power_play, axis=1).astype(int)
    shots["pp_shots"] = shots["_is_pp"]
    shots["pp_ixg"] = shots["ixg"] * shots["_is_pp"]
    shots["shot_distance"] = pd.to_numeric(shots.get("shot_distance", np.nan), errors="coerce")

    out = (
        shots.groupby(["game_id", "player_id"], as_index=False)
        .agg(
            pbp_shots=("player_id", "size"),
            pbp_goals=("pbp_goals", "sum"),
            ixg=("ixg", "sum"),
            pp_shots=("pp_shots", "sum"),
            pp_ixg=("pp_ixg", "sum"),
            avg_shot_distance=("shot_distance", "mean"),
        )
    )
    return out


def _linemate_event_overlap(pbp: pd.DataFrame) -> pd.DataFrame:
    """Approximate deployment chemistry from actual on-ice event overlap.

    SportsDataverse's full NHL PBP exposes home_on_1_id..home_on_7_id and the
    away equivalents. Counting co-presence across events is robust to projected
    lines and produces a useful deployment signal without using future games.
    """
    pair_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    event_counts: Counter[tuple[int, int]] = Counter()
    home_cols = [f"home_on_{i}_id" for i in range(1, 8)]
    away_cols = [f"away_on_{i}_id" for i in range(1, 8)]

    needed = ["game_id", *home_cols, *away_cols, "home_goalie_id", "away_goalie_id"]
    frame = pbp[[c for c in needed if c in pbp.columns]].copy()
    if "game_id" not in frame.columns:
        return pd.DataFrame(columns=["game_id", "player_id", "top_linemate_id", "top_linemate_event_share"])

    def ids_from_row(row: pd.Series, cols: list[str], goalie_col: str) -> list[int]:
        goalie = pd.to_numeric(row.get(goalie_col), errors="coerce")
        values: list[int] = []
        for col in cols:
            if col not in row.index:
                continue
            val = pd.to_numeric(row.get(col), errors="coerce")
            if pd.isna(val):
                continue
            pid = int(val)
            if not pd.isna(goalie) and pid == int(goalie):
                continue
            values.append(pid)
        return sorted(set(values))

    for _, row in frame.iterrows():
        game_raw = pd.to_numeric(row.get("game_id"), errors="coerce")
        if pd.isna(game_raw):
            continue
        game_id = int(game_raw)
        for players in (
            ids_from_row(row, home_cols, "home_goalie_id"),
            ids_from_row(row, away_cols, "away_goalie_id"),
        ):
            for player in players:
                key = (game_id, player)
                event_counts[key] += 1
                for teammate in players:
                    if teammate != player:
                        pair_counts[key][teammate] += 1

    rows: list[dict[str, float | int]] = []
    for (game_id, player_id), counts in pair_counts.items():
        if not counts:
            continue
        teammate_id, overlap = counts.most_common(1)[0]
        denominator = max(event_counts[(game_id, player_id)], 1)
        rows.append(
            {
                "game_id": game_id,
                "player_id": player_id,
                "top_linemate_id": teammate_id,
                "top_linemate_event_share": overlap / denominator,
            }
        )
    return pd.DataFrame(rows)


def build_goal_scorer_dataset(
    pbp: pd.DataFrame,
    player_box: pd.DataFrame,
    *,
    recent_games: int = 10,
    min_history_games: int = 3,
    include_linemates: bool = True,
) -> pd.DataFrame:
    """Build one leakage-safe row per skater-game for goal-scorer modelling.

    Labels come from the current game, while every rolling predictor is shifted by
    one game. This function therefore supports temporal / walk-forward training.
    """
    if recent_games < 2:
        raise ValueError("recent_games must be >= 2")

    box = player_box.copy()
    required = {"game_id", "player_id", "game_date"}
    missing = required - set(box.columns)
    if missing:
        raise ValueError(f"player_box is missing required columns: {sorted(missing)}")

    if "position" in box.columns:
        box = box[box["position"].astype(str).str.upper().ne("G")].copy()

    for col in ["game_id", "player_id"]:
        box[col] = pd.to_numeric(box[col], errors="coerce")
    box = box.dropna(subset=["game_id", "player_id"]).copy()
    box[["game_id", "player_id"]] = box[["game_id", "player_id"]].astype("int64")
    box["game_date"] = pd.to_datetime(box["game_date"], errors="coerce", utc=True)
    box["toi_seconds"] = box.get("toi", 0).map(_toi_to_seconds)

    numeric_defaults = {
        "goals": 0,
        "assists": 0,
        "shots_on_goal": 0,
        "power_play_goals": 0,
        "shifts": 0,
    }
    for col, default in numeric_defaults.items():
        if col not in box.columns:
            box[col] = default
        box[col] = pd.to_numeric(box[col], errors="coerce").fillna(default)

    shot_features = _aggregate_shot_events(pbp)
    data = box.merge(shot_features, on=["game_id", "player_id"], how="left")
    fill_zero = ["pbp_shots", "pbp_goals", "ixg", "pp_shots", "pp_ixg"]
    for col in fill_zero:
        data[col] = pd.to_numeric(data.get(col, 0), errors="coerce").fillna(0.0)

    # Prefer official boxscore shots/goals for labels; PBP is used for richer shot quality.
    data["label_goal"] = (data["goals"] > 0).astype(int)
    data = data.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    grouped = data.groupby("player_id", group_keys=False)
    rolling_sum_cols = ["goals", "shots_on_goal", "ixg", "pp_shots", "pp_ixg", "toi_seconds"]
    for col in rolling_sum_cols:
        data[f"recent_{col}_{recent_games}"] = grouped[col].transform(
            lambda s: _rolling_sum_shifted(s, recent_games, min_history_games)
        )

    data[f"recent_shooting_pct_{recent_games}"] = (
        data[f"recent_goals_{recent_games}"]
        / data[f"recent_shots_on_goal_{recent_games}"].replace(0, np.nan)
    )
    data[f"recent_goals_per60_{recent_games}"] = (
        data[f"recent_goals_{recent_games}"] * 3600.0
        / data[f"recent_toi_seconds_{recent_games}"].replace(0, np.nan)
    )
    data[f"recent_ixg_per60_{recent_games}"] = (
        data[f"recent_ixg_{recent_games}"] * 3600.0
        / data[f"recent_toi_seconds_{recent_games}"].replace(0, np.nan)
    )
    data[f"recent_shots_per60_{recent_games}"] = (
        data[f"recent_shots_on_goal_{recent_games}"] * 3600.0
        / data[f"recent_toi_seconds_{recent_games}"].replace(0, np.nan)
    )
    data[f"recent_pp_shot_share_{recent_games}"] = (
        data[f"recent_pp_shots_{recent_games}"]
        / data[f"recent_shots_on_goal_{recent_games}"].replace(0, np.nan)
    )

    # Recent average deployment is also shifted, so it is known pre-game.
    data[f"recent_toi_avg_{recent_games}"] = grouped["toi_seconds"].transform(
        lambda s: _rolling_mean_shifted(s, recent_games, min_history_games)
    )

    if include_linemates:
        overlap = _linemate_event_overlap(pbp)
        if not overlap.empty:
            data = data.merge(overlap, on=["game_id", "player_id"], how="left")
            teammate_strength = data[["game_id", "player_id", f"recent_ixg_per60_{recent_games}"]].rename(
                columns={
                    "player_id": "top_linemate_id",
                    f"recent_ixg_per60_{recent_games}": "top_linemate_recent_ixg_per60",
                }
            )
            data = data.merge(teammate_strength, on=["game_id", "top_linemate_id"], how="left")
            data["linemate_boost_score"] = (
                data["top_linemate_event_share"].fillna(0.0)
                * data["top_linemate_recent_ixg_per60"].fillna(0.0)
            )
        else:
            data["top_linemate_id"] = np.nan
            data["top_linemate_event_share"] = 0.0
            data["top_linemate_recent_ixg_per60"] = np.nan
            data["linemate_boost_score"] = 0.0

    # Keep rows with enough history for model fitting; users can retain all rows by
    # setting min_history_games=1.
    history_col = f"recent_toi_seconds_{recent_games}"
    data["history_ready"] = data[history_col].notna() & (data[history_col] > 0)
    return data
