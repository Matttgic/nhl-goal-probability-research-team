"""Synthetic end-to-end fixtures. No match forecast or economic evidence."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nhl_goal_probability_research_team.upcoming import (
    boxscore_rows, chronological_parts, feature_table, predict_upcoming,
    render_report, schedule_games, train_forecaster,
)


def fixture_history():
    rows = []
    for day in range(55):
        for pid in range(12):
            rows.append({"game_id": day+1, "game_date": (pd.Timestamp("2025-01-01")+pd.Timedelta(days=day)).date().isoformat(),
                         "player_id": pid, "goals": int((day+3*pid)%7 == 0), "shots_on_goal": (day+pid)%5,
                         "is_home": day%2, "position_D": int(pid%4 == 0), "opponent_abbrev": "AAA" if day%3 else "BBB",
                         "toi_seconds": 900 + 30*(pid%5)})
    return pd.DataFrame(rows)


class UpcomingTests(unittest.TestCase):
    def test_split_keeps_entire_dates_together(self):
        table = feature_table(fixture_history())
        train, calibration, test = chronological_parts(table)
        self.assertLess(max(train.game_date), min(calibration.game_date))
        self.assertLess(max(calibration.game_date), min(test.game_date))
        self.assertFalse(set(train.game_id) & set(calibration.game_id))

    def test_candidate_features_exclude_current_and_future_results(self):
        history = fixture_history()
        candidates = history[history.game_id.eq(40)].copy()
        before = feature_table(history, candidates)
        changed = history.copy()
        changed.loc[changed.game_id.ge(40), ["shots_on_goal", "goals"]] = 0
        pd.testing.assert_frame_equal(before, feature_table(changed, candidates))

    def test_real_pipeline_on_synthetic_fixture_and_missing_players(self):
        history = fixture_history()
        model, metrics = train_forecaster(history)
        self.assertGreater(metrics["test_rows"], 100)
        self.assertLess(metrics["calibration_through"], metrics["test_from"])
        self.assertFalse(metrics["odds_used"])
        candidates = pd.DataFrame({
            "game_id": [100]*2, "game_date": ["2025-03-02"]*2, "player_id": [0, 99],
            "player_name": ["Synthetic known", "Synthetic new"], "team_abbrev": ["AAA"]*2,
            "opponent_abbrev": ["BBB"]*2, "is_home": [1]*2, "position_D": [1, 0],
            "starts_at": ["2025-03-02T20:00:00Z"]*2,
        })
        now = pd.Timestamp("2025-03-02T12:00:00Z")
        forecast = predict_upcoming(history, candidates, model, now)
        known = forecast[forecast.player_id.eq(0)].iloc[0]
        unknown = forecast[forecast.player_id.eq(99)].iloc[0]
        self.assertTrue(0 < known.probability_goal < 1)
        self.assertTrue(pd.isna(unknown.probability_goal))
        games = [{"id":100, "startTimeUTC":"2025-03-02T20:00:00Z", "homeTeam":{"abbrev":"AAA"}, "awayTeam":{"abbrev":"BBB"}}]
        report = render_report(forecast, games, metrics, now, [])
        self.assertIn("expérimental", report)
        self.assertIn("21:00 (Paris)", report)
        self.assertIn("Synthetic known", report)
        self.assertIn("aucune probabilité inventée", report)
        self.assertIn("indisponibles", report)  # missing PP TOI is explicit
        candidates.loc[0, "starts_at"] = now.isoformat()
        with self.assertRaises(ValueError):
            predict_upcoming(history, candidates, model, now)

    def test_missing_boxscore_outcomes_block_import(self):
        game = {"id":1, "game_date":"2025-01-01"}
        box = {"gameState":"OFF", "homeTeam":{"abbrev":"AAA"}, "awayTeam":{"abbrev":"BBB"},
               "playerByGameStats":{"homeTeam":{"forwards":[{"playerId":1, "goals":0, "toi":"15:00"}]}}}
        with self.assertRaises(ValueError):
            boxscore_rows(game, box)
        box["playerByGameStats"]["homeTeam"]["forwards"][0]["sog"] = 2
        rows = boxscore_rows(game, box)
        self.assertEqual(rows[0]["shots_on_goal"], 2)
        self.assertEqual(rows[0]["toi_seconds"], 900)
        self.assertNotIn("pp_toi_seconds", rows[0])
        box["gameState"] = "LIVE"
        with self.assertRaises(ValueError):
            boxscore_rows(game, box)

    def test_regular_season_schedule_only(self):
        games = schedule_games({"gameWeek":[{"date":"2025-01-01", "games":[{"id":1,"gameType":1},{"id":2,"gameType":2},{"id":3,"gameType":3}]}]})
        self.assertEqual([g["id"] for g in games], [2])


if __name__ == "__main__":
    unittest.main()
