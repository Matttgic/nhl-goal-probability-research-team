import unittest

import pandas as pd

from nhl_goal_probability_research_team.data import build_goal_scorer_dataset


class GoalScorerFeatureTests(unittest.TestCase):
    def test_rolling_features_are_pregame_only(self):
        box = pd.DataFrame(
            {
                "game_id": [1, 2, 3, 4],
                "player_id": [10, 10, 10, 10],
                "player_name": ["Shooter"] * 4,
                "position": ["C"] * 4,
                "game_date": ["2026-01-01", "2026-01-03", "2026-01-05", "2026-01-07"],
                "team_abbrev": ["AAA"] * 4,
                "goals": [1, 0, 1, 1],
                "assists": [0, 1, 0, 0],
                "shots_on_goal": [4, 3, 5, 2],
                "power_play_goals": [0, 0, 1, 0],
                "shifts": [20, 20, 21, 22],
                "toi": ["18:00", "19:00", "20:00", "21:00"],
            }
        )
        pbp = pd.DataFrame(
            {
                "game_id": [1, 2, 3, 4],
                "game_date": ["2026-01-01", "2026-01-03", "2026-01-05", "2026-01-07"],
                "event_type": ["GOAL", "SHOT", "GOAL", "GOAL"],
                "event_player_1_id": [10, 10, 10, 10],
                "event_team_type": ["home"] * 4,
                "event_team_abbr": ["AAA"] * 4,
                "home_abbr": ["AAA"] * 4,
                "away_abbr": ["BBB"] * 4,
                "home_skaters": [5, 5, 5, 5],
                "away_skaters": [5, 5, 4, 5],
                "xg": [0.20, 0.10, 0.35, 0.25],
                "shot_distance": [18, 25, 12, 20],
            }
        )

        out = build_goal_scorer_dataset(
            pbp,
            box,
            recent_games=3,
            min_history_games=2,
            include_linemates=False,
        )
        game4 = out.loc[out["game_id"] == 4].iloc[0]

        self.assertEqual(game4["label_goal"], 1)
        # Current game goal is excluded: prior three games contain exactly two goals.
        self.assertEqual(game4["recent_goals_3"], 2)
        self.assertAlmostEqual(game4["recent_ixg_3"], 0.65, places=6)
        self.assertTrue(bool(game4["history_ready"]))


if __name__ == "__main__":
    unittest.main()
