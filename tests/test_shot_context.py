"""Synthetic invariance fixtures; these tests do not measure predictive quality."""
import unittest

import pandas as pd

from nhl_goal_probability_research_team.data.shot_context import build_shot_context_features, wilson_lower


class ShotContextTests(unittest.TestCase):
    def history(self):
        return pd.DataFrame({
            "game_id": [1, 2, 3, 4, 5], "player_id": [10]*5,
            "game_date": ["2026-01-01", "2026-01-03", "2026-01-05", "2026-01-05", "2026-01-07"],
            "shots_on_goal": [2, 4, 99, 98, 97], "goals": [0, 1, 7, 8, 9],
            "toi_seconds": [1200]*5, "pp_toi_seconds": [0, 120, 900, 900, 900],
            "is_home": [True, False, True, True, True], "opponent_abbrev": ["AAA", "BBB", "AAA", "AAA", "AAA"],
        })

    def test_future_and_same_day_results_cannot_change_features(self):
        source = self.history()
        target = source[source.game_id.eq(3)]
        original = build_shot_context_features(source, target)
        modified = source.copy()
        modified.loc[modified.game_id.ge(3), ["shots_on_goal", "goals", "pp_toi_seconds"]] = 0
        pd.testing.assert_frame_equal(original, build_shot_context_features(modified, target))
        row = original.iloc[0]
        self.assertEqual(row.shot_context_l10_games, 2)
        self.assertEqual(row.shot_context_l10_sog_avg, 3)
        self.assertEqual(row.shot_context_l10_sog_per60, 9)
        self.assertEqual(row.shot_context_venue_l10_games, 1)
        self.assertEqual(row.shot_context_matchup_l10_games, 1)
        self.assertEqual(row.shot_context_l10_pp_usage_share, .5)

    def test_order_independent(self):
        source = self.history()
        candidates = source.iloc[[4]]
        pd.testing.assert_frame_equal(build_shot_context_features(source, candidates),
                                      build_shot_context_features(source.iloc[::-1], candidates))

    def test_unknown_context_and_history_are_missing_not_zero(self):
        source = self.history().drop(columns=["is_home", "opponent_abbrev", "pp_toi_seconds"])
        row = build_shot_context_features(source).iloc[0]
        self.assertEqual(row.shot_context_history_games, 0)
        self.assertTrue(pd.isna(row.shot_context_l10_sog_avg))
        self.assertTrue(pd.isna(row.shot_context_l10_sog_2plus_rate))
        self.assertFalse(row.shot_context_venue_available)
        self.assertFalse(row.shot_context_opponent_available)
        self.assertNotIn("shot_context_l10_pp_usage_share", row.index)

    def test_missing_pp_time_is_reported(self):
        source = self.history()
        source.loc[1, "pp_toi_seconds"] = float("nan")
        row = build_shot_context_features(source, source.iloc[[2]]).iloc[0]
        self.assertEqual(row.shot_context_l10_pp_observed_games, 1)
        self.assertEqual(row.shot_context_l10_pp_usage_share, 0)

    def test_thin_perfect_record_is_not_high_confidence(self):
        self.assertLess(wilson_lower(2, 2), .35)
        self.assertLess(wilson_lower(8, 10), .51)
        self.assertTrue(pd.isna(wilson_lower(0, 0)))

    def test_duplicate_invalid_and_missing_shots_are_rejected(self):
        source = self.history()
        with self.assertRaises(ValueError):
            build_shot_context_features(pd.concat([source, source.iloc[[0]]]))
        for invalid in (-1, float("nan"), float("inf"), 1.5):
            changed = source.copy()
            changed["shots_on_goal"] = changed["shots_on_goal"].astype(float)
            changed.loc[0, "shots_on_goal"] = invalid
            with self.assertRaises(ValueError):
                build_shot_context_features(changed)


if __name__ == "__main__":
    unittest.main()
