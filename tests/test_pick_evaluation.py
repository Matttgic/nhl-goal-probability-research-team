"""Synthetic ledger cases validate accounting, not betting profitability."""
import unittest

import pandas as pd

from nhl_goal_probability_research_team.data.pick_evaluation import evaluate_picks


class PickEvaluationTests(unittest.TestCase):
    def ledger(self):
        return pd.DataFrame({
            "pick_id": ["a", "b", "c", "d", "e"], "cohort": ["value"]*5,
            "market": ["anytime_goal"]*5, "probability": [.5]*5,
            "predicted_at": ["2026-01-01T18:00:00Z"]*5,
            "starts_at": ["2026-01-01T19:00:00Z"]*5,
            "result": ["WIN", "LOSS", "VOID", "PUSH", "PENDING"],
            "decimal_odds": [3.0]*5, "odds_at": ["2026-01-01T17:59:00Z"]*5,
            "odds_source": ["synthetic_test_only"]*5,
        })

    def test_voids_pushes_pending_and_profit(self):
        row = evaluate_picks(self.ledger()).iloc[0]
        self.assertEqual(row.settled, 2)
        self.assertEqual(row.priced_settled, 2)
        self.assertEqual(row.profit_units, 1)
        self.assertEqual(row.roi, .5)
        self.assertEqual(row.brier, .25)
        self.assertAlmostEqual(row.log_loss, .69314718056)
        self.assertEqual(row.voids, 1)
        self.assertEqual(row.pushes, 1)
        self.assertEqual(row.pending, 1)

    def test_unpriced_picks_have_no_economic_claim(self):
        ledger = self.ledger().drop(columns=["decimal_odds", "odds_at", "odds_source"])
        row = evaluate_picks(ledger).iloc[0]
        self.assertEqual(row.priced_settled, 0)
        self.assertTrue(pd.isna(row.roi))
        self.assertTrue(pd.isna(row.profit_units))

    def test_only_valid_priced_rows_enter_roi(self):
        ledger = self.ledger()
        ledger.loc[1, "decimal_odds"] = float("nan")
        row = evaluate_picks(ledger).iloc[0]
        self.assertEqual(row.settled, 2)
        self.assertEqual(row.priced_settled, 1)
        self.assertEqual(row.roi, 2)

    def test_quote_after_lock_and_late_prediction_rejected(self):
        ledger = self.ledger()
        ledger.loc[0, "odds_at"] = "2026-01-01T18:01:00Z"
        with self.assertRaises(ValueError):
            evaluate_picks(ledger)
        ledger = self.ledger()
        ledger.loc[0, "predicted_at"] = ledger.loc[0, "starts_at"]
        with self.assertRaises(ValueError):
            evaluate_picks(ledger)

    def test_no_duplicate_pick_and_missing_source(self):
        ledger = self.ledger()
        with self.assertRaises(ValueError):
            evaluate_picks(pd.concat([ledger, ledger.iloc[[0]]]))
        ledger.loc[0, "odds_source"] = ""
        with self.assertRaises(ValueError):
            evaluate_picks(ledger)

    def test_naive_timestamps_do_not_assume_a_timezone(self):
        for column in ("predicted_at", "starts_at", "odds_at"):
            ledger = self.ledger()
            ledger.loc[0, column] = "2026-01-01T17:00:00"
            with self.assertRaises(ValueError):
                evaluate_picks(ledger)

    def test_separate_cohorts_and_markets(self):
        ledger = self.ledger()
        other = ledger.iloc[[0]].copy()
        other["cohort"] = "probability"
        ledger.loc[1, "market"] = "shots"
        result = evaluate_picks(pd.concat([ledger, other]))
        self.assertEqual(len(result), 3)

    def test_no_settled_results_is_not_zero_percent(self):
        row = evaluate_picks(self.ledger().iloc[[2, 3, 4]]).iloc[0]
        self.assertEqual(row.settled, 0)
        self.assertTrue(pd.isna(row.hit_rate))
        self.assertTrue(pd.isna(row.brier))

    def test_invalid_probability_odds_and_status_rejected(self):
        for column, value in [("probability", float("nan")), ("probability", 1.1),
                              ("decimal_odds", 1), ("decimal_odds", float("inf")),
                              ("result", "UNKNOWN")]:
            ledger = self.ledger()
            ledger.loc[0, column] = value
            with self.assertRaises(ValueError):
                evaluate_picks(ledger)


if __name__ == "__main__":
    unittest.main()
