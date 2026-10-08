"""Evaluate a real locked-pick ledger without using any API or provider key."""
import argparse
from pathlib import Path

import pandas as pd

from nhl_goal_probability_research_team.data.pick_evaluation import evaluate_picks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picks", required=True, help="CSV of genuine timestamped picks")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = evaluate_picks(pd.read_csv(args.picks))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
