from __future__ import annotations

import argparse
from pathlib import Path

from nhl_goal_probability_research_team.data import build_goal_scorer_dataset, load_sportsdataverse_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe NHL goal-scorer features from SportsDataverse.")
    parser.add_argument("--seasons", nargs="+", type=int, required=True, help="NHL season start years, e.g. 2023 2024")
    parser.add_argument("--recent-games", type=int, default=10)
    parser.add_argument("--min-history-games", type=int, default=3)
    parser.add_argument("--no-linemates", action="store_true", help="Skip on-ice linemate overlap features for faster runs")
    parser.add_argument("--output", default="workspace/nhl_goal_scorer_features.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pbp, box = load_sportsdataverse_frames(args.seasons)
    dataset = build_goal_scorer_dataset(
        pbp,
        box,
        recent_games=args.recent_games,
        min_history_games=args.min_history_games,
        include_linemates=not args.no_linemates,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    ready = int(dataset["history_ready"].sum()) if "history_ready" in dataset else 0
    print(f"wrote {len(dataset):,} player-game rows to {output}")
    print(f"history-ready rows: {ready:,}")
    print(f"positive goal labels: {int(dataset['label_goal'].sum()):,}")


if __name__ == "__main__":
    main()
