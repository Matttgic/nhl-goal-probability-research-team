"""Produce experimental next-48-hour forecasts and a mobile-readable report."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nhl_goal_probability_research_team.upcoming import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports")
    parser.add_argument("--cache", default="workspace/nhl_public_cache")
    parser.add_argument("--max-games", type=int, default=600)
    args = parser.parse_args()
    if not 150 <= args.max_games <= 1000:
        parser.error("--max-games must be between 150 and 1000")
    output = Path(args.output)
    now = pd.Timestamp(datetime.now(timezone.utc))
    try:
        run(output, Path(args.cache), now, args.max_games)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        message = str(exc)
        (output / "upcoming_predictions.md").write_text(
            "# Prochains matchs NHL — calcul indisponible\n\n"
            f"Tentative : {now.isoformat()}.\n\nAucune nouvelle prédiction publiée.\n\n"
            f"Motif : {message.replace(chr(10), ' ')}\n\n"
            "Consulter la dernière exécution de **Upcoming NHL experimental forecasts** dans Actions.\n",
            encoding="utf-8")
        (output / "upcoming_validation.json").write_text(json.dumps({"status":"failed", "generated_at":now.isoformat(), "error":message}))
        (output / "upcoming_predictions.csv").write_text("game_id,player_id,probability_goal,status\n")
        raise


if __name__ == "__main__":
    main()
