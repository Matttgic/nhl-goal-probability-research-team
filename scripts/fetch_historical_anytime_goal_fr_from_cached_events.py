from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd

from scripts.historical_anytime_goal_backtest_oddsapi import OddsAPIClient, historical_event_odds, extract_yes_prices

EVENT_MAP = Path(os.getenv("FR_EVENT_MAP", "cached/us/workspace/historical_anytime_goal_oddsapi/event_map.csv"))
OUT = Path(os.getenv("FR_RAW_OUT", "workspace/historical_anytime_goal_fr_raw"))
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_MINUTES_BEFORE = int(os.getenv("ODDS_PRIMARY_MINUTES_BEFORE", "60"))
CLOSE_MINUTES_BEFORE = int(os.getenv("ODDS_CLOSE_MINUTES_BEFORE", "10"))
MAX_GAMES = int(os.getenv("ODDS_MAX_TEST_GAMES", "0"))


def main() -> None:
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ODDS_API_KEY is required")
    if not EVENT_MAP.exists():
        raise RuntimeError(f"Missing cached event map: {EVENT_MAP}")

    events = pd.read_csv(EVENT_MAP)
    required = {"nhl_game_id", "odds_event_id", "odds_commence_time"}
    missing = required - set(events.columns)
    if missing:
        raise RuntimeError(f"Event map missing columns: {sorted(missing)}")

    events["start_dt"] = pd.to_datetime(events["odds_commence_time"], utc=True, errors="coerce")
    events = events.dropna(subset=["start_dt", "odds_event_id"]).sort_values("start_dt")
    if MAX_GAMES > 0:
        events = events.tail(MAX_GAMES).copy()

    client = OddsAPIClient(key)
    entry_parts: list[pd.DataFrame] = []
    close_parts: list[pd.DataFrame] = []
    errors: list[dict] = []

    for i, row in enumerate(events.itertuples(index=False), start=1):
        entry_at = row.start_dt - timedelta(minutes=PRIMARY_MINUTES_BEFORE)
        close_at = row.start_dt - timedelta(minutes=CLOSE_MINUTES_BEFORE)

        for label, at, bucket in [("entry", entry_at, entry_parts), ("close", close_at, close_parts)]:
            try:
                payload = historical_event_odds(client, str(row.odds_event_id), at)
                part = extract_yes_prices(payload, int(row.nhl_game_id), label, row.start_dt)
                if not part.empty:
                    bucket.append(part)
            except Exception as exc:
                errors.append({
                    "nhl_game_id": int(row.nhl_game_id),
                    "odds_event_id": str(row.odds_event_id),
                    "snapshot": label,
                    "error": repr(exc),
                })

        if i % 10 == 0 or i == len(events):
            print(
                f"FR historical props {i}/{len(events)} entry_parts={len(entry_parts)} "
                f"close_parts={len(close_parts)} calls={client.calls} "
                f"used={client.credits_used_header} remaining={client.credits_remaining}"
            )

    entry = pd.concat(entry_parts, ignore_index=True) if entry_parts else pd.DataFrame()
    close = pd.concat(close_parts, ignore_index=True) if close_parts else pd.DataFrame()
    if entry.empty:
        raise RuntimeError("No French historical anytime-goal entry prices returned")

    entry.to_csv(OUT / "entry_raw_prices.csv", index=False)
    if not close.empty:
        close.to_csv(OUT / "close_raw_prices.csv", index=False)
    if errors:
        (OUT / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    books = (
        entry.groupby(["bookmaker_key", "bookmaker_title"], as_index=False)
        .agg(
            quoted_rows=("odds_player_name", "size"),
            games=("nhl_game_id", "nunique"),
            players=("odds_player_name", "nunique"),
        )
        .sort_values(["games", "quoted_rows"], ascending=False)
    )
    books.to_csv(OUT / "bookmaker_coverage.csv", index=False)

    audit = {
        "region": os.getenv("ODDS_REGIONS", "fr"),
        "events_requested": int(len(events)),
        "entry_games_with_prices": int(entry.nhl_game_id.nunique()),
        "close_games_with_prices": int(close.nhl_game_id.nunique()) if not close.empty else 0,
        "entry_rows": int(len(entry)),
        "close_rows": int(len(close)),
        "api_calls": int(client.calls),
        "credits_used_header": client.credits_used_header,
        "credits_remaining": client.credits_remaining,
        "bookmakers": books.to_dict(orient="records"),
        "errors": int(len(errors)),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
