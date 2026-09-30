from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import psycopg
import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"
MARKET = "player_goal_scorer_anytime"
WINDOWS = [(60, 9), (10, 9)]
ALLOWED_BOOKS = {"betclic_fr", "netbet_fr", "pmu_fr", "unibet_fr", "winamax_fr"}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nhl_anytime_goal_fr_snapshots (
    id BIGSERIAL PRIMARY KEY,
    event_id TEXT NOT NULL,
    commence_time TIMESTAMPTZ NOT NULL,
    snapshot_at TIMESTAMPTZ NOT NULL,
    target_minutes_before INTEGER NOT NULL,
    actual_minutes_before DOUBLE PRECISION NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    bookmaker_key TEXT NOT NULL,
    bookmaker_title TEXT,
    market_last_update TIMESTAMPTZ,
    player_name TEXT NOT NULL,
    side TEXT NOT NULL,
    decimal_odds DOUBLE PRECISION NOT NULL,
    api_source TEXT NOT NULL DEFAULT 'the-odds-api',
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(event_id, target_minutes_before, bookmaker_key, player_name, side)
);
CREATE INDEX IF NOT EXISTS idx_nhl_anytime_fr_event ON nhl_anytime_goal_fr_snapshots(event_id);
CREATE INDEX IF NOT EXISTS idx_nhl_anytime_fr_commence ON nhl_anytime_goal_fr_snapshots(commence_time);
CREATE INDEX IF NOT EXISTS idx_nhl_anytime_fr_player ON nhl_anytime_goal_fr_snapshots(player_name);
"""


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def choose_window(minutes_to_start: float) -> int | None:
    candidates = [(abs(minutes_to_start - target), target) for target, tol in WINDOWS if abs(minutes_to_start - target) <= tol]
    return min(candidates)[1] if candidates else None


def main() -> None:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    db_url = os.getenv("NHL_ODDS_DATABASE_URL", "").strip()
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is required")
    if not db_url:
        raise RuntimeError("NHL_ODDS_DATABASE_URL is required")

    session = requests.Session()
    session.headers.update({"User-Agent": "nhl-goal-probability-research-team/1.0"})
    events_resp = session.get(
        f"{BASE}/sports/{SPORT}/events",
        params={"apiKey": api_key, "dateFormat": "iso"},
        timeout=30,
    )
    events_resp.raise_for_status()
    events = events_resp.json()
    now = now_utc()

    targets = []
    for event in events:
        start = datetime.fromisoformat(str(event["commence_time"]).replace("Z", "+00:00"))
        minutes = (start - now).total_seconds() / 60.0
        window = choose_window(minutes)
        if window is not None:
            targets.append((event, start, minutes, window))

    inserted = 0
    queried = 0
    coverage = {}
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            for event, start, minutes, target_window in targets:
                r = session.get(
                    f"{BASE}/sports/{SPORT}/events/{event['id']}/odds",
                    params={
                        "apiKey": api_key,
                        "regions": "fr",
                        "markets": MARKET,
                        "oddsFormat": "decimal",
                        "dateFormat": "iso",
                    },
                    timeout=30,
                )
                queried += 1
                if r.status_code == 422:
                    continue
                r.raise_for_status()
                payload = r.json()
                snapshot_at = now_utc()
                for book in payload.get("bookmakers", []) or []:
                    book_key = str(book.get("key") or "")
                    if book_key not in ALLOWED_BOOKS:
                        continue
                    for market in book.get("markets", []) or []:
                        if market.get("key") != MARKET:
                            continue
                        coverage[book_key] = coverage.get(book_key, 0) + 1
                        last_update = market.get("last_update")
                        for outcome in market.get("outcomes", []) or []:
                            player = str(outcome.get("description") or "").strip()
                            side = str(outcome.get("name") or "").strip().lower()
                            try:
                                price = float(outcome.get("price"))
                            except (TypeError, ValueError):
                                continue
                            if not player or side not in {"yes", "no"} or price <= 1:
                                continue
                            cur.execute(
                                """
                                INSERT INTO nhl_anytime_goal_fr_snapshots (
                                    event_id, commence_time, snapshot_at, target_minutes_before,
                                    actual_minutes_before, home_team, away_team, bookmaker_key,
                                    bookmaker_title, market_last_update, player_name, side, decimal_odds
                                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                                ON CONFLICT (event_id, target_minutes_before, bookmaker_key, player_name, side)
                                DO UPDATE SET
                                    snapshot_at = EXCLUDED.snapshot_at,
                                    actual_minutes_before = EXCLUDED.actual_minutes_before,
                                    bookmaker_title = EXCLUDED.bookmaker_title,
                                    market_last_update = EXCLUDED.market_last_update,
                                    decimal_odds = EXCLUDED.decimal_odds
                                """,
                                (
                                    str(event["id"]), start, snapshot_at, target_window, minutes,
                                    str(event.get("home_team") or ""), str(event.get("away_team") or ""),
                                    book_key, str(book.get("title") or book_key), last_update,
                                    player, side, price,
                                ),
                            )
                            inserted += 1
            conn.commit()

    print(json.dumps({
        "events_seen": len(events),
        "events_in_capture_window": len(targets),
        "event_prop_queries": queried,
        "rows_upserted": inserted,
        "fr_bookmaker_markets_seen": coverage,
        "windows": [{"target_minutes_before": t, "tolerance_minutes": tol} for t, tol in WINDOWS],
        "raw_prices_logged": False,
    }, indent=2))


if __name__ == "__main__":
    main()
