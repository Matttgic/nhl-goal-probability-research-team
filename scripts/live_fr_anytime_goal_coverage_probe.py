from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"
MARKET = "player_goal_scorer_anytime"


def set_output(name: str, value: str) -> None:
    path = os.getenv("GITHUB_OUTPUT", "").strip()
    if path:
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def main() -> None:
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ODDS_API_KEY is required")

    s = requests.Session()
    s.headers.update({"User-Agent": "nhl-goal-probability-research-team/1.0"})

    r = s.get(f"{BASE}/sports/{SPORT}/events", params={"apiKey": key, "dateFormat": "iso"}, timeout=30)
    r.raise_for_status()
    events = r.json()
    if not events:
        summary = {"upcoming_events": 0, "events_with_fr_anytime_goal": 0, "bookmaker_coverage": []}
        print(json.dumps(summary, indent=2))
        set_output("available", "false")
        set_output("events_with_market", "0")
        return

    audited = 0
    with_market = []
    book_games = defaultdict(int)
    book_quotes = defaultdict(int)
    remaining = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")

    for event in events:
        event_id = event.get("id")
        q = s.get(
            f"{BASE}/sports/{SPORT}/events/{event_id}/odds",
            params={
                "apiKey": key,
                "regions": "fr",
                "markets": MARKET,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
            timeout=30,
        )
        audited += 1
        remaining = q.headers.get("x-requests-remaining", remaining)
        used = q.headers.get("x-requests-used", used)
        if q.status_code == 422:
            continue
        q.raise_for_status()
        payload = q.json()
        books = []
        for book in payload.get("bookmakers", []) or []:
            count = 0
            for market in book.get("markets", []) or []:
                if market.get("key") == MARKET:
                    count += len(market.get("outcomes", []) or [])
            if count:
                key_b = str(book.get("key") or "")
                title = str(book.get("title") or key_b)
                books.append({"key": key_b, "title": title, "outcomes": count})
                book_games[key_b] += 1
                book_quotes[key_b] += count
        if books:
            with_market.append({
                "event_id": event_id,
                "commence_time": event.get("commence_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "bookmakers": books,
            })

    summary = {
        "upcoming_events": len(events),
        "events_audited": audited,
        "events_with_fr_anytime_goal": len(with_market),
        "bookmaker_coverage": [
            {"bookmaker_key": k, "games": book_games[k], "outcomes": book_quotes[k]}
            for k in sorted(book_games)
        ],
        "events_with_market": with_market,
        "quota_used_header": used,
        "quota_remaining_header": remaining,
        "note": "Coverage audit only; prices are intentionally not printed or committed.",
    }
    print(json.dumps(summary, indent=2))
    set_output("available", "true" if with_market else "false")
    set_output("events_with_market", str(len(with_market)))
    set_output("bookmakers", ",".join(sorted(book_games)))


if __name__ == "__main__":
    main()
