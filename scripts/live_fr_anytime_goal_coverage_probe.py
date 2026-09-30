from __future__ import annotations

import json
import os
from collections import defaultdict

import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"
MARKET = "player_goal_scorer_anytime"


def main() -> None:
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ODDS_API_KEY is required")

    s = requests.Session()
    s.headers.update({"User-Agent": "nhl-goal-probability-research-team/0.9"})

    r = s.get(f"{BASE}/sports/{SPORT}/events", params={"apiKey": key, "dateFormat": "iso"}, timeout=30)
    r.raise_for_status()
    events = r.json()
    if not events:
        print(json.dumps({"upcoming_events": 0, "message": "No upcoming NHL events returned"}, indent=2))
        return

    audited = []
    book_games = defaultdict(int)
    book_quotes = defaultdict(int)
    for event in events[:6]:
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
        if q.status_code == 422:
            audited.append({"event_id": event_id, "market_available": False, "status": 422})
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
        audited.append({
            "event_id": event_id,
            "commence_time": event.get("commence_time"),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
            "market_available": bool(books),
            "bookmakers": books,
        })

    summary = {
        "upcoming_events": len(events),
        "events_audited": len(audited),
        "events_with_fr_anytime_goal": sum(1 for x in audited if x.get("market_available")),
        "bookmaker_coverage": [
            {"bookmaker_key": k, "games": book_games[k], "outcomes": book_quotes[k]}
            for k in sorted(book_games)
        ],
        "events": audited,
        "note": "Coverage audit only; prices are intentionally not printed or committed.",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
