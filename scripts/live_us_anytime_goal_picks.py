from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import scripts.real_nhl_player_goal_v2 as v2

SPORT = "icehockey_nhl"
MARKET = "player_goal_scorer_anytime"
ODDS_BASE = "https://api.the-odds-api.com/v4"
EV_MIN = 0.05
MAX_ODDS = 8.0
WINDOW_MIN = int(os.getenv("LIVE_PICK_WINDOW_MINUTES", "180"))
MIN_LEAD = int(os.getenv("LIVE_PICK_MIN_LEAD_MINUTES", "15"))

FEATURES = [
    "home", "position_D", "prior_games", "days_since_prev", "career_goal_rate",
    "target_goal_avg_3", "target_goal_avg_5", "target_goal_avg_10",
    "goals_avg_3", "goals_avg_5", "goals_avg_10", "points_avg_3", "points_avg_5", "points_avg_10",
    "shots_avg_3", "shots_avg_5", "shots_avg_10", "toi_avg_3", "toi_avg_5", "toi_avg_10",
    "shots_trend_5v10", "toi_trend_5v10", "goal_form_5v10",
    "team_rest_days", "team_goals_for_5", "team_goals_for_10", "team_shots_for_5", "team_shots_for_10",
    "team_goals_allowed_5", "team_goals_allowed_10", "team_shots_allowed_5", "team_shots_allowed_10",
    "opp_rest_days", "opp_goals_for_5", "opp_goals_for_10", "opp_shots_for_5", "opp_shots_for_10",
    "opp_goals_allowed_5", "opp_goals_allowed_10", "opp_shots_allowed_5", "opp_shots_allowed_10",
]

TEAM_ABBREV = {
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
    "Carolina Hurricanes": "CAR", "Columbus Blue Jackets": "CBJ", "Calgary Flames": "CGY",
    "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL", "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
    "New Jersey Devils": "NJD", "Nashville Predators": "NSH", "New York Islanders": "NYI",
    "New York Rangers": "NYR", "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT", "Seattle Kraken": "SEA", "San Jose Sharks": "SJS",
    "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Utah Mammoth": "UTA", "Utah Hockey Club": "UTA", "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK", "Winnipeg Jets": "WPG", "Washington Capitals": "WSH",
}


def norm(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(t for t in text.split() if t not in {"jr", "sr", "ii", "iii", "iv"})


def short_key(value: str | None) -> str:
    toks = norm(value).split()
    return f"{toks[0][0]}:{toks[-1]}" if len(toks) >= 2 else ""


def build_history() -> pd.DataFrame:
    games = v2.collect_game_ids(max_games=500)
    rows: list[dict] = []
    for i, (gd, gid) in enumerate(games, start=1):
        try:
            rows.extend(v2.skater_rows(gd, gid, v2.get_json(f"gamecenter/{gid}/boxscore")))
        except Exception as exc:
            print(f"history fetch error game={gid}: {exc!r}")
        if i % 75 == 0 or i == len(games):
            print(f"history {i}/{len(games)} rows={len(rows)}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No NHL history returned")
    df = df.dropna(subset=["player_id", "game_date", "team", "opponent"]).copy()
    df["game_date"] = pd.to_datetime(df.game_date)
    df = v2.add_player_features(df)
    df = v2.add_team_features(df)
    return df.replace([np.inf, -np.inf], np.nan)


def fit_model(history: pd.DataFrame) -> Pipeline:
    train = history[history.prior_games >= 5].copy()
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2500, C=0.7)),
    ])
    model.fit(train[FEATURES], train.target_goal)
    print(f"trained rows={len(train)} games={train.game_id.nunique()} players={train.player_id.nunique()}")
    return model


def roster_map(team: str) -> dict[str, dict]:
    try:
        data = v2.get_json(f"roster/{team}/current")
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for group in ("forwards", "defensemen"):
        for p in data.get(group, []) or []:
            first = (p.get("firstName") or {}).get("default", "") if isinstance(p.get("firstName"), dict) else p.get("firstName", "")
            last = (p.get("lastName") or {}).get("default", "") if isinstance(p.get("lastName"), dict) else p.get("lastName", "")
            name = f"{first} {last}".strip()
            if name:
                out[norm(name)] = {"name": name, "player_id": p.get("id"), "position_D": 1 if group == "defensemen" else 0, "team": team}
    return out


def current_team_state(history: pd.DataFrame, team: str, event_dt: pd.Timestamp) -> dict:
    g = history[history.team == team].sort_values(["game_date", "game_id"])
    if g.empty:
        return {}
    latest = g.iloc[-1]
    days = min(14, max(0, int((event_dt.tz_localize(None) - pd.Timestamp(latest.game_date)).days)))
    return {
        "rest_days": float(days),
        "goals_for_5": float(latest.team_goals_for_5) if pd.notna(latest.team_goals_for_5) else math.nan,
        "goals_for_10": float(latest.team_goals_for_10) if pd.notna(latest.team_goals_for_10) else math.nan,
        "shots_for_5": float(latest.team_shots_for_5) if pd.notna(latest.team_shots_for_5) else math.nan,
        "shots_for_10": float(latest.team_shots_for_10) if pd.notna(latest.team_shots_for_10) else math.nan,
        "goals_allowed_5": float(latest.team_goals_allowed_5) if pd.notna(latest.team_goals_allowed_5) else math.nan,
        "goals_allowed_10": float(latest.team_goals_allowed_10) if pd.notna(latest.team_goals_allowed_10) else math.nan,
        "shots_allowed_5": float(latest.team_shots_allowed_5) if pd.notna(latest.team_shots_allowed_5) else math.nan,
        "shots_allowed_10": float(latest.team_shots_allowed_10) if pd.notna(latest.team_shots_allowed_10) else math.nan,
    }


def player_state(history: pd.DataFrame, player_id, fallback_name: str, event_dt: pd.Timestamp, position_D: int) -> dict | None:
    if player_id is not None:
        g = history[history.player_id == player_id].sort_values(["game_date", "game_id"])
    else:
        sk = short_key(fallback_name)
        keys = history.player_name.map(short_key)
        ids = history.loc[keys == sk, "player_id"].dropna().unique()
        g = history[history.player_id == ids[0]].sort_values(["game_date", "game_id"]) if len(ids) == 1 else history.iloc[0:0]
    if len(g) < 5:
        return None
    last_date = pd.Timestamp(g.game_date.iloc[-1])
    days = min(14, max(0, int((event_dt.tz_localize(None) - last_date).days)))
    target = (g.goals >= 1).astype(float)
    s: dict[str, float] = {
        "position_D": float(position_D), "prior_games": float(len(g)), "days_since_prev": float(days),
        "career_goal_rate": float(target.mean()),
    }
    for col, series in [("target_goal", target), ("goals", g.goals), ("points", g.points), ("shots", g.shots), ("toi", g.toi)]:
        for w in (3, 5, 10):
            vals = pd.to_numeric(series, errors="coerce").dropna().tail(w)
            s[f"{col}_avg_{w}"] = float(vals.mean()) if not vals.empty else math.nan
    s["shots_trend_5v10"] = s["shots_avg_5"] - s["shots_avg_10"]
    s["toi_trend_5v10"] = s["toi_avg_5"] - s["toi_avg_10"]
    s["goal_form_5v10"] = s["target_goal_avg_5"] - s["target_goal_avg_10"]
    return s


def extract_best_prices(payload: dict) -> list[dict]:
    by_player: dict[str, list[dict]] = {}
    for book in payload.get("bookmakers", []) or []:
        for market in book.get("markets", []) or []:
            if market.get("key") != MARKET:
                continue
            for o in market.get("outcomes", []) or []:
                player = str(o.get("description") or "").strip()
                side = str(o.get("name") or "").strip().lower()
                if not player or side != "yes":
                    continue
                try:
                    price = float(o.get("price"))
                except (TypeError, ValueError):
                    continue
                if price <= 1 or not np.isfinite(price):
                    continue
                by_player.setdefault(norm(player), []).append({"player": player, "odds": price, "book": str(book.get("title") or book.get("key") or "")})
    out = []
    for values in by_player.values():
        out.append(max(values, key=lambda x: x["odds"]))
    return out


def telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        raise RuntimeError("Telegram secrets are missing")
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": text, "disable_web_page_preview": True}, timeout=30)
    data = r.json() if r.content else {}
    if not r.ok or not data.get("ok"):
        raise RuntimeError(f"Telegram send failed: {data.get('description', r.status_code)}")


def main() -> None:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is missing")
    now = datetime.now(timezone.utc)
    session = requests.Session()
    evr = session.get(f"{ODDS_BASE}/sports/{SPORT}/events", params={"apiKey": api_key, "dateFormat": "iso"}, timeout=30)
    evr.raise_for_status()
    events = []
    for e in evr.json():
        start = pd.to_datetime(e.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(start):
            continue
        lead = (start.to_pydatetime() - now).total_seconds() / 60
        if MIN_LEAD <= lead <= WINDOW_MIN:
            events.append((e, start, lead))
    print(f"candidate events={len(events)}")
    if not events:
        print("No NHL games inside live window")
        return

    market_events: list[tuple[dict, pd.Timestamp, float, list[dict]]] = []
    for e, start, lead in events:
        r = session.get(
            f"{ODDS_BASE}/sports/{SPORT}/events/{e['id']}/odds",
            params={"apiKey": api_key, "regions": "us", "markets": MARKET, "oddsFormat": "decimal", "dateFormat": "iso"},
            timeout=30,
        )
        if r.status_code == 422:
            continue
        r.raise_for_status()
        prices = extract_best_prices(r.json())
        if prices:
            market_events.append((e, start, lead, prices))
            print(f"market {e.get('away_team')} @ {e.get('home_team')} lead={lead:.0f}m players={len(prices)}")
    if not market_events:
        telegram("🏒 NHL Goal Model\nAucun marché buteur US disponible pour les matchs dans les 3 prochaines heures.")
        print("No US anytime-goal markets available")
        return

    history = build_history()
    model = fit_model(history)
    roster_cache: dict[str, dict[str, dict]] = {}
    picks = []
    scored = 0

    for e, start, lead, prices in market_events:
        home = TEAM_ABBREV.get(str(e.get("home_team")))
        away = TEAM_ABBREV.get(str(e.get("away_team")))
        if not home or not away:
            print(f"unmapped teams: {e.get('away_team')} @ {e.get('home_team')}")
            continue
        for t in (home, away):
            roster_cache.setdefault(t, roster_map(t))
        roster_all = {**roster_cache[home], **roster_cache[away]}
        home_state = current_team_state(history, home, start)
        away_state = current_team_state(history, away, start)
        if not home_state or not away_state:
            continue

        for px in prices:
            nk = norm(px["player"])
            roster = roster_all.get(nk)
            if roster is None:
                sk = short_key(px["player"])
                candidates = [r for name, r in roster_all.items() if short_key(name) == sk]
                roster = candidates[0] if len(candidates) == 1 else None
            if roster is None:
                continue
            team = roster["team"]
            opp = away if team == home else home
            ps = player_state(history, roster.get("player_id"), px["player"], start, int(roster["position_D"]))
            if ps is None:
                continue
            own = home_state if team == home else away_state
            other = away_state if team == home else home_state
            row = {"home": 1.0 if team == home else 0.0, **ps}
            row.update({
                "team_rest_days": own["rest_days"],
                "team_goals_for_5": own["goals_for_5"], "team_goals_for_10": own["goals_for_10"],
                "team_shots_for_5": own["shots_for_5"], "team_shots_for_10": own["shots_for_10"],
                "team_goals_allowed_5": own["goals_allowed_5"], "team_goals_allowed_10": own["goals_allowed_10"],
                "team_shots_allowed_5": own["shots_allowed_5"], "team_shots_allowed_10": own["shots_allowed_10"],
                "opp_rest_days": other["rest_days"],
                "opp_goals_for_5": other["goals_for_5"], "opp_goals_for_10": other["goals_for_10"],
                "opp_shots_for_5": other["shots_for_5"], "opp_shots_for_10": other["shots_for_10"],
                "opp_goals_allowed_5": other["goals_allowed_5"], "opp_goals_allowed_10": other["goals_allowed_10"],
                "opp_shots_allowed_5": other["shots_allowed_5"], "opp_shots_allowed_10": other["shots_allowed_10"],
            })
            p = float(model.predict_proba(pd.DataFrame([row])[FEATURES])[:, 1][0])
            ev = p * float(px["odds"]) - 1.0
            min_fr = 1.05 / p if p > 0 else math.inf
            scored += 1
            if ev >= EV_MIN and float(px["odds"]) <= MAX_ODDS:
                picks.append({
                    "player": roster.get("name") or px["player"], "team": team, "opp": opp,
                    "home_team": home, "away_team": away, "start": start,
                    "lead": lead, "p": p, "odds": float(px["odds"]), "book": px["book"],
                    "ev": ev, "min_fr": min_fr,
                })

    picks.sort(key=lambda x: (x["start"], -x["ev"]))
    print(json.dumps({"events_with_market": len(market_events), "players_scored": scored, "qualified_picks": len(picks)}, indent=2))

    if not picks:
        telegram("🏒 NHL Goal Model\nAucun pick buteur ne passe actuellement la règle figée EV ≥ 5 % / cote US ≤ 8.00.")
        return

    lines = ["🏒 NHL — PICKS BUTEURS", "Règle: EV ≥ 5 % | cote US ≤ 8.00", ""]
    for i, p in enumerate(picks[:20], start=1):
        lines += [
            f"{i}. {p['player']} ({p['team']}) — {p['away_team']} @ {p['home_team']}",
            f"Proba modèle: {p['p']:.1%} | Réf. US: {p['odds']:.2f} ({p['book']})",
            f"EV: {p['ev']:+.1%} | Cote FR minimale: {p['min_fr']:.2f}",
            "",
        ]
    if len(picks) > 20:
        lines.append(f"+ {len(picks)-20} autres picks qualifiés non affichés")
    lines.append("La cote française doit être ≥ à la cote minimale indiquée.")
    telegram("\n".join(lines))


if __name__ == "__main__":
    main()
