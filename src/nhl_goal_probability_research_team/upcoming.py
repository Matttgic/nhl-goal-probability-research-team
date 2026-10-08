"""Experimental, odds-free pregame forecasts from public NHL boxscores."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data.shot_context import build_shot_context_features

BASE = "https://api-web.nhle.com/v1"
CORE = ["is_home", "position_D", "shot_context_history_games", "shot_context_l10_sog_avg",
        "shot_context_l10_goal_hit_rate", "shot_context_l10_toi_avg_seconds", "shot_context_l10_sog_per60"]
EXTRA = ["shot_context_l5_sog_avg", "shot_context_l5_goal_hit_rate", "shot_context_l5_toi_avg_seconds",
         "shot_context_venue_l10_games", "shot_context_venue_l10_sog_avg", "shot_context_venue_l10_goal_hit_rate",
         "shot_context_matchup_l10_games", "shot_context_matchup_l10_sog_avg", "shot_context_matchup_l10_goal_hit_rate",
         "shot_context_l10_pp_observed_games", "shot_context_l10_pp_usage_share", "shot_context_l10_pp_toi_avg_seconds"]


def utc_timestamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("A timezone-qualified timestamp is required")
    return stamp.tz_convert("UTC")


class NHLClient:
    def __init__(self, cache: Path):
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)

    def get(self, path: str, immutable: bool = False) -> dict:
        file = self.cache / (path.replace("/", "_") + ".json")
        if immutable and file.exists():
            return json.loads(file.read_text())
        error = None
        for attempt in range(3):
            try:
                request = Request(f"{BASE}/{path}", headers={"User-Agent": "nhl-research-upcoming/1.0"})
                with urlopen(request, timeout=20) as response:
                    payload = json.load(response)
                if immutable and payload.get("gameState") in {"OFF", "FINAL"}:
                    file.write_text(json.dumps(payload))
                return payload
            except Exception as exc:
                error = type(exc).__name__
                if attempt < 2:
                    time.sleep(attempt + 1)
        raise RuntimeError(f"NHL public endpoint failed: {path} ({error})")


def schedule_games(payload: dict) -> list[dict]:
    out = []
    for day in payload.get("gameWeek", []):
        for game in day.get("games", []):
            if int(game.get("gameType", 0)) == 2:
                out.append({**game, "game_date": day["date"]})
    return out


def split_slate(games: list[dict], now: pd.Timestamp) -> tuple[list[dict], list[dict]]:
    """Separate future forecasts from today's elapsed start times in Paris."""
    now = utc_timestamp(now)
    today = now.tz_convert("Europe/Paris").date()
    upcoming, started = [], []
    for game in games:
        if not game.get("startTimeUTC"):
            continue
        start = utc_timestamp(game["startTimeUTC"])
        if start <= now and start.tz_convert("Europe/Paris").date() == today:
            started.append(game)
        elif now < start <= now + pd.Timedelta(hours=48) and game.get("gameState") not in {"OFF", "FINAL", "LIVE", "CRIT"}:
            upcoming.append(game)
    return (sorted(upcoming, key=lambda g: g["startTimeUTC"]),
            sorted(started, key=lambda g: g["startTimeUTC"]))


def collect_games(client: NHLClient, now: pd.Timestamp, max_games: int = 600) -> tuple[list[dict], list[dict]]:
    """Collect completed regular-season history and the upcoming seven-day schedule."""
    found = {}
    # Weekly NHL schedules contain a week of games. Bound the training window to 14 months.
    cursor = now.date() - timedelta(days=420)
    while cursor <= now.date():
        for game in schedule_games(client.get(f"schedule/{cursor.isoformat()}")):
            found[int(game["id"])] = game
        cursor += timedelta(days=7)
    for game in schedule_games(client.get(f"schedule/{now.date().isoformat()}")):
        found[int(game["id"])] = game
    history = [g for g in found.values() if g.get("gameState") in {"OFF", "FINAL"}
               and pd.Timestamp(g["game_date"]).date() < now.date()]
    history.sort(key=lambda g: (g["game_date"], g["id"]))
    upcoming, _ = split_slate(list(found.values()), now)
    return history[-max_games:], upcoming


def _seconds(value) -> float:
    if value is None or str(value).strip() == "":
        return np.nan
    if ":" in str(value):
        minute, second = str(value).split(":", 1)
        return float(minute)*60 + float(second)
    raise ValueError("NHL time-on-ice must be in MM:SS format")


def boxscore_rows(game: dict, box: dict) -> list[dict]:
    if box.get("gameState") not in {"OFF", "FINAL"}:
        raise ValueError("Incomplete game is not eligible for training")
    rows = []
    for side, other, home in (("homeTeam", "awayTeam", 1), ("awayTeam", "homeTeam", 0)):
        team = box.get(side, {}).get("abbrev")
        opponent = box.get(other, {}).get("abbrev")
        if not team or not opponent:
            raise ValueError("Missing team identity")
        players = box.get("playerByGameStats", {}).get(side, {})
        for kind in ("forwards", "defense"):
            for player in players.get(kind, []):
                shots = player.get("sog", player.get("shots"))
                goals = player.get("goals")
                if shots is None or goals is None or player.get("playerId") is None:
                    raise ValueError("Missing observed player shots/goals/ID")
                toi = _seconds(player.get("toi"))
                if toi == 0:
                    continue  # nonparticipants must not become negative goal labels
                row = {"game_id": int(game["id"]), "game_date": game["game_date"],
                       "player_id": int(player["playerId"]), "goals": goals, "shots_on_goal": shots,
                       "team_abbrev": team, "opponent_abbrev": opponent, "is_home": home,
                       "position_D": int(kind == "defense"), "toi_seconds": toi}
                # The public boxscore often lacks PP TOI. Keep that absence explicit.
                if player.get("powerPlayToi") is not None:
                    row["pp_toi_seconds"] = _seconds(player["powerPlayToi"])
                rows.append(row)
    if not rows:
        raise ValueError("Boxscore contains no skaters")
    return rows


def load_history(client: NHLClient, games: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    rows, failed = [], []
    def fetch(game):
        return boxscore_rows(game, client.get(f"gamecenter/{game['id']}/boxscore", immutable=True))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, game): game for game in games}
        for future in as_completed(futures):
            game = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                failed.append({"game_id": game["id"], "error": str(exc)})
    if not rows:
        raise ValueError("No genuine historical NHL boxscores could be loaded")
    if failed and len(failed)/max(len(games), 1) > .05:
        raise ValueError("More than 5% of historical games failed; forecasts blocked")
    return pd.DataFrame(rows), failed


def feature_table(history: pd.DataFrame, candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    targets = history if candidates is None else candidates
    features = build_shot_context_features(history, targets)
    identity = ["game_id", "player_id", "game_date", "is_home", "position_D"]
    return targets[identity].merge(features, on=["game_id", "player_id"], validate="one_to_one")


def chronological_parts(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    days = pd.to_datetime(table["game_date"], utc=True).dt.normalize()
    dates = sorted(days.unique())
    if len(dates) < 30:
        raise ValueError("At least 30 distinct historical game dates are required")
    first, second = int(len(dates)*.6), int(len(dates)*.8)
    return (table.loc[days < dates[first]].copy(),
            table.loc[(days >= dates[first]) & (days < dates[second])].copy(),
            table.loc[days >= dates[second]].copy())


class CalibratedScorer:
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.model = Pipeline([("fill", SimpleImputer(strategy="median", add_indicator=True)),
                               ("scale", StandardScaler()),
                               ("model", LogisticRegression(C=.5, max_iter=1500, random_state=7))])
        self.calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=7)

    @staticmethod
    def logits(p):
        p = np.clip(p, 1e-6, 1-1e-6)
        return np.log(p/(1-p)).reshape(-1, 1)

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame):
        for frame in (train, calibration):
            if len(frame) < 100 or frame.label_goal.nunique() < 2:
                raise ValueError("Training/calibration samples require 100 rows and both outcomes")
        self.model.fit(train[self.columns], train.label_goal)
        raw = self.model.predict_proba(calibration[self.columns])[:, 1]
        self.calibrator.fit(self.logits(raw), calibration.label_goal)
        return self

    def predict(self, frame: pd.DataFrame):
        raw = self.model.predict_proba(frame[self.columns])[:, 1]
        return self.calibrator.predict_proba(self.logits(raw))[:, 1]


def train_forecaster(history: pd.DataFrame) -> tuple[CalibratedScorer, dict]:
    table = feature_table(history)
    labels = history[["game_id", "player_id", "goals"]].copy()
    labels["label_goal"] = labels.goals.gt(0).astype(int)
    table = table.merge(labels[["game_id", "player_id", "label_goal"]], on=["game_id", "player_id"], validate="one_to_one")
    table = table[table.shot_context_history_games >= 5].copy()
    train, calibration, test = chronological_parts(table)
    if len(test) < 100 or test.label_goal.nunique() < 2:
        raise ValueError("Test set is too small or contains only one outcome")
    all_columns = CORE + EXTRA
    present = [c for c in all_columns if c in train and train[c].notna().any()]
    models = {"reference": CalibratedScorer([c for c in CORE if c in present]).fit(train, calibration),
              "enriched": CalibratedScorer(present).fit(train, calibration)}
    metrics = {}
    y = test.label_goal.to_numpy()
    for name, model in models.items():
        p = model.predict(test)
        metrics[name] = {"brier": float(brier_score_loss(y, p)),
                         "log_loss": float(log_loss(y, p, labels=[0, 1])),
                         "mean_probability": float(p.mean()), "observed_goal_rate": float(y.mean())}
    metrics.update({"history_rows": len(history), "history_games": int(history.game_id.nunique()),
                    "training_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
                    "training_through": str(pd.to_datetime(train.game_date).max().date()),
                    "calibration_through": str(pd.to_datetime(calibration.game_date).max().date()),
                    "test_from": str(pd.to_datetime(test.game_date).min().date()),
                    "test_through": str(pd.to_datetime(test.game_date).max().date()),
                    "active_features": present, "unavailable_features": [c for c in all_columns if c not in present],
                    "status": "experimental", "odds_used": False})
    # The enriched model is pre-specified. We do not pick a winner using the test set.
    return models["enriched"], metrics


def roster_candidates(client: NHLClient, games: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    cache, rows, errors = {}, [], []
    for game in games:
        home, away = game["homeTeam"]["abbrev"], game["awayTeam"]["abbrev"]
        for team, opponent, is_home in ((home, away, 1), (away, home, 0)):
            if team not in cache:
                try:
                    cache[team] = client.get(f"roster/{team}/current")
                except Exception:
                    cache[team] = {}
                    errors.append(f"Effectif indisponible : {team}")
            for kind in ("forwards", "defensemen"):
                for player in cache[team].get(kind, []):
                    def text(value):
                        return value.get("default", "") if isinstance(value, dict) else str(value or "")
                    rows.append({"game_id": game["id"], "game_date": game["game_date"],
                                 "player_id": player["id"], "player_name": f"{text(player.get('firstName'))} {text(player.get('lastName'))}".strip(),
                                 "team_abbrev": team, "opponent_abbrev": opponent, "is_home": is_home,
                                 "position_D": int(kind == "defensemen"), "starts_at": game["startTimeUTC"],
                                 "availability": "effectif actuel ; participation non confirmée"})
    return pd.DataFrame(rows), errors


def predict_upcoming(history: pd.DataFrame, candidates: pd.DataFrame, model: CalibratedScorer, now: pd.Timestamp) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["game_id", "player_id", "player_name", "starts_at", "probability_goal", "status"])
    starts = pd.to_datetime(candidates.starts_at, utc=True)
    if (starts <= utc_timestamp(now)).any():
        raise ValueError("Started games cannot enter the upcoming forecast")
    features = feature_table(history, candidates)
    out = candidates.merge(features.drop(columns=["game_date", "is_home", "position_D"]),
                           on=["game_id", "player_id"], validate="one_to_one")
    for col in model.columns:
        if col not in out:
            out[col] = np.nan
    ready = out.shot_context_history_games.ge(5)
    out["probability_goal"] = np.nan
    if ready.any():
        out.loc[ready, "probability_goal"] = model.predict(out.loc[ready])
    out["fair_decimal_odds"] = 1 / out.probability_goal.replace(0, np.nan)
    out["predicted_at"] = utc_timestamp(now).isoformat()
    out["status"] = np.where(ready, "experimental", "historique insuffisant")
    return out.sort_values(["starts_at", "probability_goal"], ascending=[True, False])


def _safe(value):
    return str(value).replace("|", "/").replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")


def render_report(predictions: pd.DataFrame, games: list[dict], metrics: dict, now: pd.Timestamp, warnings: list[str], started_games: list[dict] | None = None) -> str:
    paris = utc_timestamp(now).tz_convert("Europe/Paris").strftime("%d/%m/%Y %H:%M")
    lines = ["# Prochains matchs NHL — modèle expérimental", "", f"Actualisé le **{paris} (Paris)**. Fenêtre : prochaines 48 heures.", "",
             "Probabilités conditionnelles à la participation du joueur. Effectifs actuels ; blessures, composition, PP1 et gardiens non confirmés.",
             "Aucune cote de bookmaker utilisée. La cote théorique ne constitue pas une recommandation de pari.", ""]
    if metrics:
        ref, ext = metrics["reference"], metrics["enriched"]
        lines += ["## Validation chronologique", "",
                  f"{metrics['history_games']} matchs historiques. Apprentissage : {metrics['training_rows']} lignes, jusqu’au {metrics['training_through']} ; calibration séparée : {metrics['calibration_rows']} lignes ; test : {metrics['test_rows']} lignes du {metrics['test_from']} au {metrics['test_through']}.", "",
                  "| Modèle | Brier, plus bas préférable | Log loss, plus basse préférable |",
                  "|---|---:|---:|", f"| Référence simple | {ref['brier']:.5f} | {ref['log_loss']:.5f} |",
                  f"| Enrichi | {ext['brier']:.5f} | {ext['log_loss']:.5f} |", "",
                  f"Fréquence de but observée : {ext['observed_goal_rate']:.1%} ; probabilité moyenne enrichie : {ext['mean_probability']:.1%}.",
                  "Ce test ne prouve pas la rentabilité. Les joueurs-matchs d’un même match sont corrélés.", ""]
        if ext["brier"] >= ref["brier"]:
            lines += ["**L’enrichissement n’améliore pas le Brier sur ce test. Les prédictions restent une expérience de recherche.**", ""]
        if any("_pp_" in col for col in metrics["unavailable_features"]):
            lines += ["Données de temps de jeu en power play indisponibles : ces variables ne sont pas utilisées.", ""]
    for warning in warnings:
        lines += [f"- {_safe(warning)}"]
    if warnings:
        lines.append("")
    if started_games:
        lines += ["## Matchs du jour dont l’heure de début est passée", "",
                  "Ces matchs restent visibles. Ce rapport ne disposait pas de pronostics verrouillés avant leur début : aucune probabilité d’avant-match n’est reconstruite après coup.", "",
                  "| Match | Début prévu (Paris) | Statut NHL | Score constaté |",
                  "|---|---|---|---|"]
        for game in started_games:
            kickoff = utc_timestamp(game["startTimeUTC"]).tz_convert("Europe/Paris").strftime("%d/%m %H:%M")
            home, away = game["homeTeam"], game["awayTeam"]
            state = game.get("gameState")
            status = "En cours" if state in {"LIVE", "CRIT"} else "Terminé" if state in {"OFF", "FINAL"} else "Heure prévue dépassée ; statut à confirmer"
            score = f"{away['score']}–{home['score']}" if "score" in away and "score" in home else "Indisponible"
            lines.append(f"| {_safe(away['abbrev'])} chez {_safe(home['abbrev'])} | {kickoff} | {status} | {_safe(score)} |")
        lines.append("")
    if not games:
        lines += ["**Aucun match de saison régulière trouvé dans les prochaines 48 heures.**", ""]
    for game in games:
        kickoff = utc_timestamp(game["startTimeUTC"]).tz_convert("Europe/Paris").strftime("%d/%m %H:%M")
        home, away = game["homeTeam"]["abbrev"], game["awayTeam"]["abbrev"]
        lines += [f"## {_safe(away)} chez {_safe(home)} — {kickoff} (Paris)", "",
                  "| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |",
                  "|---|---|---:|---:|---:|---:|"]
        group = predictions[predictions.game_id.eq(game["id"])] if not predictions.empty else pd.DataFrame()
        valid = group[group.probability_goal.notna()] if not group.empty else group
        for row in valid.head(10).to_dict("records"):
            lines.append(f"| {_safe(row['player_name'])} | {_safe(row['team_abbrev'])} | {row['probability_goal']:.1%} | {row['fair_decimal_odds']:.2f} | {row['shot_context_l10_sog_avg']:.2f} | {int(row['shot_context_history_games'])} matchs |")
        if valid.empty:
            lines += ["", "Aucune prédiction fiable disponible pour cet effectif."]
        if len(group) > len(valid):
            lines += ["", f"{len(group)-len(valid)} joueurs sans historique suffisant : aucune probabilité inventée."]
        lines.append("")
    lines += ["Tous les joueurs calculés : [CSV](upcoming_predictions.csv). Paramètres et validation : [JSON](upcoming_validation.json).", "",
              "Sources : endpoints publics NHL, matchs terminés et effectifs au moment du calcul. Le statut expérimental s’applique à toutes les lignes.", ""]
    return "\n".join(lines)


def run(output: Path, cache: Path, now: pd.Timestamp, max_games: int = 600):
    output.mkdir(parents=True, exist_ok=True)
    client = NHLClient(cache)
    historical, games = collect_games(client, now, max_games)
    history, failed = load_history(client, historical)
    model, metrics = train_forecaster(history)
    candidates, warnings = roster_candidates(client, games)
    if failed:
        warnings.append(f"{len(failed)} matchs historiques non récupérés, sur {len(historical)}.")
    # Calculation can cross a puck-drop or midnight. Refresh the Paris-day slate
    # and use the actual completion time rather than backdating predictions.
    refresh_now = pd.Timestamp.now(tz="UTC")
    day_start = refresh_now.tz_convert("Europe/Paris").normalize().tz_convert("UTC").date()
    slate = schedule_games(client.get(f"schedule/{day_start.isoformat()}"))
    report_now = pd.Timestamp.now(tz="UTC")
    games, started_games = split_slate(slate, report_now)
    if not candidates.empty:
        candidates = candidates[candidates.game_id.isin([g["id"] for g in games])].copy()
    predictions = predict_upcoming(history, candidates, model, report_now)
    report_now = pd.Timestamp.now(tz="UTC")
    games, started_games = split_slate(slate, report_now)
    predictions = predictions[predictions.game_id.isin([g["id"] for g in games])].copy()
    predictions["predicted_at"] = report_now.isoformat()
    metrics["generated_at"] = report_now.isoformat()
    metrics["failed_history_games"] = failed
    metrics["forecast_games"] = len(games)
    metrics["elapsed_start_games_today"] = len(started_games)
    predictions.to_csv(output / "upcoming_predictions.csv", index=False)
    (output / "upcoming_validation.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "upcoming_predictions.md").write_text(render_report(predictions, games, metrics, report_now, warnings, started_games), encoding="utf-8")
    print(f"Report generated: {len(games)} games, {len(predictions)} players; no odds API used")
