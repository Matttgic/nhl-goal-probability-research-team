from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import requests

OUT = Path("workspace/v4_source_probe")
OUT.mkdir(parents=True, exist_ok=True)
REPORT = Path("reports/v4_source_probe.md")
REPORT.parent.mkdir(parents=True, exist_ok=True)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "nhl-goal-probability-research-team/0.4"})
GAME_ID = 2025020001


def fetch_text(url: str) -> tuple[int, str, str]:
    r = SESSION.get(url, timeout=30)
    return r.status_code, r.headers.get("content-type", ""), r.text


def probe_csv(label: str, url: str) -> dict:
    status, ctype, text = fetch_text(url)
    out = {"label": label, "url": url, "status": status, "content_type": ctype}
    if status != 200:
        out["error_preview"] = text[:300]
        return out
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        out["parse_error"] = repr(exc)
        out["text_preview"] = text[:500]
        return out
    out["rows"] = int(len(df))
    out["columns"] = list(map(str, df.columns))
    interesting = [c for c in df.columns if any(k in c.lower() for k in ["xgoal", "expected", "situation", "player", "goalie", "ice", "toi", "line", "game", "team", "goal", "shot"])]
    out["interesting_columns"] = list(map(str, interesting))
    for key in ["situation", "position", "team", "playerPositionThatDidEvent", "event"]:
        if key in df.columns:
            vals = df[key].dropna().astype(str).unique().tolist()[:30]
            out[f"unique_{key}"] = vals
    sample_cols = [c for c in interesting[:20] if c in df.columns]
    if sample_cols:
        out["sample"] = df[sample_cols].head(3).fillna("").astype(str).to_dict(orient="records")
    return out


def probe_json(label: str, url: str) -> dict:
    r = SESSION.get(url, timeout=30)
    out = {"label": label, "url": url, "status": r.status_code, "content_type": r.headers.get("content-type", "")}
    if r.status_code != 200:
        out["error_preview"] = r.text[:300]
        return out
    try:
        payload = r.json()
    except Exception as exc:
        out["parse_error"] = repr(exc)
        out["text_preview"] = r.text[:500]
        return out
    if isinstance(payload, dict):
        out["top_keys"] = list(payload.keys())
        data = payload.get("data")
        if isinstance(data, list):
            out["data_rows"] = len(data)
            if data:
                out["data_keys"] = list(data[0].keys())
                out["sample"] = data[:2]
    return out


def main() -> None:
    probes = []
    probes.append(probe_csv(
        "MoneyPuck per-game player data",
        f"https://moneypuck.com/moneypuck/playerData/games/20252026/{GAME_ID}.csv",
    ))
    probes.append(probe_csv(
        "MoneyPuck per-game event/shot data",
        f"https://moneypuck.com/moneypuck/gameData/20252026/{GAME_ID}.csv",
    ))
    probes.append(probe_csv(
        "MoneyPuck season skaters",
        "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/skaters.csv",
    ))
    probes.append(probe_csv(
        "MoneyPuck season lines",
        "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/lines.csv",
    ))
    probes.append(probe_json(
        "NHL shift charts",
        f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={GAME_ID}",
    ))
    probes.append(probe_json(
        "NHL play-by-play",
        f"https://api-web.nhle.com/v1/gamecenter/{GAME_ID}/play-by-play",
    ))

    (OUT / "probe.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")

    lines = [
        "# V4 source probe",
        "",
        "This probe uses only public/download endpoints. MoneyPuck data is used under its published non-commercial data-use terms and should be credited in any output derived from it.",
        "",
    ]
    for p in probes:
        lines += [f"## {p['label']}", "", f"- Status: `{p['status']}`", f"- URL: `{p['url']}`"]
        if "rows" in p:
            lines.append(f"- Rows: `{p['rows']}`")
        if "interesting_columns" in p:
            lines.append(f"- Candidate columns: `{', '.join(p['interesting_columns'][:40])}`")
        if "data_keys" in p:
            lines.append(f"- Shift/event keys: `{', '.join(p['data_keys'][:40])}`")
        if p.get("parse_error") or p.get("error_preview"):
            lines.append(f"- Error: `{p.get('parse_error') or p.get('error_preview')}`")
        lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(probes, indent=2))
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
