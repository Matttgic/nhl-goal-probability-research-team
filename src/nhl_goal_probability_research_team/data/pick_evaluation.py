"""Separate probability diagnostics from timestamp-validated price evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _timestamps(values: pd.Series, field: str) -> pd.Series:
    for value in values:
        if pd.isna(value) or pd.Timestamp(value).tzinfo is None:
            raise ValueError(f"{field} requires timezone-qualified timestamps")
    return pd.to_datetime(values, utc=True, errors="raise")


def evaluate_picks(picks: pd.DataFrame) -> pd.DataFrame:
    """Summarize locked picks by cohort and market with one unit per priced bet.

    Prices require decimal_odds, odds_at and a nonempty odds_source. Missing
    prices are excluded from economic metrics, never filled or synthesized.
    Date validity cannot independently authenticate the source's underlying data.
    """
    required = {"pick_id", "cohort", "market", "probability", "predicted_at", "starts_at", "result"}
    if required - set(picks):
        raise ValueError(f"Missing columns: {sorted(required - set(picks))}")
    rows = picks.copy()
    for col in ("pick_id", "cohort", "market"):
        if rows[col].isna().any() or rows[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {col}")
    if rows.duplicated(["cohort", "pick_id"]).any():
        raise ValueError("Duplicate pick IDs within a cohort")
    for col in ("predicted_at", "starts_at"):
        rows[col] = _timestamps(rows[col], col)
        if rows[col].isna().any():
            raise ValueError(f"Missing {col}")
    if (rows["predicted_at"] >= rows["starts_at"]).any():
        raise ValueError("Predictions must be locked before puck drop")
    rows["probability"] = pd.to_numeric(rows["probability"], errors="raise")
    if not np.isfinite(rows["probability"]).all() or not rows["probability"].between(0, 1).all():
        raise ValueError("Probabilities must be finite and between zero and one")
    if not rows["result"].isin(["WIN", "LOSS", "PUSH", "VOID", "PENDING"]).all():
        raise ValueError("Unknown result; use WIN/LOSS/PUSH/VOID/PENDING")
    if "decimal_odds" not in rows:
        rows["decimal_odds"] = np.nan
    rows["decimal_odds"] = pd.to_numeric(rows["decimal_odds"], errors="raise")
    priced = rows["decimal_odds"].notna()
    if priced.any():
        if not np.isfinite(rows.loc[priced, "decimal_odds"]).all() or (rows.loc[priced, "decimal_odds"] <= 1).any():
            raise ValueError("Decimal odds must be finite and greater than one")
        if not {"odds_at", "odds_source"} <= set(rows):
            raise ValueError("Priced picks require quote timestamps and source attribution")
        quotes = _timestamps(rows.loc[priced, "odds_at"], "odds_at")
        if quotes.isna().any() or (quotes > rows.loc[priced, "predicted_at"]).any():
            raise ValueError("Price quotes must exist at or before prediction lock")
        sources = rows.loc[priced, "odds_source"]
        if sources.isna().any() or sources.astype(str).str.strip().eq("").any():
            raise ValueError("Priced picks require an odds source")
    output = []
    for (cohort, market), group in rows.groupby(["cohort", "market"], sort=True):
        decided = group[group["result"].isin(["WIN", "LOSS"])]
        y = decided["result"].eq("WIN").astype(float)
        p = decided["probability"]
        quoted = decided[decided["decimal_odds"].notna()]
        profit = np.where(quoted["result"].eq("WIN"), quoted["decimal_odds"]-1, -1).sum()
        clip = p.clip(1e-15, 1-1e-15)
        output.append({
            "cohort": cohort, "market": market, "picks": len(group),
            "settled": len(decided), "wins": int(y.sum()), "losses": int(len(y)-y.sum()),
            "pending": int(group["result"].eq("PENDING").sum()),
            "voids": int(group["result"].eq("VOID").sum()),
            "pushes": int(group["result"].eq("PUSH").sum()),
            "hit_rate": float(y.mean()) if len(y) else np.nan,
            "expected_wins": float(p.sum()) if len(y) else np.nan,
            "mean_probability": float(p.mean()) if len(y) else np.nan,
            "brier": float(((p-y)**2).mean()) if len(y) else np.nan,
            "log_loss": float(-(y*np.log(clip)+(1-y)*np.log1p(-clip)).mean()) if len(y) else np.nan,
            "priced_settled": len(quoted),
            "profit_units": float(profit) if len(quoted) else np.nan,
            "roi": float(profit/len(quoted)) if len(quoted) else np.nan,
        })
    return pd.DataFrame(output)
