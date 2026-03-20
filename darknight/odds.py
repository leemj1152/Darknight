from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class OddsPrediction:
    home_probability: float
    draw_probability: float | None
    away_probability: float
    bookmaker_margin: float


def calculate_implied_probabilities(
    home_odds: float,
    away_odds: float,
    draw_odds: float | None = None,
) -> OddsPrediction:
    if home_odds <= 1 or away_odds <= 1 or (draw_odds is not None and draw_odds <= 1):
        raise ValueError("Odds must be greater than 1.")

    implied = {"home": 1 / home_odds, "away": 1 / away_odds}
    if draw_odds is not None:
        implied["draw"] = 1 / draw_odds

    total = sum(implied.values())
    return OddsPrediction(
        home_probability=round(implied["home"] / total, 4),
        draw_probability=round(implied["draw"] / total, 4) if "draw" in implied else None,
        away_probability=round(implied["away"] / total, 4),
        bookmaker_margin=round(total - 1, 4),
    )


def calculate_implied_probabilities_from_row(
    frame: pd.DataFrame,
    gm_ts: str,
    match_seq: int,
) -> OddsPrediction:
    target = frame[(frame["gm_ts"].astype(str) == str(gm_ts)) & (frame["match_seq"] == match_seq)]
    if target.empty:
        raise ValueError("No match was found for the given gm_ts and match_seq.")

    row = target.iloc[0]
    if pd.isna(row.get("home_odds")) or pd.isna(row.get("away_odds")):
        raise ValueError("This row does not have usable odds values.")

    draw_odds = row.get("draw_odds")
    return calculate_implied_probabilities(
        home_odds=float(row["home_odds"]),
        away_odds=float(row["away_odds"]),
        draw_odds=float(draw_odds) if pd.notna(draw_odds) else None,
    )
