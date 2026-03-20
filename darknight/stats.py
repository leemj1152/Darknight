from __future__ import annotations

import pandas as pd


def add_outcome_columns(frame: pd.DataFrame) -> pd.DataFrame:
    results = frame.copy()
    results["home_win"] = (results["home_score"] > results["away_score"]).astype(int)
    results["away_win"] = (results["away_score"] > results["home_score"]).astype(int)
    results["draw"] = (results["home_score"] == results["away_score"]).astype(int)
    return results


def team_summary(frame: pd.DataFrame) -> pd.DataFrame:
    games = add_outcome_columns(frame)

    home = pd.DataFrame(
        {
            "team": games["home_team"],
            "games": 1,
            "wins": games["home_win"],
            "draws": games["draw"],
            "losses": games["away_win"],
            "goals_for": games["home_score"],
            "goals_against": games["away_score"],
        }
    )

    away = pd.DataFrame(
        {
            "team": games["away_team"],
            "games": 1,
            "wins": games["away_win"],
            "draws": games["draw"],
            "losses": games["home_win"],
            "goals_for": games["away_score"],
            "goals_against": games["home_score"],
        }
    )

    summary = (
        pd.concat([home, away], ignore_index=True)
        .groupby("team", as_index=False)
        .sum(numeric_only=True)
    )
    summary["win_rate"] = (summary["wins"] / summary["games"]).round(3)
    summary["goal_diff"] = summary["goals_for"] - summary["goals_against"]
    return summary.sort_values(["win_rate", "goal_diff"], ascending=False).reset_index(drop=True)
