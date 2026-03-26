from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from .odds import calculate_implied_probabilities
from .predictor import FormPredictor, HybridPredictor


SIDE_LABELS = {"home": "HOME", "away": "AWAY", "skip": "SKIP"}


@dataclass(slots=True)
class StrategyDecision:
    agent: str
    side: str
    selection_probability: float | None
    market_probability: float | None
    edge: float | None


def choose_market_side(row: pd.Series) -> StrategyDecision:
    home_probability = float(row["odds_home_probability"])
    away_probability = _row_away_probability(row)
    if home_probability >= away_probability:
        return StrategyDecision("odds", "home", home_probability, home_probability, 0.0)
    return StrategyDecision("odds", "away", away_probability, away_probability, 0.0)


def choose_value_side(
    agent: str,
    home_probability: float,
    row: pd.Series,
    *,
    edge_threshold: float,
) -> StrategyDecision:
    away_probability = max(0.0, min(1.0, 1.0 - home_probability))
    market_home = float(row["odds_home_probability"])
    market_away = _row_away_probability(row)
    home_edge = home_probability - market_home
    away_edge = away_probability - market_away

    best_side = "skip"
    best_probability: float | None = None
    best_market: float | None = None
    best_edge: float | None = None

    if home_edge >= edge_threshold and home_edge >= away_edge:
        best_side = "home"
        best_probability = home_probability
        best_market = market_home
        best_edge = home_edge
    elif away_edge >= edge_threshold:
        best_side = "away"
        best_probability = away_probability
        best_market = market_away
        best_edge = away_edge

    return StrategyDecision(agent, best_side, best_probability, best_market, best_edge)


def add_strategy_columns(
    report_frame: pd.DataFrame,
    *,
    edge_threshold: float,
    league_scores: pd.DataFrame | None = None,
    top_pick_count: int = 5,
) -> pd.DataFrame:
    if report_frame.empty:
        result = report_frame.copy()
        for column in (
            "odds_pick",
            "form_pick",
            "hybrid_pick",
            "form_edge",
            "hybrid_edge",
            "form_away_probability",
            "hybrid_away_probability",
            "expected_value_form_home",
            "expected_value_form_away",
            "expected_value_hybrid_home",
            "expected_value_hybrid_away",
            "expected_value_form",
            "expected_value_hybrid",
            "recommended_side",
            "recommended_model",
            "recommended_probability",
            "recommended_market_probability",
            "recommended_expected_value",
            "value_score",
            "bet_grade",
            "league_hybrid_advantage",
            "league_trust_level",
            "top_pick_rank",
        ):
            result[column] = pd.Series(dtype="object")
        return result

    enriched = report_frame.copy()
    league_adjustments = build_league_adjustments(league_scores)
    odds_picks: list[str] = []
    form_picks: list[str] = []
    hybrid_picks: list[str] = []
    form_edges: list[float | None] = []
    hybrid_edges: list[float | None] = []
    form_away_probabilities: list[float] = []
    hybrid_away_probabilities: list[float] = []
    form_home_evs: list[float | None] = []
    form_away_evs: list[float | None] = []
    hybrid_home_evs: list[float | None] = []
    hybrid_away_evs: list[float | None] = []
    form_expected_values: list[float | None] = []
    hybrid_expected_values: list[float | None] = []
    recommended_sides: list[str] = []
    recommended_models: list[str] = []
    recommended_probabilities: list[float | None] = []
    recommended_market_probabilities: list[float | None] = []
    recommended_expected_values: list[float | None] = []
    value_scores: list[float] = []
    bet_grades: list[str] = []
    league_advantages: list[float] = []
    league_levels: list[str] = []
    rank_scores: list[float] = []

    for _, row in enriched.iterrows():
        league_name = str(row.get("league", "")) if pd.notna(row.get("league")) else ""
        league_adjustment = league_adjustments.get(league_name, {"advantage": 0.0, "level": "neutral", "boost": 0})
        form_home_probability = float(row["form_home_probability"])
        hybrid_home_probability = float(row["hybrid_home_probability"])
        form_away_probability = away_probability_from_home(form_home_probability, row)
        hybrid_away_probability = away_probability_from_home(hybrid_home_probability, row)
        odds_decision = choose_market_side(row)
        form_decision = choose_value_side(
            "form",
            form_home_probability,
            row,
            edge_threshold=edge_threshold,
        )
        hybrid_decision = choose_value_side(
            "hybrid",
            hybrid_home_probability,
            row,
            edge_threshold=edge_threshold,
        )
        form_home_decision = StrategyDecision("form", "home", form_home_probability, float(row["odds_home_probability"]), form_home_probability - float(row["odds_home_probability"]))
        form_away_decision = StrategyDecision("form", "away", form_away_probability, _row_away_probability(row), form_away_probability - _row_away_probability(row))
        hybrid_home_decision = StrategyDecision("hybrid", "home", hybrid_home_probability, float(row["odds_home_probability"]), hybrid_home_probability - float(row["odds_home_probability"]))
        hybrid_away_decision = StrategyDecision("hybrid", "away", hybrid_away_probability, _row_away_probability(row), hybrid_away_probability - _row_away_probability(row))

        odds_picks.append(SIDE_LABELS[odds_decision.side])
        form_picks.append(SIDE_LABELS[form_decision.side])
        hybrid_picks.append(SIDE_LABELS[hybrid_decision.side])
        form_edges.append(None if form_decision.edge is None else round(form_decision.edge, 4))
        hybrid_edges.append(None if hybrid_decision.edge is None else round(hybrid_decision.edge, 4))
        form_away_probabilities.append(round(form_away_probability, 4))
        hybrid_away_probabilities.append(round(hybrid_away_probability, 4))
        form_home_ev = selection_expected_value(row, form_home_decision)
        form_away_ev = selection_expected_value(row, form_away_decision)
        hybrid_home_ev = selection_expected_value(row, hybrid_home_decision)
        hybrid_away_ev = selection_expected_value(row, hybrid_away_decision)
        form_home_evs.append(None if form_home_ev is None else round(form_home_ev, 4))
        form_away_evs.append(None if form_away_ev is None else round(form_away_ev, 4))
        hybrid_home_evs.append(None if hybrid_home_ev is None else round(hybrid_home_ev, 4))
        hybrid_away_evs.append(None if hybrid_away_ev is None else round(hybrid_away_ev, 4))
        form_ev = selection_expected_value(row, form_decision)
        hybrid_ev = selection_expected_value(row, hybrid_decision)
        form_expected_values.append(None if form_ev is None else round(form_ev, 4))
        hybrid_expected_values.append(None if hybrid_ev is None else round(hybrid_ev, 4))
        recommendation = choose_recommended_pick(
            row=row,
            form_home_probability=form_home_probability,
            form_away_probability=form_away_probability,
            hybrid_home_probability=hybrid_home_probability,
            hybrid_away_probability=hybrid_away_probability,
            league_boost=int(league_adjustment["boost"]),
        )
        recommended_sides.append(recommendation["side"])
        recommended_models.append(recommendation["model"])
        recommended_probabilities.append(recommendation["probability"])
        recommended_market_probabilities.append(recommendation["market_probability"])
        recommended_expected_values.append(recommendation["expected_value"])
        value_scores.append(round(float(recommendation["score"]), 4))
        bet_grades.append(
            classify_bet_grade(
                form_decision,
                hybrid_decision,
                form_confidence=abs(form_home_probability - 0.5) * 2,
                hybrid_confidence=abs(hybrid_home_probability - 0.5) * 2,
                league_boost=int(league_adjustment["boost"]),
                recommendation_score=float(recommendation["score"]),
            )
        )
        league_advantages.append(round(float(league_adjustment["advantage"]), 4))
        league_levels.append(str(league_adjustment["level"]))
        rank_scores.append(float(recommendation["score"]))

    enriched["odds_pick"] = odds_picks
    enriched["form_pick"] = form_picks
    enriched["hybrid_pick"] = hybrid_picks
    enriched["form_edge"] = form_edges
    enriched["hybrid_edge"] = hybrid_edges
    enriched["form_away_probability"] = form_away_probabilities
    enriched["hybrid_away_probability"] = hybrid_away_probabilities
    enriched["expected_value_form_home"] = form_home_evs
    enriched["expected_value_form_away"] = form_away_evs
    enriched["expected_value_hybrid_home"] = hybrid_home_evs
    enriched["expected_value_hybrid_away"] = hybrid_away_evs
    enriched["expected_value_form"] = form_expected_values
    enriched["expected_value_hybrid"] = hybrid_expected_values
    enriched["recommended_side"] = recommended_sides
    enriched["recommended_model"] = recommended_models
    enriched["recommended_probability"] = recommended_probabilities
    enriched["recommended_market_probability"] = recommended_market_probabilities
    enriched["recommended_expected_value"] = recommended_expected_values
    enriched["value_score"] = value_scores
    enriched["bet_grade"] = bet_grades
    enriched["league_hybrid_advantage"] = league_advantages
    enriched["league_trust_level"] = league_levels
    enriched["top_pick_rank"] = assign_top_pick_ranks(enriched, rank_scores, top_pick_count)
    return enriched


def simulate_betting_agents(
    frame: pd.DataFrame,
    *,
    recent_games: int,
    lookback_days: int,
    edge_threshold: float,
    stake: float,
    sport: str | None = None,
    league: str | None = None,
    form_predictor: FormPredictor | None = None,
    hybrid_predictor: HybridPredictor | None = None,
) -> dict[str, pd.DataFrame]:
    scoped = frame.copy()
    scoped["played_at"] = pd.to_datetime(scoped["played_at"])
    if sport:
        scoped = scoped[scoped["sport"] == sport]
    if league:
        scoped = scoped[scoped["league"] == league]
    scoped = scoped.sort_values("played_at").reset_index(drop=True)
    scoped = scoped[pd.notna(scoped.get("home_odds")) & pd.notna(scoped.get("away_odds"))].reset_index(drop=True)
    if scoped.empty:
        raise ValueError("No rows with usable odds are available for simulation.")

    last_played = scoped["played_at"].max()
    start_time = last_played - timedelta(days=lookback_days)
    evaluation = scoped[scoped["played_at"] >= start_time].reset_index(drop=True)
    if evaluation.empty:
        raise ValueError("No rows were found inside the requested simulation window.")

    if form_predictor is None or hybrid_predictor is None:
        training = scoped[scoped["played_at"] < evaluation["played_at"].min()].reset_index(drop=True)
        if len(training) < 50:
            raise ValueError("Not enough history before the simulation window to fit the models.")

        form_predictor = FormPredictor(recent_games=recent_games)
        form_predictor.fit(training)
        hybrid_predictor = HybridPredictor(recent_games=recent_games)
        hybrid_predictor.fit(training)

    rows: list[dict[str, object]] = []
    full_history = scoped.reset_index(drop=True)
    full_total = len(evaluation)
    for index, (_, match) in enumerate(evaluation.iterrows(), start=1):
        if index == 1 or index % 100 == 0 or index == full_total:
            print(f"[simulate] scoring matches {index}/{full_total}")

        prior = full_history[full_history["played_at"] < match["played_at"]]
        if prior.empty:
            continue

        implied = calculate_implied_probabilities(
            home_odds=float(match["home_odds"]),
            away_odds=float(match["away_odds"]),
            draw_odds=float(match["draw_odds"]) if pd.notna(match.get("draw_odds")) else None,
        )
        row = pd.Series(
            {
                "odds_home_probability": implied.home_probability,
                "draw_odds": match.get("draw_odds"),
                "odds_draw_probability": implied.draw_probability,
                "odds_away_probability": implied.away_probability,
                "home_odds": match["home_odds"],
                "away_odds": match["away_odds"],
            }
        )

        form_prediction = form_predictor.predict(
            prior,
            str(match["home_team"]),
            str(match["away_team"]),
            sport=str(match["sport"]) if pd.notna(match.get("sport")) else None,
            league=str(match["league"]) if pd.notna(match.get("league")) else None,
        )
        hybrid_prediction = hybrid_predictor.predict(
            prior,
            str(match["home_team"]),
            str(match["away_team"]),
            home_odds=float(match["home_odds"]),
            draw_odds=float(match["draw_odds"]) if pd.notna(match.get("draw_odds")) else None,
            away_odds=float(match["away_odds"]),
            sport=str(match["sport"]) if pd.notna(match.get("sport")) else None,
            league=str(match["league"]) if pd.notna(match.get("league")) else None,
        )

        decisions = [
            choose_market_side(row),
            choose_value_side(
                "form",
                form_prediction.home_win_probability,
                row,
                edge_threshold=edge_threshold,
            ),
            choose_value_side(
                "hybrid",
                hybrid_prediction.hybrid_home_probability,
                row,
                edge_threshold=edge_threshold,
            ),
        ]

        for decision in decisions:
            rows.append(
                build_simulation_row(
                    match=match,
                    decision=decision,
                    stake=stake,
                )
            )

    bets = pd.DataFrame(rows)
    if bets.empty:
        raise ValueError("Simulation did not produce any bet rows.")

    settled = bets[bets["side"] != "skip"].copy()
    summary = (
        settled.groupby("agent", dropna=False)
        .agg(
            bets=("profit", "size"),
            wins=("won", "sum"),
            stake_total=("stake", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )
    if not summary.empty:
        summary["roi"] = summary["profit"] / summary["stake_total"]
        summary["win_rate"] = summary["wins"] / summary["bets"]
    skipped = (
        bets.groupby("agent", dropna=False)
        .agg(skipped=("side", lambda values: int((values == "skip").sum())))
        .reset_index()
    )
    summary = summary.merge(skipped, on="agent", how="outer").fillna({"bets": 0, "wins": 0, "stake_total": 0.0, "profit": 0.0, "roi": 0.0, "win_rate": 0.0, "skipped": 0})

    daily = (
        settled.assign(play_date=pd.to_datetime(settled["played_at"]).dt.date)
        .groupby(["play_date", "agent"], dropna=False)
        .agg(daily_profit=("profit", "sum"), bets=("profit", "size"))
        .reset_index()
        .sort_values(["agent", "play_date"])
    )
    if not daily.empty:
        daily["cumulative_profit"] = daily.groupby("agent")["daily_profit"].cumsum()
        daily["cumulative_bets"] = daily.groupby("agent")["bets"].cumsum()
        daily["cumulative_stake"] = daily["cumulative_bets"] * stake
        daily["cumulative_roi"] = daily["cumulative_profit"] / daily["cumulative_stake"]

    return {
        "summary": summary.sort_values("roi", ascending=False).reset_index(drop=True),
        "bets": bets.sort_values(["played_at", "agent"]).reset_index(drop=True),
        "daily": daily.reset_index(drop=True),
    }


def build_simulation_row(
    *,
    match: pd.Series,
    decision: StrategyDecision,
    stake: float,
) -> dict[str, object]:
    actual_home_win = bool(match["home_score"] > match["away_score"])
    actual_away_win = bool(match["away_score"] > match["home_score"])
    won = False
    profit = 0.0
    odds_used: float | None = None

    if decision.side == "home":
        odds_used = float(match["home_odds"])
        won = actual_home_win
        profit = stake * (odds_used - 1.0) if won else -stake
    elif decision.side == "away":
        odds_used = float(match["away_odds"])
        won = actual_away_win
        profit = stake * (odds_used - 1.0) if won else -stake

    return {
        "played_at": match["played_at"],
        "sport": match.get("sport", ""),
        "league": match.get("league", ""),
        "home_team": match["home_team"],
        "away_team": match["away_team"],
        "home_score": match["home_score"],
        "away_score": match["away_score"],
        "agent": decision.agent,
        "side": decision.side,
        "selection_probability": decision.selection_probability,
        "market_probability": decision.market_probability,
        "edge": decision.edge,
        "odds_used": odds_used,
        "stake": stake if decision.side != "skip" else 0.0,
        "won": int(won),
        "profit": round(profit, 4),
        "gm_ts": match.get("gm_ts", ""),
        "match_seq": match.get("match_seq", 0),
    }


def _row_away_probability(row: pd.Series) -> float:
    if pd.notna(row.get("odds_away_probability")):
        return float(row["odds_away_probability"])
    draw_probability = float(row["odds_draw_probability"]) if pd.notna(row.get("odds_draw_probability")) else 0.0
    return max(0.0, 1.0 - float(row["odds_home_probability"]) - draw_probability)


def selection_expected_value(row: pd.Series, decision: StrategyDecision) -> float | None:
    if decision.side == "skip" or decision.selection_probability is None:
        return None

    if decision.side == "home":
        odds = row.get("home_odds")
    else:
        odds = row.get("away_odds")

    if pd.isna(odds):
        return None

    return float(odds) * float(decision.selection_probability) - 1.0


def classify_bet_grade(
    form_decision: StrategyDecision,
    hybrid_decision: StrategyDecision,
    form_confidence: float,
    hybrid_confidence: float,
    league_boost: int = 0,
    recommendation_score: float | None = None,
) -> str:
    score = recommendation_score if recommendation_score is not None else score_pick_candidate(form_decision, hybrid_decision, form_confidence, hybrid_confidence, league_boost)
    if score >= 4:
        return "A"
    if score >= 2.5:
        return "B"
    if score >= 1:
        return "C"
    return "PASS"


def score_pick_candidate(
    form_decision: StrategyDecision,
    hybrid_decision: StrategyDecision,
    form_confidence: float,
    hybrid_confidence: float,
    league_boost: int,
) -> float:
    score = float(league_boost)
    if hybrid_decision.side != "skip":
        score += 1.5 + hybrid_confidence * 3
    if form_decision.side != "skip":
        score += 1.0 + form_confidence * 2
    if hybrid_decision.side != "skip" and form_decision.side == hybrid_decision.side:
        score += 1.0
    return score


def build_league_adjustments(league_scores: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if league_scores is None or league_scores.empty:
        return {}

    pivot = (
        league_scores.pivot_table(index="league", columns="model", values=["accuracy", "rows"], aggfunc="first")
        .sort_index()
    )
    adjustments: dict[str, dict[str, object]] = {}
    for league_name in pivot.index:
        hybrid_accuracy = _pivot_value(pivot, league_name, "accuracy", "hybrid")
        odds_accuracy = _pivot_value(pivot, league_name, "accuracy", "odds")
        hybrid_rows = _pivot_value(pivot, league_name, "rows", "hybrid")
        advantage = hybrid_accuracy - odds_accuracy
        boost = 0
        level = "neutral"
        if hybrid_rows >= 40 and advantage >= 0.03:
            boost = 1
            level = "strong"
        elif hybrid_rows >= 25 and advantage >= 0.0:
            boost = 0
            level = "neutral"
        elif hybrid_rows < 25 or advantage <= -0.03:
            boost = -1
            level = "weak"
        adjustments[str(league_name)] = {
            "advantage": advantage,
            "level": level,
            "boost": boost,
        }
    return adjustments


def _pivot_value(pivot: pd.DataFrame, league_name: object, metric: str, model: str) -> float:
    try:
        value = pivot.loc[league_name, (metric, model)]
    except KeyError:
        return 0.0
    return 0.0 if pd.isna(value) else float(value)


def assign_top_pick_ranks(report_frame: pd.DataFrame, scores: list[float], top_pick_count: int) -> list[object]:
    ranking_frame = report_frame.copy()
    ranking_frame["_score"] = scores
    ranking_frame["_row_id"] = range(len(ranking_frame))
    ranking_frame = ranking_frame[ranking_frame["bet_grade"] != "PASS"].copy()
    if ranking_frame.empty:
        return [None] * len(report_frame)

    ranking_frame = ranking_frame.sort_values(
        ["_score", "expected_value_hybrid", "expected_value_form", "played_at"],
        ascending=[False, False, False, True],
    ).head(top_pick_count)
    rank_map = {int(row["_row_id"]): rank for rank, (_, row) in enumerate(ranking_frame.iterrows(), start=1)}
    return [rank_map.get(index) for index in range(len(report_frame))]


def away_probability_from_home(home_probability: float, row: pd.Series) -> float:
    draw_probability = float(row.get("odds_draw_probability")) if pd.notna(row.get("odds_draw_probability")) else 0.0
    return max(0.0, 1.0 - home_probability - draw_probability)


def choose_recommended_pick(
    *,
    row: pd.Series,
    form_home_probability: float,
    form_away_probability: float,
    hybrid_home_probability: float,
    hybrid_away_probability: float,
    league_boost: int,
) -> dict[str, object]:
    candidates = [
        build_confidence_candidate("HYBRID", "home", hybrid_home_probability, float(row["odds_home_probability"])),
        build_confidence_candidate("HYBRID", "away", hybrid_away_probability, _row_away_probability(row)),
        build_confidence_candidate("FORM", "home", form_home_probability, float(row["odds_home_probability"])),
        build_confidence_candidate("FORM", "away", form_away_probability, _row_away_probability(row)),
    ]
    for candidate in candidates:
        score = candidate["confidence"] * 10 + league_boost
        if candidate["model"] == "HYBRID":
            score += 1.0
        if candidate["side"] == ("home" if hybrid_home_probability >= hybrid_away_probability else "away") and candidate["model"] == "HYBRID":
            score += 0.5
        candidate["score"] = round(score, 4)
    best = max(candidates, key=lambda item: item["score"])
    if best["confidence"] < 0.06:
        return {
            "side": "SKIP",
            "model": "none",
            "probability": None,
            "market_probability": None,
            "expected_value": None,
            "score": 0.0,
        }
    return {
        "side": SIDE_LABELS[str(best["side"])],
        "model": str(best["model"]),
        "probability": round(float(best["probability"]), 4),
        "market_probability": round(float(best["market_probability"]), 4),
        "expected_value": None,
        "score": float(best["score"]),
    }


def build_value_candidate(model: str, side: str, probability: float, market_probability: float, odds: float) -> dict[str, float | str]:
    return {
        "model": model,
        "side": side,
        "probability": probability,
        "market_probability": market_probability,
        "edge": probability - market_probability,
        "expected_value": odds * probability - 1.0,
        "odds": odds,
    }


def build_confidence_candidate(model: str, side: str, probability: float, market_probability: float) -> dict[str, float | str]:
    confidence = abs(probability - 0.5) * 2
    return {
        "model": model,
        "side": side,
        "probability": probability,
        "market_probability": market_probability,
        "confidence": confidence,
    }
