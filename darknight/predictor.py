from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .odds import calculate_implied_probabilities


FORM_FEATURE_COLUMNS = [
    "handicap_line",
    "home_win_rate_all",
    "away_win_rate_all",
    "home_recent_win_rate",
    "away_recent_win_rate",
    "home_avg_margin",
    "away_avg_margin",
    "home_home_win_rate",
    "away_away_win_rate",
    "head_to_head_home_win_rate",
]

HYBRID_FEATURE_COLUMNS = FORM_FEATURE_COLUMNS + [
    "odds_home_probability",
    "odds_away_probability",
    "odds_draw_probability",
    "bookmaker_margin",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return default
    return float(numeric)


def _sanitize_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    sanitized = frame.copy()
    for column in columns:
        if column not in sanitized.columns:
            sanitized[column] = 0.0
        sanitized[column] = pd.to_numeric(sanitized[column], errors="coerce").fillna(0.0)
    return sanitized[columns]


@dataclass(slots=True)
class FormPrediction:
    home_team: str
    away_team: str
    home_win_probability: float
    features: dict[str, float]


@dataclass(slots=True)
class HybridPrediction:
    home_team: str
    away_team: str
    odds_home_probability: float
    form_home_probability: float
    hybrid_home_probability: float
    bookmaker_margin: float


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    results = frame.copy()
    results["played_at"] = pd.to_datetime(results["played_at"])
    if "sport" not in results.columns:
        results["sport"] = ""
    if "league" not in results.columns:
        results["league"] = ""
    if "handicap_line" not in results.columns:
        results["handicap_line"] = 0.0
    results["handicap_line"] = pd.to_numeric(results["handicap_line"], errors="coerce").fillna(0.0)
    return results.sort_values("played_at").reset_index(drop=True)


def adjusted_scores(game: pd.Series) -> tuple[float, float]:
    handicap_line = _safe_float(game.get("handicap_line", 0.0), 0.0)
    return float(game["home_score"]) + handicap_line, float(game["away_score"])


def home_covers(game: pd.Series) -> int:
    adjusted_home, adjusted_away = adjusted_scores(game)
    return int(adjusted_home > adjusted_away)


def _team_games(frame: pd.DataFrame, team: str) -> pd.DataFrame:
    return frame[(frame["home_team"] == team) | (frame["away_team"] == team)].copy()


def _apply_context(frame: pd.DataFrame, sport: str | None, league: str | None) -> pd.DataFrame:
    scoped = frame
    if sport:
        scoped = scoped[scoped["sport"] == sport]
    if league:
        scoped = scoped[scoped["league"] == league]
    return scoped


def _compute_team_features(
    frame: pd.DataFrame,
    team: str,
    *,
    use_home_only: bool = False,
    use_away_only: bool = False,
    recent_games: int = 5,
) -> dict[str, float]:
    games = _team_games(frame, team)
    if use_home_only:
        games = games[games["home_team"] == team]
    if use_away_only:
        games = games[games["away_team"] == team]

    if games.empty:
        return {
            "games": 0.0,
            "win_rate": 0.5,
            "recent_win_rate": 0.5,
            "avg_margin": 0.0,
        }

    margins: list[float] = []
    wins: list[int] = []
    for _, game in games.iterrows():
        is_home = game["home_team"] == team
        adjusted_home, adjusted_away = adjusted_scores(game)
        goals_for = adjusted_home if is_home else adjusted_away
        goals_against = adjusted_away if is_home else adjusted_home
        margins.append(float(goals_for - goals_against))
        wins.append(int(goals_for > goals_against))

    recent = wins[-recent_games:] if wins else []
    return {
        "games": float(len(games)),
        "win_rate": float(sum(wins) / len(wins)),
        "recent_win_rate": float(sum(recent) / len(recent)) if recent else 0.5,
        "avg_margin": float(sum(margins) / len(margins)),
    }


def _compute_head_to_head(frame: pd.DataFrame, home_team: str, away_team: str) -> float:
    games = frame[
        ((frame["home_team"] == home_team) & (frame["away_team"] == away_team))
        | ((frame["home_team"] == away_team) & (frame["away_team"] == home_team))
    ]
    if games.empty:
        return 0.5

    home_side_wins = 0
    for _, game in games.iterrows():
        adjusted_home, adjusted_away = adjusted_scores(game)
        if game["home_team"] == home_team and adjusted_home > adjusted_away:
            home_side_wins += 1
        elif game["away_team"] == home_team and adjusted_away > adjusted_home:
            home_side_wins += 1

    return float(home_side_wins / len(games))


def build_form_features(
    frame: pd.DataFrame,
    home_team: str,
    away_team: str,
    *,
    handicap_line: float = 0.0,
    sport: str | None = None,
    league: str | None = None,
    recent_games: int = 5,
) -> pd.DataFrame:
    scoped = _apply_context(_normalize_frame(frame), sport, league)
    home_all = _compute_team_features(scoped, home_team, recent_games=recent_games)
    away_all = _compute_team_features(scoped, away_team, recent_games=recent_games)
    home_home = _compute_team_features(
        scoped,
        home_team,
        use_home_only=True,
        recent_games=recent_games,
    )
    away_away = _compute_team_features(
        scoped,
        away_team,
        use_away_only=True,
        recent_games=recent_games,
    )
    h2h = _compute_head_to_head(scoped, home_team, away_team)

    return pd.DataFrame(
        [
            {
                "handicap_line": _safe_float(handicap_line, 0.0),
                "home_win_rate_all": home_all["win_rate"],
                "away_win_rate_all": away_all["win_rate"],
                "home_recent_win_rate": home_all["recent_win_rate"],
                "away_recent_win_rate": away_all["recent_win_rate"],
                "home_avg_margin": home_all["avg_margin"],
                "away_avg_margin": away_all["avg_margin"],
                "home_home_win_rate": home_home["win_rate"],
                "away_away_win_rate": away_away["win_rate"],
                "head_to_head_home_win_rate": h2h,
            }
        ]
    )


class FormPredictor:
    def __init__(self, recent_games: int = 5) -> None:
        self.recent_games = recent_games
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> None:
        dataset = self.get_training_dataset(frame)
        if dataset.empty:
            raise ValueError("Not enough training rows for the form model.")
        self.model.fit(_sanitize_feature_frame(dataset, FORM_FEATURE_COLUMNS), dataset["home_win"])

    def predict(
        self,
        frame: pd.DataFrame,
        home_team: str,
        away_team: str,
        *,
        handicap_line: float = 0.0,
        sport: str | None = None,
        league: str | None = None,
    ) -> FormPrediction:
        features = build_form_features(
            frame,
            home_team,
            away_team,
            handicap_line=handicap_line,
            sport=sport,
            league=league,
            recent_games=self.recent_games,
        )
        probability = float(self.model.predict_proba(_sanitize_feature_frame(features, FORM_FEATURE_COLUMNS))[0][1])
        return FormPrediction(
            home_team=home_team,
            away_team=away_team,
            home_win_probability=round(probability, 4),
            features={key: round(float(value), 4) for key, value in features.iloc[0].to_dict().items()},
        )

    def _build_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, float | int]] = []
        for index in range(1, len(frame)):
            current_game = frame.iloc[index]
            prior_games = frame.iloc[:index]
            scoped_prior = _apply_context(prior_games, current_game.get("sport"), current_game.get("league"))
            features = build_form_features(
                scoped_prior,
                current_game["home_team"],
                current_game["away_team"],
                handicap_line=_safe_float(current_game.get("handicap_line", 0.0), 0.0),
                sport=current_game.get("sport"),
                league=current_game.get("league"),
                recent_games=self.recent_games,
            )
            rows.append(
                {
                    **features.iloc[0].to_dict(),
                    "home_win": home_covers(current_game),
                }
            )
        return pd.DataFrame(rows)

    def get_training_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._build_dataset(_normalize_frame(frame))

    def feature_importance(self) -> pd.DataFrame:
        classifier = self.model.named_steps["classifier"]
        coefficients = classifier.coef_[0]
        total = sum(abs(value) for value in coefficients) or 1.0
        return pd.DataFrame(
            {
                "feature": FORM_FEATURE_COLUMNS,
                "coefficient": coefficients,
                "importance_share": [abs(value) / total for value in coefficients],
            }
        ).sort_values("importance_share", ascending=False).reset_index(drop=True)


class HybridPredictor:
    def __init__(self, recent_games: int = 5) -> None:
        self.recent_games = recent_games
        self.form_predictor = FormPredictor(recent_games=recent_games)
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> None:
        normalized = _normalize_frame(frame)
        self.form_predictor.fit(normalized)
        dataset = self.get_training_dataset(normalized)
        if dataset.empty:
            raise ValueError("Not enough training rows for the hybrid model.")
        self.model.fit(_sanitize_feature_frame(dataset, HYBRID_FEATURE_COLUMNS), dataset["home_win"])

    def predict(
        self,
        frame: pd.DataFrame,
        home_team: str,
        away_team: str,
        *,
        home_odds: float,
        away_odds: float,
        draw_odds: float | None = None,
        handicap_line: float = 0.0,
        sport: str | None = None,
        league: str | None = None,
    ) -> HybridPrediction:
        normalized = _normalize_frame(frame)
        odds_prediction = calculate_implied_probabilities(
            home_odds=home_odds,
            away_odds=away_odds,
            draw_odds=draw_odds,
        )
        form_prediction = self.form_predictor.predict(
            normalized,
            home_team,
            away_team,
            handicap_line=handicap_line,
            sport=sport,
            league=league,
        )
        features = pd.DataFrame(
            [
                {
                    **form_prediction.features,
                    "odds_home_probability": odds_prediction.home_probability,
                    "odds_away_probability": odds_prediction.away_probability,
                    "odds_draw_probability": odds_prediction.draw_probability or 0.0,
                    "bookmaker_margin": odds_prediction.bookmaker_margin,
                }
            ]
        )
        hybrid_probability = float(self.model.predict_proba(_sanitize_feature_frame(features, HYBRID_FEATURE_COLUMNS))[0][1])
        return HybridPrediction(
            home_team=home_team,
            away_team=away_team,
            odds_home_probability=round(odds_prediction.home_probability, 4),
            form_home_probability=round(form_prediction.home_win_probability, 4),
            hybrid_home_probability=round(hybrid_probability, 4),
            bookmaker_margin=round(odds_prediction.bookmaker_margin, 4),
        )

    def _build_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, float | int]] = []
        for index in range(1, len(frame)):
            current_game = frame.iloc[index]
            if pd.isna(current_game.get("home_odds")) or pd.isna(current_game.get("away_odds")):
                continue

            prior_games = frame.iloc[:index]
            form_prediction = self.form_predictor.predict(
                prior_games,
                current_game["home_team"],
                current_game["away_team"],
                sport=current_game.get("sport"),
                league=current_game.get("league"),
            )
            odds_prediction = calculate_implied_probabilities(
                home_odds=float(current_game["home_odds"]),
                away_odds=float(current_game["away_odds"]),
                draw_odds=(
                    float(current_game["draw_odds"])
                    if not pd.isna(current_game.get("draw_odds"))
                    else None
                ),
            )
            rows.append(
                {
                    **form_prediction.features,
                    "odds_home_probability": odds_prediction.home_probability,
                    "odds_away_probability": odds_prediction.away_probability,
                    "odds_draw_probability": odds_prediction.draw_probability or 0.0,
                    "bookmaker_margin": odds_prediction.bookmaker_margin,
                    "home_win": home_covers(current_game),
                }
            )
        return pd.DataFrame(rows)

    def get_training_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._build_dataset(_normalize_frame(frame))

    def feature_importance(self) -> pd.DataFrame:
        classifier = self.model.named_steps["classifier"]
        coefficients = classifier.coef_[0]
        total = sum(abs(value) for value in coefficients) or 1.0
        return pd.DataFrame(
            {
                "feature": HYBRID_FEATURE_COLUMNS,
                "coefficient": coefficients,
                "importance_share": [abs(value) / total for value in coefficients],
            }
        ).sort_values("importance_share", ascending=False).reset_index(drop=True)


MatchPredictor = FormPredictor
