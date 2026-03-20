from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class GameResult:
    played_at: datetime
    sport: str
    league: str
    game_type: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    venue: str = ""
    match_seq: int = 0
    home_odds: float | None = None
    draw_odds: float | None = None
    away_odds: float | None = None

    @property
    def winner(self) -> str:
        if self.home_score > self.away_score:
            return self.home_team
        if self.away_score > self.home_score:
            return self.away_team
        return "DRAW"


@dataclass(slots=True)
class UpcomingMatch:
    played_at: datetime
    close_at: datetime | None
    sport: str
    league: str
    game_type: str
    status: str
    home_team: str
    away_team: str
    venue: str = ""
    match_seq: int = 0
    home_odds: float | None = None
    draw_odds: float | None = None
    away_odds: float | None = None
