from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import Response

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    PlaywrightTimeoutError = TimeoutError
    sync_playwright = None

from .config import ScraperConfig
from .models import GameResult, UpcomingMatch


EXPECTED_COLUMNS = [
    "played_at",
    "league",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
]

RESULT_PUBLISHED_TEXT = "결과발표"
NORMAL_GAME_TYPE = "일반"
HANDICAP_GAME_TYPE = "핸디캡"
WIN_LABEL = "승"
DRAW_LABEL = "무"
LOSE_LABEL = "패"


class BatmanScraper:
    def __init__(self, config: ScraperConfig | None = None) -> None:
        self.config = config or ScraperConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})

    def fetch_html(self, url: str, use_browser: bool = False, headed: bool = False) -> str:
        if use_browser:
            return self.fetch_html_with_browser(url, headed=headed)
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.text

    def fetch_game_html(
        self,
        gm_ts: int,
        gm_id: str | None = None,
        use_browser: bool = False,
        headed: bool = False,
    ) -> str:
        url = self.build_game_url(gm_ts, gm_id=gm_id)
        return self.fetch_html(url, use_browser=use_browser, headed=headed)

    def fetch_upcoming_html(
        self,
        gm_ts: int,
        gm_id: str | None = None,
        use_browser: bool = False,
        headed: bool = False,
        base_url: str | None = None,
    ) -> str:
        url = self.build_upcoming_url(gm_ts, gm_id=gm_id, base_url=base_url)
        return self.fetch_html(url, use_browser=use_browser, headed=headed)

    def fetch_game_response(self, gm_ts: int, gm_id: str | None = None) -> Response:
        response = self.session.get(
            self.config.base_url,
            timeout=30,
            params={"gmId": gm_id or self.config.gm_id, "gmTs": format_gmts(gm_ts)},
        )
        response.raise_for_status()
        return response

    def build_game_url(self, gm_ts: int, gm_id: str | None = None) -> str:
        return f"{self.config.base_url}?gmId={gm_id or self.config.gm_id}&gmTs={format_gmts(gm_ts)}"

    def build_upcoming_url(
        self,
        gm_ts: int,
        gm_id: str | None = None,
        base_url: str | None = None,
    ) -> str:
        raw_url = base_url or self.config.upcoming_base_url
        parts = urlsplit(raw_url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["gmId"] = gm_id or query.get("gmId") or self.config.gm_id
        query["gmTs"] = format_gmts(gm_ts)
        if "frameType" not in query:
            query["frameType"] = "typeA"
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

    def fetch_html_with_browser(self, url: str, headed: bool = False) -> str:
        if sync_playwright is None:
            raise RuntimeError(
                "Playwright is not installed. Run 'pip install -r requirements.txt' "
                "and 'python -m playwright install chromium'."
            )

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=not headed)
            page = browser.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_load_state("networkidle", timeout=30000)
                page.wait_for_selector("#tbl_gmBuySlipList", timeout=15000)
                return page.content()
            except PlaywrightTimeoutError as exc:
                raise ValueError("The Betman table did not render in the browser.") from exc
            finally:
                browser.close()

    def load_html_file(self, html_path: str | Path) -> str:
        return Path(html_path).read_text(encoding="utf-8")

    def parse_results(self, html: str, gm_ts: int | str | None = None) -> pd.DataFrame:
        rows = self._select_rows(html)
        games = [game for game in (self._build_game_result_from_row(row, gm_ts) for row in rows) if game is not None]
        if not games:
            raise ValueError("No valid result rows were parsed.")

        frame = pd.DataFrame(
            [
                {
                    "played_at": game.played_at,
                    "sport": game.sport,
                    "league": game.league,
                    "game_type": game.game_type,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "home_score": game.home_score,
                    "away_score": game.away_score,
                    "handicap_line": game.handicap_line,
                    "venue": game.venue,
                    "match_seq": game.match_seq,
                    "home_odds": game.home_odds,
                    "draw_odds": game.draw_odds,
                    "away_odds": game.away_odds,
                }
                for game in games
            ]
        )
        return frame.sort_values(["played_at", "match_seq"]).reset_index(drop=True)

    def parse_upcoming_matches(self, html: str, gm_ts: int | str | None = None) -> pd.DataFrame:
        rows = self._select_rows(html)
        matches = [
            match for match in (self._build_upcoming_match_from_row(row, gm_ts) for row in rows) if match is not None
        ]
        if not matches:
            raise ValueError("No valid upcoming rows were parsed.")

        frame = pd.DataFrame(
            [
                {
                    "played_at": match.played_at,
                    "close_at": match.close_at,
                    "sport": match.sport,
                    "league": match.league,
                    "game_type": match.game_type,
                    "status": match.status,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "handicap_line": match.handicap_line,
                    "venue": match.venue,
                    "match_seq": match.match_seq,
                    "home_odds": match.home_odds,
                    "draw_odds": match.draw_odds,
                    "away_odds": match.away_odds,
                }
                for match in matches
            ]
        )
        return frame.sort_values(["played_at", "match_seq"]).reset_index(drop=True)

    def save_results(self, frame: pd.DataFrame, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False, encoding="utf-8-sig")
        return output

    def _select_rows(self, html: str) -> list[object]:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.select_one("#tbl_gmBuySlipList") or soup.select_one(self.config.table_selector)
        if table is None:
            raise ValueError("Could not find the Betman table.")
        return list(table.select("tbody tr") or table.select(self.config.row_selector))

    def _build_game_result_from_row(self, row: object, gm_ts: int | str | None) -> GameResult | None:
        cells = row.select("td")
        if len(cells) < 8:
            return None

        status = self._clean_text(cells[1])
        if RESULT_PUBLISHED_TEXT not in status:
            return None

        sport = self._clean_text(cells[2].select_one(".icoGame"))
        league = self._clean_text(cells[2].select_one(".db"))
        game_type = self._clean_text(cells[3].select_one(".badge"))
        if game_type and game_type not in {NORMAL_GAME_TYPE, HANDICAP_GAME_TYPE}:
            return None

        home_team, away_team = self._extract_teams(cells[4])
        home_score, away_score = self._extract_scores(cells[4])
        if not home_team or not away_team or home_score is None or away_score is None:
            return None

        played_at = self._extract_datetime_from_cell(cells[6], gm_ts)
        if played_at is None:
            return None

        home_odds, draw_odds, away_odds = self._extract_odds(cells[5])
        venue = self._extract_venue(cells[7])
        handicap_line = self._extract_handicap_line(cells[4])
        match_seq = int(row.attrs.get("data-matchseq", 0) or 0)

        return GameResult(
            played_at=played_at,
            sport=sport,
            league=league,
            game_type=game_type or NORMAL_GAME_TYPE,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            handicap_line=handicap_line,
            venue=venue,
            match_seq=match_seq,
            home_odds=home_odds,
            draw_odds=draw_odds,
            away_odds=away_odds,
        )

    def _build_upcoming_match_from_row(self, row: object, gm_ts: int | str | None) -> UpcomingMatch | None:
        cells = row.select("td")
        if len(cells) < 8:
            return None

        status = self._clean_text(cells[1])
        if RESULT_PUBLISHED_TEXT in status:
            return None

        sport = self._clean_text(cells[2].select_one(".icoGame"))
        league = self._clean_text(cells[2].select_one(".db"))
        game_type = self._clean_text(cells[3].select_one(".badge"))
        if game_type and game_type not in {NORMAL_GAME_TYPE, HANDICAP_GAME_TYPE}:
            return None

        home_team, away_team = self._extract_teams(cells[4])
        if not home_team or not away_team:
            return None

        played_at = self._extract_datetime_from_cell(cells[6], gm_ts)
        if played_at is None:
            return None

        close_at = self._extract_datetime_from_cell(cells[1], gm_ts)
        home_odds, draw_odds, away_odds = self._extract_odds(cells[5])
        venue = self._extract_venue(cells[7])
        handicap_line = self._extract_handicap_line(cells[4])
        match_seq = int(row.attrs.get("data-matchseq", 0) or 0)

        return UpcomingMatch(
            played_at=played_at,
            close_at=close_at,
            sport=sport,
            league=league,
            game_type=game_type or NORMAL_GAME_TYPE,
            status=status,
            home_team=home_team,
            away_team=away_team,
            handicap_line=handicap_line,
            venue=venue,
            match_seq=match_seq,
            home_odds=home_odds,
            draw_odds=draw_odds,
            away_odds=away_odds,
        )

    def _extract_teams(self, score_cell: object) -> tuple[str, str]:
        team_cells = score_cell.select(".scoreDiv .cell")
        if len(team_cells) < 2:
            return "", ""
        return self._extract_team_name(team_cells[0]), self._extract_team_name(team_cells[1])

    def _extract_team_name(self, team_cell: object) -> str:
        direct_text = " ".join(
            part.strip()
            for part in team_cell.find_all(string=True, recursive=False)
            if part and part.strip()
        )
        if direct_text:
            return direct_text

        spans: list[str] = []
        for span in team_cell.select("span"):
            classes = span.get("class") or []
            if "blind" in classes:
                continue
            text = self._clean_text(span)
            if text and not text.startswith(("H ", "U/O", "+", "-")):
                spans.append(text)
        return spans[-1] if spans else ""

    def _extract_scores(self, score_cell: object) -> tuple[int | None, int | None]:
        score_nodes = score_cell.select(".scoreDiv .score")
        if len(score_nodes) < 2:
            return None, None
        return (
            self._extract_int(score_nodes[0].get_text(" ", strip=True)),
            self._extract_int(score_nodes[1].get_text(" ", strip=True)),
        )

    def _extract_datetime_from_cell(self, cell: object, gm_ts: int | str | None) -> datetime | None:
        raw_text = cell.get_text(" ", strip=True)
        full_match = re.search(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2}).*?(\d{1,2}):(\d{2})", raw_text)
        if full_match:
            year, month, day, hour, minute = map(int, full_match.groups())
            return datetime(year, month, day, hour, minute)

        partial_match = re.search(r"(\d{1,2})\.(\d{1,2}).*?(\d{1,2}):(\d{2})", raw_text)
        if not partial_match:
            return None

        if gm_ts is None:
            year = datetime.now().year
        else:
            year_prefix, _ = parse_gmts(gm_ts)
            year = 2000 + year_prefix

        month, day, hour, minute = map(int, partial_match.groups())
        return datetime(year, month, day, hour, minute)

    def _extract_odds(self, odds_cell: object) -> tuple[float | None, float | None, float | None]:
        parsed: list[tuple[str, float]] = []
        for button in odds_cell.select(".btnChk"):
            label = self._extract_odds_label(button)
            value = self._extract_float(self._clean_text(button.select_one(".db")))
            if label and value is not None:
                parsed.append((label, value))

        if not parsed:
            return None, None, None

        home_odds = draw_odds = away_odds = None
        for label, value in parsed:
            if label == WIN_LABEL:
                home_odds = value
            elif label == DRAW_LABEL:
                draw_odds = value
            elif label == LOSE_LABEL:
                away_odds = value

        if len(parsed) == 2:
            if home_odds is None:
                home_odds = parsed[0][1]
            if away_odds is None:
                away_odds = parsed[-1][1]

        return home_odds, draw_odds, away_odds

    def _extract_odds_label(self, button: object) -> str:
        labels: list[str] = []
        for span in button.select("span"):
            classes = span.get("class") or []
            if "blind" in classes or "db" in classes:
                continue
            text = self._clean_text(span)
            if text:
                labels.append(text)
        return labels[0] if labels else ""

    def _extract_venue(self, venue_cell: object) -> str:
        return self._clean_text(venue_cell.select_one(".ttHLayer span")) or self._clean_text(venue_cell)

    def _extract_handicap_line(self, score_cell: object) -> float | None:
        badge_text = self._clean_text(score_cell.select_one(".udPointBox .udPoint"))
        if not badge_text:
            return None
        match = re.search(r"([+-]?\d+(?:\.\d+)?)", badge_text)
        if not match:
            return None
        value = float(match.group(1))
        if "H" in badge_text.upper():
            return value
        return None

    def _extract_int(self, text: str) -> int | None:
        match = re.search(r"-?\d+", text)
        return int(match.group(0)) if match else None

    def _extract_float(self, text: str) -> float | None:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else None

    def _clean_text(self, node: object | None) -> str:
        if node is None:
            return ""
        return " ".join(node.get_text(" ", strip=True).split())


def load_results_csv(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    frame["played_at"] = pd.to_datetime(frame["played_at"])
    for column in ("home_odds", "draw_odds", "away_odds"):
        if column not in frame.columns:
            frame[column] = pd.NA
    if "handicap_line" not in frame.columns:
        frame["handicap_line"] = pd.NA
    if "sport" not in frame.columns:
        frame["sport"] = ""
    if "league" not in frame.columns:
        frame["league"] = ""
    return frame


def parse_gmts(gm_ts: int | str) -> tuple[int, int]:
    text = f"{int(str(gm_ts)):06d}"
    return int(text[:2]), int(text[2:])


def format_gmts(gm_ts: int | str) -> str:
    year, round_no = parse_gmts(gm_ts)
    return f"{year:02d}{round_no:04d}"


def next_gmts(gm_ts: int | str) -> int:
    year, round_no = parse_gmts(gm_ts)
    return int(f"{year:02d}{round_no + 1:04d}")


def next_year_gmts(gm_ts: int | str) -> int:
    year, _ = parse_gmts(gm_ts)
    return int(f"{year + 1:02d}0001")
