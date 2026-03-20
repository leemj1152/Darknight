from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ScraperConfig:
    base_url: str = "https://www.betman.co.kr/main/mainPage/gamebuy/closedGameSlip.do"
    upcoming_base_url: str = "https://www.betman.co.kr/main/mainPage/gamebuy/gameSlip.do"
    gm_id: str = "G101"
    table_selector: str = "table"
    row_selector: str = "tr"
    cell_selector: str = "td"
    date_format: str = "%Y-%m-%d %H:%M"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
