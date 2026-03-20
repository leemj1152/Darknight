from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_command() -> list[str]:
    command = [
        "python",
        "main.py",
        "predict-today",
        "--input",
        os.getenv("INPUT_CSV", "data/results.csv"),
        "--url",
        os.getenv("UPCOMING_URL", "https://www.betman.co.kr/main/mainPage/gamebuy/gameSlip.do?frameType=typeA&gmId=G101"),
        "--output-dir",
        os.getenv("OUTPUT_DIR", "reports"),
        "--cache-dir",
        os.getenv("CACHE_DIR", ".cache"),
        "--recent-games",
        os.getenv("RECENT_GAMES", "5"),
        "--search-window",
        os.getenv("SEARCH_WINDOW", "12"),
    ]

    gmts = os.getenv("UPCOMING_GMTS")
    if gmts:
        command.extend(["--gmts", gmts])
    if env_flag("USE_BROWSER", True):
        command.append("--browser")
    if env_flag("HEADED", False):
        command.append("--headed")

    return command


def build_sync_command() -> list[str]:
    command = [
        "python",
        "main.py",
        "sync-results",
        "--input",
        os.getenv("INPUT_CSV", "data/results.csv"),
        "--output",
        os.getenv("INPUT_CSV", "data/results.csv"),
        "--gm-id",
        os.getenv("GM_ID", "G101"),
        "--probe-count",
        os.getenv("SYNC_PROBE_COUNT", "6"),
        "--stop-after-miss",
        os.getenv("SYNC_STOP_AFTER_MISS", "2"),
    ]
    if env_flag("USE_BROWSER", True):
        command.append("--browser")
    if env_flag("HEADED", False):
        command.append("--headed")
    return command


def next_run_time(now: datetime, hour: int, minute: int) -> datetime:
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def run_prediction() -> None:
    Path(os.getenv("OUTPUT_DIR", "reports")).mkdir(parents=True, exist_ok=True)
    Path(os.getenv("CACHE_DIR", ".cache")).mkdir(parents=True, exist_ok=True)

    sync_command = build_sync_command()
    print(f"[scheduler] syncing: {' '.join(sync_command)}", flush=True)
    sync_completed = subprocess.run(sync_command, check=False)
    print(f"[scheduler] sync exit code: {sync_completed.returncode}", flush=True)

    command = build_command()
    print(f"[scheduler] running: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    print(f"[scheduler] exit code: {completed.returncode}", flush=True)


def main() -> int:
    timezone = ZoneInfo(os.getenv("TZ", "Asia/Seoul"))
    hour = int(os.getenv("REPORT_HOUR", "8"))
    minute = int(os.getenv("REPORT_MINUTE", "0"))
    run_on_start = env_flag("RUN_ON_START", False)

    print(
        f"[scheduler] timezone={timezone.key} report_time={hour:02d}:{minute:02d} "
        f"run_on_start={run_on_start}",
        flush=True,
    )

    if run_on_start:
        run_prediction()

    while True:
        now = datetime.now(timezone)
        target = next_run_time(now, hour, minute)
        sleep_seconds = max(1, int((target - now).total_seconds()))
        print(
            f"[scheduler] now={now.isoformat()} next_run={target.isoformat()} "
            f"sleep={sleep_seconds}s",
            flush=True,
        )
        time.sleep(sleep_seconds)
        run_prediction()


if __name__ == "__main__":
    raise SystemExit(main())
