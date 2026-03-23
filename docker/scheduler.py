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


def build_prediction_command() -> list[str]:
    command = [
        "python",
        "main.py",
        "predict-all",
        "--input",
        os.getenv("INPUT_CSV", "data/results.csv"),
        "--url",
        os.getenv("UPCOMING_URL", "https://www.betman.co.kr/main/mainPage/gamebuy/gameSlip.do?frameType=typeA&gmId=G101"),
        "--output-dir",
        os.getenv("OUTPUT_DIR", "reports"),
        "--analysis-dir",
        os.getenv("ANALYSIS_DIR", "analysis"),
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


def build_backtest_command() -> list[str]:
    command = [
        "python",
        "main.py",
        "backtest",
        "--input",
        os.getenv("INPUT_CSV", "data/results.csv"),
        "--output-dir",
        os.getenv("ANALYSIS_DIR", "analysis"),
        "--recent-games",
        os.getenv("RECENT_GAMES", "5"),
        "--test-ratio",
        os.getenv("BACKTEST_TEST_RATIO", "0.2"),
        "--by-league",
    ]
    sport = os.getenv("BACKTEST_SPORT")
    if sport:
        command.extend(["--sport", sport])
    league = os.getenv("BACKTEST_LEAGUE")
    if league:
        command.extend(["--league", league])
    return command


def build_simulation_command() -> list[str]:
    command = [
        "python",
        "main.py",
        "simulate-bets",
        "--input",
        os.getenv("INPUT_CSV", "data/results.csv"),
        "--output-dir",
        os.getenv("ANALYSIS_DIR", "analysis"),
        "--recent-games",
        os.getenv("RECENT_GAMES", "5"),
        "--lookback-days",
        os.getenv("SIMULATION_LOOKBACK_DAYS", "30"),
        "--edge-threshold",
        os.getenv("SIMULATION_EDGE_THRESHOLD", "0.05"),
        "--stake",
        os.getenv("SIMULATION_STAKE", "1.0"),
        "--cache-dir",
        os.getenv("CACHE_DIR", ".cache"),
    ]
    sport = os.getenv("SIMULATION_SPORT")
    if sport:
        command.extend(["--sport", sport])
    league = os.getenv("SIMULATION_LEAGUE")
    if league:
        command.extend(["--league", league])
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


def build_settle_command() -> list[str]:
    return [
        "python",
        "main.py",
        "settle-reports",
        "--input",
        os.getenv("INPUT_CSV", "data/results.csv"),
        "--reports-dir",
        os.getenv("OUTPUT_DIR", "reports"),
        "--output-dir",
        os.getenv("ANALYSIS_DIR", "analysis"),
    ]


def run_pipeline(reason: str) -> None:
    Path(os.getenv("OUTPUT_DIR", "reports")).mkdir(parents=True, exist_ok=True)
    Path(os.getenv("CACHE_DIR", ".cache")).mkdir(parents=True, exist_ok=True)
    Path(os.getenv("ANALYSIS_DIR", "analysis")).mkdir(parents=True, exist_ok=True)
    lock_path = Path(os.getenv("SCHEDULER_LOCK_PATH", "analysis/scheduler.lock"))

    if lock_path.exists():
        print(f"[scheduler] skip reason={reason} lock={lock_path}", flush=True)
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(f"{reason}|{datetime.now().isoformat()}", encoding="utf-8")

    try:
        print(f"[scheduler] pipeline start reason={reason}", flush=True)

        sync_command = build_sync_command()
        print(f"[scheduler] syncing: {' '.join(sync_command)}", flush=True)
        sync_completed = subprocess.run(sync_command, check=False)
        print(f"[scheduler] sync exit code: {sync_completed.returncode}", flush=True)

        settle_command = build_settle_command()
        print(f"[scheduler] settling: {' '.join(settle_command)}", flush=True)
        settle_completed = subprocess.run(settle_command, check=False)
        print(f"[scheduler] settle exit code: {settle_completed.returncode}", flush=True)

        backtest_command = build_backtest_command()
        print(f"[scheduler] backtesting: {' '.join(backtest_command)}", flush=True)
        backtest_completed = subprocess.run(backtest_command, check=False)
        print(f"[scheduler] backtest exit code: {backtest_completed.returncode}", flush=True)

        simulation_command = build_simulation_command()
        print(f"[scheduler] simulating: {' '.join(simulation_command)}", flush=True)
        simulation_completed = subprocess.run(simulation_command, check=False)
        print(f"[scheduler] simulation exit code: {simulation_completed.returncode}", flush=True)

        prediction_command = build_prediction_command()
        print(f"[scheduler] running: {' '.join(prediction_command)}", flush=True)
        prediction_completed = subprocess.run(prediction_command, check=False)
        print(f"[scheduler] predict exit code: {prediction_completed.returncode}", flush=True)
    finally:
        if lock_path.exists():
            lock_path.unlink()


def next_run_time(now: datetime, hour: int, minute: int) -> datetime:
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def is_round_refresh_day(now: datetime, round_days: set[int]) -> bool:
    return now.weekday() in round_days


def parse_weekdays(raw: str) -> set[int]:
    mapping = {
        "mon": 0,
        "monday": 0,
        "tue": 1,
        "tuesday": 1,
        "wed": 2,
        "wednesday": 2,
        "thu": 3,
        "thursday": 3,
        "fri": 4,
        "friday": 4,
        "sat": 5,
        "saturday": 5,
        "sun": 6,
        "sunday": 6,
    }
    weekdays: set[int] = set()
    for part in raw.split(","):
        normalized = part.strip().lower()
        if normalized in mapping:
            weekdays.add(mapping[normalized])
    return weekdays or {0, 2, 4}


def main() -> int:
    timezone = ZoneInfo(os.getenv("TZ", "Asia/Seoul"))
    daily_hour = int(os.getenv("REPORT_HOUR", "8"))
    daily_minute = int(os.getenv("REPORT_MINUTE", "0"))
    round_hour = int(os.getenv("ROUND_REPORT_HOUR", "14"))
    round_minute = int(os.getenv("ROUND_REPORT_MINUTE", "0"))
    round_days = parse_weekdays(os.getenv("ROUND_REPORT_WEEKDAYS", "mon,wed,fri"))
    poll_minutes = int(os.getenv("POLL_MINUTES", "10"))
    run_on_start = env_flag("RUN_ON_START", False)
    state_path = Path(os.getenv("SCHEDULER_STATE_PATH", "analysis/scheduler_state.txt"))

    print(
        f"[scheduler] timezone={timezone.key} daily_time={daily_hour:02d}:{daily_minute:02d} "
        f"round_time={round_hour:02d}:{round_minute:02d} round_days={sorted(round_days)} "
        f"poll_minutes={poll_minutes} run_on_start={run_on_start}",
        flush=True,
    )

    last_mark = state_path.read_text(encoding="utf-8").strip() if state_path.exists() else ""
    if run_on_start:
        run_pipeline("startup")
        last_mark = f"startup:{datetime.now(timezone).isoformat()}"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(last_mark, encoding="utf-8")

    while True:
        now = datetime.now(timezone)
        daily_key = f"daily:{now.date().isoformat()}"
        round_key = f"round:{now.date().isoformat()}"
        daily_due = (
            last_mark != daily_key
            and now >= now.replace(hour=daily_hour, minute=daily_minute, second=0, microsecond=0)
        )
        round_due = (
            last_mark != round_key
            and is_round_refresh_day(now, round_days)
            and now >= now.replace(hour=round_hour, minute=round_minute, second=0, microsecond=0)
        )

        if daily_due:
            run_pipeline("daily")
            last_mark = daily_key
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(last_mark, encoding="utf-8")
        elif round_due:
            run_pipeline("round_refresh")
            last_mark = round_key
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(last_mark, encoding="utf-8")

        sleep_seconds = max(60, poll_minutes * 60)
        print(
            f"[scheduler] now={now.isoformat()} next_daily={next_run_time(now, daily_hour, daily_minute).isoformat()} "
            f"sleep={sleep_seconds}s",
            flush=True,
        )
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
