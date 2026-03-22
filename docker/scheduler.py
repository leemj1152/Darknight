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


def next_run_time(now: datetime, hour: int, minute: int) -> datetime:
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def run_prediction() -> None:
    Path(os.getenv("OUTPUT_DIR", "reports")).mkdir(parents=True, exist_ok=True)
    Path(os.getenv("CACHE_DIR", ".cache")).mkdir(parents=True, exist_ok=True)
    Path(os.getenv("ANALYSIS_DIR", "analysis")).mkdir(parents=True, exist_ok=True)

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
