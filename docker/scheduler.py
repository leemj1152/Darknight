from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
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


def build_probe_command() -> list[str]:
    command = [
        "python",
        "main.py",
        "probe-round",
        "--input",
        os.getenv("INPUT_CSV", "data/results.csv"),
        "--url",
        os.getenv("UPCOMING_URL", "https://www.betman.co.kr/main/mainPage/gamebuy/gameSlip.do?frameType=typeA&gmId=G101"),
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


def run_pipeline(reason: str) -> None:
    Path(os.getenv("OUTPUT_DIR", "reports")).mkdir(parents=True, exist_ok=True)
    Path(os.getenv("CACHE_DIR", ".cache")).mkdir(parents=True, exist_ok=True)
    Path(os.getenv("ANALYSIS_DIR", "analysis")).mkdir(parents=True, exist_ok=True)

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


def probe_round_snapshot() -> dict[str, str] | None:
    command = build_probe_command()
    print(f"[scheduler] probing round: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        print(f"[scheduler] probe failed rc={completed.returncode}", flush=True)
        if completed.stdout:
            print(completed.stdout, flush=True)
        if completed.stderr:
            print(completed.stderr, flush=True)
        return None

    snapshot: dict[str, str] = {}
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        snapshot[key.strip()] = value.strip()
    if snapshot:
        print(
            f"[scheduler] probe gmTs={snapshot.get('gmTs', '')} close_at_max={snapshot.get('close_at_max', '')}",
            flush=True,
        )
    return snapshot or None


def next_daily_run(now: datetime, hour: int, minute: int) -> datetime:
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    timezone = ZoneInfo(os.getenv("TZ", "Asia/Seoul"))
    hour = int(os.getenv("REPORT_HOUR", "8"))
    minute = int(os.getenv("REPORT_MINUTE", "0"))
    poll_minutes = int(os.getenv("POLL_MINUTES", "15"))
    run_on_start = env_flag("RUN_ON_START", False)
    state_path = Path(os.getenv("SCHEDULER_STATE_PATH", "analysis/scheduler_state.json"))

    print(
        f"[scheduler] timezone={timezone.key} report_time={hour:02d}:{minute:02d} "
        f"poll_minutes={poll_minutes} run_on_start={run_on_start}",
        flush=True,
    )

    state = load_state(state_path)
    if run_on_start:
        run_pipeline("startup")
        state["last_daily_run_date"] = datetime.now(timezone).date().isoformat()
        save_state(state_path, state)

    while True:
        now = datetime.now(timezone)
        target = next_daily_run(now, hour, minute)
        snapshot = probe_round_snapshot()
        daily_due = state.get("last_daily_run_date") != now.date().isoformat() and now >= now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        round_changed = False
        if snapshot and snapshot.get("gmTs"):
            last_gmts = state.get("last_round_gmts")
            current_gmts = snapshot.get("gmTs")
            round_changed = current_gmts != last_gmts and last_gmts is not None
            state["last_seen_round_gmts"] = current_gmts
            state["last_seen_round_close_at"] = snapshot.get("close_at_max", "")

        if daily_due:
            run_pipeline("daily")
            state["last_daily_run_date"] = now.date().isoformat()
            if snapshot and snapshot.get("gmTs"):
                state["last_round_gmts"] = snapshot["gmTs"]
            save_state(state_path, state)
        elif round_changed:
            run_pipeline("round_change")
            state["last_round_gmts"] = snapshot["gmTs"]
            state["last_round_change_run_at"] = now.isoformat()
            save_state(state_path, state)
        elif snapshot and state.get("last_round_gmts") is None:
            state["last_round_gmts"] = snapshot["gmTs"]
            save_state(state_path, state)

        sleep_seconds = max(60, poll_minutes * 60)
        print(
            f"[scheduler] now={now.isoformat()} next_daily={target.isoformat()} "
            f"sleep={sleep_seconds}s",
            flush=True,
        )
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
