# Darknight

Betman match collector and prediction toolkit.

## What it does

- Collects closed match results from Betman pages
- Iterates over `gmTs` ranges year by year
- Saves result rows with scores and odds
- Computes team summary stats
- Supports three prediction modes:
  - `predict-odds`: bookmaker implied probability only
  - `predict-form`: team form and matchup features only
  - `predict-hybrid`: odds + form combined

## Install

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## Data columns

Collected rows include these fields when available:

- `played_at`
- `sport`
- `league`
- `game_type`
- `home_team`
- `away_team`
- `home_score`
- `away_score`
- `venue`
- `match_seq`
- `home_odds`
- `draw_odds`
- `away_odds`
- `gm_id`
- `gm_ts`

## gmTs format

- `gmTs` format is `YYRRRR`
- `YY` is the year
- `RRRR` is the round number
- Example: `260033` means year `26`, round `0033`

`collect-range` moves through the range sequentially. If a year has already produced rows and then the next round is missing, it can move to the next year with `--stop-after-miss`.

The default is `2`, so the collector only moves to the next year after two consecutive missing rounds within a year that has already produced data.

## Collect examples

Single local HTML:

```bash
python main.py collect --html-file sample.html --gmts 260033 --output data/results.csv
```

Single Betman page with Chromium rendering:

```bash
python main.py collect --url "https://www.betman.co.kr/main/mainPage/gamebuy/closedGameSlip.do?gmId=G101&gmTs=260033" --gmts 260033 --browser --output data/results.csv
```

Range collection with Chromium:

```bash
python main.py collect-range --start-gmts 200001 --end-gmts 260145 --gm-id G101 --browser --output data/results.csv
```

## Stats example

```bash
python main.py stats --input data/results.csv
```

## Incremental sync

You can update `data/results.csv` incrementally from the latest known round before generating predictions.

```bash
python main.py sync-results --input data/results.csv --output data/results.csv --browser
```

## Prediction examples

Odds only:

```bash
python main.py predict-odds --home-odds 2.50 --draw-odds 3.15 --away-odds 2.25
python main.py predict-odds --input data/results.csv --gm-ts 200001 --match-seq 1
```

Form only:

```bash
python main.py predict-form --input data/results.csv --home-team "샬럿H" --away-team "보스턴C" --sport "농구" --league "NBA"
```

Hybrid:

```bash
python main.py predict-hybrid --input data/results.csv --home-team "샬럿H" --away-team "보스턴C" --sport "농구" --league "NBA" --home-odds 2.10 --away-odds 1.72
python main.py predict-hybrid --input data/results.csv --home-team "샬럿H" --away-team "보스턴C" --gm-ts 200001 --match-seq 1
```

`predict` is an alias for `predict-hybrid`.

## Backtest

You can evaluate how the models perform on historical data using a time-based train/test split.

```bash
python main.py backtest --input data/results.csv --sport "농구" --league "NBA" --test-ratio 0.2
python main.py backtest --input data/results.csv --sport "축구" --by-league --output-dir analysis
```

## Mock Betting

You can simulate three agents over the recent historical window.

```bash
python main.py simulate-bets --input data/results.csv --lookback-days 30 --output-dir analysis
```

This writes:

- `analysis/simulation_summary.csv`
- `analysis/simulation_bets.csv`
- `analysis/simulation_daily.csv`

## Model cache

`predict-form`, `predict-hybrid`, and `predict-today` can store trained models in `.cache` by default.

- The cache key uses the CSV file path, size, modified time, predictor type, and `recent-games`
- If `data/results.csv` changes, a new cache file is created automatically
- If the CSV stays the same, later predictions reuse the cached model and start much faster

Example:

```bash
python main.py predict-form --input data/results.csv --home-team "샬럿H" --away-team "보스턴C" --sport "농구" --league "NBA" --cache-dir .cache
```

## Daily report

You can generate a report for today's upcoming matches from an open Betman page.

```bash
python main.py predict-today --input data/results.csv --url "https://www.betman.co.kr/main/mainPage/gamebuy/gameSlip.do?frameType=typeA&gmId=G101" --browser --output-dir reports
python main.py predict-round --input data/results.csv --url "https://www.betman.co.kr/main/mainPage/gamebuy/gameSlip.do?frameType=typeA&gmId=G101" --browser --output-dir reports
python main.py predict-all --input data/results.csv --url "https://www.betman.co.kr/main/mainPage/gamebuy/gameSlip.do?frameType=typeA&gmId=G101" --browser --output-dir reports
```

Outputs:

- `reports/daily_predictions_YYYY-MM-DD.csv`
- `reports/daily_predictions_YYYY-MM-DD.md`

The project also includes [run_daily_report.ps1](/c:/Users/lee33/Documents/Darknight/run_daily_report.ps1) as a starter script for Windows Task Scheduler.

Example scheduler command:

```powershell
schtasks /Create /SC DAILY /TN "Darknight Daily Report" /TR "powershell -ExecutionPolicy Bypass -File C:\Users\lee33\Documents\Darknight\run_daily_report.ps1" /ST 08:00
```

## Docker scheduler

The project includes a Docker scheduler that runs `predict-today` every day at 08:00 by default.

Build:

```bash
docker build -t darknight .
```

Run with Docker Compose:

```bash
docker compose up -d --build
```

Or run directly:

```bash
docker run -d \
  --name darknight \
  -e TZ=Asia/Seoul \
  -e REPORT_HOUR=8 \
  -e REPORT_MINUTE=0 \
  -e RUN_ON_START=false \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/reports:/app/reports \
  -v ${PWD}/.cache:/app/.cache \
  darknight
```

Default container behavior:

- runs `python docker/scheduler.py`
- waits until the next scheduled time
- runs `sync-results` first
- runs `backtest` and writes analysis files
- runs `simulate-bets` and writes mock betting ROI files
- executes `predict-all`
- saves analysis under `/app/analysis`
- saves both daily and round reports under `/app/reports`
- reuses model caches under `/app/.cache`

Useful environment variables:

- `TZ`: timezone, default `Asia/Seoul`
- `REPORT_HOUR`: report hour, default `8`
- `REPORT_MINUTE`: report minute, default `0`
- `RUN_ON_START`: if `true`, run once immediately on container start
- `INPUT_CSV`: historical results file, default `data/results.csv`
- `UPCOMING_URL`: page for today's matches
- `UPCOMING_GMTS`: optional fixed gmTs. If omitted, the app probes forward from the latest known round and collects all target-date matches it finds
- `OUTPUT_DIR`: report directory, default `reports`
- `ANALYSIS_DIR`: backtest output directory, default `analysis`
- `SIMULATION_LOOKBACK_DAYS`: recent history window for mock betting, default `30`
- `SIMULATION_EDGE_THRESHOLD`: minimum edge for form/hybrid bets, default `0.05`
- `SIMULATION_STAKE`: flat unit stake per simulated bet, default `1.0`
- `CACHE_DIR`: cache directory, default `.cache`
- `RECENT_GAMES`: recent game window, default `5`
- `SEARCH_WINDOW`: how many future `gmTs` values to scan for today's matches
- `BACKTEST_TEST_RATIO`: test split fraction for the daily backtest job
- `SYNC_PROBE_COUNT`: how many completed `gmTs` values to probe ahead from the latest stored result round
- `SYNC_STOP_AFTER_MISS`: stop sync probing after this many consecutive misses
- `USE_BROWSER`: if `true`, use Chromium rendering
- `HEADED`: if `true`, open the browser window

`predict-today` does not need a hardcoded round in normal use. It can probe multiple `gmTs` values ahead, detect which pages contain the target date, and include those matches in the report.

The repository includes `compose.yaml`, so on Windows the simplest path is usually:

```bash
docker compose up -d --build
```

## Current parsing rules

- Parses `#tbl_gmBuySlipList` first
- Uses rows marked as `결과발표`
- Keeps `일반` game type rows for the main dataset
- Reads `승/무/패` odds when present
- Uses `gmTs` to fill the year when the page only shows month/day

If Betman changes the page structure, update the selectors and parsing logic in `darknight/scraper.py`.
