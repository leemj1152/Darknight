from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from .cache import build_cache_key, csv_signature, ensure_cache_dir, load_joblib, save_joblib
from .odds import calculate_implied_probabilities, calculate_implied_probabilities_from_row
from .predictor import FormPredictor, HybridPredictor
from .reporting import filter_matches_for_date, save_daily_report
from .scraper import (
    BatmanScraper,
    format_gmts,
    load_results_csv,
    next_gmts,
    next_year_gmts,
    parse_gmts,
)
from .stats import team_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Betman results, compute stats, and run predictions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect results from a single page.")
    collect.add_argument("--url", help="Source URL.")
    collect.add_argument("--html-file", help="Local HTML file path.")
    collect.add_argument("--gmts", type=int, help="gmTs used to infer the year when the page omits it.")
    collect.add_argument("--browser", action="store_true", help="Render the page with Chromium.")
    collect.add_argument("--headed", action="store_true", help="Open the browser window for debugging.")
    collect.add_argument("--output", default="data/results.csv", help="CSV output path.")

    collect_range = subparsers.add_parser(
        "collect-range",
        help="Collect results by iterating through a gmTs range.",
    )
    collect_range.add_argument("--start-gmts", required=True, type=int, help="Start gmTs, e.g. 200001.")
    collect_range.add_argument("--end-gmts", required=True, type=int, help="End gmTs, e.g. 260145.")
    collect_range.add_argument("--gm-id", default="G101", help="Betman game id.")
    collect_range.add_argument("--browser", action="store_true", help="Render each page with Chromium.")
    collect_range.add_argument("--headed", action="store_true", help="Open the browser window for debugging.")
    collect_range.add_argument("--output", default="data/results.csv", help="CSV output path.")
    collect_range.add_argument(
        "--stop-after-miss",
        default=2,
        type=int,
        help="Advance to the next year after this many misses once a year has started producing data.",
    )

    stats = subparsers.add_parser("stats", help="Show team summary stats.")
    stats.add_argument("--input", default="data/results.csv", help="Input CSV path.")

    sync_results = subparsers.add_parser(
        "sync-results",
        help="Incrementally append newly finished results to the existing CSV.",
    )
    sync_results.add_argument("--input", default="data/results.csv", help="Existing results CSV path.")
    sync_results.add_argument("--output", default="data/results.csv", help="Output CSV path.")
    sync_results.add_argument("--gm-id", default="G101", help="Betman game id.")
    sync_results.add_argument("--browser", action="store_true", help="Render each page with Chromium.")
    sync_results.add_argument("--headed", action="store_true", help="Open the browser window for debugging.")
    sync_results.add_argument("--probe-count", default=6, type=int, help="How many gmTs values to probe ahead from the latest known round.")
    sync_results.add_argument("--stop-after-miss", default=2, type=int, help="Stop after this many consecutive misses while probing.")

    predict_form = subparsers.add_parser("predict-form", help="Form-based home win probability.")
    add_predict_common_args(predict_form)

    predict_odds = subparsers.add_parser("predict-odds", help="Odds-only implied probabilities.")
    predict_odds.add_argument("--home-odds", type=float, help="Home win odds.")
    predict_odds.add_argument("--draw-odds", type=float, help="Draw odds.")
    predict_odds.add_argument("--away-odds", type=float, help="Away win odds.")
    predict_odds.add_argument("--input", help="CSV path containing odds columns.")
    predict_odds.add_argument("--gm-ts", help="gmTs to lookup in the CSV.")
    predict_odds.add_argument("--match-seq", type=int, help="match_seq to lookup in the CSV.")

    predict_hybrid = subparsers.add_parser("predict-hybrid", help="Combined odds + form prediction.")
    add_predict_common_args(predict_hybrid)
    predict_hybrid.add_argument("--home-odds", type=float, help="Home win odds.")
    predict_hybrid.add_argument("--draw-odds", type=float, help="Draw odds.")
    predict_hybrid.add_argument("--away-odds", type=float, help="Away win odds.")
    predict_hybrid.add_argument("--gm-ts", help="Optional gmTs to lookup odds from the CSV.")
    predict_hybrid.add_argument("--match-seq", type=int, help="Optional match_seq to lookup odds from the CSV.")

    predict = subparsers.add_parser("predict", help="Alias for predict-hybrid.")
    add_predict_common_args(predict)
    predict.add_argument("--home-odds", type=float, help="Home win odds.")
    predict.add_argument("--draw-odds", type=float, help="Draw odds.")
    predict.add_argument("--away-odds", type=float, help="Away win odds.")
    predict.add_argument("--gm-ts", help="Optional gmTs to lookup odds from the CSV.")
    predict.add_argument("--match-seq", type=int, help="Optional match_seq to lookup odds from the CSV.")

    predict_today = subparsers.add_parser(
        "predict-today",
        help="Generate a report for today's matches from a Betman page.",
    )
    predict_today.add_argument("--input", default="data/results.csv", help="Historical results CSV path.")
    predict_today.add_argument("--url", help="Upcoming matches page URL.")
    predict_today.add_argument("--html-file", help="Local HTML file path for upcoming matches.")
    predict_today.add_argument("--gmts", type=int, help="gmTs used to infer the year when the page omits it.")
    predict_today.add_argument("--browser", action="store_true", help="Render the page with Chromium.")
    predict_today.add_argument("--headed", action="store_true", help="Open the browser window for debugging.")
    predict_today.add_argument("--date", help="Target date in YYYY-MM-DD. Defaults to today.")
    predict_today.add_argument("--search-window", default=12, type=int, help="How many gmTs values to probe ahead.")
    predict_today.add_argument("--output-dir", default="reports", help="Directory for daily report files.")
    predict_today.add_argument("--cache-dir", default=".cache", help="Directory for trained model caches.")
    predict_today.add_argument("--recent-games", default=5, type=int, help="Recent game window for form features.")

    return parser


def add_predict_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default="data/results.csv", help="Input CSV path.")
    parser.add_argument("--home-team", required=True, help="Home team.")
    parser.add_argument("--away-team", required=True, help="Away team.")
    parser.add_argument("--sport", help="Optional sport filter.")
    parser.add_argument("--league", help="Optional league filter.")
    parser.add_argument("--recent-games", default=5, type=int, help="Recent game window for form features.")
    parser.add_argument("--cache-dir", default=".cache", help="Directory for trained model caches.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    scraper = BatmanScraper()

    if args.command == "collect":
        if not args.url and not args.html_file:
            parser.error("collect requires either --url or --html-file.")

        html = (
            scraper.fetch_html(args.url, use_browser=args.browser, headed=args.headed)
            if args.url
            else scraper.load_html_file(args.html_file)
        )
        frame = scraper.parse_results(html, gm_ts=args.gmts)
        output_path = scraper.save_results(frame, args.output)
        print(f"Collected: {output_path}")
        print(frame.head(5).to_string(index=False))
        return 0

    if args.command == "collect-range":
        frame = collect_gmts_range(
            scraper=scraper,
            start_gmts=args.start_gmts,
            end_gmts=args.end_gmts,
            gm_id=args.gm_id,
            stop_after_miss=args.stop_after_miss,
            use_browser=args.browser,
            headed=args.headed,
        )
        output_path = scraper.save_results(frame, args.output)
        print(f"Collected: {output_path}")
        print(f"Rows: {len(frame)}")
        print(frame.head(5).to_string(index=False))
        return 0

    if args.command == "stats":
        frame = load_results_csv(args.input)
        summary = team_summary(frame)
        print(summary.to_string(index=False))
        return 0

    if args.command == "sync-results":
        frame = sync_recent_results(
            scraper=scraper,
            input_path=args.input,
            output_path=args.output,
            gm_id=args.gm_id,
            use_browser=args.browser,
            headed=args.headed,
            probe_count=args.probe_count,
            stop_after_miss=args.stop_after_miss,
        )
        print(f"Synced rows: {len(frame)}")
        return 0

    if args.command == "predict-odds":
        prediction = resolve_odds_prediction(args)
        print_odds_prediction(prediction)
        return 0

    if args.command == "predict-form":
        frame = load_results_csv(args.input)
        predictor = load_or_fit_form_predictor(
            csv_path=args.input,
            frame=frame,
            recent_games=args.recent_games,
            cache_dir=args.cache_dir,
        )
        prediction = predictor.predict(
            frame,
            args.home_team,
            args.away_team,
            sport=args.sport,
            league=args.league,
        )
        print(f"Home win probability: {prediction.home_win_probability:.2%}")
        for key, value in prediction.features.items():
            print(f"{key}: {value:.4f}")
        return 0

    if args.command in {"predict-hybrid", "predict"}:
        frame = load_results_csv(args.input)
        predictor = load_or_fit_hybrid_predictor(
            csv_path=args.input,
            frame=frame,
            recent_games=args.recent_games,
            cache_dir=args.cache_dir,
        )
        home_odds, draw_odds, away_odds = resolve_odds_values(args, frame)
        prediction = predictor.predict(
            frame,
            args.home_team,
            args.away_team,
            home_odds=home_odds,
            draw_odds=draw_odds,
            away_odds=away_odds,
            sport=args.sport,
            league=args.league,
        )
        print(f"Odds home probability: {prediction.odds_home_probability:.2%}")
        print(f"Form home probability: {prediction.form_home_probability:.2%}")
        print(f"Hybrid home probability: {prediction.hybrid_home_probability:.2%}")
        print(f"Bookmaker margin: {prediction.bookmaker_margin:.2%}")
        return 0

    if args.command == "predict-today":
        if not args.url and not args.html_file:
            parser.error("predict-today requires either --url or --html-file.")

        historical = load_results_csv(args.input)
        predictor = load_or_fit_hybrid_predictor(
            csv_path=args.input,
            frame=historical,
            recent_games=args.recent_games,
            cache_dir=args.cache_dir,
        )
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else datetime.now().date()
        todays_matches = resolve_todays_matches(
            scraper=scraper,
            historical=historical,
            target_date=target_date,
            url=args.url,
            html_file=args.html_file,
            gm_ts=args.gmts,
            use_browser=args.browser,
            headed=args.headed,
            search_window=args.search_window,
        )
        report = build_daily_prediction_report(historical, todays_matches, predictor)
        csv_path, md_path = save_daily_report(report, args.output_dir, target_date)
        print(f"Daily report CSV: {csv_path}")
        print(f"Daily report Markdown: {md_path}")
        print(f"Matches found: {len(report)}")
        if not report.empty:
            print(report.to_string(index=False))
        return 0

    return 1


def resolve_odds_prediction(args: argparse.Namespace):
    if args.input and args.gm_ts and args.match_seq:
        frame = pd.read_csv(args.input)
        return calculate_implied_probabilities_from_row(frame, args.gm_ts, args.match_seq)

    if args.home_odds is None or args.away_odds is None:
        raise ValueError("predict-odds requires either CSV lookup args or direct odds values.")

    return calculate_implied_probabilities(
        home_odds=args.home_odds,
        draw_odds=args.draw_odds,
        away_odds=args.away_odds,
    )


def resolve_odds_values(
    args: argparse.Namespace,
    frame: pd.DataFrame,
) -> tuple[float, float | None, float]:
    if args.home_odds is not None and args.away_odds is not None:
        return float(args.home_odds), float(args.draw_odds) if args.draw_odds is not None else None, float(args.away_odds)

    if args.gm_ts and args.match_seq:
        target = frame[
            (frame["gm_ts"].astype(str) == str(args.gm_ts))
            & (frame["match_seq"] == args.match_seq)
        ]
        if target.empty:
            raise ValueError("No matching row for the given gmTs and match_seq.")
        row = target.iloc[0]
        if pd.isna(row.get("home_odds")) or pd.isna(row.get("away_odds")):
            raise ValueError("The selected row does not have usable odds values.")
        return (
            float(row["home_odds"]),
            float(row["draw_odds"]) if pd.notna(row.get("draw_odds")) else None,
            float(row["away_odds"]),
        )

    raise ValueError("predict-hybrid requires direct odds or a CSV row lookup.")


def print_odds_prediction(prediction) -> None:
    print(f"Home probability: {prediction.home_probability:.2%}")
    if prediction.draw_probability is not None:
        print(f"Draw probability: {prediction.draw_probability:.2%}")
    print(f"Away probability: {prediction.away_probability:.2%}")
    print(f"Bookmaker margin: {prediction.bookmaker_margin:.2%}")


def predictor_cache_path(
    *,
    cache_dir: str,
    csv_path: str,
    recent_games: int,
    predictor_name: str,
) -> Path:
    ensure_cache_dir(cache_dir)
    key = build_cache_key(
        {
            "predictor": predictor_name,
            "recent_games": recent_games,
            "csv": csv_signature(csv_path),
        }
    )
    return Path(cache_dir) / f"{predictor_name}_{key}.joblib"


def load_or_fit_form_predictor(
    *,
    csv_path: str,
    frame: pd.DataFrame,
    recent_games: int,
    cache_dir: str,
) -> FormPredictor:
    cache_path = predictor_cache_path(
        cache_dir=cache_dir,
        csv_path=csv_path,
        recent_games=recent_games,
        predictor_name="form",
    )
    cached = load_joblib(cache_path)
    if cached is not None:
        return cached

    predictor = FormPredictor(recent_games=recent_games)
    predictor.fit(frame)
    save_joblib(cache_path, predictor)
    return predictor


def load_or_fit_hybrid_predictor(
    *,
    csv_path: str,
    frame: pd.DataFrame,
    recent_games: int,
    cache_dir: str,
) -> HybridPredictor:
    cache_path = predictor_cache_path(
        cache_dir=cache_dir,
        csv_path=csv_path,
        recent_games=recent_games,
        predictor_name="hybrid",
    )
    cached = load_joblib(cache_path)
    if cached is not None:
        return cached

    predictor = HybridPredictor(recent_games=recent_games)
    predictor.fit(frame)
    save_joblib(cache_path, predictor)
    return predictor


def build_daily_prediction_report(
    historical: pd.DataFrame,
    todays_matches: pd.DataFrame,
    predictor: HybridPredictor,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, match in todays_matches.iterrows():
        if pd.isna(match.get("home_odds")) or pd.isna(match.get("away_odds")):
            continue

        prediction = predictor.predict(
            historical,
            str(match["home_team"]),
            str(match["away_team"]),
            home_odds=float(match["home_odds"]),
            draw_odds=float(match["draw_odds"]) if pd.notna(match.get("draw_odds")) else None,
            away_odds=float(match["away_odds"]),
            sport=str(match["sport"]) if pd.notna(match.get("sport")) else None,
            league=str(match["league"]) if pd.notna(match.get("league")) else None,
        )
        rows.append(
            {
                "played_at": match["played_at"],
                "close_at": match.get("close_at"),
                "sport": match.get("sport", ""),
                "league": match.get("league", ""),
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "status": match.get("status", ""),
                "home_odds": match.get("home_odds"),
                "draw_odds": match.get("draw_odds"),
                "away_odds": match.get("away_odds"),
                "odds_home_probability": prediction.odds_home_probability,
                "form_home_probability": prediction.form_home_probability,
                "hybrid_home_probability": prediction.hybrid_home_probability,
                "bookmaker_margin": prediction.bookmaker_margin,
                "venue": match.get("venue", ""),
                "match_seq": match.get("match_seq", 0),
                "gm_ts": match.get("gm_ts", ""),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "played_at",
                "close_at",
                "sport",
                "league",
                "home_team",
                "away_team",
                "status",
                "home_odds",
                "draw_odds",
                "away_odds",
                "odds_home_probability",
                "form_home_probability",
                "hybrid_home_probability",
                "bookmaker_margin",
                "venue",
                "match_seq",
                "gm_ts",
            ]
        )

    report = pd.DataFrame(rows)
    return report.sort_values(["played_at", "sport", "league", "match_seq"]).reset_index(drop=True)


def resolve_todays_matches(
    *,
    scraper: BatmanScraper,
    historical: pd.DataFrame,
    target_date,
    url: str | None,
    html_file: str | None,
    gm_ts: int | None,
    use_browser: bool,
    headed: bool,
    search_window: int,
) -> pd.DataFrame:
    if html_file:
        html = scraper.load_html_file(html_file)
        frame = scraper.parse_upcoming_matches(html, gm_ts=gm_ts)
        return filter_matches_for_date(frame, target_date)

    if not url:
        raise ValueError("An upcoming URL is required when html-file is not provided.")

    start_gmts = gm_ts or infer_upcoming_seed_gmts(historical, target_date.year)
    frames: list[pd.DataFrame] = []
    current = start_gmts
    found_target = False

    for _ in range(search_window):
        try:
            html = scraper.fetch_upcoming_html(
                current,
                use_browser=use_browser,
                headed=headed,
                base_url=url,
            )
            frame = scraper.parse_upcoming_matches(html, gm_ts=current)
        except (requests.RequestException, ValueError) as exc:
            print(f"[upcoming-miss] gmTs={format_gmts(current)} ({exc})")
            current = next_gmts(current)
            continue

        frame = frame.assign(gm_ts=format_gmts(current))
        todays = filter_matches_for_date(frame, target_date)
        if not todays.empty:
            print(f"[upcoming-hit] gmTs={format_gmts(current)} rows={len(todays)}")
            frames.append(todays)
            found_target = True
        else:
            min_date = pd.to_datetime(frame["played_at"]).dt.date.min()
            if found_target and min_date and min_date > target_date:
                break

        current = next_gmts(current)

    if not frames:
        return pd.DataFrame(
            columns=[
                "played_at",
                "close_at",
                "sport",
                "league",
                "game_type",
                "status",
                "home_team",
                "away_team",
                "venue",
                "match_seq",
                "home_odds",
                "draw_odds",
                "away_odds",
                "gm_ts",
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["gm_ts", "match_seq", "home_team", "away_team"])
    return combined.sort_values(["played_at", "gm_ts", "match_seq"]).reset_index(drop=True)


def infer_upcoming_seed_gmts(historical: pd.DataFrame, year: int) -> int:
    year_prefix = year % 100
    if "gm_ts" not in historical.columns:
        return int(f"{year_prefix:02d}0001")

    gm_series = historical["gm_ts"].dropna().astype(str).str.zfill(6)
    year_matches = gm_series[gm_series.str.startswith(f"{year_prefix:02d}")]
    if year_matches.empty:
        return int(f"{year_prefix:02d}0001")

    return int(year_matches.max())


def sync_recent_results(
    *,
    scraper: BatmanScraper,
    input_path: str,
    output_path: str,
    gm_id: str,
    use_browser: bool,
    headed: bool,
    probe_count: int,
    stop_after_miss: int,
) -> pd.DataFrame:
    existing = load_results_csv(input_path)
    if "gm_ts" not in existing.columns or existing["gm_ts"].dropna().empty:
        raise ValueError("The existing CSV must contain gm_ts values for incremental sync.")

    latest_gmts = int(existing["gm_ts"].dropna().astype(str).str.zfill(6).max())
    current = latest_gmts
    misses = 0
    frames: list[pd.DataFrame] = [existing.copy()]

    for _ in range(probe_count):
        print(f"[sync-check] gmTs={format_gmts(current)}")
        try:
            html = scraper.fetch_game_html(
                current,
                gm_id=gm_id,
                use_browser=use_browser,
                headed=headed,
            )
            frame = scraper.parse_results(html, gm_ts=current)
        except (requests.RequestException, ValueError) as exc:
            print(f"[sync-miss] gmTs={format_gmts(current)} ({exc})")
            misses += 1
            if misses >= stop_after_miss:
                break
            current = next_gmts(current)
            continue

        frame = frame.assign(gm_id=gm_id, gm_ts=format_gmts(current))
        frames.append(frame)
        misses = 0
        current = next_gmts(current)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=[
            "played_at",
            "league",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "gm_ts",
            "match_seq",
        ]
    )
    combined = combined.sort_values(["played_at", "gm_ts", "match_seq"]).reset_index(drop=True)
    scraper.save_results(combined, output_path)
    print(f"Synced file: {output_path}")
    return combined


def collect_gmts_range(
    scraper: BatmanScraper,
    start_gmts: int,
    end_gmts: int,
    gm_id: str,
    stop_after_miss: int,
    use_browser: bool,
    headed: bool,
) -> pd.DataFrame:
    start_year, _ = parse_gmts(start_gmts)
    end_year, _ = parse_gmts(end_gmts)
    if start_gmts > end_gmts:
        raise ValueError("start-gmts cannot be greater than end-gmts.")
    if start_year > end_year:
        raise ValueError("start-gmts cannot be in a later year than end-gmts.")

    current = start_gmts
    frames: list[pd.DataFrame] = []
    hits_in_year = 0
    misses_in_year = 0

    while current <= end_gmts:
        print(f"[check] gmTs={format_gmts(current)}")
        try:
            html = scraper.fetch_game_html(
                current,
                gm_id=gm_id,
                use_browser=use_browser,
                headed=headed,
            )
            frame = scraper.parse_results(html, gm_ts=current)
        except (requests.RequestException, ValueError) as exc:
            print(f"[miss] gmTs={format_gmts(current)} ({exc})")
            misses_in_year += 1
            if hits_in_year > 0 and misses_in_year >= stop_after_miss:
                current = next_year_gmts(current)
                hits_in_year = 0
                misses_in_year = 0
                continue
            current = next_gmts(current)
            continue

        frame = frame.assign(gm_id=gm_id, gm_ts=format_gmts(current))
        frames.append(frame)
        hits_in_year += 1
        misses_in_year = 0
        current = next_gmts(current)

    if not frames:
        raise ValueError("No rows were collected in the requested gmTs range.")

    valid_frames = [frame for frame in frames if not frame.empty and not frame.isna().all(axis=None)]
    if not valid_frames:
        raise ValueError("Collected frames were empty after filtering invalid rows.")

    combined = pd.concat(valid_frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=[
            "played_at",
            "league",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "gm_ts",
            "match_seq",
        ]
    )
    return combined.sort_values(["played_at", "gm_ts", "match_seq"]).reset_index(drop=True)
