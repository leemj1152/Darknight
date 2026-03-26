from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .cache import build_cache_key, csv_signature, ensure_cache_dir, load_joblib, save_joblib
from .odds import calculate_implied_probabilities, calculate_implied_probabilities_from_row
from .predictor import FormPredictor, HybridPredictor
from .reporting import filter_matches_for_date, save_daily_report, save_round_report
from .simulation import add_strategy_columns, simulate_betting_agents
from .scraper import (
    BatmanScraper,
    format_gmts,
    load_results_csv,
    next_gmts,
    next_year_gmts,
    parse_gmts,
)
from .stats import team_summary
from .tracking import settle_prediction_reports


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

    backtest = subparsers.add_parser(
        "backtest",
        help="Evaluate form, odds, and hybrid models on historical data.",
    )
    backtest.add_argument("--input", default="data/results.csv", help="Input CSV path.")
    backtest.add_argument("--sport", help="Optional sport filter.")
    backtest.add_argument("--league", help="Optional league filter.")
    backtest.add_argument("--recent-games", default=5, type=int, help="Recent game window for form features.")
    backtest.add_argument("--test-ratio", default=0.2, type=float, help="Fraction of rows used for the time-based test split.")
    backtest.add_argument("--output-dir", default="analysis", help="Directory for saved backtest outputs.")
    backtest.add_argument("--by-league", action="store_true", help="Also run backtests separately for each league in the filtered dataset.")

    simulate = subparsers.add_parser(
        "simulate-bets",
        help="Run mock betting for three agents over the recent historical window.",
    )
    simulate.add_argument("--input", default="data/results.csv", help="Input CSV path.")
    simulate.add_argument("--sport", help="Optional sport filter.")
    simulate.add_argument("--league", help="Optional league filter.")
    simulate.add_argument("--recent-games", default=5, type=int, help="Recent game window for form features.")
    simulate.add_argument("--lookback-days", default=30, type=int, help="Historical days to simulate.")
    simulate.add_argument("--edge-threshold", default=0.05, type=float, help="Minimum edge required for form/hybrid bets.")
    simulate.add_argument("--stake", default=1.0, type=float, help="Flat stake per simulated bet.")
    simulate.add_argument("--cache-dir", default=".cache", help="Directory for trained model caches.")
    simulate.add_argument("--output-dir", default="analysis", help="Directory for saved simulation outputs.")

    settle_reports = subparsers.add_parser(
        "settle-reports",
        help="Match saved prediction reports against finished results and track hit rate / ROI.",
    )
    settle_reports.add_argument("--input", default="data/results.csv", help="Historical results CSV path.")
    settle_reports.add_argument("--reports-dir", default="reports", help="Directory containing prediction report CSV files.")
    settle_reports.add_argument("--output-dir", default="analysis", help="Directory for settled outputs.")

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

    probe_round = subparsers.add_parser(
        "probe-round",
        help="Resolve the currently active round and print its close window.",
    )
    probe_round.add_argument("--input", default="data/results.csv", help="Historical results CSV path.")
    probe_round.add_argument("--url", required=True, help="Upcoming matches page base URL.")
    probe_round.add_argument("--gmts", type=int, help="Optional fixed gmTs.")
    probe_round.add_argument("--browser", action="store_true", help="Render the page with Chromium.")
    probe_round.add_argument("--headed", action="store_true", help="Open the browser window for debugging.")
    probe_round.add_argument("--search-window", default=6, type=int, help="How many gmTs values to probe ahead.")

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
    predict_today.add_argument("--analysis-dir", default="analysis", help="Directory containing backtest league summaries.")
    predict_today.add_argument("--cache-dir", default=".cache", help="Directory for trained model caches.")
    predict_today.add_argument("--recent-games", default=5, type=int, help="Recent game window for form features.")

    predict_round = subparsers.add_parser(
        "predict-round",
        help="Generate a report for the current or requested round.",
    )
    predict_round.add_argument("--input", default="data/results.csv", help="Historical results CSV path.")
    predict_round.add_argument("--url", required=True, help="Upcoming matches page base URL.")
    predict_round.add_argument("--gmts", type=int, help="Requested round gmTs. If omitted, infer the active round.")
    predict_round.add_argument("--browser", action="store_true", help="Render the page with Chromium.")
    predict_round.add_argument("--headed", action="store_true", help="Open the browser window for debugging.")
    predict_round.add_argument("--search-window", default=6, type=int, help="How many gmTs values to probe ahead.")
    predict_round.add_argument("--output-dir", default="reports", help="Directory for round report files.")
    predict_round.add_argument("--analysis-dir", default="analysis", help="Directory containing backtest league summaries.")
    predict_round.add_argument("--cache-dir", default=".cache", help="Directory for trained model caches.")
    predict_round.add_argument("--recent-games", default=5, type=int, help="Recent game window for form features.")

    predict_all = subparsers.add_parser(
        "predict-all",
        help="Generate both today's report and the current round report.",
    )
    predict_all.add_argument("--input", default="data/results.csv", help="Historical results CSV path.")
    predict_all.add_argument("--url", required=True, help="Upcoming matches page base URL.")
    predict_all.add_argument("--gmts", type=int, help="Optional fixed gmTs for the round report.")
    predict_all.add_argument("--browser", action="store_true", help="Render the page with Chromium.")
    predict_all.add_argument("--headed", action="store_true", help="Open the browser window for debugging.")
    predict_all.add_argument("--date", help="Target date in YYYY-MM-DD. Defaults to today.")
    predict_all.add_argument("--search-window", default=12, type=int, help="How many gmTs values to probe ahead.")
    predict_all.add_argument("--output-dir", default="reports", help="Directory for report files.")
    predict_all.add_argument("--analysis-dir", default="analysis", help="Directory containing backtest league summaries.")
    predict_all.add_argument("--cache-dir", default=".cache", help="Directory for trained model caches.")
    predict_all.add_argument("--recent-games", default=5, type=int, help="Recent game window for form features.")

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

    if args.command == "probe-round":
        historical = load_results_csv(args.input)
        round_matches, round_gmts = resolve_round_matches(
            scraper=scraper,
            historical=historical,
            url=args.url,
            gm_ts=args.gmts,
            use_browser=args.browser,
            headed=args.headed,
            search_window=args.search_window,
        )
        close_times = pd.to_datetime(round_matches["close_at"], errors="coerce")
        played_times = pd.to_datetime(round_matches["played_at"], errors="coerce")
        print(f"gmTs: {round_gmts}")
        print(f"rows: {len(round_matches)}")
        print(f"close_at_max: {close_times.max().isoformat() if close_times.notna().any() else ''}")
        print(f"played_at_min: {played_times.min().isoformat() if played_times.notna().any() else ''}")
        print(f"played_at_max: {played_times.max().isoformat() if played_times.notna().any() else ''}")
        return 0

    if args.command == "predict-odds":
        prediction = resolve_odds_prediction(args)
        print_odds_prediction(prediction)
        return 0

    if args.command == "backtest":
        print("[backtest] loading results")
        frame = load_results_csv(args.input)
        scoped = frame.copy()
        if args.sport:
            scoped = scoped[scoped["sport"] == args.sport]
        if args.league:
            scoped = scoped[scoped["league"] == args.league]
        print(f"[backtest] rows after filter: {len(scoped)}")
        results = run_backtest(scoped, recent_games=args.recent_games, test_ratio=args.test_ratio)
        for name, metrics in results["metrics"].items():
            print(
                f"{name}: accuracy={metrics['accuracy']:.4f} "
                f"brier={metrics['brier']:.4f} log_loss={metrics['log_loss']:.4f} "
                f"rows={metrics['rows']}"
            )
        print("\nForm importance")
        print(results["form_importance"].to_string(index=False))
        print("\nHybrid importance")
        print(results["hybrid_importance"].to_string(index=False))
        output_paths = save_backtest_outputs(
            output_dir=args.output_dir,
            prefix=build_backtest_prefix(args.sport, args.league),
            results=results,
        )
        print(f"\nSaved summary: {output_paths['summary']}")
        print(f"Saved form importance: {output_paths['form_importance']}")
        print(f"Saved hybrid importance: {output_paths['hybrid_importance']}")
        if args.by_league:
            league_summary = run_backtest_by_league(scoped, recent_games=args.recent_games, test_ratio=args.test_ratio)
            league_path = save_league_backtest_summary(
                output_dir=args.output_dir,
                prefix=build_backtest_prefix(args.sport, args.league),
                summary=league_summary,
            )
            print(f"Saved league summary: {league_path}")
            if not league_summary.empty:
                print("\nLeague summary")
                print(league_summary.to_string(index=False))
        return 0

    if args.command == "simulate-bets":
        print("[simulate] loading results")
        frame = load_results_csv(args.input)
        print("[simulate] loading cached predictors")
        form_predictor = load_or_fit_form_predictor(
            csv_path=args.input,
            frame=frame,
            recent_games=args.recent_games,
            cache_dir=args.cache_dir,
        )
        hybrid_predictor = load_or_fit_hybrid_predictor(
            csv_path=args.input,
            frame=frame,
            recent_games=args.recent_games,
            cache_dir=args.cache_dir,
        )
        print("[simulate] running three-agent mock betting")
        results = simulate_betting_agents(
            frame,
            recent_games=args.recent_games,
            lookback_days=args.lookback_days,
            edge_threshold=args.edge_threshold,
            stake=args.stake,
            sport=args.sport,
            league=args.league,
            form_predictor=form_predictor,
            hybrid_predictor=hybrid_predictor,
        )
        summary_path, bets_path, daily_path = save_simulation_outputs(
            output_dir=args.output_dir,
            prefix=build_simulation_prefix(args.sport, args.league),
            results=results,
        )
        print(results["summary"].to_string(index=False))
        print(f"Saved simulation summary: {summary_path}")
        print(f"Saved simulation bets: {bets_path}")
        print(f"Saved simulation daily: {daily_path}")
        return 0

    if args.command == "settle-reports":
        print("[settle] loading results")
        frame = load_results_csv(args.input)
        outputs = settle_prediction_reports(
            results_frame=frame,
            reports_dir=args.reports_dir,
            output_dir=args.output_dir,
        )
        print(f"Settled predictions: {outputs.settled_path}")
        print(f"Settled summary: {outputs.summary_path}")
        summary = pd.read_csv(outputs.summary_path)
        if not summary.empty:
            print(summary.to_string(index=False))
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

        print("[predict-today] loading historical results")
        historical = load_results_csv(args.input)
        print("[predict-today] loading or training hybrid model")
        predictor = load_or_fit_hybrid_predictor(
            csv_path=args.input,
            frame=historical,
            recent_games=args.recent_games,
            cache_dir=args.cache_dir,
        )
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else datetime.now().date()
        print(f"[predict-today] resolving matches for {target_date.isoformat()}")
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
        print(f"[predict-today] matches resolved: {len(todays_matches)}")
        print("[predict-today] building prediction report")
        report = build_daily_prediction_report(
            historical,
            todays_matches,
            predictor,
            league_scores=load_league_backtest_summary(args.analysis_dir),
        )
        csv_path, md_path = save_daily_report(report, args.output_dir, target_date)
        print(f"Daily report CSV: {csv_path}")
        print(f"Daily report Markdown: {md_path}")
        print(f"Matches found: {len(report)}")
        if not report.empty:
            print(report.to_string(index=False))
        return 0

    if args.command == "predict-round":
        print("[predict-round] loading historical results")
        historical = load_results_csv(args.input)
        print("[predict-round] loading or training hybrid model")
        predictor = load_or_fit_hybrid_predictor(
            csv_path=args.input,
            frame=historical,
            recent_games=args.recent_games,
            cache_dir=args.cache_dir,
        )
        print("[predict-round] resolving round matches")
        round_matches, round_gmts = resolve_round_matches(
            scraper=scraper,
            historical=historical,
            url=args.url,
            gm_ts=args.gmts,
            use_browser=args.browser,
            headed=args.headed,
            search_window=args.search_window,
        )
        print(f"[predict-round] round gmTs={round_gmts} matches={len(round_matches)}")
        print("[predict-round] building prediction report")
        report = build_daily_prediction_report(
            historical,
            round_matches,
            predictor,
            league_scores=load_league_backtest_summary(args.analysis_dir),
        )
        csv_path, md_path = save_round_report(report, args.output_dir, round_gmts)
        print(f"Round report CSV: {csv_path}")
        print(f"Round report Markdown: {md_path}")
        print(f"gmTs: {round_gmts}")
        print(f"Matches found: {len(report)}")
        if not report.empty:
            print(report.to_string(index=False))
        return 0

    if args.command == "predict-all":
        print("[predict-all] loading historical results")
        historical = load_results_csv(args.input)
        print("[predict-all] loading or training hybrid model")
        predictor = load_or_fit_hybrid_predictor(
            csv_path=args.input,
            frame=historical,
            recent_games=args.recent_games,
            cache_dir=args.cache_dir,
        )
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else datetime.now().date()

        print(f"[predict-all] resolving daily matches for {target_date.isoformat()}")
        todays_matches = resolve_todays_matches(
            scraper=scraper,
            historical=historical,
            target_date=target_date,
            url=args.url,
            html_file=None,
            gm_ts=args.gmts,
            use_browser=args.browser,
            headed=args.headed,
            search_window=args.search_window,
        )
        print(f"[predict-all] daily matches resolved: {len(todays_matches)}")
        print("[predict-all] building daily report")
        league_scores = load_league_backtest_summary(args.analysis_dir)
        daily_report = build_daily_prediction_report(
            historical,
            todays_matches,
            predictor,
            league_scores=league_scores,
        )
        daily_csv, daily_md = save_daily_report(daily_report, args.output_dir, target_date)

        print("[predict-all] resolving round matches")
        round_matches, round_gmts = resolve_round_matches(
            scraper=scraper,
            historical=historical,
            url=args.url,
            gm_ts=args.gmts,
            use_browser=args.browser,
            headed=args.headed,
            search_window=args.search_window,
        )
        print(f"[predict-all] round gmTs={round_gmts} matches={len(round_matches)}")
        print("[predict-all] building round report")
        round_report = build_daily_prediction_report(
            historical,
            round_matches,
            predictor,
            league_scores=league_scores,
        )
        round_csv, round_md = save_round_report(round_report, args.output_dir, round_gmts)

        print(f"Daily report CSV: {daily_csv}")
        print(f"Daily report Markdown: {daily_md}")
        print(f"Daily matches found: {len(daily_report)}")
        print(f"Round report CSV: {round_csv}")
        print(f"Round report Markdown: {round_md}")
        print(f"Round gmTs: {round_gmts}")
        print(f"Round matches found: {len(round_report)}")
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

    fallback = load_latest_predictor_cache(cache_dir=cache_dir, predictor_name="form")
    if fallback is not None:
        save_joblib(cache_path, fallback)
        return fallback

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

    fallback = load_latest_predictor_cache(cache_dir=cache_dir, predictor_name="hybrid")
    if fallback is not None:
        save_joblib(cache_path, fallback)
        return fallback

    predictor = HybridPredictor(recent_games=recent_games)
    predictor.fit(frame)
    save_joblib(cache_path, predictor)
    return predictor


def load_latest_predictor_cache(
    *,
    cache_dir: str,
    predictor_name: str,
):
    cache_path = Path(cache_dir)
    candidates = sorted(
        cache_path.glob(f"{predictor_name}_*.joblib"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        cached = load_joblib(candidate)
        if cached is not None:
            print(f"[cache] using fallback {candidate.name}")
            return cached
    return None


def build_daily_prediction_report(
    historical: pd.DataFrame,
    todays_matches: pd.DataFrame,
    predictor: HybridPredictor,
    *,
    league_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(todays_matches)
    for index, (_, match) in enumerate(todays_matches.iterrows(), start=1):
        if index == 1 or index % 10 == 0 or index == total:
            print(f"[report] building predictions {index}/{total}")
        if pd.isna(match.get("home_odds")) or pd.isna(match.get("away_odds")):
            continue

        implied = calculate_implied_probabilities(
            home_odds=float(match["home_odds"]),
            draw_odds=float(match["draw_odds"]) if pd.notna(match.get("draw_odds")) else None,
            away_odds=float(match["away_odds"]),
        )

        prediction = predictor.predict(
            historical,
            str(match["home_team"]),
            str(match["away_team"]),
            home_odds=float(match["home_odds"]),
            draw_odds=float(match["draw_odds"]) if pd.notna(match.get("draw_odds")) else None,
            away_odds=float(match["away_odds"]),
            handicap_line=float(match.get("handicap_line", 0.0) or 0.0),
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
                "game_type": match.get("game_type", ""),
                "handicap_line": match.get("handicap_line"),
                "home_odds": match.get("home_odds"),
                "draw_odds": match.get("draw_odds"),
                "away_odds": match.get("away_odds"),
                "odds_home_probability": implied.home_probability,
                "odds_draw_probability": implied.draw_probability,
                "odds_away_probability": implied.away_probability,
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
                "game_type",
                "handicap_line",
                "home_odds",
                "draw_odds",
                "away_odds",
                "odds_home_probability",
                "odds_draw_probability",
                "odds_away_probability",
                "form_home_probability",
                "hybrid_home_probability",
                "bookmaker_margin",
                "venue",
                "match_seq",
                "gm_ts",
            ]
        )

    report = pd.DataFrame(rows)
    report = report.sort_values(["played_at", "sport", "league", "match_seq"]).reset_index(drop=True)
    return add_strategy_columns(
        report,
        edge_threshold=0.05,
        league_scores=league_scores,
        top_pick_count=5,
    )


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
                "handicap_line",
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


def resolve_round_matches(
    *,
    scraper: BatmanScraper,
    historical: pd.DataFrame,
    url: str,
    gm_ts: int | None,
    use_browser: bool,
    headed: bool,
    search_window: int,
) -> tuple[pd.DataFrame, str]:
    if gm_ts is not None:
        html = scraper.fetch_upcoming_html(
            gm_ts,
            use_browser=use_browser,
            headed=headed,
            base_url=url,
        )
        frame = scraper.parse_upcoming_matches(html, gm_ts=gm_ts).assign(gm_ts=format_gmts(gm_ts))
        return frame, format_gmts(gm_ts)

    start_gmts = infer_upcoming_seed_gmts(historical, datetime.now().year)
    current = start_gmts
    now = datetime.now()
    fallback_frame: pd.DataFrame | None = None
    fallback_gmts: str | None = None

    for _ in range(search_window):
        try:
            html = scraper.fetch_upcoming_html(
                current,
                use_browser=use_browser,
                headed=headed,
                base_url=url,
            )
            frame = scraper.parse_upcoming_matches(html, gm_ts=current).assign(gm_ts=format_gmts(current))
        except (requests.RequestException, ValueError) as exc:
            print(f"[round-miss] gmTs={format_gmts(current)} ({exc})")
            current = next_gmts(current)
            continue

        if fallback_frame is None:
            fallback_frame = frame
            fallback_gmts = format_gmts(current)

        close_times = pd.to_datetime(frame["close_at"], errors="coerce")
        if close_times.notna().any() and close_times.max() >= now:
            print(f"[round-hit] gmTs={format_gmts(current)} rows={len(frame)}")
            return frame, format_gmts(current)

        current = next_gmts(current)

    if fallback_frame is not None and fallback_gmts is not None:
        return fallback_frame, fallback_gmts

    raise ValueError("Could not resolve an active round from upcoming pages.")


def run_backtest(
    frame: pd.DataFrame,
    *,
    recent_games: int,
    test_ratio: float,
) -> dict[str, object]:
    if frame.empty:
        raise ValueError("No rows available for backtest.")

    ordered = frame.sort_values("played_at").reset_index(drop=True)
    split_index = max(1, int(len(ordered) * (1 - test_ratio)))
    train = ordered.iloc[:split_index].reset_index(drop=True)
    test = ordered.iloc[split_index:].reset_index(drop=True)
    if test.empty:
        raise ValueError("Test split is empty. Increase the dataset or test ratio.")

    print(f"[backtest] train rows: {len(train)} test rows: {len(test)}")
    print("[backtest] training form model")

    form_predictor = FormPredictor(recent_games=recent_games)
    form_predictor.fit(train)
    print("[backtest] training hybrid model")
    hybrid_predictor = HybridPredictor(recent_games=recent_games)
    hybrid_predictor.fit(train)

    form_rows: list[tuple[int, float]] = []
    hybrid_rows: list[tuple[int, float]] = []
    odds_rows: list[tuple[int, float]] = []

    total = len(test)
    for index, (_, row) in enumerate(test.iterrows(), start=1):
        if index == 1 or index % 250 == 0 or index == total:
            print(f"[backtest] scoring rows {index}/{total}")
        actual = int(row["home_score"] > row["away_score"])

        form_prediction = form_predictor.predict(
            train,
            row["home_team"],
            row["away_team"],
            sport=row.get("sport"),
            league=row.get("league"),
        )
        form_rows.append((actual, form_prediction.home_win_probability))

        if pd.notna(row.get("home_odds")) and pd.notna(row.get("away_odds")):
            odds_prediction = calculate_implied_probabilities(
                home_odds=float(row["home_odds"]),
                away_odds=float(row["away_odds"]),
                draw_odds=float(row["draw_odds"]) if pd.notna(row.get("draw_odds")) else None,
            )
            odds_rows.append((actual, odds_prediction.home_probability))

            hybrid_prediction = hybrid_predictor.predict(
                train,
                row["home_team"],
                row["away_team"],
                home_odds=float(row["home_odds"]),
                away_odds=float(row["away_odds"]),
                draw_odds=float(row["draw_odds"]) if pd.notna(row.get("draw_odds")) else None,
                sport=row.get("sport"),
                league=row.get("league"),
            )
            hybrid_rows.append((actual, hybrid_prediction.hybrid_home_probability))

    return {
        "metrics": {
            "form": classification_metrics(form_rows),
            "odds": classification_metrics(odds_rows),
            "hybrid": classification_metrics(hybrid_rows),
        },
        "form_importance": form_predictor.feature_importance(),
        "hybrid_importance": hybrid_predictor.feature_importance(),
    }


def classification_metrics(rows: list[tuple[int, float]]) -> dict[str, float | int]:
    if not rows:
        return {"accuracy": float("nan"), "brier": float("nan"), "log_loss": float("nan"), "rows": 0}

    y_true = [row[0] for row in rows]
    y_prob = [min(max(row[1], 1e-6), 1 - 1e-6) for row in rows]
    y_pred = [int(prob >= 0.5) for prob in y_prob]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "rows": len(rows),
    }


def build_backtest_prefix(sport: str | None, league: str | None) -> str:
    parts = ["backtest"]
    if sport:
        parts.append(sanitize_filename(sport))
    if league:
        parts.append(sanitize_filename(league))
    return "_".join(parts)


def sanitize_filename(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_") or "all"


def build_simulation_prefix(sport: str | None, league: str | None) -> str:
    parts = ["simulation"]
    if sport:
        parts.append(sanitize_filename(sport))
    if league:
        parts.append(sanitize_filename(league))
    return "_".join(parts)


def save_backtest_outputs(
    *,
    output_dir: str,
    prefix: str,
    results: dict[str, object],
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_frame = pd.DataFrame(
        [
            {
                "model": model_name,
                **metrics,
            }
            for model_name, metrics in results["metrics"].items()
        ]
    )
    summary_path = output_path / f"{prefix}_summary.csv"
    form_path = output_path / f"{prefix}_form_importance.csv"
    hybrid_path = output_path / f"{prefix}_hybrid_importance.csv"

    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    results["form_importance"].to_csv(form_path, index=False, encoding="utf-8-sig")
    results["hybrid_importance"].to_csv(hybrid_path, index=False, encoding="utf-8-sig")

    return {
        "summary": summary_path,
        "form_importance": form_path,
        "hybrid_importance": hybrid_path,
    }


def save_simulation_outputs(
    *,
    output_dir: str,
    prefix: str,
    results: dict[str, pd.DataFrame],
) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / f"{prefix}_summary.csv"
    bets_path = output_path / f"{prefix}_bets.csv"
    daily_path = output_path / f"{prefix}_daily.csv"
    results["summary"].to_csv(summary_path, index=False, encoding="utf-8-sig")
    results["bets"].to_csv(bets_path, index=False, encoding="utf-8-sig")
    results["daily"].to_csv(daily_path, index=False, encoding="utf-8-sig")
    return summary_path, bets_path, daily_path


def load_league_backtest_summary(analysis_dir: str) -> pd.DataFrame | None:
    path = Path(analysis_dir) / "backtest_league_summary.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def run_backtest_by_league(
    frame: pd.DataFrame,
    *,
    recent_games: int,
    test_ratio: float,
) -> pd.DataFrame:
    if frame.empty or "league" not in frame.columns:
        return pd.DataFrame()

    summaries: list[dict[str, object]] = []
    grouped = frame.groupby("league", dropna=False)
    for league_name, group in grouped:
        if len(group) < 30:
            continue
        try:
            result = run_backtest(group, recent_games=recent_games, test_ratio=test_ratio)
        except ValueError:
            continue

        for model_name, metrics in result["metrics"].items():
            summaries.append(
                {
                    "league": league_name if pd.notna(league_name) else "",
                    "model": model_name,
                    **metrics,
                }
            )

    if not summaries:
        return pd.DataFrame()

    return pd.DataFrame(summaries).sort_values(["league", "model"]).reset_index(drop=True)


def save_league_backtest_summary(
    *,
    output_dir: str,
    prefix: str,
    summary: pd.DataFrame,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"{prefix}_league_summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    return path


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
