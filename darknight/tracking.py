from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPORT_GLOBS = [
    "daily_predictions_*.csv",
    "round_predictions_*.csv",
    "round_predictions_*_enriched.csv",
]


@dataclass(slots=True)
class SettledReportOutputs:
    settled_path: Path
    summary_path: Path


def settle_prediction_reports(
    *,
    results_frame: pd.DataFrame,
    reports_dir: str | Path,
    output_dir: str | Path,
) -> SettledReportOutputs:
    reports_path = Path(reports_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = normalize_results(results_frame)
    report_files = discover_report_files(reports_path)
    settled_frames: list[pd.DataFrame] = []

    for report_file in report_files:
        try:
            report = pd.read_csv(report_file)
        except Exception:
            continue
        if report.empty or "home_team" not in report.columns or "away_team" not in report.columns:
            continue
        settled = settle_single_report(report, results, report_file.name)
        if not settled.empty:
            settled_frames.append(settled)

    if settled_frames:
        settled_frame = pd.concat(settled_frames, ignore_index=True)
    else:
        settled_frame = pd.DataFrame(columns=settled_columns())

    summary = summarize_settled_reports(settled_frame)
    settled_path = output_path / "settled_predictions.csv"
    summary_path = output_path / "settled_summary.csv"
    settled_frame.to_csv(settled_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return SettledReportOutputs(settled_path=settled_path, summary_path=summary_path)


def discover_report_files(reports_path: Path) -> list[Path]:
    seen: dict[Path, None] = {}
    for pattern in REPORT_GLOBS:
        for file_path in reports_path.glob(pattern):
            if file_path.is_file():
                seen[file_path] = None
    return sorted(seen.keys())


def normalize_results(frame: pd.DataFrame) -> pd.DataFrame:
    results = frame.copy()
    results["played_at"] = pd.to_datetime(results["played_at"], errors="coerce")
    results["gm_ts"] = results["gm_ts"].astype(str).str.zfill(6)
    if "handicap_line" not in results.columns:
        results["handicap_line"] = 0.0
    results["handicap_line"] = pd.to_numeric(results["handicap_line"], errors="coerce").fillna(0.0)
    results["actual_outcome"] = results.apply(compute_actual_outcome, axis=1)
    return results


def settle_single_report(report: pd.DataFrame, results: pd.DataFrame, report_name: str) -> pd.DataFrame:
    working = report.copy()
    working["played_at"] = pd.to_datetime(working["played_at"], errors="coerce")
    if "gm_ts" in working.columns:
        working["gm_ts"] = working["gm_ts"].astype(str).str.zfill(6)
    if "match_seq" in working.columns:
        working["match_seq"] = pd.to_numeric(working["match_seq"], errors="coerce")

    merged = working.merge(
        results[
            [
                "gm_ts",
                "match_seq",
                "played_at",
                "sport",
                "league",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "handicap_line",
                "actual_outcome",
            ]
        ],
        on=["gm_ts", "match_seq", "home_team", "away_team"],
        how="left",
        suffixes=("", "_result"),
    )

    unresolved = merged["actual_outcome"].isna()
    if unresolved.any():
        fallback = working.loc[unresolved].merge(
            results[
                [
                    "played_at",
                    "sport",
                    "league",
                    "home_team",
                    "away_team",
                    "home_score",
                    "away_score",
                    "handicap_line",
                    "actual_outcome",
                ]
            ],
            on=["played_at", "sport", "league", "home_team", "away_team"],
            how="left",
        )
        for column in ("home_score", "away_score", "actual_outcome"):
            merged.loc[unresolved, column] = fallback[column].values

    if merged["actual_outcome"].notna().sum() == 0:
        return pd.DataFrame(columns=settled_columns())

    merged["report_name"] = report_name
    merged["recommended_hit"] = merged.apply(
        lambda row: evaluate_pick_hit(row.get("recommended_side"), row.get("actual_outcome")),
        axis=1,
    )
    merged["recommended_profit"] = merged.apply(compute_recommended_profit, axis=1)
    merged["recommended_full_side"] = merged.apply(resolve_full_recommended_side, axis=1)
    merged["recommended_full_hit"] = merged.apply(
        lambda row: evaluate_pick_hit(row.get("recommended_full_side"), row.get("actual_outcome")),
        axis=1,
    )
    merged["odds_pick_hit"] = merged.apply(lambda row: evaluate_pick_hit(row.get("odds_pick"), row.get("actual_outcome")), axis=1)
    merged["form_pick_hit"] = merged.apply(lambda row: evaluate_pick_hit(row.get("form_pick"), row.get("actual_outcome")), axis=1)
    merged["hybrid_pick_hit"] = merged.apply(lambda row: evaluate_pick_hit(row.get("hybrid_pick"), row.get("actual_outcome")), axis=1)
    return merged.reindex(columns=settled_columns())


def compute_actual_outcome(row: pd.Series) -> str:
    adjusted_home_score = float(row["home_score"]) + float(row.get("handicap_line", 0.0) or 0.0)
    if adjusted_home_score > float(row["away_score"]):
        return "HOME"
    if float(row["away_score"]) > adjusted_home_score:
        return "AWAY"
    return "DRAW"


def evaluate_pick_hit(predicted_side: object, actual_outcome: object) -> object:
    if pd.isna(predicted_side) or str(predicted_side).upper() == "SKIP":
        return None
    return int(str(predicted_side).upper() == str(actual_outcome).upper())


def compute_recommended_profit(row: pd.Series) -> float | None:
    side = row.get("recommended_side")
    if pd.isna(side) or str(side).upper() == "SKIP":
        return None
    if str(side).upper() == "HOME":
        odds = row.get("home_odds")
    elif str(side).upper() == "AWAY":
        odds = row.get("away_odds")
    else:
        return None
    if pd.isna(odds):
        return None
    return round(float(odds) - 1.0, 4) if evaluate_pick_hit(side, row.get("actual_outcome")) == 1 else -1.0


def resolve_full_recommended_side(row: pd.Series) -> str:
    side = row.get("recommended_side")
    if pd.notna(side) and str(side).upper() != "SKIP":
        return str(side).upper()

    model = str(row.get("recommended_model", "")).upper()
    if model == "FORM":
        home_probability = float(row.get("form_home_probability", 0.5))
        away_probability = float(row.get("form_away_probability", 1.0 - home_probability))
        return "HOME" if home_probability >= away_probability else "AWAY"
    if model == "HYBRID":
        home_probability = float(row.get("hybrid_home_probability", 0.5))
        away_probability = float(row.get("hybrid_away_probability", 1.0 - home_probability))
        return "HOME" if home_probability >= away_probability else "AWAY"

    hybrid_home = float(row.get("hybrid_home_probability", 0.5))
    hybrid_away = float(row.get("hybrid_away_probability", 1.0 - hybrid_home))
    return "HOME" if hybrid_home >= hybrid_away else "AWAY"


def summarize_settled_reports(settled: pd.DataFrame) -> pd.DataFrame:
    if settled.empty:
        return pd.DataFrame(
            columns=[
                "bucket",
                "name",
                "settled_rows",
                "odds_accuracy",
                "form_accuracy",
                "hybrid_accuracy",
                "recommended_full_accuracy",
                "recommended_bets",
                "recommended_hits",
                "recommended_accuracy",
                "recommended_profit",
                "recommended_roi",
            ]
        )

    summaries: list[dict[str, object]] = []
    summaries.extend(summarize_bucket(settled, "overall", "all"))
    for report_name, group in settled.groupby("report_name", dropna=False):
        summaries.extend(summarize_bucket(group, "report", str(report_name)))
    return pd.DataFrame(summaries)


def summarize_bucket(frame: pd.DataFrame, bucket: str, name: str) -> list[dict[str, object]]:
    recommended = frame[frame["recommended_side"].fillna("SKIP") != "SKIP"].copy()
    recommended_hits = recommended["recommended_hit"].dropna().astype(int)
    recommended_profit = recommended["recommended_profit"].dropna().sum() if not recommended.empty else 0.0
    recommended_bets = int(len(recommended))
    recommended_accuracy = float(recommended_hits.mean()) if not recommended_hits.empty else 0.0
    recommended_roi = float(recommended_profit / recommended_bets) if recommended_bets else 0.0
    odds_accuracy = summarize_hit_column(frame, "odds_pick_hit")
    form_accuracy = summarize_hit_column(frame, "form_pick_hit")
    hybrid_accuracy = summarize_hit_column(frame, "hybrid_pick_hit")
    recommended_full_accuracy = summarize_hit_column(frame, "recommended_full_hit")

    return [
        {
            "bucket": bucket,
            "name": name,
            "settled_rows": int(len(frame)),
            "odds_accuracy": odds_accuracy,
            "form_accuracy": form_accuracy,
            "hybrid_accuracy": hybrid_accuracy,
            "recommended_full_accuracy": recommended_full_accuracy,
            "recommended_bets": recommended_bets,
            "recommended_hits": int(recommended_hits.sum()) if not recommended_hits.empty else 0,
            "recommended_accuracy": recommended_accuracy,
            "recommended_profit": float(recommended_profit),
            "recommended_roi": recommended_roi,
        }
    ]


def summarize_hit_column(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    series = frame[column].dropna()
    if series.empty:
        return 0.0
    return float(series.astype(int).mean())


def settled_columns() -> list[str]:
    return [
        "report_name",
        "played_at",
        "sport",
        "league",
        "home_team",
        "away_team",
        "gm_ts",
        "match_seq",
        "home_odds",
        "away_odds",
        "recommended_side",
        "recommended_full_side",
        "recommended_model",
        "recommended_expected_value",
        "bet_grade",
        "actual_outcome",
        "home_score",
        "away_score",
        "recommended_hit",
        "recommended_full_hit",
        "recommended_profit",
        "odds_pick",
        "odds_pick_hit",
        "form_pick",
        "form_pick_hit",
        "hybrid_pick",
        "hybrid_pick_hit",
    ]
