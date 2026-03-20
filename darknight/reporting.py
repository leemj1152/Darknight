from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def filter_matches_for_date(frame: pd.DataFrame, target_date: date) -> pd.DataFrame:
    matches = frame.copy()
    matches["played_at"] = pd.to_datetime(matches["played_at"])
    return matches[matches["played_at"].dt.date == target_date].sort_values(["played_at", "match_seq"]).reset_index(drop=True)


def save_daily_report(
    report_frame: pd.DataFrame,
    output_dir: str | Path,
    target_date: date,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"daily_predictions_{target_date.isoformat()}.csv"
    md_path = output_path / f"daily_predictions_{target_date.isoformat()}.md"
    report_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(to_markdown_report(report_frame, target_date), encoding="utf-8")
    return csv_path, md_path


def to_markdown_report(report_frame: pd.DataFrame, target_date: date) -> str:
    lines = [f"# Daily Predictions {target_date.isoformat()}", ""]
    if report_frame.empty:
        lines.append("No matches found.")
        return "\n".join(lines)

    ordered = report_frame.sort_values(["played_at", "sport", "league", "match_seq"])
    for _, row in ordered.iterrows():
        lines.append(
            f"- close {pd.to_datetime(row['close_at']).strftime('%H:%M') if pd.notna(row.get('close_at')) else '--:--'} "
            f"play {pd.to_datetime(row['played_at']).strftime('%H:%M')} "
            f"{row['sport']} {row['league']} "
            f"{row['home_team']} vs {row['away_team']} | "
            f"odds {row['odds_home_probability']:.2%} | "
            f"form {row['form_home_probability']:.2%} | "
            f"hybrid {row['hybrid_home_probability']:.2%} | "
            f"gmTs {row.get('gm_ts', '')}"
        )
    return "\n".join(lines)
