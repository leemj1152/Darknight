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


def save_round_report(
    report_frame: pd.DataFrame,
    output_dir: str | Path,
    gm_ts: str,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"round_predictions_{gm_ts}.csv"
    md_path = output_path / f"round_predictions_{gm_ts}.md"
    report_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(to_markdown_round_report(report_frame, gm_ts), encoding="utf-8")
    return csv_path, md_path


def to_markdown_report(report_frame: pd.DataFrame, target_date: date) -> str:
    lines = [f"# Daily Predictions {target_date.isoformat()}", ""]
    if report_frame.empty:
        lines.append("No matches found.")
        return "\n".join(lines)

    ordered = report_frame.sort_values(["played_at", "sport", "league", "match_seq"])
    top_picks = ordered[ordered["top_pick_rank"].notna()].sort_values("top_pick_rank") if "top_pick_rank" in ordered.columns else ordered.iloc[0:0]
    if not top_picks.empty:
        lines.append("## Top Picks")
        lines.append("")
        for _, row in top_picks.iterrows():
            lines.append(
                f"- #{int(row['top_pick_rank'])} {row['home_team']} vs {row['away_team']} | "
                f"pick {row.get('recommended_side', '')} via {row.get('recommended_model', '')} | "
                f"grade {row.get('bet_grade', '')} | "
                f"prob={_format_optional_percent(row.get('recommended_probability'))}"
            )
        lines.append("")
    for _, row in ordered.iterrows():
        lines.append(
            f"- close {pd.to_datetime(row['close_at']).strftime('%H:%M') if pd.notna(row.get('close_at')) else '--:--'} "
            f"play {pd.to_datetime(row['played_at']).strftime('%H:%M')} "
            f"{row['sport']} {row['league']} {row.get('game_type', '')} "
            f"{row['home_team']} vs {row['away_team']} | "
            f"handicap={_format_optional_number(row.get('handicap_line'))} | "
            f"market H={row['odds_home_probability']:.2%} A={_format_optional_percent(row.get('odds_away_probability'))} | "
            f"form H={row['form_home_probability']:.2%} A={_format_optional_percent(row.get('form_away_probability'))} | "
            f"hybrid H={row['hybrid_home_probability']:.2%} A={_format_optional_percent(row.get('hybrid_away_probability'))} | "
            f"best {row.get('recommended_side', '')} via {row.get('recommended_model', '')} prob={_format_optional_percent(row.get('recommended_probability'))} | "
            f"grade {row.get('bet_grade', '')} rank={_format_rank(row.get('top_pick_rank'))} trust={row.get('league_trust_level', '')} | "
            f"gmTs {row.get('gm_ts', '')}"
        )
    return "\n".join(lines)


def to_markdown_round_report(report_frame: pd.DataFrame, gm_ts: str) -> str:
    lines = [f"# Round Predictions {gm_ts}", ""]
    if report_frame.empty:
        lines.append("No matches found.")
        return "\n".join(lines)

    ordered = report_frame.sort_values(["played_at", "sport", "league", "match_seq"])
    top_picks = ordered[ordered["top_pick_rank"].notna()].sort_values("top_pick_rank") if "top_pick_rank" in ordered.columns else ordered.iloc[0:0]
    if not top_picks.empty:
        lines.append("## Top Picks")
        lines.append("")
        for _, row in top_picks.iterrows():
            lines.append(
                f"- #{int(row['top_pick_rank'])} {row['home_team']} vs {row['away_team']} | "
                f"pick {row.get('recommended_side', '')} via {row.get('recommended_model', '')} | "
                f"grade {row.get('bet_grade', '')} | "
                f"prob={_format_optional_percent(row.get('recommended_probability'))}"
            )
        lines.append("")
    for _, row in ordered.iterrows():
        lines.append(
            f"- close {pd.to_datetime(row['close_at']).strftime('%H:%M') if pd.notna(row.get('close_at')) else '--:--'} "
            f"play {pd.to_datetime(row['played_at']).strftime('%m-%d %H:%M')} "
            f"{row['sport']} {row['league']} {row.get('game_type', '')} "
            f"{row['home_team']} vs {row['away_team']} | "
            f"handicap={_format_optional_number(row.get('handicap_line'))} | "
            f"market H={row['odds_home_probability']:.2%} A={_format_optional_percent(row.get('odds_away_probability'))} | "
            f"form H={row['form_home_probability']:.2%} A={_format_optional_percent(row.get('form_away_probability'))} | "
            f"hybrid H={row['hybrid_home_probability']:.2%} A={_format_optional_percent(row.get('hybrid_away_probability'))} | "
            f"best {row.get('recommended_side', '')} via {row.get('recommended_model', '')} prob={_format_optional_percent(row.get('recommended_probability'))} | "
            f"grade {row.get('bet_grade', '')} rank={_format_rank(row.get('top_pick_rank'))} trust={row.get('league_trust_level', '')}"
        )
    return "\n".join(lines)


def _format_optional_percent(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2%}"


def _format_rank(value: object) -> str:
    if pd.isna(value):
        return "-"
    return str(int(float(value)))


def _format_optional_number(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.1f}"
