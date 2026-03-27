from __future__ import annotations

import json
import shutil
from datetime import datetime
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
    metadata = _build_report_metadata(
        report_frame,
        report_type="daily",
        csv_path=csv_path,
        md_path=md_path,
        extra={"target_date": target_date.isoformat()},
    )
    _archive_report(output_path, csv_path, md_path, report_type="daily", report_key=target_date.isoformat())
    _write_latest_metadata(output_path, report_type="daily", metadata=metadata)
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
    metadata = _build_report_metadata(
        report_frame,
        report_type="round",
        csv_path=csv_path,
        md_path=md_path,
        extra={"gm_ts": gm_ts},
    )
    _archive_report(output_path, csv_path, md_path, report_type="round", report_key=gm_ts)
    _write_latest_metadata(output_path, report_type="round", metadata=metadata)
    return csv_path, md_path


def _build_report_metadata(
    report_frame: pd.DataFrame,
    *,
    report_type: str,
    csv_path: Path,
    md_path: Path,
    extra: dict[str, str],
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "report_type": report_type,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(report_frame)),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        **extra,
    }
    if not report_frame.empty:
        played_at = pd.to_datetime(report_frame["played_at"], errors="coerce")
        metadata["first_played_at"] = played_at.min().isoformat() if played_at.notna().any() else ""
        metadata["last_played_at"] = played_at.max().isoformat() if played_at.notna().any() else ""
        if "close_at" in report_frame.columns:
            close_at = pd.to_datetime(report_frame["close_at"], errors="coerce")
            metadata["last_close_at"] = close_at.max().isoformat() if close_at.notna().any() else ""
    return metadata


def _archive_report(
    output_path: Path,
    csv_path: Path,
    md_path: Path,
    *,
    report_type: str,
    report_key: str,
) -> None:
    archive_dir = output_path / "archive" / report_type
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_csv = archive_dir / f"{report_type}_{report_key}_{stamp}.csv"
    archived_md = archive_dir / f"{report_type}_{report_key}_{stamp}.md"
    shutil.copy2(csv_path, archived_csv)
    shutil.copy2(md_path, archived_md)


def _write_latest_metadata(output_path: Path, *, report_type: str, metadata: dict[str, object]) -> None:
    latest_path = output_path / "latest_reports.json"
    payload: dict[str, object] = {}
    if latest_path.exists():
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload[report_type] = metadata
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
