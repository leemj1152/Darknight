from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
ANALYSIS_DIR = BASE_DIR / "analysis"
DATA_DIR = BASE_DIR / "data"
LATEST_REPORTS_PATH = REPORTS_DIR / "latest_reports.json"


st.set_page_config(page_title="Darknight", layout="wide")
st.title("Darknight Prediction Console")


@st.cache_data(ttl=60)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=60)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_file(pattern: str, directory: Path) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def recent_files(pattern: str, directory: Path, limit: int = 20) -> list[Path]:
    return sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]


def filter_prediction_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    sports = ["전체"] + sorted(str(value) for value in frame.get("sport", pd.Series(dtype="object")).dropna().unique())
    leagues = ["전체"] + sorted(str(value) for value in frame.get("league", pd.Series(dtype="object")).dropna().unique())
    grades = ["전체"] + sorted(str(value) for value in frame.get("bet_grade", pd.Series(dtype="object")).dropna().unique())

    with st.sidebar:
        st.header(f"필터: {scope}")
        sport = st.selectbox("종목", sports, index=0, key=f"{scope}_sport")
        league = st.selectbox("리그", leagues, index=0, key=f"{scope}_league")
        grade = st.selectbox("등급", grades, index=0, key=f"{scope}_grade")
        only_top = st.checkbox("Top picks만 보기", value=False, key=f"{scope}_only_top")

    filtered = frame.copy()
    if sport != "전체":
        filtered = filtered[filtered["sport"] == sport]
    if league != "전체":
        filtered = filtered[filtered["league"] == league]
    if grade != "전체" and "bet_grade" in filtered.columns:
        filtered = filtered[filtered["bet_grade"] == grade]
    if only_top and "top_pick_rank" in filtered.columns:
        filtered = filtered[filtered["top_pick_rank"].notna()]
    return filtered


def render_report_metadata(report_type: str, metadata: dict) -> None:
    if not metadata:
        return
    cols = st.columns(4)
    cols[0].metric("생성 시각", str(metadata.get("generated_at", "-")))
    cols[1].metric("행 수", str(metadata.get("rows", 0)))
    if report_type == "round":
        cols[2].metric("회차", str(metadata.get("gm_ts", "-")))
    else:
        cols[2].metric("대상 날짜", str(metadata.get("target_date", "-")))
    cols[3].metric("마지막 마감", str(metadata.get("last_close_at", "-")))


latest_reports = load_json(LATEST_REPORTS_PATH)

tab_daily, tab_round, tab_simulation, tab_tracking, tab_data = st.tabs(
    ["오늘 리포트", "회차 리포트", "모의 베팅", "정산", "데이터"]
)

with tab_daily:
    st.subheader("오늘 경기 리포트")
    render_report_metadata("daily", latest_reports.get("daily", {}))
    daily_file = latest_file("daily_predictions_*.csv", REPORTS_DIR)
    st.caption(str(daily_file) if daily_file else "오늘 리포트 파일이 없습니다.")
    daily_frame = load_csv(daily_file) if daily_file else pd.DataFrame()
    filtered_daily = filter_prediction_frame(daily_frame, "daily") if not daily_frame.empty else daily_frame
    if filtered_daily.empty:
        st.info("표시할 오늘 리포트가 없습니다.")
    else:
        st.dataframe(filtered_daily, use_container_width=True, hide_index=True)
        st.download_button(
            "오늘 리포트 CSV 다운로드",
            filtered_daily.to_csv(index=False).encode("utf-8-sig"),
            "daily_predictions_filtered.csv",
            "text/csv",
            key="download_daily",
        )

with tab_round:
    st.subheader("회차 리포트")
    render_report_metadata("round", latest_reports.get("round", {}))
    round_files = recent_files("round_predictions_*.csv", REPORTS_DIR, limit=50)
    options = {file.name: file for file in round_files}
    selected_name = st.selectbox("회차 파일", list(options.keys()), key="round_file_select") if options else None
    round_frame = load_csv(options[selected_name]) if selected_name else pd.DataFrame()
    if round_frame.empty:
        st.info("표시할 회차 리포트가 없습니다.")
    else:
        filtered_round = filter_prediction_frame(round_frame, "round")
        if "top_pick_rank" in filtered_round.columns:
            top_picks = filtered_round[filtered_round["top_pick_rank"].notna()].sort_values("top_pick_rank")
        else:
            top_picks = pd.DataFrame()
        if not top_picks.empty:
            st.markdown("### Top picks")
            columns = [
                "top_pick_rank",
                "sport",
                "league",
                "home_team",
                "away_team",
                "recommended_side",
                "recommended_model",
                "recommended_probability",
                "bet_grade",
            ]
            available = [column for column in columns if column in top_picks.columns]
            st.dataframe(top_picks[available], use_container_width=True, hide_index=True)
        st.dataframe(filtered_round, use_container_width=True, hide_index=True)
        st.download_button(
            "회차 리포트 CSV 다운로드",
            filtered_round.to_csv(index=False).encode("utf-8-sig"),
            "round_predictions_filtered.csv",
            "text/csv",
            key="download_round",
        )
        archive_dir = REPORTS_DIR / "archive" / "round"
        archive_files = recent_files("*.csv", archive_dir, limit=20)
        if archive_files:
            st.markdown("### Recent round snapshots")
            st.dataframe(
                pd.DataFrame(
                    {
                        "file": [item.name for item in archive_files],
                        "modified_at": [pd.to_datetime(item.stat().st_mtime, unit="s") for item in archive_files],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

with tab_simulation:
    st.subheader("3개 전략 모의 베팅")
    summary_frame = load_csv(ANALYSIS_DIR / "simulation_summary.csv")
    daily_frame = load_csv(ANALYSIS_DIR / "simulation_daily.csv")
    bets_frame = load_csv(ANALYSIS_DIR / "simulation_bets.csv")
    if summary_frame.empty:
        st.info("모의 베팅 데이터가 없습니다.")
    else:
        metric_columns = st.columns(max(len(summary_frame), 1))
        for index, (_, row) in enumerate(summary_frame.iterrows()):
            with metric_columns[index]:
                st.metric(
                    f"{str(row['agent']).upper()} ROI",
                    f"{float(row['roi']):.2%}",
                    delta=f"profit {float(row['profit']):.2f}",
                )
                st.caption(
                    f"bets {int(float(row['bets']))} | win rate {float(row['win_rate']):.2%} | skipped {int(float(row['skipped']))}"
                )
        st.dataframe(summary_frame, use_container_width=True, hide_index=True)

        if not daily_frame.empty:
            chart_frame = daily_frame.copy()
            chart_frame["play_date"] = pd.to_datetime(chart_frame["play_date"])
            profit_chart = chart_frame.pivot(index="play_date", columns="agent", values="cumulative_profit").sort_index()
            roi_chart = chart_frame.pivot(index="play_date", columns="agent", values="cumulative_roi").sort_index()
            st.markdown("### 누적 수익")
            st.line_chart(profit_chart, use_container_width=True)
            st.markdown("### 누적 ROI")
            st.line_chart(roi_chart, use_container_width=True)

        if not bets_frame.empty:
            agents = ["전체"] + sorted(str(value) for value in bets_frame["agent"].dropna().unique())
            selected_agent = st.selectbox("전략", agents, index=0, key="simulation_agent")
            filtered_bets = bets_frame.copy()
            if selected_agent != "전체":
                filtered_bets = filtered_bets[filtered_bets["agent"] == selected_agent]
            st.markdown("### 최근 베팅 내역")
            st.dataframe(
                filtered_bets.sort_values("played_at", ascending=False).head(200),
                use_container_width=True,
                hide_index=True,
            )

with tab_tracking:
    st.subheader("최근 정산")
    settled_summary = load_csv(ANALYSIS_DIR / "settled_summary.csv")
    settled_predictions = load_csv(ANALYSIS_DIR / "settled_predictions.csv")
    if settled_summary.empty:
        st.info("정산 데이터가 없습니다.")
    else:
        overall = settled_summary[settled_summary["bucket"] == "overall"]
        if not overall.empty:
            row = overall.iloc[0]
            metric_columns = st.columns(4)
            metric_columns[0].metric("Odds 정확도", f"{float(row['odds_accuracy']):.2%}")
            metric_columns[1].metric("Form 정확도", f"{float(row['form_accuracy']):.2%}")
            metric_columns[2].metric("Hybrid 정확도", f"{float(row['hybrid_accuracy']):.2%}")
            metric_columns[3].metric("최종 추천 정확도", f"{float(row['recommended_full_accuracy']):.2%}")
            roi_columns = st.columns(3)
            roi_columns[0].metric("추천 ROI", f"{float(row['recommended_roi']):.2%}")
            roi_columns[1].metric("추천 베팅 수", f"{int(row['recommended_bets'])}")
            roi_columns[2].metric("정산 경기 수", f"{int(row['settled_rows'])}")
        st.dataframe(settled_summary, use_container_width=True, hide_index=True)
        if not settled_predictions.empty:
            st.markdown("### 최근 정산 내역")
            st.dataframe(
                settled_predictions.sort_values("played_at", ascending=False).head(200),
                use_container_width=True,
                hide_index=True,
            )

with tab_data:
    st.subheader("기초 데이터")
    results_path = DATA_DIR / "results.csv"
    results_frame = load_csv(results_path)
    st.caption(str(results_path))
    if results_frame.empty:
        st.info("결과 데이터가 없습니다.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("결과 행 수", f"{len(results_frame):,}")
        col2.metric("마지막 gmTs", str(results_frame["gm_ts"].astype(str).max()))
        col3.metric("마지막 경기 시각", str(pd.to_datetime(results_frame["played_at"]).max()))
        st.dataframe(results_frame.tail(200), use_container_width=True, hide_index=True)
