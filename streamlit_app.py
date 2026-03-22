from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
ANALYSIS_DIR = BASE_DIR / "analysis"
DATA_DIR = BASE_DIR / "data"


st.set_page_config(page_title="Darknight", layout="wide")
st.title("Darknight Prediction Console")


@st.cache_data(ttl=60)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def latest_file(pattern: str, directory: Path) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def filter_prediction_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    sports = ["전체"] + sorted(str(value) for value in frame["sport"].dropna().unique())
    leagues = ["전체"] + sorted(str(value) for value in frame["league"].dropna().unique())
    grades = ["전체"] + sorted(
        str(value) for value in frame.get("bet_grade", pd.Series(dtype="object")).dropna().unique()
    )

    with st.sidebar:
        st.header(f"필터: {scope}")
        sport = st.selectbox("종목", sports, index=0, key=f"{scope}_sport")
        league = st.selectbox("리그", leagues, index=0, key=f"{scope}_league")
        grade = st.selectbox("등급", grades, index=0, key=f"{scope}_grade")
        only_top = st.checkbox("Top picks만 보기", value=False, key=f"{scope}_only_top")
        min_ev = st.slider(
            "최소 추천 EV",
            min_value=-0.2,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key=f"{scope}_min_ev",
        )

    filtered = frame.copy()
    if sport != "전체":
        filtered = filtered[filtered["sport"] == sport]
    if league != "전체":
        filtered = filtered[filtered["league"] == league]
    if grade != "전체":
        filtered = filtered[filtered["bet_grade"] == grade]
    if only_top and "top_pick_rank" in filtered.columns:
        filtered = filtered[filtered["top_pick_rank"].notna()]
    if "recommended_expected_value" in filtered.columns:
        filtered = filtered[filtered["recommended_expected_value"].fillna(-999) >= min_ev]
    return filtered


tab_reports, tab_round, tab_simulation, tab_tracking, tab_data = st.tabs(
    ["오늘 리포트", "회차 리포트", "모의 베팅", "정산", "데이터"]
)

with tab_reports:
    daily_file = latest_file("daily_predictions_*.csv", REPORTS_DIR)
    st.subheader("오늘 리포트")
    st.caption(str(daily_file) if daily_file else "리포트 파일이 없습니다.")
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
    round_files = sorted(
        REPORTS_DIR.glob("round_predictions_*.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    options = {file.name: file for file in round_files}
    selected_name = (
        st.selectbox("회차 파일", list(options.keys()), key="round_file_select")
        if options
        else None
    )
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
            st.dataframe(
                top_picks[
                    [
                        "top_pick_rank",
                        "sport",
                        "league",
                        "home_team",
                        "away_team",
                        "recommended_side",
                        "recommended_model",
                        "recommended_expected_value",
                        "bet_grade",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        st.dataframe(filtered_round, use_container_width=True, hide_index=True)
        st.download_button(
            "회차 리포트 CSV 다운로드",
            filtered_round.to_csv(index=False).encode("utf-8-sig"),
            "round_predictions_filtered.csv",
            "text/csv",
            key="download_round",
        )

with tab_simulation:
    summary_frame = load_csv(ANALYSIS_DIR / "simulation_summary.csv")
    daily_frame = load_csv(ANALYSIS_DIR / "simulation_daily.csv")
    bets_frame = load_csv(ANALYSIS_DIR / "simulation_bets.csv")

    st.subheader("3개 전략 모의 베팅")
    if summary_frame.empty:
        st.info("모의 베팅 데이터가 없습니다. `python main.py simulate-bets`를 먼저 실행하세요.")
    else:
        st.markdown("### 전략 요약")
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
            st.markdown("### 누적 수익")
            chart_frame = daily_frame.copy()
            chart_frame["play_date"] = pd.to_datetime(chart_frame["play_date"])
            pivot_profit = chart_frame.pivot(index="play_date", columns="agent", values="cumulative_profit").sort_index()
            pivot_roi = chart_frame.pivot(index="play_date", columns="agent", values="cumulative_roi").sort_index()
            st.line_chart(pivot_profit, use_container_width=True)
            st.markdown("### 누적 ROI")
            st.line_chart(pivot_roi, use_container_width=True)

        if not bets_frame.empty:
            st.markdown("### 최근 베팅 내역")
            agents = ["전체"] + sorted(str(value) for value in bets_frame["agent"].dropna().unique())
            selected_agent = st.selectbox("전략", agents, index=0, key="simulation_agent")
            filtered_bets = bets_frame.copy()
            if selected_agent != "전체":
                filtered_bets = filtered_bets[filtered_bets["agent"] == selected_agent]
            st.dataframe(
                filtered_bets.sort_values("played_at", ascending=False).head(200),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "모의 베팅 CSV 다운로드",
                filtered_bets.to_csv(index=False).encode("utf-8-sig"),
                "simulation_bets_filtered.csv",
                "text/csv",
                key="download_simulation_bets",
            )

with tab_tracking:
    settled_summary = load_csv(ANALYSIS_DIR / "settled_summary.csv")
    settled_predictions = load_csv(ANALYSIS_DIR / "settled_predictions.csv")
    st.subheader("예측 정산")
    if settled_summary.empty:
        st.info("정산 데이터가 없습니다. `python main.py settle-reports`를 먼저 실행하세요.")
    else:
        st.dataframe(settled_summary, use_container_width=True, hide_index=True)
        if not settled_predictions.empty:
            st.markdown("### 최근 정산")
            st.dataframe(
                settled_predictions.sort_values("played_at", ascending=False).head(200),
                use_container_width=True,
                hide_index=True,
            )

with tab_data:
    results_path = DATA_DIR / "results.csv"
    results_frame = load_csv(results_path)
    st.subheader("기초 데이터")
    st.caption(str(results_path))
    if results_frame.empty:
        st.info("결과 데이터가 없습니다.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("결과 행 수", f"{len(results_frame):,}")
        col2.metric("마지막 gmTs", str(results_frame["gm_ts"].astype(str).max()))
        col3.metric("마지막 경기일", str(pd.to_datetime(results_frame["played_at"]).max()))
        st.dataframe(results_frame.tail(200), use_container_width=True, hide_index=True)
