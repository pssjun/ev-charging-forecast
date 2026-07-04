import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="수요 예측", page_icon="🔮", layout="wide")

# 자동 번역 방지
st.markdown(
    '<meta name="google" content="notranslate">',
    unsafe_allow_html=True
)

# =========================
# 데이터 로딩
# =========================
@st.cache_data
def load_data():
    result_path = Path("data/processed/result_df.parquet")
    result_df = pd.read_parquet(result_path) if result_path.exists() else None
    return result_df

result_df = load_data()

if result_df is None:
    st.error("`data/processed/result_df.parquet` 파일이 없습니다. 노트북에서 저장했는지 확인하세요.")
    st.stop()

# 데이터 타입 정리
result_df["date_hour"] = pd.to_datetime(result_df["date_hour"])
result_df = result_df.sort_values("date_hour").reset_index(drop=True)

# =========================
# 헤더
# =========================
st.title("🔮 수요 예측")
st.caption("최종 모델(XGBoost + Weather + Optuna)의 Test 구간 예측 결과를 인터랙티브하게 조회합니다.")
st.divider()

# =========================
# 사이드바 필터
# =========================
st.sidebar.header("🔍 조회 조건")

min_date = result_df["date_hour"].min().date()
max_date = result_df["date_hour"].max().date()

st.sidebar.markdown(f"**Test 데이터 기간**  \n{min_date} ~ {max_date}")

# 조회 방식 선택
view_mode = st.sidebar.radio(
    "조회 방식",
    ["기간 선택", "특정일 24시간"],
    help="기간 선택: 여러 일 범위 조회 / 특정일: 하루의 24시간 예측"
)

if view_mode == "기간 선택":
    date_range = st.sidebar.date_input(
        "날짜 범위",
        value=(min_date, min(min_date + pd.Timedelta(days=7), max_date)),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (
            (result_df["date_hour"].dt.date >= start_date) &
            (result_df["date_hour"].dt.date <= end_date)
        )
        view_df = result_df[mask].copy()
    else:
        view_df = result_df.copy()

else:  # 특정일 24시간
    target_date = st.sidebar.date_input(
        "날짜 선택",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    mask = result_df["date_hour"].dt.date == target_date
    view_df = result_df[mask].copy()

if len(view_df) == 0:
    st.warning("선택한 조건의 데이터가 없습니다.")
    st.stop()

# =========================
# 상단 요약 카드
# =========================
mae = np.mean(np.abs(view_df["actual"] - view_df["predicted"]))
peak_actual_time = view_df.loc[view_df["actual"].idxmax(), "date_hour"]
peak_actual_value = view_df["actual"].max()
peak_pred_time = view_df.loc[view_df["predicted"].idxmax(), "date_hour"]
peak_pred_value = view_df["predicted"].max()

c1, c2, c3, c4 = st.columns(4)
c1.metric("조회 구간", f"{len(view_df):,} 시간")
c2.metric("이 구간 MAE", f"{mae:.2f} kWh")
c3.metric(
    "실제 피크",
    f"{peak_actual_value:.1f} kWh",
    help=f"시점: {peak_actual_time}"
)
c4.metric(
    "예측 피크",
    f"{peak_pred_value:.1f} kWh",
    help=f"시점: {peak_pred_time}"
)

st.divider()

# =========================
# 실제 vs 예측 그래프
# =========================
st.subheader("📈 실제값 vs 예측값")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=view_df["date_hour"],
    y=view_df["actual"],
    mode="lines",
    name="실제값",
    line=dict(color="#1f77b4", width=2)
))

fig.add_trace(go.Scatter(
    x=view_df["date_hour"],
    y=view_df["predicted"],
    mode="lines",
    name="예측값",
    line=dict(color="#ff7f0e", width=2, dash="dash")
))

fig.update_layout(
    xaxis_title="시간",
    yaxis_title="총 충전량 (kWh)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# 잔차 그래프
# =========================
st.subheader("📉 잔차 (실제값 - 예측값)")

view_df["residual"] = view_df["actual"] - view_df["predicted"]

fig_resid = go.Figure()

fig_resid.add_trace(go.Bar(
    x=view_df["date_hour"],
    y=view_df["residual"],
    marker_color=view_df["residual"].apply(
        lambda x: "#d62728" if x > 0 else "#2ca02c"
    ),
    name="잔차"
))

fig_resid.add_hline(y=0, line_dash="dash", line_color="gray")

fig_resid.update_layout(
    xaxis_title="시간",
    yaxis_title="잔차 (kWh)",
    hovermode="x unified",
    height=350,
    showlegend=False
)

st.plotly_chart(fig_resid, use_container_width=True)

with st.expander("💡 잔차 해석"):
    st.markdown("""
    - **양의 잔차(빨간색)**: 실제 수요 > 예측 → 모델이 **과소 예측**한 구간
    - **음의 잔차(초록색)**: 실제 수요 < 예측 → 모델이 **과대 예측**한 구간
    - 양의 방향으로 큰 잔차가 몰린 구간은 갑작스러운 충전 수요 급증을 모델이 못 잡은 케이스
    - 이런 급증은 지역 이벤트, 특정 충전소 집중 이용 등 현재 Feature로는 설명이 어려운 요인이 원인일 수 있음
    """)

# =========================
# 상세 데이터 테이블
# =========================
with st.expander("📋 상세 데이터 보기"):
    display_df = view_df[["date_hour", "actual", "predicted", "residual"]].copy()
    display_df.columns = ["시간", "실제값 (kWh)", "예측값 (kWh)", "잔차 (kWh)"]
    display_df["실제값 (kWh)"] = display_df["실제값 (kWh)"].round(2)
    display_df["예측값 (kWh)"] = display_df["예측값 (kWh)"].round(2)
    display_df["잔차 (kWh)"] = display_df["잔차 (kWh)"].round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # CSV 다운로드
    csv = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "CSV로 다운로드",
        csv,
        f"prediction_{min(view_df['date_hour']).date()}_{max(view_df['date_hour']).date()}.csv",
        "text/csv"
    )

st.divider()
st.caption("본 페이지는 최종 모델(XGBoost + Weather + Optuna 튜닝)의 Test set 예측 결과를 기반으로 합니다.")