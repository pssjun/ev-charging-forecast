import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="EDA 대시보드", page_icon="📊", layout="wide")

# =========================
# 데이터 로딩
# =========================
@st.cache_data
def load_data():
    df_path = Path("data/processed/df_clean.parquet")
    hourly_path = Path("data/processed/hourly_ts.parquet")

    df = pd.read_parquet(df_path) if df_path.exists() else None
    hourly_ts = pd.read_parquet(hourly_path) if hourly_path.exists() else None

    return df, hourly_ts

df, hourly_ts = load_data()

if df is None or hourly_ts is None:
    st.error("데이터 파일이 없습니다. `data/processed/` 폴더를 확인하세요.")
    st.stop()

# 날짜 변환
df["chargBgngDt"] = pd.to_datetime(df["chargBgngDt"])
df["date"] = df["chargBgngDt"].dt.date
hourly_ts["date_hour"] = pd.to_datetime(hourly_ts["date_hour"])

# =========================
# 헤더
# =========================
st.title("📊 EDA 대시보드")
st.caption("EV 충전 수요의 시간대·요일·계절·지역·충전기 유형별 패턴을 인터랙티브하게 탐색합니다.")
st.divider()

# =========================
# 사이드바 필터
# =========================
st.sidebar.header("🔍 필터")

# 지역 필터
regions = ["전체"] + sorted(df["rgnNm"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("지역 선택", regions)

# 충전기 유형 필터
charger_types = ["전체"] + sorted(df["chargTypeNm"].dropna().unique().tolist())
selected_charger = st.sidebar.selectbox("충전기 유형", charger_types)

# 월 범위 필터
month_range = st.sidebar.slider("월 범위", 1, 12, (1, 12))

# 필터 적용
filtered_df = df.copy()
if selected_region != "전체":
    filtered_df = filtered_df[filtered_df["rgnNm"] == selected_region]
if selected_charger != "전체":
    filtered_df = filtered_df[filtered_df["chargTypeNm"] == selected_charger]
filtered_df = filtered_df[
    (filtered_df["month"] >= month_range[0]) &
    (filtered_df["month"] <= month_range[1])
]

# 필터 결과 요약
st.sidebar.divider()
st.sidebar.metric("필터링된 건수", f"{len(filtered_df):,}")
st.sidebar.metric("총 충전량", f"{filtered_df['totChargRcngQnt'].sum():,.0f} kWh")

# =========================
# 상단 요약 카드
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 충전 건수", f"{len(filtered_df):,}")
c2.metric("총 충전량", f"{filtered_df['totChargRcngQnt'].sum():,.0f} kWh")
c3.metric("건당 평균 충전량", f"{filtered_df['totChargRcngQnt'].mean():.2f} kWh")
c4.metric("평균 충전 시간", f"{filtered_df['charging_minutes'].mean():.1f} 분")

st.divider()

# =========================
# 시간대별 패턴
# =========================
st.subheader("⏰ 시간대별 충전 수요")

hour_agg = (
    filtered_df.groupby("hour")["totChargRcngQnt"]
    .sum()
    .reset_index()
)

fig_hour = px.line(
    hour_agg,
    x="hour", y="totChargRcngQnt",
    markers=True,
    labels={"hour": "시간(hour)", "totChargRcngQnt": "총 충전량 (kWh)"},
    title="시간대별 총 충전량"
)
fig_hour.update_layout(hovermode="x unified")
st.plotly_chart(fig_hour, use_container_width=True)

with st.expander("💡 인사이트"):
    st.markdown("""
    - 시간대 변수는 충전 수요 설명에 일부 기여하지만, 단독으로는 충분히 설명하지 못한다.
    - Feature Importance에서도 `hour`보다 `rolling_24h`, `lag_1` 같은 과거 수요 정보가 더 중요하게 나타났다.
    - 충전 수요는 현재 시각보다 최근 충전 패턴의 영향을 더 크게 받는 것으로 판단된다.
    """)

# =========================
# 요일별 패턴
# =========================
st.subheader("📅 요일별 충전 수요")

weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
weekday_agg = (
    filtered_df.groupby("weekday")["totChargRcngQnt"]
    .mean()
    .reset_index()
)
weekday_agg["weekday_name"] = weekday_agg["weekday"].map(weekday_map)

fig_weekday = px.bar(
    weekday_agg,
    x="weekday_name", y="totChargRcngQnt",
    labels={"weekday_name": "요일", "totChargRcngQnt": "평균 충전량 (kWh)"},
    title="요일별 평균 충전량",
    color="totChargRcngQnt",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig_weekday, use_container_width=True)

with st.expander("💡 인사이트"):
    st.markdown("""
    - 금요일이 가장 높고 일요일이 가장 낮게 나타남 (금요일이 일요일 대비 약 40% 높음).
    - 목→금 증가 경향 → 주말 이동 대비 사전 충전 수요의 영향으로 해석 가능.
    - `weekday` 변수는 예측 모델에 활용 가능한 유의미한 설명 변수로 판단됨.
    """)

# =========================
# 월별 패턴
# =========================
st.subheader("🗓️ 월별 충전 수요")

month_agg = (
    filtered_df.groupby("month")["totChargRcngQnt"]
    .sum()
    .reset_index()
)

fig_month = px.line(
    month_agg,
    x="month", y="totChargRcngQnt",
    markers=True,
    labels={"month": "월", "totChargRcngQnt": "총 충전량 (kWh)"},
    title="월별 총 충전량"
)
fig_month.update_xaxes(dtick=1)
st.plotly_chart(fig_month, use_container_width=True)

with st.expander("💡 인사이트"):
    st.markdown("""
    - 6월에 가장 높은 충전량, 7월 일시 감소 후 다시 증가하는 계절성 확인.
    - API 수집 특성상 1월 데이터가 상대적으로 적어 과소 추정된 것으로 판단됨.
    - 계절적 요인이 존재하므로 `month`를 예측 변수로 활용.
    """)

# =========================
# 충전기 유형별
# =========================
st.subheader("🔌 충전기 유형별 충전량")

type_agg = (
    filtered_df.groupby("chargTypeNm")["totChargRcngQnt"]
    .sum()
    .reset_index()
    .sort_values(by="totChargRcngQnt", ascending=False)
)

fig_type = px.bar(
    type_agg,
    x="chargTypeNm", y="totChargRcngQnt",
    labels={"chargTypeNm": "충전기 유형", "totChargRcngQnt": "총 충전량 (kWh)"},
    title="충전기 유형별 총 충전량",
    color="totChargRcngQnt",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig_type, use_container_width=True)

with st.expander("💡 인사이트"):
    st.markdown("""
    - DC급속 충전기가 전체 충전량의 대부분 차지 → 이용자들의 급속 충전 선호.
    - DC콤보(완속)는 상대적으로 낮은 이용량.
    - 충전기 유형을 Feature로 활용 시 예측 성능 개선 가능성 존재.
    """)

# =========================
# 지역별 Top 10
# =========================
st.subheader("🗺️ 지역별 충전 수요 Top 10")

region_agg = (
    df.groupby("rgnNm")["totChargRcngQnt"]  # 지역 필터 무시하고 전체 순위
    .sum()
    .reset_index()
    .sort_values(by="totChargRcngQnt", ascending=False)
    .head(10)
)

fig_region = px.bar(
    region_agg,
    x="rgnNm", y="totChargRcngQnt",
    labels={"rgnNm": "지역", "totChargRcngQnt": "총 충전량 (kWh)"},
    title="지역별 충전 수요 Top 10",
    color="totChargRcngQnt",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig_region, use_container_width=True)

with st.expander("💡 인사이트"):
    st.markdown("""
    - 경기도가 가장 높고, 경상북도·경상남도가 뒤를 이음.
    - 서울특별시는 Top 10에 포함되지만 상대적으로 낮음.
    - EV 보급률, 충전 인프라 규모, 차량 이동량 차이의 영향으로 해석.
    """)

st.divider()
st.caption("모든 시각화는 사이드바 필터에 따라 인터랙티브하게 업데이트됩니다.")