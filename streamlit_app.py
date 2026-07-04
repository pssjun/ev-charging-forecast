import streamlit as st
import pandas as pd
from pathlib import Path

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="EV 충전소 수요 예측",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 데이터 로딩
# =========================
@st.cache_data
def load_data():
    final_compare_path = Path("data/processed/final_compare.csv")
    hourly_path = Path("data/processed/hourly_ts.parquet")

    final_compare = pd.read_csv(final_compare_path) if final_compare_path.exists() else None
    hourly_ts = pd.read_parquet(hourly_path) if hourly_path.exists() else None

    return final_compare, hourly_ts

final_compare, hourly_ts = load_data()

# =========================
# 헤더
# =========================
st.title("⚡ EV 충전소 시간대별 수요 예측")
st.markdown("#### 공공데이터 OpenAPI 기반 전기차 충전 수요 분석 및 예측 프로젝트")
st.divider()

# =========================
# 핵심 지표
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("분석 기간", "2024.01 ~ 2024.12")

with col2:
    if hourly_ts is not None:
        st.metric("시간 단위 데이터", f"{len(hourly_ts):,}시간")
    else:
        st.metric("시간 단위 데이터", "—")

if final_compare is not None and len(final_compare) > 0:
    best_row = final_compare.loc[final_compare["R2"].idxmax()]
    with col3:
        st.metric("최종 R²", f"{best_row['R2']:.3f}")
    with col4:
        st.metric("최종 RMSE", f"{best_row['RMSE']:.2f}")
else:
    with col3:
        st.metric("최종 R²", "—")
    with col4:
        st.metric("최종 RMSE", "—")

st.divider()

# =========================
# 문제 정의
# =========================
st.header("📌 문제 정의")

st.markdown("""
전기차 보급이 확대되면서 충전 인프라의 효율적 운영이 중요해지고 있습니다.  
충전 수요는 시간대, 요일, 계절, 날씨, 과거 충전 패턴에 따라 달라질 수 있습니다.

본 프로젝트는 2024년 EV 충전 이력 데이터를 시간 단위로 집계하고,  
시간당 총 충전량(`total_charge_kwh`)을 예측하는 머신러닝 모델을 구축했습니다.
""")

st.info(
    "핵심 목표: 시간대별 총 충전량(kWh)을 예측하여 피크 시간대 수요 대응과 충전 인프라 운영 최적화 가능성을 확인합니다."
)

# =========================
# 접근 방법
# =========================
st.header("🔬 분석 및 모델링 흐름")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📦 데이터 구성")
    st.markdown("""
    - **EV 충전 이력 데이터**: 약 60,000건
    - **시간 단위 집계 데이터**: `hourly_ts`
    - **기상 데이터**: 기온, 강수량, 풍속, 습도
    - **공휴일 변수**: 한국 공휴일 여부
    """)

with col_right:
    st.subheader("🚀 모델링 단계")
    st.markdown("""
    1. Baseline 모델 비교
    2. XGBoost 기본 모델
    3. XGBoost + Weather 변수 추가
    4. Optuna 기반 하이퍼파라미터 튜닝
    5. Feature Importance 및 Residual Analysis
    """)

# =========================
# 주요 Feature
# =========================
st.header("🧩 주요 Feature")

st.markdown("""
| 구분 | 변수 |
|---|---|
| 시간 변수 | `hour`, `weekday`, `month`, `day`, `is_weekend`, `is_holiday` |
| 과거 수요 변수 | `kwh_lag_1`, `kwh_lag_24`, `kwh_lag_168` |
| 이동평균 변수 | `kwh_rolling_24h`, `kwh_rolling_168h` |
| 날씨 변수 | `temp`, `rainfall`, `wind_speed`, `humidity` |
""")

# =========================
# 평가 지표
# =========================
st.header("📐 평가 지표")

st.markdown("""
| 지표 | 의미 |
|---|---|
| **MAE** | 평균적으로 몇 kWh 정도 예측이 빗나갔는지 |
| **RMSE** | 큰 오차에 더 민감한 지표 |
| **R²** | 모델이 충전량 변동성을 얼마나 설명하는지 |
| **MAPE** | 실제값 대비 평균 오차율 |
| **Peak_MAE** | 상위 20% 피크 수요 구간에서의 MAE |
""")

st.success(
    "시계열 데이터 특성을 고려하여 Train / Validation / Test를 시간 순서대로 60% / 20% / 20%로 분할했습니다."
)

# =========================
# 성능 요약
# =========================
if final_compare is not None:
    st.header("🏆 모델 성능 요약")
    st.dataframe(final_compare, use_container_width=True)

    st.markdown("""
    최종 모델은 Baseline, 기본 XGBoost, Weather XGBoost와 비교하여 평가했습니다.  
    단순히 전체 성능뿐 아니라 피크 시간대 오차(`Peak_MAE`)도 함께 확인하여 운영 관점의 실용성을 검토했습니다.
    """)

# =========================
# 페이지 안내
# =========================
st.divider()
st.header("📋 페이지 안내")

st.markdown("""
좌측 사이드바에서 분석 단계별 페이지로 이동할 수 있습니다.

- **프로젝트 개요**: 데이터와 문제 정의
- **EDA 대시보드**: 시간대, 요일, 월별 충전 수요 패턴
- **수요 예측**: 실제값과 예측값 비교
- **모델 성능 비교**: Baseline, XGBoost, Optuna 성능 비교
- **결론과 한계**: 핵심 인사이트와 개선 방향
""")

# =========================
# 푸터
# =========================
st.divider()

st.caption(
    "Data Source: 공공데이터포털 EV 충전 이력 API, 기상청 ASOS | "
    "Built with Streamlit | "
    "[GitHub](https://github.com/pssjun/ev-charging-forecast)"
)