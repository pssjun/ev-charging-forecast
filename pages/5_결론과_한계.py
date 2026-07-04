import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="결론과 한계", page_icon="📝", layout="wide")

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
    final_path = Path("data/processed/final_compare.csv")
    return pd.read_csv(final_path) if final_path.exists() else None

final_compare = load_data()

# =========================
# 헤더
# =========================
st.title("📝 결론과 한계")
st.caption("프로젝트의 핵심 발견, 한계 그리고 향후 개선 방향을 정리합니다.")
st.divider()

# =========================
# 핵심 성과
# =========================
st.header("🏆 핵심 성과")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 데이터 & 파이프라인
    - 공공데이터포털 API로 EV 충전 이력 **60,000건** 수집  
      (aiohttp 비동기 병렬 처리 + 월별 parquet 체크포인트)
    - 기상청 ASOS 시간자료 API로 날씨 데이터 통합
    - 시간 단위 집계 + Lag/Rolling Feature Engineering
    - **Train/Val/Test = 60/20/20** 시간 순서 보존 분할
    """)

with col2:
    st.markdown("""
    ### 모델링
    - **Baseline 5종** (Naive/Seasonal/Rolling) 기준선 정립
    - **XGBoost 4단계** 점진적 성능 개선  
      (Basic → +Weather → +Optuna)
    - **Optuna 하이퍼파라미터 튜닝**  
      (Validation 기반, Test set 누수 방지)
    - **다중 지표 평가**  
      (MAE / RMSE / R² / MAPE / **Peak_MAE**)
    """)

if final_compare is not None:
    st.markdown("### 📊 최종 성능")

    best_row = final_compare.loc[final_compare["R2"].idxmax()]
    baseline_row = final_compare.iloc[0]  # 첫 행이 Best Baseline

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최고 모델", best_row["Model"])
    c2.metric(
        "R²",
        f"{best_row['R2']:.3f}",
        f"{best_row['R2'] - baseline_row['R2']:+.3f} vs Baseline"
    )
    c3.metric(
        "RMSE",
        f"{best_row['RMSE']:.2f}",
        f"{best_row['RMSE'] - baseline_row['RMSE']:+.2f} vs Baseline",
        delta_color="inverse"
    )
    c4.metric(
        "Peak MAE",
        f"{best_row['Peak_MAE']:.2f}",
        f"{best_row['Peak_MAE'] - baseline_row['Peak_MAE']:+.2f} vs Baseline",
        delta_color="inverse"
    )

st.divider()

# =========================
# 주요 인사이트
# =========================
st.header("💡 주요 인사이트")

with st.container(border=True):
    st.markdown("""
    #### 1. 과거 수요 패턴이 예측의 핵심이었다
    Feature Importance 분석 결과, `kwh_rolling_24h` (최근 24시간 평균)와  
    `kwh_lag_1` (직전 1시간)이 가장 강력한 예측 변수였다.
    → EV 충전 수요는 **현재 시각 자체보다 최근의 충전 관성**에 더 크게 의존한다.
    """)

with st.container(border=True):
    st.markdown("""
    #### 2. 요일 효과는 뚜렷했지만 공휴일 효과는 미미했다
    - 금요일 평균 충전량이 일요일 대비 **약 40% 높게** 관측됨
      → 주말 이동 대비 사전 충전 수요로 해석
    - 공휴일 효과는 평일 대비 **약 2%만** 증가
      → 예상과 달리 이벤트성 요인보다 **일상적 이동 패턴**의 영향이 지배적
    """)

with st.container(border=True):
    st.markdown("""
    #### 3. 지역·충전기 유형별 격차가 크다
    - **경기도**가 총 충전량 1위 (경상북·남도 뒤이음)
    - **DC급속** 충전기가 전체 충전량의 대부분 차지
      → 이용자들의 급속 충전 선호가 명확
    - 향후 지역·충전기 유형별 개별 모델 or 이를 Feature로 활용 시 개선 여지
    """)

with st.container(border=True):
    st.markdown("""
    #### 4. 복잡한 모델이 항상 우세하지는 않았다
    초기 XGBoost 기본 모델이 Rolling 24h Baseline보다 성능이 낮게 나왔다.  
    → **Baseline 검증의 중요성**을 재확인한 사례  
    → 이후 Weather 변수 추가 + Optuna 튜닝으로 최종 개선 확보
    """)

st.divider()

# =========================
# 한계 (핵심)
# =========================
st.header("⚠️ 한계")

st.markdown("R²가 0.3 수준에 머무는 이유를 정직하게 분석합니다.")

with st.container(border=True):
    st.markdown("""
    #### 1. 데이터 자체의 한계
    - **합성데이터 사용**: 원본이 아닌 한국환경공단의 AI 학습용 합성데이터  
      → 실제 측정 데이터의 통계적 특성은 반영되지만,  
        실제 이벤트(지역 축제, 프로모션, 도로 통제 등)와의 인과관계는 약화됨
    - **API 호출 제한**으로 월별 5,000건 균등 샘플링  
      → 시간당 평균 약 7건으로 **시계열 신호가 노이즈에 묻히는 구간** 존재
    - **1월 데이터 수집량이 특히 부족**  
      → 월별 EDA에서 1월 평균 충전량이 과소 추정됨
    """)

with st.container(border=True):
    st.markdown("""
    #### 2. Feature의 한계
    - 현재는 **시간·과거 수요·날씨 변수만** 사용  
    - 실제 수요에 영향을 줄 수 있는 요인 미포함:  
      - 지역 이벤트, 프로모션, 명절 이동
      - 유가, 전기차 신차 출시
      - 충전소 단위 특성 (개별 충전소 신설·고장 등)
    - 잔차 분석에서 **양의 방향으로 긴 꼬리** 관찰됨  
      → 급격한 수요 증가를 현재 Feature로는 설명 불가
    """)

with st.container(border=True):
    st.markdown("""
    #### 3. 모델링 접근의 한계
    - 전국 단위로 집계한 시계열 하나만 예측  
      → **충전소별·지역별 개별 예측 미실시**
    - 딥러닝(LSTM, TFT) 등 시퀀스 모델 미비교  
      → XGBoost 계열만 다룸
    """)

st.divider()

# =========================
# 향후 개선 방향
# =========================
st.header("🚀 향후 개선 방향")

improvement_data = [
    {
        "우선순위": "🔴 High",
        "항목": "실측 데이터 확보",
        "설명": "한국전력공사 원본 데이터(활용신청 후 이메일 수령)로 대체하여 R² 개선 검증"
    },
    {
        "우선순위": "🔴 High",
        "항목": "충전소·지역별 개별 모델",
        "설명": "전국 통합이 아닌 지역·충전기 유형별 시계열로 분리해 예측 → 세분화된 인사이트 확보"
    },
    {
        "우선순위": "🟡 Mid",
        "항목": "외생 변수 추가",
        "설명": "지역 이벤트, 명절, 프로모션, 유가 데이터 등 통합해 급증 구간 설명력 향상"
    },
    {
        "우선순위": "🟡 Mid",
        "항목": "딥러닝 모델 비교",
        "설명": "LSTM, TFT 등 시퀀스 모델과 성능 비교 → 두 번째 프로젝트로 확장"
    },
    {
        "우선순위": "🟢 Low",
        "항목": "실시간 서빙 파이프라인",
        "설명": "일 단위 배치 재학습 + 시간 단위 추론 구조 설계 → 실서비스 시나리오 구현"
    },
]

improvement_df = pd.DataFrame(improvement_data)
st.dataframe(improvement_df, use_container_width=True, hide_index=True)

st.divider()

# =========================
# 배운 점 (Lesson & Learn)
# =========================
st.header("🎓 Lesson & Learn")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("""
        #### 기술적 배운 점
        - **Baseline 없이는 개선 판단이 불가능**하다는 것을 실감했다.
        - 시계열 데이터는 **시간 순서 보존 분할**이 필수임을 재확인했다.
        - Optuna 튜닝 시 **Test set 누수**를 방지하는 검증 구조의 중요성을 학습했다.
        - 비동기(aiohttp) 처리로 대량 API 수집을 안정적으로 수행하는 실무 감각을 얻었다.
        """)

with col2:
    with st.container(border=True):
        st.markdown("""
        #### 태도적 배운 점
        - 모델 성능이 낮을 때 **원인을 데이터·Feature·방법론 차원에서 분리해 해석**하는 습관.
        - **좋은 결과보다 정직한 실험 기록**이 프로젝트 가치를 높인다는 것.
        - 신입 포트폴리오는 **R² 절대값보다 문제 접근 방법론과 한계 인식**이 중요하다는 것.
        - 완벽한 결과 대신 **실행 가능한 End-to-End 파이프라인**의 가치.
        """)

st.divider()

# =========================
# 마무리
# =========================
st.markdown("""
### 📌 마무리
본 프로젝트는 공공데이터 기반 EV 충전소 수요 예측이라는 문제를,  
**데이터 수집 → EDA → Feature Engineering → 모델링 → 배포**의 End-to-End 흐름으로 수행했습니다.

R² 0.3 수준의 결과가 절대적 성능으로는 만족스럽지 않지만,  
그 원인을 **데이터 특성·Feature 한계·모델 선택** 관점에서 명확히 분석하고,  
**향후 개선 로드맵**을 정리했다는 점에서 의미 있는 첫 프로젝트라고 생각합니다.

두 번째 프로젝트에서는 이 한계를 반영해 **딥러닝 기반 시계열 모델(LSTM/TFT)** 또는  
**자율주행 관련 CV 데이터셋** 등 분석 방법론의 다각화를 계획하고 있습니다.
""")

st.caption("📌 프로젝트 상세는 좌측 사이드바 각 페이지 및 [GitHub 저장소](https://github.com/pssjun/ev-charging-forecast)에서 확인 가능합니다.")