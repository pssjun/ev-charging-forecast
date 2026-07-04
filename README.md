# ⚡ EV 충전소 시간대별 수요 예측

> 공공데이터 OpenAPI 기반 전기차 충전 수요 분석 및 예측 프로젝트

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ev-charging-forecast-cert26wrpydmhpk6nfy4ap.streamlit.app/)

**🔗 [Live Demo 바로가기](https://ev-charging-forecast-cert26wrpydmhpk6nfy4ap.streamlit.app/)**

---

## 📌 프로젝트 개요

전기차 보급이 빠르게 확대되면서 충전 인프라 운영 효율성이 사회적 과제로 떠오르고 있다.  
본 프로젝트는 EV 충전소의 **시간대별 총 충전량(kWh)** 을 예측하여 충전 인프라 운영 최적화에 활용할 수 있는 분석 모델을 구축한다.

- **기간**: 2024.01 ~ 2024.12
- **데이터**: 한국환경공단 EV 충전 이력 합성데이터 (약 60,000건) + 기상청 ASOS 시간자료
- **문제 유형**: 시계열 회귀 (Regression)
- **타깃 변수**: 시간당 총 충전량 (`total_charge_kwh`)

---

## 🎯 주요 성과

| 항목 | 값 |
|---|---|
| 최종 R² | **0.285** |
| 최종 RMSE | **62.45** kWh |
| Baseline 대비 R² 개선 | +0.015 |
| 데이터 수집 | 60K 건 (aiohttp 비동기 병렬) |
| 모델링 단계 | 4단계 (Baseline → XGBoost → +Weather → +Optuna) |

---

## 🛠️ 사용 기술

- **언어**: Python 3.14
- **데이터 수집**: `aiohttp` (비동기 병렬), `requests`
- **데이터 처리**: `pandas`, `numpy`, `pyarrow`
- **모델링**: `xgboost`, `scikit-learn`, `optuna`
- **시각화**: `matplotlib`, `seaborn`, `plotly`
- **배포**: `streamlit`, Streamlit Cloud

---

## 📂 프로젝트 구조
ev-charging-forecast/
├── streamlit_app.py               # 홈 (프로젝트 개요)
├── pages/
│   ├── 1_📊_EDA_대시보드.py       # EDA 인터랙티브 시각화
│   ├── 2_🔮_수요_예측.py          # 예측 결과 조회
│   ├── 3_📈_모델_성능_비교.py     # 단계별 모델 비교
│   └── 4_📝_결론과_한계.py        # 결론 및 향후 개선 방향
├── data/
│   ├── raw/                       # 원본 API 수집 데이터 (.gitignore)
│   └── processed/                 # 학습·시각화용 가공 데이터
├── models/                        # 학습된 모델 (.pkl)
├── notebooks/                     # 분석 노트북
└── requirements.txt
---

## 🚀 실행 방법

### 로컬 실행

```bash
# 저장소 클론
git clone https://github.com/pssjun/ev-charging-forecast.git
cd ev-charging-forecast

# 가상환경 & 패키지 설치
pip install -r requirements.txt

# Streamlit 실행
streamlit run streamlit_app.py
```

### 배포 URL
- **Streamlit Cloud**: https://ev-charging-forecast-cert26wrpydmhpk6nfy4ap.streamlit.app/

---

## 🔬 분석 및 모델링 흐름

### 1. 데이터 수집
- 공공데이터포털 API를 `aiohttp` 비동기 병렬 처리로 호출
- 월별 parquet 체크포인트 저장 → 중단·재시작 안정성 확보
- 기상청 ASOS 시간자료 API로 날씨 데이터 통합

### 2. 전처리 & Feature Engineering
- 결측치 처리, 이상치 필터링 (충전 시간 0분 이하 or 24시간 초과 제외)
- 시간 단위(hourly) 집계 후 결측 시간 채움 (`pd.date_range` reindex)
- **Lag Features**: `kwh_lag_1`, `kwh_lag_24`, `kwh_lag_168`
- **Rolling Features**: `kwh_rolling_24h`, `kwh_rolling_168h`
- **외부 변수**: 날씨(기온·강수·풍속·습도), 공휴일 여부

### 3. 데이터 분할
- 시계열 특성 고려: **Train / Validation / Test = 60 / 20 / 20 시간 순서 보존**
- Optuna 튜닝은 Validation set으로 → Test set 데이터 누수 방지

### 4. 모델링 (4단계 점진적 향상)
| 단계 | 모델 | 목적 |
|---|---|---|
| 1 | Baseline 5종 (Naive / Seasonal / Rolling) | 기준선 정립 |
| 2 | XGBoost 기본 | 시간 + Lag/Rolling 변수 |
| 3 | XGBoost + Weather | 외부 변수 효과 검증 |
| 4 | XGBoost + Weather + Optuna | 하이퍼파라미터 튜닝 |

### 5. 평가 지표
- **MAE / RMSE / R² / MAPE**: 일반 회귀 지표
- **Peak_MAE**: 실제 수요 상위 20% 구간의 MAE → 운영 관점 핵심 지표

---

## 💡 주요 인사이트

1. **과거 수요 패턴이 예측의 핵심** — Rolling 24h와 Lag 1h가 가장 강력한 Feature
2. **요일 효과는 뚜렷** — 금요일이 일요일 대비 약 40% 높음 (주말 대비 사전 충전)
3. **공휴일 효과는 미미** — 평일 대비 2% 증가 수준 → 일상적 이동 패턴이 지배적
4. **복잡한 모델이 항상 우세하지는 않다** — 초기 XGBoost가 Rolling Baseline보다 낮음

---

## ⚠️ 한계 & 향후 개선

### 한계
- **합성데이터 특성**: 실제 이벤트(축제·프로모션·도로 통제)와의 인과관계 약화
- **60K 건 샘플 한계**: 시간당 평균 약 7건으로 시계열 신호가 노이즈에 묻힘
- **Feature 부족**: 지역 이벤트·유가·충전소 단위 특성 등 미포함

### 향후 개선 방향
- 🔴 **High**: 한전 실측 데이터 확보 후 성능 재검증
- 🔴 **High**: 충전소·지역별 개별 예측 모델 구축
- 🟡 **Mid**: 외생 변수(이벤트·프로모션·유가) 통합
- 🟡 **Mid**: LSTM / TFT 등 딥러닝 시퀀스 모델 비교

---

## 📊 데이터 출처

- [공공데이터포털 - 한국환경공단 EV 충전 이력 합성데이터](https://www.data.go.kr/)
- [기상청 ASOS 시간자료 API](https://www.data.go.kr/data/15057210/openapi.do)

---

## ⚠️ 참고 사항

- 실행 시 브라우저 자동 번역 기능을 **끄고** 접속해 주세요. (한글 UI 왜곡 방지)
- Streamlit Cloud 배포 앱은 최초 접속 시 sleep 상태에서 깨어나는 데 30초~1분 소요될 수 있습니다.

---

## 👤 About

- **작성자**: pssjun
- **프로젝트 유형**: 데이터 사이언스 포트폴리오 (신입 취업 준비)
- **작업 기간**: 2026.01~2026.06