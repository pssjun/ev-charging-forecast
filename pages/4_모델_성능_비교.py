import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="모델 성능 비교", page_icon="📈", layout="wide")

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
    baseline_path = Path("data/processed/baseline_compare.csv")
    importance_path = Path("data/processed/feature_importance.csv")
    result_path = Path("data/processed/result_df.parquet")

    final_compare = pd.read_csv(final_path) if final_path.exists() else None
    baseline_compare = pd.read_csv(baseline_path) if baseline_path.exists() else None
    importance = pd.read_csv(importance_path) if importance_path.exists() else None
    result_df = pd.read_parquet(result_path) if result_path.exists() else None

    return final_compare, baseline_compare, importance, result_df

final_compare, baseline_compare, importance, result_df = load_data()

if final_compare is None:
    st.error("`data/processed/final_compare.csv` 파일이 없습니다. 노트북에서 저장했는지 확인하세요.")
    st.stop()

# =========================
# 헤더
# =========================
st.title("📈 모델 성능 비교")
st.caption("Baseline → XGBoost → +Weather → +Optuna 단계별 성능 향상 과정을 비교합니다.")
st.divider()

# =========================
# 최종 요약 지표 카드
# =========================
best_row = final_compare.loc[final_compare["R2"].idxmax()]
worst_row = final_compare.loc[final_compare["R2"].idxmin()]

r2_improvement = best_row["R2"] - worst_row["R2"]
rmse_improvement = worst_row["RMSE"] - best_row["RMSE"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("최고 모델", best_row["Model"])
c2.metric("최종 R²", f"{best_row['R2']:.3f}")
c3.metric("최종 RMSE", f"{best_row['RMSE']:.2f}")
c4.metric(
    "R² 개선",
    f"+{r2_improvement:.3f}",
    help=f"최악 모델 대비 R² 개선폭"
)

st.divider()

# =========================
# 최종 모델 비교 테이블
# =========================
st.subheader("🏆 최종 모델 비교")

# 소수점 정리
display_final = final_compare.copy()
for col in ["MAE", "RMSE", "R2", "MAPE", "Peak_MAE"]:
    if col in display_final.columns:
        display_final[col] = display_final[col].round(3)

st.dataframe(
    display_final.style
        .highlight_min(subset=["MAE", "RMSE", "MAPE", "Peak_MAE"], color="lightgreen")
        .highlight_max(subset=["R2"], color="lightgreen"),
    use_container_width=True,
    hide_index=True
)

st.caption("💡 초록색: 각 지표별 최고 성능")

with st.expander("📖 지표 설명"):
    st.markdown("""
    - **MAE**: 평균 절대 오차 (kWh 단위 직관적 해석)
    - **RMSE**: 제곱근 평균 제곱 오차 (큰 오차에 민감)
    - **R²**: 결정계수 (모델이 수요 변동성을 얼마나 설명하는지, 1에 가까울수록 좋음)
    - **MAPE**: 평균 절대 백분율 오차 (%)
    - **Peak_MAE**: 실제 수요 상위 20% 구간의 MAE (**운영 관점 핵심 지표**)
    """)

st.divider()

# =========================
# 모델별 지표 시각화
# =========================
st.subheader("📊 지표별 성능 비교")

metric_col = st.radio(
    "비교할 지표 선택",
    ["R2", "RMSE", "MAE", "MAPE", "Peak_MAE"],
    horizontal=True
)

# R2는 높을수록 좋고, 나머지는 낮을수록 좋음
is_higher_better = metric_col == "R2"

fig_metric = px.bar(
    final_compare,
    x="Model", y=metric_col,
    color=metric_col,
    color_continuous_scale="Blues" if is_higher_better else "Reds_r",
    text=final_compare[metric_col].round(3)
)
fig_metric.update_traces(textposition="outside")
fig_metric.update_layout(
    xaxis_title="",
    yaxis_title=metric_col,
    showlegend=False,
    height=450
)
st.plotly_chart(fig_metric, use_container_width=True)

st.divider()

# =========================
# Baseline 5종 상세 비교
# =========================
if baseline_compare is not None:
    st.subheader("🎯 Baseline 5종 상세 비교")
    st.caption("XGBoost 도입 이전, 단순 예측 모델들의 성능을 먼저 비교했습니다.")

    display_baseline = baseline_compare.copy()
    for col in ["MAE", "RMSE", "R2", "MAPE", "Peak_MAE"]:
        if col in display_baseline.columns:
            display_baseline[col] = display_baseline[col].round(3)

    st.dataframe(
        display_baseline.style
            .highlight_min(subset=["MAE", "RMSE", "MAPE", "Peak_MAE"], color="lightgreen")
            .highlight_max(subset=["R2"], color="lightgreen"),
        use_container_width=True,
        hide_index=True
    )

    with st.expander("💡 Baseline 5종 설명"):
        st.markdown("""
        - **Naive lag_1**: 직전 1시간 값이 다음 시간에도 유지된다고 가정
        - **Seasonal naive lag_24**: 24시간 전(하루 전 같은 시간) 값을 예측값으로
        - **Seasonal naive lag_168**: 168시간 전(1주일 전 같은 시간) 값을 예측값으로
        - **Rolling 24h**: 직전 24시간 이동평균
        - **Rolling 168h**: 직전 168시간(1주일) 이동평균

        시계열 모델링에서는 이런 단순 baseline이 예상외로 강력할 수 있습니다.  
        복잡한 ML 모델이 baseline보다 나은 성능을 내는지 반드시 검증해야 합니다.
        """)

    st.divider()

# =========================
# Feature Importance
# =========================
if importance is not None:
    st.subheader("🧩 Feature Importance (최종 모델 기준)")

    importance_sorted = importance.sort_values("importance", ascending=True)

    fig_imp = px.bar(
        importance_sorted,
        x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues"
    )
    fig_imp.update_layout(
        xaxis_title="Importance",
        yaxis_title="",
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("💡 Feature Importance 해석"):
        top_feature = importance.iloc[0]["feature"]
        st.markdown(f"""
        - 가장 중요한 변수는 **`{top_feature}`** 였습니다.
        - 상위 변수 대부분이 **과거 충전량 기반 Lag/Rolling 변수**로 나타났으며,  
          이는 EV 충전 수요가 최근 충전 패턴에 강하게 의존한다는 것을 의미합니다.
        - 시간(hour), 요일(weekday) 같은 시간 변수의 절대 중요도는 상대적으로 낮았으나,  
          다른 변수들과의 상호작용을 통해 예측에 기여했습니다.
        - 날씨 변수(temp, rainfall 등)는 예상보다 영향력이 크지 않았는데,  
          이는 합성데이터 특성상 실제 기상 이벤트와의 연관성이 약할 수 있음을 시사합니다.
        """)

    st.divider()

# =========================
# 잔차 분포
# =========================
if result_df is not None:
    st.subheader("📉 잔차 분포 (최종 모델)")

    result_df["residual"] = result_df["actual"] - result_df["predicted"]

    fig_resid = px.histogram(
        result_df,
        x="residual",
        nbins=40,
        color_discrete_sequence=["#1f77b4"]
    )
    fig_resid.add_vline(x=0, line_dash="dash", line_color="red")
    fig_resid.update_layout(
        xaxis_title="잔차 (실제값 - 예측값)",
        yaxis_title="빈도",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_resid, use_container_width=True)

    # 잔차 요약 통계
    residual_stats = {
        "평균": result_df["residual"].mean(),
        "중앙값": result_df["residual"].median(),
        "표준편차": result_df["residual"].std(),
        "최댓값": result_df["residual"].max(),
        "최솟값": result_df["residual"].min()
    }

    stat_cols = st.columns(5)
    for i, (label, value) in enumerate(residual_stats.items()):
        stat_cols[i].metric(label, f"{value:.2f}")

    with st.expander("💡 잔차 분포 해석"):
        st.markdown("""
        - 잔차 평균이 0에 가까워 전반적인 예측 편향은 크지 않습니다.
        - 다만 **양의 방향으로 긴 꼬리**가 관찰되며,  
          이는 일부 시간대에 발생한 급격한 수요 증가를 모델이 충분히 설명하지 못했음을 의미합니다.
        - 향후 지역 정보, 충전기 유형, 특별 이벤트 정보 등 외생 변수를 추가하면  
          이런 급증 예측 성능을 개선할 수 있을 것으로 판단됩니다.
        """)

st.divider()

# =========================
# 종합 해석
# =========================
st.subheader("📝 종합 해석")

st.markdown("""
1. **Baseline이 예상보다 강력했다**  
   단순 이동평균(Rolling 24h)이 초기 XGBoost와 비슷하거나 오히려 나은 결과를 냈습니다.  
   시계열 예측에서 baseline 검증의 중요성을 확인한 사례였습니다.

2. **날씨 변수의 효과는 제한적이었다**  
   기온·강수량 등 외부 변수를 추가했지만 성능 개선폭은 크지 않았습니다.  
   합성데이터 특성상 실제 기상 이벤트와의 연관성이 약할 수 있습니다.

3. **Optuna 튜닝으로 안정적 개선**  
   Validation set 기반 튜닝(Test set 누수 방지)을 통해 최종 성능을 확보했습니다.

4. **한계는 명확했다**  
   R² 절대값이 여전히 낮은 것은 60K건 샘플 데이터와 합성데이터 자체의 통계적 특성 한계에서 비롯된 것으로 판단됩니다.  
   실측 데이터 전량 확보 시 상당한 개선이 가능할 것으로 예상됩니다.
""")

st.caption("📌 자세한 결론과 한계 논의는 좌측 사이드바의 **결론과 한계** 페이지를 참조하세요.")