"""OrgPilot AI 대시보드.

실행:  streamlit run app/dashboard.py

모든 수치는 src/ 분석 모듈(pipeline.run_all)이 계산한다. 대시보드는 표시만 한다.
AI 설명은 저장된(수치 검증을 통과한) Gemini 응답만 보여 주고, API를 호출하지 않는다.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ai_explain as A
from src import staffing as S
from src.config import AS_OF, AVAILABLE_HOURS_PER_FTE
from src.kpi import OVERLOAD_THRESHOLD, UNDERUSE_THRESHOLD
from src.pipeline import run_all

st.set_page_config(page_title="OrgPilot AI", page_icon="📊", layout="wide")

STATUS_COLOR = {"과부하": "#e34948", "적정": "#a3a29b", "저활용": "#2a78d6"}
ACTUAL, FORECAST, MUTED = "#2a78d6", "#eb6834", "#8a8984"
SOURCE_LABEL = {"cache": "Gemini 생성 · 수치 검증 통과", "gemini": "Gemini 생성 · 수치 검증 통과",
                "fallback": "계산 결과 기반 기본 설명"}


@st.cache_resource(show_spinner="분석 모듈 실행 중…")
def results():
    return run_all()


def base_layout(fig, height=360, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(font_size=13), **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)")
    return fig


res = results()
names = res.staff.set_index("department_id")["name"]

st.title("OrgPilot AI")
st.caption(f"업무량 데이터 기반 조직 진단 · 적정 정원(TO) 산정 · 기준시점 {AS_OF}  |  "
           "⚠ 모든 데이터는 가상 기업 *Aster Digital*의 **합성 데이터**입니다.")

tab1, tab2 = st.tabs(["① 조직 현황", "② 적정 TO 산정"])

# ═════════════════════════════════════════════════════════════════
# ① 조직 현황 (Module A)
# ═════════════════════════════════════════════════════════════════
with tab1:
    k = res.kpis.sort_values("load_ratio", ascending=False)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전사 현원", f"{k['headcount'].sum()}명", help="휴직자 포함")
    c2.metric("현행 정원(TO)", f"{k['approved_to'].sum()}명")
    c3.metric("과부하 조직", f"{(k['status'] == '과부하').sum()}개", help=f"업무부하율 > {OVERLOAD_THRESHOLD}")
    c4.metric("저활용 조직", f"{(k['status'] == '저활용').sum()}개", help=f"업무부하율 < {UNDERUSE_THRESHOLD}")

    left, right = st.columns([1, 1.25])
    with left:
        kk = k.sort_values("load_ratio")
        fig = go.Figure(go.Bar(
            y=kk["name"], x=kk["load_ratio"] - 1, base=1, orientation="h",
            marker_color=kk["status"].map(STATUS_COLOR),
            text=[f"{v:.2f} {s}" for v, s in zip(kk["load_ratio"], kk["status"])], textposition="outside",
            customdata=kk[["status", "required_fte", "active_headcount", "overtime_per_person"]],
            hovertemplate="<b>%{y}</b><br>업무부하율 %{text}<br>필요 FTE %{customdata[1]:.1f} / 가용인원 %{customdata[2]}명"
                          "<br>1인당 월 초과근무 추정 %{customdata[3]:.1f}h<extra></extra>",
        ))
        fig.add_vline(x=1, line_color=MUTED, line_width=1)
        base_layout(fig, title="조직별 업무부하율 (최근 12개월, 1.0 = 적정)", showlegend=False)
        fig.update_xaxes(range=[0.75, 1.5])
        st.plotly_chart(fig, width="stretch")

    with right:
        default = k["name"].head(2).tolist() + k["name"].tail(1).tolist()
        pick = st.multiselect("월별 추이를 볼 조직", k["name"].tolist(), default=default)
        fig = go.Figure()
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
        order = res.staff["name"].tolist()  # 색은 조직에 고정 (선택 순서와 무관)
        for nm in pick:
            d = names[names == nm].index[0]
            g = res.monthly[res.monthly["department_id"] == d]
            fig.add_scatter(x=g["period"], y=g["load_ratio"], mode="lines+markers", name=nm,
                            line=dict(width=2, color=palette[order.index(nm)]), marker=dict(size=6),
                            hovertemplate=f"<b>{nm}</b> %{{x}}<br>업무부하율 %{{y:.2f}}<extra></extra>")
        fig.add_hline(y=1, line_dash="dash", line_color=MUTED, line_width=1)
        base_layout(fig, title="월별 업무부하율 — 분기말(3·6·9·12월) 피크", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")

    st.subheader("조직별 KPI")
    st.dataframe(
        k[["name", "status", "headcount", "active_headcount", "approved_to", "required_fte", "load_ratio",
           "peak_load_ratio", "overtime_per_person", "overload_months", "workload_yoy"]],
        hide_index=True, width="stretch",
        column_config={
            "name": "조직", "status": "상태", "headcount": "현원", "active_headcount": "가용인원",
            "approved_to": "현행 정원", "required_fte": st.column_config.NumberColumn("필요 FTE", format="%.1f"),
            "load_ratio": st.column_config.NumberColumn("업무부하율", format="%.2f"),
            "peak_load_ratio": st.column_config.NumberColumn("최대 월부하율", format="%.2f"),
            "overtime_per_person": st.column_config.NumberColumn("1인당 월 초과(h)", format="%.1f"),
            "overload_months": "과부하 개월수",
            "workload_yoy": st.column_config.NumberColumn("업무량 YoY", format="percent"),
        },
    )
    with st.expander("KPI 정의"):
        st.markdown(f"""
- **필요 FTE** = 업무건수 × 건당 표준시간 / 1인당 월 가용시간({AVAILABLE_HOURS_PER_FTE:.0f}h)
- **업무부하율** = 필요시간 / 가용시간 (가용시간 = 현원 × 휴직 제외 비율 × {AVAILABLE_HOURS_PER_FTE:.0f}h), 12개월 합계 기준
- **1인당 월 초과근무 추정** = max(0, 필요시간 − 가용시간) / 가용인원 — 부족분 전부를 초과근무로 메운다고 가정한 상한
- **상태**: 부하율 > {OVERLOAD_THRESHOLD} 과부하, < {UNDERUSE_THRESHOLD} 저활용
""")

# ═════════════════════════════════════════════════════════════════
# ② 적정 TO 산정 (Module B + C + AI 설명)
# ═════════════════════════════════════════════════════════════════
with tab2:
    staff = res.staff.sort_values("headcount_gap", ascending=False)
    # 슬라이더가 카드보다 먼저 와야 카드 수치도 같은 가정으로 계산된다
    err = st.slider("예측 오차 가정 — 실제 업무량이 예측보다 이만큼 많다면 (−: 적다면)", -20, 20, 0, step=5,
                    format="%d%%", help="0% = 예측대로. 검증된 오차(6개월 앞, 90분위)는 약 5%") / 100
    view = staff.copy()
    view["recommended_to"] = S.recommended_to(view["forecast_fte"] * (1 + err), view["active_ratio"])
    view["headcount_gap"] = view["recommended_to"] - view["headcount"]
    view["action"] = view["headcount_gap"].map(S.action_label)
    stab = res.stability.set_index("department_id")["stable"]
    view["stable"] = view["department_id"].map(stab).map({True: "안정", False: "재검토"})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("권장 정원 합계", f"{view['recommended_to'].sum()}명",
              f"{view['recommended_to'].sum() - view['approved_to'].sum():+d}명 vs 현행 정원", delta_color="off")
    c2.metric("충원 필요", f"+{view['headcount_gap'].clip(lower=0).sum()}명")
    c3.metric("재배치 검토", f"{view['headcount_gap'].clip(upper=0).sum()}명")
    bt = res.backtest.loc[S.F.MAIN_MODEL]
    c4.metric("예측 오차 (WAPE)", f"{bt['monthly_wape']:.1%}",
              f"현행 방식 {res.backtest.loc['baseline_mean', 'monthly_wape']:.1%}", delta_color="off",
              help="Rolling-origin 검증, 1~6개월 앞 월 업무량 기준")
    if err:
        st.caption(f"⚠ 예측 오차 {err:+.0%} 가정이 적용된 수치입니다. 아래 '조직 상세'의 AI 설명·스킬 매칭은 예측대로(0%) 기준입니다.")

    left, right = st.columns([1.1, 1])
    with left:
        vv = view.sort_values("headcount_gap")
        fig = go.Figure(go.Bar(
            y=vv["name"], x=vv["headcount_gap"], orientation="h",
            marker_color=["#e34948" if g > 0 else "#2a78d6" if g < 0 else "#a3a29b" for g in vv["headcount_gap"]],
            text=[f"{g:+d}명" if g else "0명" for g in vv["headcount_gap"]], textposition="outside",
            customdata=vv[["recommended_to", "headcount", "approved_to", "stable"]],
            hovertemplate="<b>%{y}</b><br>인원 격차 %{text}<br>권장 정원 %{customdata[0]}명 · 현원 %{customdata[1]}명"
                          " · 현행 정원 %{customdata[2]}명<br>±10% 오차 내 판단: %{customdata[3]}<extra></extra>",
        ))
        fig.add_vline(x=0, line_color=MUTED, line_width=1)
        lim = max(abs(vv["headcount_gap"]).max() + 5, 10)
        base_layout(fig, title=f"인원 격차 (권장 정원 − 현원) · 예측 오차 {err:+.0%} 가정", showlegend=False)
        fig.update_xaxes(range=[-lim, lim])
        st.plotly_chart(fig, width="stretch")
    with right:
        st.dataframe(
            view[["name", "headcount", "forecast_fte", "recommended_to", "headcount_gap", "action", "stable"]],
            hide_index=True, width="stretch", height=330,
            column_config={"name": "조직", "headcount": "현원", "approved_to": "현행 정원",
                           "forecast_fte": st.column_config.NumberColumn("필요 FTE", format="%.1f"),
                           "recommended_to": "권장 정원", "headcount_gap": "인원 격차", "action": "판단",
                           "stable": "±10% 판단"},
        )
    st.caption("빨강 = 충원 필요 · 파랑 = 재배치 검토. '±10% 판단'이 '재검토'인 조직은 예측 오차 범위 안에서 판단 방향이 바뀝니다.")

    st.divider()
    sel = st.selectbox("조직 상세", staff["name"].tolist())
    d = names[names == sel].index[0]
    row = res.staff.set_index("department_id").loc[d]

    left, right = st.columns([1.3, 1])
    with left:
        h = res.monthly[res.monthly["department_id"] == d]
        f = res.forecast[res.forecast["department_id"] == d]
        ff = f["task_count_pred"] * f["standard_hours"] / AVAILABLE_HOURS_PER_FTE
        fig = go.Figure()
        fig.add_scatter(x=h["period"], y=h["required_fte"], name="실적 필요 FTE", mode="lines",
                        line=dict(color=ACTUAL, width=2), hovertemplate="%{x}<br>실적 %{y:.1f} FTE<extra></extra>")
        fig.add_scatter(x=[h["period"].iloc[-1], *f["period"]], y=[h["required_fte"].iloc[-1], *ff], name="예측 필요 FTE",
                        mode="lines", line=dict(color=FORECAST, width=2, dash="dash"),
                        hovertemplate="%{x}<br>예측 %{y:.1f} FTE<extra></extra>")
        fig.add_scatter(x=[*f["period"], *f["period"][::-1]], y=[*(ff * 1.1), *(ff * 0.9)[::-1]], fill="toself",
                        fillcolor="rgba(235,104,52,0.12)", line=dict(width=0), name="예측 ±10%", hoverinfo="skip")
        fig.add_hline(y=row["active_headcount"], line_dash="dot", line_color=MUTED,
                      annotation_text=f"가용인원 {row['active_headcount']}명", annotation_position="top left")
        base_layout(fig, title=f"{sel} — 필요 FTE 실적과 12개월 예측", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("**AI 설명**")
        e = A.explain_gap(res, d, model=A.model_name(), allow_call=False)  # 대시보드는 API를 호출하지 않음
        st.info(e.text)
        st.caption(f"{SOURCE_LABEL[e.source]}" + (f" · {e.model}" if e.source != "fallback" else "")
                   + " · 모든 수치는 분석 모듈의 계산값이며, 응답 속 숫자가 계산 결과에 없으면 표시하지 않습니다.")

    st.markdown("**스킬 기반 충원 방식** (규칙 기반 후보 — 확정안 아님)")
    gap = res.skill_gap[res.skill_gap["department_id"] == d]
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.dataframe(gap[["skill", "required", "supply", "gap", "status"]], hide_index=True, width="stretch",
                     column_config={"skill": "필요 스킬", "required": "필요 인원", "supply": "보유(숙련도 3+)",
                                    "gap": "격차", "status": "상태"})
    with c2:
        fp = res.fill_plan.set_index("department_id")
        if d in fp.index:
            p = fp.loc[d]
            m1, m2, m3 = st.columns(3)
            m1.metric("내부 재배치", f"{p['internal_redeploy']}명")
            m2.metric("외부 채용", f"{p['external_hire']}명")
            m3.metric("육성 필요(최소)", f"{p['upskill_needed']}명")
            st.caption(f"채용 시 우선 스킬: {p['skills_to_hire_for']}")
            inbound = res.redeploy[res.redeploy["to_dept"] == d]
            if len(inbound):
                st.dataframe(inbound[["employee_id", "from_name", "fills_skills", "proficiency"]], hide_index=True,
                             width="stretch",
                             column_config={"employee_id": "사번", "from_name": "현 소속", "fills_skills": "메우는 부족 스킬",
                                            "proficiency": "숙련도"})
        else:
            outbound = res.redeploy[res.redeploy["from_dept"] == d]
            st.metric("재배치 후보", f"{len(outbound)}명", help="충원 필요 조직의 부족 스킬을 보유하고, 빠져도 이 조직에 스킬 부족이 생기지 않는 인원")
            if len(outbound):
                st.dataframe(outbound[["employee_id", "to_name", "fills_skills", "proficiency"]], hide_index=True,
                             width="stretch",
                             column_config={"employee_id": "사번", "to_name": "후보 조직", "fills_skills": "메우는 부족 스킬",
                                            "proficiency": "숙련도"})
            else:
                st.caption("재배치 대상 아님")

    with st.expander("산정 방법"):
        st.markdown("""
- **예측**: 조직별 `log(업무건수) = a + b·t + c·분기말 + d·1월` 회귀, Rolling-origin 검증(1~6개월 앞)
- **필요 FTE** = 예측 업무건수 × 건당 표준시간 / 140h (향후 12개월 평균)
- **권장 정원** = ceil(필요 FTE / 가용비율) — 정원에는 휴직자도 포함되므로 머릿수로 환산
- **인원 격차** = 권장 정원 − 현원 (+ 충원 필요 / − 재배치 검토)
- **판단 안정성**: 예측 오차 ±10% 안에서 충원/재배치 방향이 바뀌지 않으면 '안정'
- **스킬 매칭**: 공급 = 숙련도 3 이상 가용 인원, 수요 = 필요 비율 × 권장 정원. 재배치 후보는 빠져도 원 조직에 스킬 부족이 생기지 않는 인원만
- 자세한 내용: `docs/methodology.md`
""")
