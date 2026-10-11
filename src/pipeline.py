"""모든 분석 모듈을 한 번에 실행해 결과를 묶는다. 대시보드·AI 설명이 같은 결과를 공유하도록 하는 단일 진입점."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import forecast as F
from src import kpi
from src import skill_match as C
from src import staffing as S
from src.data_gen import SCHEMA, load


@dataclass
class Results:
    tables: dict[str, pd.DataFrame]
    kpis: pd.DataFrame          # Module A 스냅샷
    monthly: pd.DataFrame       # Module A 월별
    coef: pd.DataFrame          # Module B 회귀계수
    backtest: pd.DataFrame      # Module B 검증 요약
    forecast: pd.DataFrame      # Module B 월별 예측
    staff: pd.DataFrame         # Module B 정원 표
    sensitivity: pd.DataFrame   # Module B 민감도
    stability: pd.DataFrame     # Module B 판단 안정성
    skill_gap: pd.DataFrame     # Module C
    redeploy: pd.DataFrame      # Module C 재배치 후보
    fill_plan: pd.DataFrame     # Module C 충원 방식


def run_all(tables: dict[str, pd.DataFrame] | None = None, backtest: bool = True) -> Results:
    tables = tables or {n: load(n) for n in SCHEMA}
    df = F.prepare(tables["workload"])
    fc, staff, sens = S.run(tables)
    redeploy, fill_plan = C.redeployment_plan(tables, staff)
    return Results(
        tables=tables,
        kpis=kpi.department_kpis(tables),
        monthly=kpi.monthly_metrics(tables),
        coef=F.fit_seasonal_trend(df),
        backtest=F.summarize_backtest(F.backtest(df)) if backtest else pd.DataFrame(),
        forecast=fc,
        staff=staff,
        sensitivity=sens,
        stability=S.decision_stability(sens),
        skill_gap=C.skill_gap(tables, staff),
        redeploy=redeploy,
        fill_plan=fill_plan,
    )
