"""Module A. 조직 생산성 진단 KPI.

정의 (조직 d, 월 t):
  필요시간      required_hours[t] = task_count[t] × standard_hours
  가용인원      available_hc[t]   = actual_headcount[t] × active_ratio
                active_ratio      = 기준시점 active 인원 / 현원  (휴직 이력이 없어 기준시점 비율이 유지된다고 가정)
  가용시간      capacity_hours[t] = available_hc[t] × AVAILABLE_HOURS_PER_FTE
  업무부하율    load_ratio[t]     = required_hours[t] / capacity_hours[t]
  초과근무 추정 overtime_hours[t] = max(0, required_hours[t] − capacity_hours[t])
                                    (모자란 시간을 기존 인원이 초과근무로 메운다고 가정한 상한 추정치)

스냅샷 KPI는 최근 12개월(계절성 1주기)을 집계해 계산한다.
"""
from __future__ import annotations

import pandas as pd

from src.config import AS_OF, AVAILABLE_HOURS_PER_FTE

OVERLOAD_THRESHOLD = 1.05   # 부하율이 이보다 크면 '과부하'
UNDERUSE_THRESHOLD = 0.95   # 이보다 작으면 '저활용'
WINDOW_MONTHS = 12

Tables = dict[str, pd.DataFrame]


def active_ratio(employees: pd.DataFrame) -> pd.Series:
    """조직별 가용(active) 인원 비율."""
    is_active = employees["status"].eq("active")
    return is_active.groupby(employees["department_id"]).mean().rename("active_ratio")


def classify_load(load_ratio: float) -> str:
    if load_ratio > OVERLOAD_THRESHOLD:
        return "과부하"
    if load_ratio < UNDERUSE_THRESHOLD:
        return "저활용"
    return "적정"


def monthly_metrics(t: Tables) -> pd.DataFrame:
    """조직 × 월 단위 KPI."""
    df = t["workload"].merge(t["staffing_plan"], on=["department_id", "period"], validate="1:1")
    df = df.merge(active_ratio(t["employees"]), left_on="department_id", right_index=True, validate="m:1")

    df["required_hours"] = df["task_count"] * df["standard_hours"]
    df["available_hc"] = df["actual_headcount"] * df["active_ratio"]
    df["capacity_hours"] = df["available_hc"] * AVAILABLE_HOURS_PER_FTE
    df["required_fte"] = df["required_hours"] / AVAILABLE_HOURS_PER_FTE
    df["load_ratio"] = df["required_hours"] / df["capacity_hours"]
    df["tasks_per_fte"] = df["task_count"] / df["available_hc"]
    df["overtime_hours"] = (df["required_hours"] - df["capacity_hours"]).clip(lower=0)
    df["overtime_per_person"] = df["overtime_hours"] / df["available_hc"]
    return df.sort_values(["department_id", "period"]).reset_index(drop=True)


def department_kpis(t: Tables, as_of: str = AS_OF, window: int = WINDOW_MONTHS) -> pd.DataFrame:
    """기준시점 직전 `window`개월을 집계한 조직별 KPI 스냅샷."""
    m = monthly_metrics(t)
    periods = sorted(m["period"].unique())
    if as_of not in periods:
        raise ValueError(f"as_of {as_of} not in data")
    end = periods.index(as_of)
    if end + 1 < 2 * window:
        raise ValueError(f"전년 대비 성장률 계산에 {2 * window}개월이 필요합니다")
    recent = m[m["period"].isin(periods[end - window + 1 : end + 1])]
    prior = m[m["period"].isin(periods[end - 2 * window + 1 : end - window + 1])]

    g = recent.groupby("department_id")
    out = pd.DataFrame({
        "required_fte": g["required_fte"].mean(),
        # 비율의 평균이 아니라 합계의 비율: 업무량이 큰 달에 가중치가 더 실리도록
        "load_ratio": g["required_hours"].sum() / g["capacity_hours"].sum(),
        "peak_load_ratio": g["load_ratio"].max(),
        "tasks_per_fte": g["task_count"].sum() / g["available_hc"].sum(),
        "overtime_per_person": g["overtime_per_person"].mean(),
        "overload_months": g["load_ratio"].apply(lambda s: int((s > 1).sum())),
        "workload_yoy": g["required_hours"].sum() / prior.groupby("department_id")["required_hours"].sum() - 1,
    })

    now = m[m["period"] == as_of].set_index("department_id")
    dept = t["departments"].set_index("department_id")
    out.insert(0, "name", dept["name"])
    out.insert(1, "headcount", dept["current_headcount"])
    out.insert(2, "active_headcount", t["employees"].query("status == 'active'").groupby("department_id").size())
    out.insert(3, "approved_to", now["approved_to"])
    out["status"] = out["load_ratio"].map(classify_load)
    return out.reset_index()


if __name__ == "__main__":
    from src.data_gen import SCHEMA, load

    tables = {n: load(n) for n in SCHEMA}
    k = department_kpis(tables)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(k.to_string(index=False))
