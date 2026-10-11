"""Module B-2. 예측 업무량 → 필요 FTE → 적정 정원(TO) → 인력 격차 + 민감도.

  필요 FTE      = 예측 업무건수 × 건당 표준시간 / 1인당 월 가용시간      (향후 12개월 평균)
  권장 정원     = ceil(필요 FTE / active_ratio)
                  정원에는 휴직자도 포함되므로 가용 비율로 나눠 '머릿수'로 환산.
                  정원은 정수이고 인력 부족의 비용(과부하·이탈)이 더 크다고 보고 올림.
  FTE 격차      = 필요 FTE − 가용인원           (계획서 정의: 지금 실제로 일하는 인원 대비)
  인원 격차     = 권장 정원 − 현원              (+: 충원 필요, −: 재배치 검토)
  정원 조정폭   = 권장 정원 − 현재 승인 정원    (현행 TO가 얼마나 어긋나 있는지)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import forecast as F
from src.config import AS_OF, AVAILABLE_HOURS_PER_FTE
from src.kpi import active_ratio

SENSITIVITY_ERRORS = (-0.20, -0.10, 0.0, 0.10, 0.20)

Tables = dict[str, pd.DataFrame]


def required_fte(fc: pd.DataFrame) -> pd.DataFrame:
    """월별 예측 → 조직별 향후 12개월 평균·최대 필요 FTE."""
    fte = fc["task_count_pred"] * fc["standard_hours"] / AVAILABLE_HOURS_PER_FTE
    g = fc.assign(fte=fte).groupby("department_id")["fte"]
    return pd.DataFrame({"forecast_fte": g.mean(), "peak_fte": g.max()})


def recommended_to(fte: pd.Series | float, ratio: pd.Series | float):
    # 부동소수 오차로 12.000000001 → 13이 되지 않도록 소수 6자리에서 반올림 후 올림
    return np.ceil(np.round(fte / ratio, 6)).astype(int)


def action_label(gap: int) -> str:
    if gap > 0:
        return f"충원 필요 (+{gap})"
    if gap < 0:
        return f"재배치 검토 ({gap})"
    return "유지"


def staffing_table(t: Tables, fc: pd.DataFrame) -> pd.DataFrame:
    dept = t["departments"].set_index("department_id")
    emp = t["employees"]
    sp = t["staffing_plan"]
    now = sp[sp["period"] == sp["period"].max()].set_index("department_id")

    out = required_fte(fc)
    out.insert(0, "name", dept["name"])
    out.insert(1, "headcount", dept["current_headcount"])
    out.insert(2, "active_headcount", emp.query("status == 'active'").groupby("department_id").size())
    out.insert(3, "approved_to", now["approved_to"])
    out["active_ratio"] = active_ratio(emp)
    out["recommended_to"] = recommended_to(out["forecast_fte"], out["active_ratio"])
    out["fte_gap"] = out["forecast_fte"] - out["active_headcount"]
    out["headcount_gap"] = out["recommended_to"] - out["headcount"]
    out["to_change"] = out["recommended_to"] - out["approved_to"]
    out["action"] = out["headcount_gap"].map(action_label)
    return out.reset_index()


def sensitivity(table: pd.DataFrame, errors=SENSITIVITY_ERRORS) -> pd.DataFrame:
    """예측이 e만큼 틀렸다면(실제 = 예측 × (1+e)) 권장 정원·인원 격차가 어떻게 바뀌는가."""
    rows = []
    for _, r in table.iterrows():
        for e in errors:
            rec = int(recommended_to(r["forecast_fte"] * (1 + e), r["active_ratio"]))
            rows.append((r["department_id"], r["name"], e, rec, rec - r["headcount"]))
    return pd.DataFrame(rows, columns=["department_id", "name", "error", "recommended_to", "headcount_gap"])


def decision_stability(sens: pd.DataFrame, band: float = 0.10) -> pd.DataFrame:
    """±band 오차 안에서 판단 방향(충원/유지/재배치)이 바뀌지 않으면 '안정'."""
    s = sens[sens["error"].abs() <= band + 1e-9]
    g = s.groupby("department_id")["headcount_gap"]
    out = pd.DataFrame({"gap_min": g.min(), "gap_max": g.max()})
    direction = np.sign(out)
    out["stable"] = direction["gap_min"] == direction["gap_max"]
    return out.reset_index()


def run(t: Tables, model: str = F.MAIN_MODEL, horizon: int = 12, as_of: str = AS_OF):
    """대시보드·AI 설명에서 쓰는 단일 진입점: (월별 예측, 정원 표, 민감도 표)."""
    df = F.prepare(t["workload"])
    fc = F.forecast(df, model, horizon, as_of)
    table = staffing_table(t, fc)
    return fc, table, sensitivity(table)


if __name__ == "__main__":
    from src.data_gen import SCHEMA, load

    tables = {n: load(n) for n in SCHEMA}
    fc, table, sens = run(tables)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(table.drop(columns=["active_ratio"]).to_string(index=False))
        print()
        print(sens.pivot(index="name", columns="error", values="headcount_gap").to_string())
        print()
        print(decision_stability(sens).to_string(index=False))
