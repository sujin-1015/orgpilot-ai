import pandas as pd
import pytest

from src import kpi
from src.config import AVAILABLE_HOURS_PER_FTE as H
from src.data_gen import generate, make_periods

PERIODS = make_periods()


def _toy_tables(task_count: int = 100, std_hours: float = 14.0, headcount: int = 10, n_leave: int = 0):
    """1개 조직, 업무량 일정: 손으로 계산 가능한 최소 데이터."""
    n = len(PERIODS)
    return {
        "departments": pd.DataFrame({"department_id": ["X"], "name": ["테스트팀"], "function": ["T"],
                                     "current_headcount": [headcount]}),
        "employees": pd.DataFrame({"employee_id": [f"E{i}" for i in range(headcount)],
                                   "department_id": "X", "job_family": "개발", "tenure": 1.0,
                                   "status": ["on_leave"] * n_leave + ["active"] * (headcount - n_leave)}),
        "workload": pd.DataFrame({"department_id": "X", "period": PERIODS,
                                  "task_count": [task_count] * n, "standard_hours": std_hours}),
        "staffing_plan": pd.DataFrame({"department_id": "X", "period": PERIODS,
                                       "approved_to": headcount, "actual_headcount": headcount}),
    }


def test_balanced_department_hand_calculation():
    # 100건 × 14h = 1,400h = 10명 × 140h → 부하율 정확히 1.0, 초과근무 0
    k = kpi.department_kpis(_toy_tables()).iloc[0]
    assert k["required_fte"] == pytest.approx(1400 / H)
    assert k["load_ratio"] == pytest.approx(1.0)
    assert k["overtime_per_person"] == 0
    assert k["tasks_per_fte"] == pytest.approx(10.0)
    assert k["workload_yoy"] == pytest.approx(0.0)
    assert k["status"] == "적정"


def test_overload_and_overtime():
    # 120건 × 14h = 1,680h, 가용 1,400h → 부하율 1.2, 1인당 초과 28h
    k = kpi.department_kpis(_toy_tables(task_count=120)).iloc[0]
    assert k["load_ratio"] == pytest.approx(1.2)
    assert k["overtime_per_person"] == pytest.approx(28.0)
    assert k["overload_months"] == 12
    assert k["status"] == "과부하"


def test_on_leave_reduces_capacity():
    # 10명 중 2명 휴직 → 가용 8명 × 140h = 1,120h, 필요 1,400h → 1.25
    k = kpi.department_kpis(_toy_tables(n_leave=2)).iloc[0]
    assert k["active_headcount"] == 8
    assert k["load_ratio"] == pytest.approx(1.25)


def test_classify_thresholds():
    assert kpi.classify_load(1.06) == "과부하"
    assert kpi.classify_load(1.05) == "적정"
    assert kpi.classify_load(0.95) == "적정"
    assert kpi.classify_load(0.94) == "저활용"


def test_requires_two_years_for_yoy():
    with pytest.raises(ValueError):
        kpi.department_kpis(_toy_tables(), as_of=PERIODS[20])


def test_synthetic_data_scenario():
    """생성 시 의도한 시나리오(AI·클라우드 과부하, 경영지원·SI2 저활용)가 KPI로 드러나는지."""
    k = kpi.department_kpis(generate()).set_index("department_id")
    assert len(k) == 8
    assert set(k.query("status == '과부하'").index) >= {"D03", "D04"}
    assert set(k.query("status == '저활용'").index) >= {"D02", "D08"}
    assert k["active_headcount"].sum() <= k["headcount"].sum()
