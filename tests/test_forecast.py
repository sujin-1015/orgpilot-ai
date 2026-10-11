import numpy as np
import pandas as pd
import pytest

from src import forecast as F
from src import staffing as S
from src.config import AS_OF, AVAILABLE_HOURS_PER_FTE as H
from src.data_gen import DEPT_SPECS, _season_factor, generate, make_periods


@pytest.fixture(scope="module")
def tables():
    return generate()


@pytest.fixture(scope="module")
def df(tables):
    return F.prepare(tables["workload"])


def _noiseless_workload(a=5.0, b=0.01, c=0.1, d=-0.05):
    periods = make_periods()
    rows = []
    for t, p in enumerate(periods):
        m = int(p[5:7])
        y = np.exp(a + b * t + c * (m in (3, 6, 9, 12)) + d * (m == 1))
        rows.append(("X", p, y, 10.0))
    return pd.DataFrame(rows, columns=["department_id", "period", "task_count", "standard_hours"])


# ── 예측 모델 ─────────────────────────────────────────────

def test_seasonal_trend_recovers_known_parameters():
    df = F.prepare(_noiseless_workload())
    coef = F.fit_seasonal_trend(df).loc["X"]
    assert coef.to_numpy() == pytest.approx([5.0, 0.01, 0.1, -0.05], abs=1e-9)


def test_rolling_origin_never_trains_on_future(df):
    """모델에 전달되는 학습 데이터가 예측 대상보다 항상 과거인지, 모델 함수 안에서 직접 확인."""
    calls = []

    def spy(train, future):
        assert train["t"].max() < future["t"].min()
        assert "task_count" not in future.columns  # 정답 컬럼이 예측 입력에 섞이지 않음
        calls.append(len(train))
        return np.ones(len(future))

    bt = F.rolling_origin(df, spy, min_train=12, horizon=6)
    assert len(calls) == 12  # 기준점 t=11..22
    assert bt["h"].between(1, 6).all()


def test_forecast_uses_only_data_up_to_as_of(df):
    fc = F.forecast(df, horizon=12)
    assert fc["period"].min() > AS_OF
    assert len(fc) == 8 * 12


def test_main_model_beats_current_practice_baseline(df):
    summary = F.summarize_backtest(F.backtest(df, {k: F.MODELS[k] for k in ["baseline_mean", "seasonal_trend"]}))
    assert summary.loc["seasonal_trend", "monthly_wape"] < summary.loc["baseline_mean", "monthly_wape"]


def test_forecast_close_to_generator_ground_truth(tables, df):
    """합성 데이터라서 가능한 검증: 생성기에 숨겨 둔 '진짜 필요 FTE'와 예측 필요 FTE 비교 (오차 5% 이내)."""
    fc = F.forecast(df, horizon=12)
    pred = S.required_fte(fc)["forecast_fte"]
    future = [str(p) for p in pd.period_range(pd.Period(AS_OF, "M") + 1, periods=12, freq="M")]
    for spec in DEPT_SPECS:
        truth = np.mean([spec["need_end"] * (1 + spec["growth"]) ** (k + 1) * _season_factor(int(p[5:7]), spec["season_amp"])
                         for k, p in enumerate(future)])
        assert pred[spec["id"]] == pytest.approx(truth, rel=0.05), spec["name"]


# ── FTE · 정원 산정 ───────────────────────────────────────

def _toy_tables(active=8, leave=2):
    n = active + leave
    return {
        "departments": pd.DataFrame({"department_id": ["X"], "name": ["테스트팀"], "function": ["T"], "current_headcount": [n]}),
        "employees": pd.DataFrame({"employee_id": [f"E{i}" for i in range(n)], "department_id": "X", "job_family": "개발",
                                   "tenure": 1.0, "status": ["active"] * active + ["on_leave"] * leave}),
        "staffing_plan": pd.DataFrame({"department_id": ["X"], "period": [AS_OF], "approved_to": [11], "actual_headcount": [n]}),
    }


def _toy_forecast(tasks=100.0, std=14.0):
    return pd.DataFrame({"department_id": "X", "period": ["2026-10", "2026-11"],
                         "task_count_pred": [tasks, tasks], "standard_hours": std})


def test_fte_and_recommended_to_hand_calculation():
    # 100건 × 14h / 140h = 10 FTE. 가용비율 0.8 → 정원 12.5 → 올림 13
    row = S.staffing_table(_toy_tables(), _toy_forecast()).iloc[0]
    assert row["forecast_fte"] == pytest.approx(100 * 14 / H)
    assert row["recommended_to"] == 13
    assert row["fte_gap"] == pytest.approx(10 - 8)
    assert row["headcount_gap"] == 3
    assert row["to_change"] == 2
    assert row["action"] == "충원 필요 (+3)"


def test_recommended_to_no_float_roundup():
    # 정확히 12.0이어야 할 값이 부동소수 오차로 13이 되면 안 됨
    assert S.recommended_to(0.1 * 3 * 40, 1.0) == 12


def test_sensitivity_scales_linearly():
    table = S.staffing_table(_toy_tables(active=10, leave=0), _toy_forecast())  # 10 FTE, 가용비율 1.0
    sens = S.sensitivity(table).set_index("error")["recommended_to"]
    assert sens.to_dict() == {-0.2: 8, -0.1: 9, 0.0: 10, 0.1: 11, 0.2: 12}


def test_decision_stability_flags_direction_flip():
    sens = pd.DataFrame({"department_id": ["A"] * 3 + ["B"] * 3, "error": [-0.1, 0, 0.1] * 2,
                         "headcount_gap": [2, 5, 8, -2, 1, 3]})
    st = S.decision_stability(sens).set_index("department_id")["stable"]
    assert bool(st["A"]) and not bool(st["B"])


def test_scenario_growth_departments_need_hiring(tables):
    _, table, _ = S.run(tables)
    t = table.set_index("department_id")
    assert (t.loc[["D03", "D04"], "headcount_gap"] > 0).all()
    assert (t.loc[["D02", "D08"], "headcount_gap"] < 0).all()
