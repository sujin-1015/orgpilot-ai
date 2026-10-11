"""합성 데이터 무결성 검증.

각 check 함수는 실패 시 AssertionError를 던진다. pytest(tests/test_data.py)와
CLI(python -m src.validate) 양쪽에서 같은 함수를 사용한다.
"""
from __future__ import annotations

import pandas as pd

from src.config import AS_OF, N_MONTHS
from src.data_gen import SCHEMA, make_periods

Tables = dict[str, pd.DataFrame]

PRIMARY_KEYS = {
    "departments": ["department_id"],
    "employees": ["employee_id"],
    "workload": ["department_id", "period"],
    "skills": ["employee_id", "skill"],
    "staffing_plan": ["department_id", "period"],
}

FOREIGN_KEYS = [  # (자식 테이블, 컬럼, 부모 테이블)
    ("employees", "department_id", "departments"),
    ("workload", "department_id", "departments"),
    ("staffing_plan", "department_id", "departments"),
    ("skills", "employee_id", "employees"),
]


def check_schema(t: Tables) -> None:
    """정해진 컬럼만 존재 — 생성기의 숨은 정답(필요 FTE 등)이 새어 나가지 않았는지 확인."""
    for name, cols in SCHEMA.items():
        assert list(t[name].columns) == cols, f"{name}: columns {list(t[name].columns)} != {cols}"


def check_primary_keys(t: Tables) -> None:
    for name, pk in PRIMARY_KEYS.items():
        dup = t[name].duplicated(subset=pk).sum()
        assert dup == 0, f"{name}: PK {pk} 중복 {dup}건"


def check_foreign_keys(t: Tables) -> None:
    for child, col, parent in FOREIGN_KEYS:
        orphans = set(t[child][col]) - set(t[parent][col])
        assert not orphans, f"{child}.{col} -> {parent}: 고아 키 {sorted(orphans)[:5]}"


def check_no_nulls(t: Tables) -> None:
    for name, df in t.items():
        n = int(df.isna().sum().sum())
        assert n == 0, f"{name}: 결측치 {n}건"


def check_headcount_consistency(t: Tables) -> None:
    """departments.current_headcount = 직원 수 = 기준시점 staffing_plan.actual_headcount."""
    emp_counts = t["employees"].groupby("department_id").size()
    dept = t["departments"].set_index("department_id")["current_headcount"]
    diff = (dept - emp_counts.reindex(dept.index, fill_value=0))
    assert (diff == 0).all(), f"부서 현원 ≠ 직원 수: {diff[diff != 0].to_dict()}"

    sp = t["staffing_plan"]
    latest = sp[sp["period"] == AS_OF].set_index("department_id")["actual_headcount"]
    diff2 = dept - latest.reindex(dept.index)
    assert (diff2 == 0).all(), f"부서 현원 ≠ {AS_OF} 실제 현원: {diff2[diff2 != 0].to_dict()}"


def check_period_coverage(t: Tables) -> None:
    """모든 조직이 동일한 24개월 연속 기간을 가지며, 기준시점 이후(미래) 데이터가 없다."""
    expected = make_periods()
    assert len(expected) == N_MONTHS
    for name in ("workload", "staffing_plan"):
        df = t[name]
        future = df[df["period"] > AS_OF]
        assert future.empty, f"{name}: 기준시점({AS_OF}) 이후 데이터 {len(future)}건 — 미래 정보 누출"
        for dept_id, g in df.groupby("department_id"):
            assert sorted(g["period"]) == expected, f"{name}/{dept_id}: 기간 불일치"


def check_value_ranges(t: Tables) -> None:
    sk = t["skills"]
    assert sk["proficiency"].between(1, 5).all(), "proficiency 범위(1~5) 위반"
    per_emp = sk.groupby("employee_id").size()
    assert per_emp.between(3, 5).all(), "직원별 스킬 개수(3~5) 위반"
    assert set(per_emp.index) == set(t["employees"]["employee_id"]), "스킬이 없는 직원 존재"

    wl = t["workload"]
    assert (wl["task_count"] > 0).all() and (wl["standard_hours"] > 0).all(), "업무량/표준시간 ≤ 0"

    emp = t["employees"]
    assert (emp["tenure"] >= 0).all(), "근속연수 음수"
    assert set(emp["status"]) <= {"active", "on_leave"}, f"알 수 없는 status {set(emp['status'])}"

    sp = t["staffing_plan"]
    assert (sp["approved_to"] > 0).all() and (sp["actual_headcount"] > 0).all(), "정원/현원 ≤ 0"


ALL_CHECKS = [
    check_schema, check_primary_keys, check_foreign_keys, check_no_nulls,
    check_headcount_consistency, check_period_coverage, check_value_ranges,
]


def run_all(t: Tables) -> list[tuple[str, bool, str]]:
    results = []
    for fn in ALL_CHECKS:
        try:
            fn(t)
            results.append((fn.__name__, True, ""))
        except AssertionError as e:
            results.append((fn.__name__, False, str(e)))
    return results


if __name__ == "__main__":
    from src.data_gen import load

    tables = {name: load(name) for name in SCHEMA}
    results = run_all(tables)
    for name, ok, msg in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}")
    raise SystemExit(0 if all(ok for _, ok, _ in results) else 1)
