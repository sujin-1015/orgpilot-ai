import pandas as pd
import pytest

from src import skill_match as C
from src import staffing as S
from src.data_gen import generate


@pytest.fixture(scope="module")
def tables():
    return generate()


@pytest.fixture(scope="module")
def staff(tables):
    return S.run(tables)[1]


@pytest.fixture(scope="module")
def plan(tables, staff):
    return C.redeployment_plan(tables, staff)


def test_supply_counts_only_active_and_proficient(tables):
    sup = C.supply_matrix(tables)
    emp, sk = tables["employees"], tables["skills"]
    x = sk.merge(emp, on="employee_id")
    expected = len(x[(x.department_id == "D04") & (x.skill == "Python") & (x.status == "active") & (x.proficiency >= 3)])
    assert sup.at["D04", "Python"] == expected


def test_demand_rounds_up(staff):
    dem = C.demand_matrix(staff)
    rec = staff.set_index("department_id").at["D04", "recommended_to"]
    assert dem.at["D04", "Python"] == -(-int(round(0.70 * rec * 1e6)) // 1_000_000)  # ceil
    assert dem.at["D04", "Java"] == 0  # 필요 스킬이 아니면 수요 0


def test_requirements_cover_all_departments(tables):
    assert set(C.SKILL_REQUIREMENTS) == set(tables["departments"]["department_id"])
    for req in C.SKILL_REQUIREMENTS.values():
        assert all(0 < r <= 1 for r in req.values())


def test_redeployment_respects_capacity(staff, plan):
    assignments, summary = plan
    gap = staff.set_index("department_id")["headcount_gap"]
    assert assignments["employee_id"].is_unique
    for src, n in assignments["from_dept"].value_counts().items():
        assert gap[src] < 0 and n <= -gap[src]
    for tgt, n in assignments["to_dept"].value_counts().items():
        assert gap[tgt] > 0 and n <= gap[tgt]
    assert (summary["internal_redeploy"] + summary["external_hire"] == summary["hiring_need"]).all()


def test_redeployed_people_actually_have_the_skills(tables, plan):
    assignments, _ = plan
    sk = tables["skills"].set_index(["employee_id", "skill"])["proficiency"]
    for r in assignments.itertuples():
        for s in r.fills_skills.split(", "):
            assert sk[(r.employee_id, s)] >= C.PROFICIENT


def test_redeployment_creates_no_new_shortage_at_source(tables, staff, plan):
    """재배치 후 출발 조직의 스킬 격차를 다시 계산해, 원래 충족이던 스킬이 부족해지지 않았는지."""
    assignments, _ = plan
    before = C.skill_gap(tables, staff).set_index(["department_id", "skill"])["gap"]
    moved = tables["employees"].copy()
    moved.loc[moved["employee_id"].isin(assignments["employee_id"]), "department_id"] = "MOVED"
    after = C.skill_gap({**tables, "employees": moved}, staff).set_index(["department_id", "skill"])["gap"]
    for src in assignments["from_dept"].unique():
        for (d, s), g in before.items():
            if d == src and g >= 0:
                assert after[(d, s)] >= 0, f"{d}/{s} 재배치로 부족 발생"


def test_plan_is_deterministic(tables, staff, plan):
    again, _ = C.redeployment_plan(tables, staff)
    pd.testing.assert_frame_equal(plan[0], again)


def test_upskill_needed_matches_remaining_shortage(plan):
    """육성 필요 인원 = max(0, 남은 스킬 부족 최대치 − 외부 채용 인원). 모듈 출력의 문자열에서 역산해 대조."""
    _, summary = plan
    for r in summary.itertuples():
        counts = [int(x.rsplit("(", 1)[1].rstrip(")")) for x in r.skills_to_hire_for.split(", ")] if r.skills_to_hire_for != "-" else []
        assert r.upskill_needed == max([0, *(c - r.external_hire for c in counts)])
    ai = summary.set_index("department_id").loc["D04"]
    assert ai["upskill_needed"] > 0  # AI팀은 ML/DL 부족이 채용 인원보다 큼


def test_scenario_ai_ml_shortage_flagged(tables, staff):
    g = C.skill_gap(tables, staff).set_index(["department_id", "skill"])
    assert g.at[("D04", "ML/DL"), "status"] == "부족"
    assert g.at[("D03", "Cloud(AWS/Azure)"), "status"] == "부족"
