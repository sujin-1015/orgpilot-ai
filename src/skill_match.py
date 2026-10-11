"""Module C. 스킬 기반 인력 매칭 (규칙 기반, 최적화 없음).

  공급  supply[d, s] = 조직 d의 가용(active) 인원 중 스킬 s 숙련도 ≥ 3(실무 가능)인 인원
  수요  demand[d, s] = ceil(SKILL_REQUIREMENTS[d][s] × 권장 정원[d])   (Module B 결과)
  격차  gap = supply − demand    (−: 부족, +: 여유)

재배치 규칙 (그리디, 결정적):
  1. 후보 = '재배치 검토' 조직(인원 격차 < 0)의 가용 인원
  2. 대상 = '충원 필요' 조직, 후보가 대상 조직의 부족 스킬을 1개 이상 실무 수준으로 보유
  3. 우선순위 = 보유한 부족 스킬 수 ↓, 해당 스킬 숙련도 합 ↓, employee_id ↑
  4. 다음 중 하나라도 어기면 배정하지 않음
     - 출발 조직의 재배치 가능 인원(|인원 격차|) 초과
     - 대상 조직의 충원 필요 인원 초과
     - 대상 조직에서 그 사람이 메울 부족 스킬이 이미 다 채워짐
     - 그 사람이 빠지면 출발 조직에 새로운 스킬 부족이 생김
"""
from __future__ import annotations

import math

import pandas as pd

PROFICIENT = 3  # 1~5 중 '실무 가능' 기준

# 조직별 필요 스킬: 권장 정원 중 해당 스킬을 실무 수준으로 보유해야 하는 인원 비율.
# 직무 특성에 근거한 가정값이며, 실제 적용 시 현업 인터뷰·직무기술서로 정해야 한다.
SKILL_REQUIREMENTS: dict[str, dict[str, float]] = {
    "D01": {"Java": 0.40, "SQL": 0.40, "Python": 0.30, "ERP/Business": 0.20},
    "D02": {"Java": 0.40, "SQL": 0.40, "Python": 0.30, "ERP/Business": 0.20},
    "D03": {"Cloud(AWS/Azure)": 0.60, "Network": 0.40, "Kubernetes": 0.35, "ITSM": 0.20},
    "D04": {"Python": 0.70, "ML/DL": 0.40, "Data Analysis": 0.35, "Cloud(AWS/Azure)": 0.25, "SQL": 0.25},
    "D05": {"Project Management": 0.50, "ERP/Business": 0.40, "Data Analysis": 0.20},
    "D06": {"ITSM": 0.50, "Network": 0.40, "Cloud(AWS/Azure)": 0.25, "SQL": 0.20},
    "D07": {"Information Security": 0.60, "Network": 0.40, "Cloud(AWS/Azure)": 0.20},
    "D08": {"ERP/Business": 0.40, "Data Analysis": 0.35, "Project Management": 0.20, "SQL": 0.20},
}

Tables = dict[str, pd.DataFrame]


def proficient_skills(t: Tables) -> pd.DataFrame:
    """가용 인원의 실무 수준 스킬 (employee_id, department_id, skill, proficiency)."""
    active = t["employees"].query("status == 'active'")[["employee_id", "department_id"]]
    return t["skills"].query("proficiency >= @PROFICIENT").merge(active, on="employee_id")


def supply_matrix(t: Tables) -> pd.DataFrame:
    ps = proficient_skills(t)
    return ps.pivot_table(index="department_id", columns="skill", values="employee_id",
                          aggfunc="count", fill_value=0)


def demand_matrix(staff: pd.DataFrame) -> pd.DataFrame:
    rec = staff.set_index("department_id")["recommended_to"]
    rows = {d: {s: math.ceil(round(r * rec[d], 6)) for s, r in req.items()} for d, req in SKILL_REQUIREMENTS.items()}
    return pd.DataFrame(rows).T.fillna(0).astype(int)


def skill_gap(t: Tables, staff: pd.DataFrame) -> pd.DataFrame:
    """조직 × 필요 스킬별 공급·수요·격차 (필요 스킬로 정의된 조합만)."""
    sup, dem = supply_matrix(t), demand_matrix(staff)
    names = staff.set_index("department_id")["name"]
    rows = []
    for d, req in SKILL_REQUIREMENTS.items():
        for s in req:
            supply = int(sup.at[d, s]) if (d in sup.index and s in sup.columns) else 0
            rows.append((d, names[d], s, int(dem.at[d, s]), supply, supply - int(dem.at[d, s])))
    out = pd.DataFrame(rows, columns=["department_id", "name", "skill", "required", "supply", "gap"])
    out["status"] = out["gap"].map(lambda g: "부족" if g < 0 else ("충족" if g == 0 else "여유"))
    return out


def skill_comparison(t: Tables, staff: pd.DataFrame) -> pd.DataFrame:
    """스킬별 규칙 기반 비교표: 어느 조직이 부족하고, 어느 조직이 (자기 수요를 넘어) 여유가 있는가.

    여유는 해당 스킬이 필요 스킬이 아닌 조직의 보유 인원도 포함한다(그 조직 입장에선 전부 여유).
    """
    sup, dem = supply_matrix(t), demand_matrix(staff)
    dem = dem.reindex(index=sup.index, columns=sup.columns, fill_value=0)
    surplus = sup - dem
    names = staff.set_index("department_id")["name"]
    gap = skill_gap(t, staff)
    rows = []
    for s in sorted(gap["skill"].unique()):
        short = gap[(gap["skill"] == s) & (gap["gap"] < 0)]
        spare = surplus[s][surplus[s] > 0].sort_values(ascending=False)
        rows.append({
            "skill": s,
            "shortage_total": int(-short["gap"].sum()),
            "shortage_depts": ", ".join(f"{n}({g})" for n, g in zip(short["name"], short["gap"])) or "-",
            "surplus_total": int(spare.sum()),
            "surplus_depts": ", ".join(f"{names[d]}(+{v})" for d, v in spare.head(3).items()) or "-",
        })
    return pd.DataFrame(rows).sort_values("shortage_total", ascending=False).reset_index(drop=True)


def redeployment_plan(t: Tables, staff: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(배정 목록, 대상 조직별 요약) — 내부 재배치로 채울 수 있는 인원과 외부 채용 필요 인원."""
    gap = skill_gap(t, staff)
    remaining_short = {(r.department_id, r.skill): -r.gap for r in gap.itertuples() if r.gap < 0}
    source_margin = {(r.department_id, r.skill): r.gap for r in gap.itertuples()}  # 출발 조직의 스킬 여유
    hc_gap = staff.set_index("department_id")["headcount_gap"]
    src_cap = {d: -int(g) for d, g in hc_gap.items() if g < 0}
    tgt_need = {d: int(g) for d, g in hc_gap.items() if g > 0}

    ps = proficient_skills(t)
    cand = ps[ps["department_id"].isin(src_cap)]
    pairs = []
    for emp, g in cand.groupby("employee_id"):
        src = g["department_id"].iloc[0]
        prof = dict(zip(g["skill"], g["proficiency"]))
        for tgt in tgt_need:
            matched = sorted(s for s in prof if (tgt, s) in remaining_short)
            if matched:
                pairs.append((len(matched), sum(prof[s] for s in matched), emp, src, tgt, matched, prof))
    pairs.sort(key=lambda p: (-p[0], -p[1], p[2], p[4]))

    assigned, rows = set(), []
    for _, _, emp, src, tgt, matched, prof in pairs:
        if emp in assigned or src_cap[src] == 0 or tgt_need[tgt] == 0:
            continue
        fills = [s for s in matched if remaining_short.get((tgt, s), 0) > 0]
        if not fills:
            continue
        if any(source_margin.get((src, s), 1) <= 0 for s in prof):  # 빠지면 출발 조직에 부족 발생
            continue
        for s in fills:
            remaining_short[(tgt, s)] -= 1
        for s in prof:
            if (src, s) in source_margin:
                source_margin[(src, s)] -= 1
        src_cap[src] -= 1
        tgt_need[tgt] -= 1
        assigned.add(emp)
        rows.append((emp, src, tgt, ", ".join(fills), ", ".join(f"{s}({prof[s]})" for s in fills)))

    names = staff.set_index("department_id")["name"]
    plan = pd.DataFrame(rows, columns=["employee_id", "from_dept", "to_dept", "fills_skills", "proficiency"])
    plan.insert(2, "from_name", plan["from_dept"].map(names))
    plan.insert(4, "to_name", plan["to_dept"].map(names))

    summary = []
    for d, need in ((d, int(g)) for d, g in hc_gap.items() if g > 0):
        internal = int((plan["to_dept"] == d).sum())
        still_short = {s: n for (dd, s), n in remaining_short.items() if dd == d and n > 0}
        external = need - internal
        summary.append({
            "department_id": d, "name": names[d], "hiring_need": need,
            "internal_redeploy": internal, "external_hire": external,
            "skills_to_hire_for": ", ".join(f"{s}({n})" for s, n in sorted(still_short.items(), key=lambda x: -x[1])) or "-",
            # 채용 인원이 모두 해당 스킬을 갖춰도 남는 부족분 → 기존 인력 육성(업스킬링) 최소 인원
            "upskill_needed": max([0, *(n - external for n in still_short.values())]),
        })
    return plan, pd.DataFrame(summary).sort_values("hiring_need", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from src import staffing as S
    from src.data_gen import SCHEMA, load

    tables = {n: load(n) for n in SCHEMA}
    _, staff, _ = S.run(tables)
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        g = skill_gap(tables, staff)
        print(g.pivot(index="name", columns="skill", values="gap").fillna("").to_string(), "\n")
        print(skill_comparison(tables, staff).to_string(index=False), "\n")
        plan, summary = redeployment_plan(tables, staff)
        print(plan.to_string(index=False), "\n")
        print(summary.to_string(index=False))
