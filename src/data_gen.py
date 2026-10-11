"""가상 IT 서비스 기업 'Aster Digital'의 합성 HR 데이터 생성기.

⚠ 이 모듈이 만드는 모든 데이터는 합성(synthetic) 데이터다. 실존 기업·인물과 무관하다.

생성 로직 요약 (자세한 내용은 docs/data_generation.md):
  1. 조직별 '진짜 필요 FTE'(숨은 정답)를 추세 × 계절성 × 노이즈로 만든다.
  2. 필요 FTE × 1인당 가용시간 / 건당 표준처리시간 → 월별 업무건수(task_count)로 변환한다.
  3. 숨은 정답(필요 FTE, 성장률 등)은 테이블에 저장하지 않는다. 분석은 업무건수에서 출발해야 한다.
  4. 정원(TO)은 '전년도 말 현원 + 부서 요청분'이라는 과거 관행대로 만든다
     → 업무량 기반 필요 인원과 어긋나는 상황을 재현한다.

실행:  python -m src.data_gen
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from src.config import AS_OF, AVAILABLE_HOURS_PER_FTE, DATA_DIR, DB_PATH, N_MONTHS, SEED

SKILLS = [
    "Python", "Java", "SQL", "Cloud(AWS/Azure)", "Kubernetes", "ML/DL",
    "Data Analysis", "Project Management", "Information Security",
    "Network", "ITSM", "ERP/Business",
]

# 각 조직의 생성 파라미터.
#   headcount_end : 기준시점(AS_OF) 현원        headcount_start : 24개월 전 현원
#   need_end      : 기준시점의 계절성 제외 '진짜' 필요 FTE (숨은 정답, 저장하지 않음)
#   growth        : 필요 FTE의 월 성장률        season_amp : 분기말 업무량 증가폭
#   std_hours     : 업무 1건당 표준처리시간(h)  to_request : 연초 정원 산정 시 부서 증원 요청분
#   skill_w       : 스킬 보유 확률 가중치(SKILLS 순서)
DEPT_SPECS = [
    dict(id="D01", name="SI사업1팀", function="SI", headcount_start=54, headcount_end=55,
         need_end=56.0, growth=0.004, season_amp=0.15, std_hours=24.0, to_request=2,
         job_families={"개발": 0.75, "PM": 0.10, "데이터": 0.15},
         skill_w=[6, 9, 7, 3, 1, 0.5, 2, 2, 1, 1, 1, 4]),
    dict(id="D02", name="SI사업2팀", function="SI", headcount_start=50, headcount_end=48,
         need_end=40.0, growth=-0.003, season_amp=0.15, std_hours=24.0, to_request=2,
         job_families={"개발": 0.80, "PM": 0.10, "데이터": 0.10},
         skill_w=[7, 9, 7, 3, 1, 1, 3, 2, 1, 1, 1, 4]),
    dict(id="D03", name="클라우드인프라팀", function="Cloud", headcount_start=33, headcount_end=38,
         need_end=45.0, growth=0.012, season_amp=0.08, std_hours=6.0, to_request=3,
         job_families={"인프라": 0.85, "개발": 0.15},
         skill_w=[3, 1, 2, 9, 6, 0.5, 1, 1, 3, 6, 3, 0.5]),
    dict(id="D04", name="AI서비스팀", function="AI", headcount_start=18, headcount_end=24,
         need_end=32.0, growth=0.020, season_amp=0.05, std_hours=40.0, to_request=3,
         job_families={"데이터/AI": 0.80, "개발": 0.20},
         skill_w=[9, 1, 6, 4, 2, 5, 6, 1, 0.5, 0.5, 0.5, 1]),
    dict(id="D05", name="PMO팀", function="PM", headcount_start=30, headcount_end=30,
         need_end=29.0, growth=0.002, season_amp=0.12, std_hours=16.0, to_request=1,
         job_families={"PM": 0.90, "경영지원": 0.10},
         skill_w=[1, 1, 2, 2, 0.5, 0.5, 4, 9, 1, 0.5, 2, 6]),
    dict(id="D06", name="IT운영팀", function="Ops", headcount_start=54, headcount_end=50,
         need_end=44.0, growth=-0.006, season_amp=0.05, std_hours=2.5, to_request=1,
         job_families={"운영": 0.85, "인프라": 0.15},
         skill_w=[2, 2, 4, 4, 2, 0.5, 1, 1, 2, 5, 9, 2]),
    dict(id="D07", name="정보보안팀", function="Security", headcount_start=20, headcount_end=22,
         need_end=24.0, growth=0.006, season_amp=0.08, std_hours=5.0, to_request=1,
         job_families={"보안": 0.90, "인프라": 0.10},
         skill_w=[3, 1, 2, 4, 2, 0.5, 1, 1, 9, 6, 3, 0.5]),
    dict(id="D08", name="경영지원팀", function="Support", headcount_start=34, headcount_end=33,
         need_end=28.0, growth=0.000, season_amp=0.10, std_hours=3.0, to_request=1,
         job_families={"경영지원": 0.90, "데이터": 0.10},
         skill_w=[1, 0.5, 4, 0.5, 0.2, 0.2, 6, 4, 1, 0.5, 1, 9]),
]

# 각 테이블의 공식 스키마. 이 외의 컬럼(특히 숨은 정답)이 저장되면 검증에서 실패한다.
SCHEMA = {
    "departments": ["department_id", "name", "function", "current_headcount"],
    "employees": ["employee_id", "department_id", "job_family", "tenure", "status"],
    "workload": ["department_id", "period", "task_count", "standard_hours"],
    "skills": ["employee_id", "skill", "proficiency"],
    "staffing_plan": ["department_id", "period", "approved_to", "actual_headcount"],
}


def make_periods(as_of: str = AS_OF, n: int = N_MONTHS) -> list[str]:
    return [str(p) for p in pd.period_range(end=as_of, periods=n, freq="M")]


def _season_factor(month: int, amp: float) -> float:
    """분기말(3·6·9·12월) 업무량 증가, 1월은 비수기."""
    if month in (3, 6, 9, 12):
        return 1.0 + amp
    if month == 1:
        return 1.0 - amp / 2
    return 1.0


def _gen_workload(rng: np.random.Generator, periods: list[str]) -> pd.DataFrame:
    rows = []
    n = len(periods)
    for spec in DEPT_SPECS:
        for t, period in enumerate(periods):
            month = int(period[5:7])
            trend = spec["need_end"] * (1 + spec["growth"]) ** (t - (n - 1))
            need_fte = trend * _season_factor(month, spec["season_amp"]) * rng.lognormal(0, 0.04)
            task_count = round(need_fte * AVAILABLE_HOURS_PER_FTE / spec["std_hours"])
            rows.append((spec["id"], period, task_count, spec["std_hours"]))
    return pd.DataFrame(rows, columns=SCHEMA["workload"])


def _gen_headcount_path(rng: np.random.Generator, start: int, end: int, n: int) -> list[int]:
    """24개월 현원 경로: 시작→끝 선형 추세 + ±1명 변동, 마지막 달은 정확히 현재 현원."""
    path = []
    for t in range(n):
        base = start + (end - start) * t / (n - 1)
        jitter = 0 if t in (0, n - 1) else int(rng.integers(-1, 2))
        path.append(int(round(base)) + jitter)
    return path


def _gen_staffing_plan(rng: np.random.Generator, periods: list[str]) -> pd.DataFrame:
    """정원(TO) = 직전 연말(첫해는 관측 첫 달) 현원 + 부서 요청분. 연초에만 갱신되는 과거 관행."""
    rows = []
    for spec in DEPT_SPECS:
        hc = _gen_headcount_path(rng, spec["headcount_start"], spec["headcount_end"], len(periods))
        to_by_year: dict[str, int] = {}
        for t, period in enumerate(periods):
            year = period[:4]
            if year not in to_by_year:
                prev = hc[t - 1] if t > 0 else hc[0]
                to_by_year[year] = prev + spec["to_request"]
            rows.append((spec["id"], period, to_by_year[year], hc[t]))
    return pd.DataFrame(rows, columns=SCHEMA["staffing_plan"])


def _gen_employees_and_skills(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    emp_rows, skill_rows = [], []
    eid = 0
    for spec in DEPT_SPECS:
        families = list(spec["job_families"])
        fam_p = np.array(list(spec["job_families"].values()))
        skill_p = np.array(spec["skill_w"], dtype=float)
        skill_p /= skill_p.sum()
        for _ in range(spec["headcount_end"]):
            eid += 1
            emp_id = f"E{eid:04d}"
            tenure = float(np.clip(rng.gamma(2.0, 3.5), 0.1, 30.0).round(1))
            status = "on_leave" if rng.random() < 0.03 else "active"
            emp_rows.append((emp_id, spec["id"], rng.choice(families, p=fam_p / fam_p.sum()), tenure, status))

            n_skills = int(rng.integers(3, 6))  # 3~5개
            for skill in rng.choice(SKILLS, size=n_skills, replace=False, p=skill_p):
                prof = int(np.clip(round(rng.normal(2.3 + 0.12 * tenure, 0.9)), 1, 5))
                skill_rows.append((emp_id, str(skill), prof))
    employees = pd.DataFrame(emp_rows, columns=SCHEMA["employees"])
    skills = pd.DataFrame(skill_rows, columns=SCHEMA["skills"])
    return employees, skills


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """seed가 같으면 항상 동일한 5개 테이블을 반환한다."""
    rng = np.random.default_rng(seed)
    periods = make_periods()

    employees, skills = _gen_employees_and_skills(rng)
    departments = pd.DataFrame(
        [(s["id"], s["name"], s["function"]) for s in DEPT_SPECS],
        columns=["department_id", "name", "function"],
    )
    counts = employees.groupby("department_id").size().rename("current_headcount")
    departments = departments.join(counts, on="department_id")

    tables = {
        "departments": departments,
        "employees": employees,
        "workload": _gen_workload(rng, periods),
        "skills": skills,
        "staffing_plan": _gen_staffing_plan(rng, periods),
    }
    return {name: df[SCHEMA[name]].reset_index(drop=True) for name, df in tables.items()}


def save(tables: dict[str, pd.DataFrame]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    with sqlite3.connect(DB_PATH) as conn:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)


def load(name: str) -> pd.DataFrame:
    """SQLite에서 테이블 하나를 읽는다."""
    if name not in SCHEMA:  # 테이블명은 SQL 파라미터로 바인딩할 수 없으므로 화이트리스트로 제한
        raise ValueError(f"unknown table: {name}")
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(f"SELECT * FROM {name}", conn)


if __name__ == "__main__":
    tables = generate()
    save(tables)
    for name, df in tables.items():
        print(f"{name:<14} {len(df):>5} rows  -> {DATA_DIR / (name + '.csv')}")
    print(f"SQLite: {DB_PATH}")
