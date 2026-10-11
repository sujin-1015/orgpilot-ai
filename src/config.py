"""프로젝트 공통 설정값.

여기 있는 값들은 '가정(assumption)'이다. 실제 조직에 적용할 때는
근무제도·휴가·교육시간 등을 반영해 다시 산정해야 한다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic"
DB_PATH = DATA_DIR / "orgpilot.db"

SEED = 42

# 분석 기준 시점: 이 시점까지의 데이터만 '관측된 과거'로 존재한다.
AS_OF = "2026-09"
N_MONTHS = 24  # 2024-10 ~ 2026-09

# 1인당 월 가용시간(FTE 환산 기준).
# 월 소정근로 약 165h 중 교육·회의·휴가·행정 등 비직접업무 약 15%를 제외한 값.
AVAILABLE_HOURS_PER_FTE = 140.0

TABLES = ["departments", "employees", "workload", "skills", "staffing_plan"]
