import pandas as pd
import pytest

from src import validate
from src.data_gen import SCHEMA, generate, load


@pytest.fixture(scope="module")
def tables():
    return generate()


@pytest.mark.parametrize("check", validate.ALL_CHECKS, ids=lambda f: f.__name__)
def test_generated_data_passes(check, tables):
    check(tables)


def test_reproducible_with_same_seed():
    a, b = generate(seed=7), generate(seed=7)
    for name in SCHEMA:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_different_seed_changes_data():
    a, b = generate(seed=1), generate(seed=2)
    assert not a["workload"].equals(b["workload"])


def test_saved_db_matches_generator(tables):
    """디스크의 SQLite가 현재 생성 코드로 만든 결과와 같은지 (오래된 DB 사용 방지)."""
    for name in SCHEMA:
        pd.testing.assert_frame_equal(load(name), tables[name], check_dtype=False)


# 검증 함수가 실제로 오류를 잡아내는지 — 일부러 데이터를 망가뜨려 확인한다.
def test_detects_orphan_foreign_key(tables):
    broken = {k: v.copy() for k, v in tables.items()}
    broken["employees"].loc[0, "department_id"] = "D99"
    with pytest.raises(AssertionError, match="고아 키"):
        validate.check_foreign_keys(broken)


def test_detects_future_leakage(tables):
    broken = {k: v.copy() for k, v in tables.items()}
    row = broken["workload"].iloc[[0]].assign(period="2026-10")
    broken["workload"] = pd.concat([broken["workload"], row], ignore_index=True)
    with pytest.raises(AssertionError, match="미래 정보 누출"):
        validate.check_period_coverage(broken)


def test_detects_hidden_column_leak(tables):
    broken = {k: v.copy() for k, v in tables.items()}
    broken["workload"]["true_need_fte"] = 1.0
    with pytest.raises(AssertionError):
        validate.check_schema(broken)


def test_detects_headcount_mismatch(tables):
    broken = {k: v.copy() for k, v in tables.items()}
    broken["departments"].loc[0, "current_headcount"] += 1
    with pytest.raises(AssertionError, match="현원"):
        validate.check_headcount_consistency(broken)
