"""AI 설명 기능 테스트 — 실제 API를 호출하지 않는다(가짜 LLM 주입). 캐시는 임시 경로를 쓴다."""
import pytest

from src import ai_explain as A
from src.pipeline import run_all


@pytest.fixture(scope="module")
def res():
    from src.data_gen import generate
    return run_all(generate(), backtest=False)


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "CACHE_PATH", tmp_path / "cache.json")


def fake_llm(text):
    calls = []

    def llm(system, prompt):
        calls.append(prompt)
        return text
    llm.calls = calls
    return llm


# ── 수치가 계산 결과에서 온다 ─────────────────────────────

def test_metrics_come_from_calculation_modules(res):
    m = A.get_department_metrics(res, "D04")
    k = res.kpis.set_index("department_id").loc["D04"]
    s = res.staff.set_index("department_id").loc["D04"]
    assert m["최근 12개월 업무부하율"] == f"{k['load_ratio']:.2f}"
    assert m["권장 정원"] == f"{int(s['recommended_to'])}명"
    assert m["인원 격차(권장 정원 - 현원)"] == f"{int(s['headcount_gap']):+d}명"
    assert m["향후 12개월 필요 FTE"] == f"{s['forecast_fte']:.1f}"


def test_prompt_contains_facts_and_no_numbers_rule(res):
    m = A.get_department_metrics(res, "D03")
    p = A.build_prompt(m)
    for k, v in m.items():
        if k != "department_id":
            assert f"{k}: {v}" in p
    assert "새로 계산" in A.SYSTEM_INSTRUCTION


# ── 수치 일치 검증 (계획서 요구: 계산 결과와 AI 응답 수치 일치) ──

def test_number_extraction():
    assert A.extract_numbers("부하율 1.37, +15명, -3명, 1,200건, 2026-09 기준") == [1.37, 15.0, -3.0, 1200.0]


def test_llm_answer_with_correct_numbers_is_accepted(res):
    m = A.get_department_metrics(res, "D04")
    text = (f"{m['조직']}은 업무부하율 {m['최근 12개월 업무부하율']}로 과부하 상태이며 "
            f"권장 정원 {m['권장 정원']} 기준 {m['인원 격차(권장 정원 - 현원)']} 충원이 필요합니다.")
    e = A.explain_gap(res, "D04", llm=fake_llm(text))
    assert e.source == "gemini" and e.text == text and e.unknown_numbers == []


def test_hallucinated_number_is_rejected_and_replaced(res):
    """LLM이 계산에 없는 숫자(부하율 1.52, 20명)를 만들면 버리고 템플릿으로 대체한다."""
    e = A.explain_gap(res, "D04", llm=fake_llm("AI서비스팀은 부하율 1.52로 20명 충원이 필요합니다."))
    assert e.source == "fallback"
    assert set(e.rejected_numbers) == {1.52, 20.0}
    assert A.verify_numbers(e.text, A.get_department_metrics(res, "D04")) == []


def test_llm_arithmetic_is_rejected(res):
    """사실끼리 계산한 숫자(예: 권장 정원 39 − 현행 정원 26 = 13)도 사실 목록에 없으면 거부."""
    m = A.get_department_metrics(res, "D04")
    assert "13" not in A.facts_text(m)  # 전제 확인
    e = A.explain_gap(res, "D04", llm=fake_llm("현행 정원보다 13명 많은 정원이 필요합니다."))
    assert e.source == "fallback" and e.rejected_numbers == [13.0]


@pytest.mark.parametrize("dept", ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"])
def test_fallback_text_numbers_match_for_every_department(res, dept):
    m = A.get_department_metrics(res, dept)
    assert A.verify_numbers(A.fallback_explanation(m), m) == []


# ── 실패·비용 방지 ────────────────────────────────────────

def test_api_error_falls_back_without_raising(res):
    def broken(system, prompt):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")
    e = A.explain_gap(res, "D03", llm=broken)
    assert e.source == "fallback" and "429" in e.error


def test_cache_prevents_second_call(res):
    m = A.get_department_metrics(res, "D04")
    llm = fake_llm(f"{m['조직']}의 권장 정원은 {m['권장 정원']}입니다.")
    first = A.explain_gap(res, "D04", llm=llm)
    second = A.explain_gap(res, "D04", llm=llm)
    assert first.source == "gemini" and second.source == "cache"
    assert len(llm.calls) == 1


def test_no_call_mode_never_calls(res):
    def must_not_call(system, prompt):
        raise AssertionError("API가 호출됨")
    e = A.explain_gap(res, "D02", llm=must_not_call, allow_call=False)
    assert e.source == "fallback"


def test_truncated_response_is_not_used(res):
    """토큰 한도로 끊긴 응답(IncompleteResponse)은 수치가 맞아도 쓰지 않고, 캐시에도 넣지 않는다."""
    def truncated(system, prompt):
        raise A.IncompleteResponse("finish_reason=MAX_TOKENS")
    e = A.explain_gap(res, "D04", llm=truncated)
    assert e.source == "fallback" and "MAX_TOKENS" in e.error
    assert not A.CACHE_PATH.exists()


def test_zero_gap_has_no_plus_sign():
    assert A._signed(0) == "0명" and A._signed(3) == "+3명" and A._signed(-2) == "-2명"


def test_committed_cache_passes_number_check(res, monkeypatch):
    """저장소에 커밋된 실제 Gemini 응답 캐시가 현재 계산 결과와 수치가 일치하는지 (API 호출 없음)."""
    from src.config import DATA_DIR
    monkeypatch.setattr(A, "CACHE_PATH", DATA_DIR / "ai_cache" / "explanations.json")
    if not A.CACHE_PATH.exists():
        pytest.skip("캐시 없음")
    model = A.json.loads(A.CACHE_PATH.read_text(encoding="utf-8"))
    model = next(iter(model.values()))["model"]
    for d in res.staff["department_id"]:
        e = A.explain_gap(res, d, model=model, allow_call=False)
        assert e.source == "cache", f"{d}: 캐시가 현재 계산 결과와 맞지 않음(재생성 필요)"
        assert e.unknown_numbers == []
