"""대시보드 스모크 테스트: 브라우저 없이 Streamlit 앱을 실행해 오류 없이 그려지는지, 슬라이더 연동이 맞는지 확인.
API 호출 없음(대시보드는 저장된 AI 설명만 표시)."""
import pytest
from streamlit.testing.v1 import AppTest

from src import staffing as S
from src.config import ROOT
from src.data_gen import SCHEMA, load


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180)
    at.run()
    return at


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_renders_without_exception(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == ["① 조직 현황", "② 적정 TO 산정"]


def test_cards_match_calculation_modules(app):
    _, staff, _ = S.run({n: load(n) for n in SCHEMA})
    assert _metric(app, "전사 현원") == f"{staff['headcount'].sum()}명"
    assert _metric(app, "권장 정원 합계") == f"{staff['recommended_to'].sum()}명"
    assert _metric(app, "충원 필요") == f"+{staff['headcount_gap'].clip(lower=0).sum()}명"


def test_slider_updates_cards(app):
    """슬라이더 +10% → 카드도 같은 가정으로 재계산 (상단 카드가 고정돼 있던 버그 회귀 방지)."""
    _, staff, sens = S.run({n: load(n) for n in SCHEMA})
    expected = sens[sens["error"] == 0.10]["headcount_gap"]
    app.slider[0].set_value(10).run()
    assert not app.exception
    assert _metric(app, "충원 필요") == f"+{expected.clip(lower=0).sum()}명"
    assert _metric(app, "재배치 검토") == f"{expected.clip(upper=0).sum()}명"


def test_ai_explanation_shown_from_verified_cache(app):
    app.slider[0].set_value(0).run()
    assert any("수치 검증 통과" in c.value for c in app.caption)
