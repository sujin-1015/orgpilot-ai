"""README용 대시보드 화면 캡처.

사전 조건: 대시보드 실행 중 (streamlit run app/dashboard.py, 기본 포트 8501)
실행:      python scripts/capture_screenshots.py
PC에 설치된 Microsoft Edge를 Playwright로 headless 실행한다(별도 브라우저 다운로드 없음).
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"


def wait_ready(page, text: str, plots: int) -> None:
    page.get_by_text(text, exact=False).first.wait_for(timeout=120_000)
    page.wait_for_function(f"document.querySelectorAll('.js-plotly-plot').length >= {plots}", timeout=60_000)
    page.wait_for_timeout(2500)  # 차트 애니메이션·표 렌더링 대기


def fit_height(page) -> None:
    """Streamlit은 내부 컨테이너가 스크롤되므로, 뷰포트를 콘텐츠 높이에 맞춰 한 장에 담는다."""
    h = page.evaluate("Math.max(...[...document.querySelectorAll('[data-testid=stMainBlockContainer]')].map(e => e.scrollHeight))")
    page.set_viewport_size({"width": 1440, "height": int(h) + 80})
    page.wait_for_timeout(1500)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5, locale="ko-KR")
    page.goto(URL)

    wait_ready(page, "조직별 KPI", plots=2)
    fit_height(page)
    page.screenshot(path=OUT / "dashboard_overview.png", full_page=True)

    page.set_viewport_size({"width": 1440, "height": 900})
    page.get_by_role("tab", name="② 적정 TO 산정").click()
    wait_ready(page, "AI 설명", plots=4)
    fit_height(page)
    page.screenshot(path=OUT / "dashboard_staffing.png", full_page=True)

    browser.close()
    print("saved:", *(OUT / n for n in ["dashboard_overview.png", "dashboard_staffing.png"]), sep="\n  ")
