"""AI 설명 기능: 계산된 수치를 LLM(Gemini)이 자연어로 설명한다.

원칙
  - 모든 수치는 Python(분석 모듈)이 계산한다. LLM은 받은 사실(facts)을 설명만 한다.
  - LLM 응답에 facts에 없는 숫자가 하나라도 있으면 그 응답은 버리고, 계산 결과로 만든
    템플릿 설명(fallback)을 대신 쓴다 → 화면에 나가는 숫자는 항상 계산 결과와 일치.
  - 비용 방지: 웹 검색(grounding) 등 도구를 쓰지 않고, 응답은 파일 캐시에 저장해 재호출하지 않는다.
    API 오류(무료 한도 초과 429 포함) 시 예외를 던지지 않고 fallback으로 대체한다.

실행:  python -m src.ai_explain            # 8개 조직 설명 생성(캐시 있으면 재사용) + 수치 검증 결과 출력
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from src.config import DATA_DIR, ROOT
from src.pipeline import Results

CACHE_PATH = DATA_DIR / "ai_cache" / "explanations.json"
DEFAULT_MODEL = "gemini-3.8-flash"  # 2026-10 확인: 2.5-flash는 신규 사용자에게 404

SYSTEM_INSTRUCTION = """당신은 기업 HR 조직의 인력계획 분석가입니다.
주어진 '사실' 목록만 근거로, 이 조직의 인력 판단(충원 필요/재배치 검토/유지)이 왜 나왔는지 한국어로 설명하세요.

규칙:
1. 숫자는 '사실' 목록에 적힌 숫자를 적힌 그대로만 쓰세요. 새로 계산하거나(더하기·빼기·비율 변환 포함), 반올림을 바꾸거나, 추측한 숫자를 쓰지 마세요.
2. 개수·순서를 말할 때도 숫자 대신 한글로 쓰세요(예: '두 가지', '첫째').
3. 사실에 없는 원인(예: 특정 프로젝트, 퇴사 이유)을 지어내지 마세요.
4. 4~6문장, 하나의 문단. 결론 → 근거(업무부하·업무량 추세) → 예측 불확실성 → 충원·재배치 방식 순서로 쓰세요.
5. 마크다운, 제목, 목록 기호를 쓰지 마세요."""

LLM = Callable[[str, str], str]  # (system_instruction, prompt) -> text


# ── 1. 계산 결과 → 설명에 쓸 사실 ─────────────────────────────

def _pct(x: float) -> str:
    return f"{x * 100:+.0f}%"


def _signed(n: int) -> str:
    return f"{n:+d}명" if n else "0명"


def get_department_metrics(res: Results, dept_id: str) -> dict:
    """한 조직에 대한 계산 결과를 '표시 형식'까지 확정해 모은다. LLM과 템플릿 모두 이 값만 쓴다."""
    k = res.kpis.set_index("department_id").loc[dept_id]
    s = res.staff.set_index("department_id").loc[dept_id]
    st = res.stability.set_index("department_id").loc[dept_id]
    c = res.coef.loc[dept_id]
    load = float(k["load_ratio"])
    gap = res.skill_gap[(res.skill_gap["department_id"] == dept_id) & (res.skill_gap["gap"] < 0)]
    gap = gap.sort_values("gap")

    m = {
        "department_id": dept_id,
        "조직": k["name"],
        "현원": f"{int(k['headcount'])}명",
        "가용인원(휴직 제외)": f"{int(k['active_headcount'])}명",
        "현행 정원": f"{int(k['approved_to'])}명",
        "최근 12개월 업무부하율": f"{load:.2f}",
        "업무부하 해석": (f"필요시간이 가용시간보다 {abs(load - 1) * 100:.0f}% 많음" if load > 1
                       else f"필요시간이 가용시간보다 {abs(load - 1) * 100:.0f}% 적음" if load < 1 else "필요시간과 가용시간이 같음"),
        "부하 상태": k["status"],
        "1인당 월 초과근무 추정": f"{k['overtime_per_person']:.1f}시간",
        "업무량 전년 대비": _pct(k["workload_yoy"]),
        "추정 월 업무량 성장률": _pct(np.expm1(c["t"])) if abs(np.expm1(c["t"])) >= 0.005 else f"{np.expm1(c['t']) * 100:+.1f}%",
        "분기말 업무량 증가폭": _pct(np.expm1(c["qend"])),
        "향후 12개월 필요 FTE": f"{s['forecast_fte']:.1f}",
        "권장 정원": f"{int(s['recommended_to'])}명",
        "인원 격차(권장 정원 - 현원)": _signed(int(s['headcount_gap'])),
        "판단": s["action"],
        "예측 오차 ±10% 시 인원 격차 범위": f"{_signed(int(st['gap_min']))} ~ {_signed(int(st['gap_max']))}",
        "판단 안정성": "안정(±10% 오차에도 판단 방향 동일)" if st["stable"] else "재검토 필요(±10% 오차 안에서 판단 방향이 바뀜)",
        "부족 스킬": ", ".join(f"{r.skill} {-r.gap}명" for r in gap.itertuples()) or "없음",
    }
    fp = res.fill_plan.set_index("department_id")
    if dept_id in fp.index:
        f = fp.loc[dept_id]
        m["충원 방식"] = (f"내부 재배치 {int(f['internal_redeploy'])}명, 외부 채용 {int(f['external_hire'])}명"
                      + (f", 기존 인력 육성 최소 {int(f['upskill_needed'])}명" if f["upskill_needed"] else ""))
    out = res.redeploy[res.redeploy["from_dept"] == dept_id]
    if len(out):
        dest = out["to_name"].value_counts()
        m["재배치 후보"] = ", ".join(f"{n} {v}명" for n, v in dest.items())
    return m


def facts_text(metrics: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in metrics.items() if k != "department_id")


def build_prompt(metrics: dict) -> str:
    return (f"[사실] (합성 데이터 기반 분석 결과, 기준시점 2026-09)\n{facts_text(metrics)}\n\n"
            f"위 사실만 사용해 {metrics['조직']}의 인력 판단 근거를 설명하세요.")


# ── 2. 수치 검증 ──────────────────────────────────────────────

_DATE = re.compile(r"\d{4}-\d{2}")
_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_numbers(text: str) -> list[float]:
    """텍스트 속 숫자를 모두 뽑는다. YYYY-MM 날짜는 하나의 토큰으로 보고 제외한다."""
    text = _DATE.sub(" ", text)
    return [float(x.replace(",", "")) for x in _NUM.findall(text)]


def allowed_numbers(metrics: dict) -> set[float]:
    """facts에 등장하는 숫자와 그 절댓값(부호 없이 '15명 부족'처럼 쓰는 경우)만 허용."""
    nums = extract_numbers(build_prompt(metrics))
    return {round(abs(n), 6) for n in nums} | {round(n, 6) for n in nums}


def verify_numbers(text: str, metrics: dict) -> list[float]:
    """facts에 없는 숫자 목록을 반환 (빈 리스트면 통과)."""
    allowed = allowed_numbers(metrics)
    return [n for n in extract_numbers(text) if round(n, 6) not in allowed and round(abs(n), 6) not in allowed]


# ── 3. 템플릿 설명 (LLM 실패 시) ───────────────────────────────

def fallback_explanation(m: dict) -> str:
    parts = [
        f"{m['조직']}의 판단은 '{m['판단']}'입니다.",
        f"최근 12개월 업무부하율이 {m['최근 12개월 업무부하율']}({m['업무부하 해석']})이고, 업무량은 전년 대비 {m['업무량 전년 대비']} 변했습니다.",
        f"향후 12개월 필요 FTE {m['향후 12개월 필요 FTE']} 기준 권장 정원은 {m['권장 정원']}이며, 현원 대비 {m['인원 격차(권장 정원 - 현원)']}입니다.",
        f"예측이 ±10% 틀려도 인원 격차는 {m['예측 오차 ±10% 시 인원 격차 범위']} 범위이며, 판단은 {m['판단 안정성']}입니다.",
    ]
    if "충원 방식" in m:
        parts.append(f"충원 방식은 {m['충원 방식']}이며, 부족 스킬은 {m['부족 스킬']}입니다.")
    if "재배치 후보" in m:
        parts.append(f"스킬 조건을 만족하는 재배치 후보(대상 조직별)는 {m['재배치 후보']}입니다.")
    return " ".join(parts)


# ── 4. Gemini 호출 + 캐시 ─────────────────────────────────────

def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")


def model_name() -> str:
    load_env()
    return os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def gemini_llm(model: str | None = None) -> LLM:
    """Gemini 호출 함수. 도구(검색 grounding 등)는 쓰지 않는다 — 무료 한도 밖 과금 요소 차단."""
    from google import genai
    from google.genai import types

    load_env()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY가 .env에 없습니다")
    client = genai.Client(api_key=key)
    model = model or model_name()

    def call(system: str, prompt: str) -> str:
        import time

        from google.genai import errors

        for wait in (10, 30, None):  # 503(일시 과부하) 같은 서버 오류만 재시도. 429(한도)·404는 재시도하지 않음
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    # Gemini 3는 출력 토큰 한도 안에서 내부 추론(thinking)도 하므로 한도를 넉넉히 둔다
                    config=types.GenerateContentConfig(system_instruction=system, temperature=0.2, max_output_tokens=8192),
                )
                finish = resp.candidates[0].finish_reason if resp.candidates else None
                if finish != types.FinishReason.STOP:  # 토큰 한도 등으로 중간에 끊긴 응답은 쓰지 않는다
                    raise IncompleteResponse(f"finish_reason={finish}")
                return (resp.text or "").strip()
            except errors.ServerError:
                if wait is None:
                    raise
                time.sleep(wait)

    return call


class IncompleteResponse(RuntimeError):
    """LLM 응답이 끝까지 생성되지 않음(예: MAX_TOKENS)."""


@dataclass
class Explanation:
    department_id: str
    text: str
    source: str                       # "gemini" | "cache" | "fallback"
    model: str = ""
    unknown_numbers: list[float] = field(default_factory=list)  # 최종 text 기준 (항상 비어 있어야 함)
    rejected: str = ""                # 수치 검증에 실패해 버린 LLM 응답 (투명성 목적)
    rejected_numbers: list[float] = field(default_factory=list)
    error: str = ""


def _cache_key(metrics: dict, model: str) -> str:
    payload = json.dumps([SYSTEM_INSTRUCTION, build_prompt(metrics), model], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def explain_gap(res: Results, dept_id: str, llm: LLM | None = None, model: str | None = None,
                use_cache: bool = True, allow_call: bool = True) -> Explanation:
    """조직의 인력 격차 설명. 캐시 → LLM → (수치 검증 실패/오류 시) 템플릿 순서.

    캐시 키는 프롬프트·시스템 지시·모델의 해시라서, 계산 결과가 바뀌면 자동으로 새로 생성된다.
    allow_call=False면 API를 호출하지 않는다(캐시 또는 템플릿만).
    """
    metrics = get_department_metrics(res, dept_id)
    model = model or (model_name() if llm is None else "injected")
    key = _cache_key(metrics, model)
    cache = _load_cache() if use_cache else {}

    if key in cache:
        hit = Explanation(**cache[key])
        hit.unknown_numbers = verify_numbers(hit.text, metrics)  # 캐시도 매번 재검증
        if not hit.unknown_numbers:
            hit.source = "cache"
            return hit

    if not allow_call:
        return Explanation(dept_id, fallback_explanation(metrics), "fallback", error="API 호출 비활성화")

    try:
        text = (llm or gemini_llm(model))(SYSTEM_INSTRUCTION, build_prompt(metrics))
    except Exception as e:  # 무료 한도 초과(429), 네트워크, 키 오류 등 → 앱이 멈추지 않도록 대체
        return Explanation(dept_id, fallback_explanation(metrics), "fallback", model, error=f"{type(e).__name__}: {str(e)[:200]}")

    bad = verify_numbers(text, metrics)
    if bad or not text:
        result = Explanation(dept_id, fallback_explanation(metrics), "fallback", model,
                             rejected=text, rejected_numbers=bad, error="수치 검증 실패" if bad else "빈 응답")
    else:
        result = Explanation(dept_id, text, "gemini", model)

    if use_cache and result.source == "gemini":
        cache[key] = asdict(result)
        _save_cache(cache)
    return result


if __name__ == "__main__":
    import sys

    from src.pipeline import run_all

    allow = "--no-call" not in sys.argv
    only = [a for a in sys.argv[1:] if a.startswith("D")]
    res = run_all(backtest=False)
    for d in only or res.staff["department_id"]:
        e = explain_gap(res, d, allow_call=allow)
        if e.error.startswith("ClientError"):  # 404(모델 없음)·429(한도 초과) 등은 반복해도 같으므로 이후 호출 중단
            allow = False
        print(f"[{d}] source={e.source} model={e.model} unknown_numbers={e.unknown_numbers} {e.error}")
        if e.rejected:
            print(f"  rejected (숫자 {e.rejected_numbers}): {e.rejected[:200]}")
        print(f"  {e.text}\n")
