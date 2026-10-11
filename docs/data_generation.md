# 합성 데이터 생성 문서

> ⚠ **모든 데이터는 합성(synthetic) 데이터입니다.** 가상 기업 *Aster Digital*을 가정해 `src/data_gen.py`로 생성했으며, 실존 기업·인물·실제 인사 데이터와 무관합니다.

## 재현 방법

```bash
python -m src.data_gen     # data/synthetic/*.csv + orgpilot.db 생성 (seed=42 고정)
python -m src.validate     # 무결성 검증
pytest tests/test_data.py  # 검증 + 재현성 테스트
```

## 시나리오

- 기준시점(AS_OF): **2026-09**, 관측기간 24개월(2024-10 ~ 2026-09)
- 8개 조직, 300명
- 기존 정원(TO)은 **"직전 연말 현원 + 부서 증원 요청분"** 으로 연초에 한 번 정해지는 과거 관행을 따른다. 그 결과 업무량이 늘어나는 조직(AI·클라우드)은 정원이 부족하고, 업무량이 줄어드는 조직(SI2·운영·경영지원)은 정원이 남는 상황이 만들어진다.

## 테이블

| 테이블 | PK | 컬럼 | 행 수 |
|---|---|---|---|
| `departments` | department_id | name, function, current_headcount | 8 |
| `employees` | employee_id | department_id, job_family, tenure(년), status(active/on_leave) | 300 |
| `workload` | (department_id, period) | task_count(월 업무건수), standard_hours(건당 표준처리시간) | 192 |
| `skills` | (employee_id, skill) | proficiency(1~5) | 약 1,200 |
| `staffing_plan` | (department_id, period) | approved_to(정원), actual_headcount(현원) | 192 |

## 생성 로직

### 1) 업무량 (`workload`)

조직마다 기준시점의 **'진짜' 필요 FTE**(`need_end`)를 정하고, 시점 t(0~23)의 필요 FTE를 다음과 같이 만든다.

```
need_fte[t] = need_end × (1 + growth)^(t − 23) × season(month) × LogNormal(0, 0.04)
season      = 1 + amp   (3·6·9·12월, 분기말 마감)
              1 − amp/2 (1월, 비수기)
              1         (그 외)
task_count  = round(need_fte × 140h / standard_hours)
```

- 140h = 1인당 월 가용시간 가정 (`src/config.py`의 `AVAILABLE_HOURS_PER_FTE`)
- **숨은 정답(`need_end`, `growth`, `season_amp`)은 테이블에 저장하지 않는다.** 분석은 `task_count`에서 출발해야 하며, `check_schema`가 정해진 컬럼 외 저장을 막는다.

### 2) 조직별 파라미터

| ID | 조직 | 현원(24개월 전→현재) | 필요 FTE(기준시점) | 월 성장률 | 분기말 증가폭 | 건당 표준시간 | 의도한 상황 |
|---|---|---|---|---|---|---|---|
| D01 | SI사업1팀 | 54 → 55 | 56 | +0.4% | 15% | 24h | 균형~약간 과부하 |
| D02 | SI사업2팀 | 50 → 48 | 40 | −0.3% | 15% | 24h | 저활용 |
| D03 | 클라우드인프라팀 | 33 → 38 | 45 | +1.2% | 8% | 6h | 과부하(성장) |
| D04 | AI서비스팀 | 18 → 24 | 32 | +2.0% | 5% | 40h | 심한 과부하(고성장) |
| D05 | PMO팀 | 30 → 30 | 29 | +0.2% | 12% | 16h | 균형 |
| D06 | IT운영팀 | 54 → 50 | 44 | −0.6% | 5% | 2.5h | 저활용(자동화로 감소) |
| D07 | 정보보안팀 | 20 → 22 | 24 | +0.6% | 8% | 5h | 약간 과부하 |
| D08 | 경영지원팀 | 34 → 33 | 28 | 0% | 10% | 3h | 저활용 |

### 3) 직원·스킬 (`employees`, `skills`)

- 직무군(job_family): 조직별 가중치로 추출 (예: AI서비스팀은 데이터/AI 80%, 개발 20%)
- 근속연수: Gamma(2, 3.5), 0.1~30년으로 제한
- 휴직(on_leave): 약 3%
- 스킬: 직원당 3~5개, 조직별 가중치로 비복원 추출 → **조직 간 스킬 분포 불균형**을 의도적으로 반영 (예: ML/DL 보유자는 적고 AI서비스팀에 몰려 있음)
- 숙련도: `round(Normal(2.3 + 0.12 × 근속연수, 0.9))`, 1~5로 제한 → 근속이 길수록 숙련도가 높은 경향

### 4) 정원·현원 (`staffing_plan`)

- 현원: 24개월 전 현원 → 현재 현원으로 선형 추세 + 월 ±1명 변동. 마지막 달은 `departments.current_headcount`와 정확히 일치
- 정원(TO): 연초에만 갱신. `직전 연말 현원 + to_request`(부서 증원 요청분 1~3명). 첫 해(2024)는 관측 첫 달 현원 기준

## 검증 항목 (`src/validate.py`)

| 검증 | 내용 |
|---|---|
| check_schema | 정해진 컬럼만 존재 (숨은 정답 누출 방지) |
| check_primary_keys | PK 중복 없음 |
| check_foreign_keys | 모든 FK가 부모 테이블에 존재 |
| check_no_nulls | 결측치 없음 |
| check_headcount_consistency | 부서 현원 = 직원 수 = 기준시점 staffing_plan 현원 |
| check_period_coverage | 모든 조직이 동일한 24개월 보유, 기준시점 이후 데이터 없음 (미래 정보 누출 방지) |
| check_value_ranges | 숙련도 1~5, 직원별 스킬 3~5개, 업무량·정원 > 0 등 |

`tests/test_data.py`는 위 검증 외에 ① 같은 seed → 동일 데이터(재현성), ② 디스크 DB가 현재 코드의 생성 결과와 일치, ③ **일부러 망가뜨린 데이터(고아 FK, 미래 시점 행, 숨은 컬럼, 현원 불일치)를 검증 함수가 실제로 잡아내는지**를 테스트한다.

## 한계

- 업무량 = 건수 × 고정 표준시간이라는 단순 구조. 실제로는 업무 난이도·숙련도에 따라 처리시간이 달라진다.
- 필요 FTE 구조를 생성자가 알고 설계했으므로, 이 데이터에서 예측이 잘 맞는 것이 실제 조직에서도 잘 맞는다는 근거는 되지 않는다.
- 이직·채용 이벤트, 직원 단위 업무 배분 데이터는 포함하지 않았다.
