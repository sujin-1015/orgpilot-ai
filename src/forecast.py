"""Module B-1. 조직별 월 업무량(task_count) 예측과 Rolling-origin 검증.

모델 (모두 같은 시그니처: fit_predict(train, future) -> 예측 task_count 배열)
  baseline_mean     : 최근 12개월 평균 — "작년 수준 유지"라는 현행 관행의 대리 지표
  seasonal_trend    : 조직별 log(task_count) = a + b·t + c·분기말 + d·1월  (최소제곱, 주 모델)
  xgboost_model     : 조직·t·월·분기말 특성의 XGBoost (비교용)

검증: 학습 12개월부터 시작해 기준점을 한 달씩 옮기며 1~6개월 뒤를 예측 (Rolling-origin).
기준점 이후 데이터는 학습에 절대 사용하지 않는다.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.config import AS_OF

Model = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def prepare(workload: pd.DataFrame) -> pd.DataFrame:
    """workload → 모델 입력 (t: 첫 달=0인 월 인덱스, 계절 더미)."""
    df = workload[["department_id", "period", "task_count", "standard_hours"]].copy()
    df = df.sort_values(["department_id", "period"]).reset_index(drop=True)
    periods = sorted(df["period"].unique())
    df["t"] = df["period"].map({p: i for i, p in enumerate(periods)})
    return add_calendar(df)


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    month = df["period"].str[5:7].astype(int)
    return df.assign(month=month, qend=month.isin([3, 6, 9, 12]).astype(int), jan=(month == 1).astype(int))


def future_frame(df: pd.DataFrame, horizon: int, as_of: str = AS_OF) -> pd.DataFrame:
    """기준시점 다음 달부터 horizon개월의 빈 예측 프레임."""
    periods = [str(p) for p in pd.period_range(pd.Period(as_of, "M") + 1, periods=horizon, freq="M")]
    last_t = df.loc[df["period"] == as_of, "t"].iloc[0]
    std = df.groupby("department_id")["standard_hours"].last()
    rows = [(d, p, last_t + i + 1, std[d]) for d in sorted(df["department_id"].unique()) for i, p in enumerate(periods)]
    return add_calendar(pd.DataFrame(rows, columns=["department_id", "period", "t", "standard_hours"]))


# ── 모델 ──────────────────────────────────────────────────────────────

def baseline_mean(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    level = train.sort_values("t").groupby("department_id")["task_count"].apply(lambda s: s.tail(12).mean())
    return future["department_id"].map(level).to_numpy(dtype=float)


SEASONAL_TREND_FEATURES = ["const", "t", "qend", "jan"]


def fit_seasonal_trend(train: pd.DataFrame) -> pd.DataFrame:
    """조직별 회귀계수 표 (행: 조직, 열: const/t/qend/jan). 해석·설명용으로도 쓴다."""
    coefs = {}
    for dept, g in train.groupby("department_id"):
        X = np.column_stack([np.ones(len(g)), g["t"], g["qend"], g["jan"]])
        coefs[dept], *_ = np.linalg.lstsq(X, np.log(g["task_count"].to_numpy(dtype=float)), rcond=None)
    return pd.DataFrame(coefs, index=SEASONAL_TREND_FEATURES).T


def seasonal_trend(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    c = fit_seasonal_trend(train).loc[future["department_id"]].to_numpy()
    X = np.column_stack([np.ones(len(future)), future["t"], future["qend"], future["jan"]])
    return np.exp((X * c).sum(axis=1))


def xgboost_model(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    from xgboost import XGBRegressor

    depts = sorted(train["department_id"].unique())

    def feats(df):
        X = pd.get_dummies(pd.Categorical(df["department_id"], categories=depts), dtype=float)
        return X.assign(t=df["t"].to_numpy(), month=df["month"].to_numpy(), qend=df["qend"].to_numpy())

    model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
    model.fit(feats(train), np.log(train["task_count"]))
    return np.exp(model.predict(feats(future)))


MODELS: dict[str, Model] = {
    "baseline_mean": baseline_mean,
    "seasonal_trend": seasonal_trend,
    "xgboost": xgboost_model,
}
MAIN_MODEL = "seasonal_trend"


# ── 검증 ──────────────────────────────────────────────────────────────

def rolling_origin(df: pd.DataFrame, model: Model, min_train: int = 12, horizon: int = 6) -> pd.DataFrame:
    """기준점 o(학습 마지막 t)를 min_train−1부터 옮기며 o+1..o+horizon을 예측."""
    rows = []
    max_t = df["t"].max()
    for origin in range(min_train - 1, max_t):
        train = df[df["t"] <= origin]
        test = df[(df["t"] > origin) & (df["t"] <= origin + horizon)]
        assert train["t"].max() < test["t"].min()  # 미래 정보 누출 방지
        pred = model(train, test.drop(columns="task_count"))
        rows.append(test.assign(origin=origin, h=test["t"] - origin, pred=pred))
    return pd.concat(rows, ignore_index=True)


def wape(actual: pd.Series, pred: pd.Series) -> float:
    """Weighted APE = Σ|오차| / Σ실제. 조직 규모가 달라도 해석이 쉬움."""
    return float(np.abs(actual - pred).sum() / actual.sum())


def backtest(df: pd.DataFrame, models: dict[str, Model] = MODELS, **kw) -> pd.DataFrame:
    return pd.concat([rolling_origin(df, m, **kw).assign(model=name) for name, m in models.items()],
                     ignore_index=True)


def summarize_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    """모델별: 월 단위 WAPE, 그리고 TO 산정에 실제로 쓰이는 '향후 기간 평균 수준'의 오차."""
    out = {}
    for name, g in bt.groupby("model"):
        level = g.groupby(["origin", "department_id"])[["task_count", "pred"]].mean()
        level_err = (level["pred"] / level["task_count"] - 1)
        out[name] = {
            "monthly_wape": wape(g["task_count"], g["pred"]),
            "level_wape": wape(level["task_count"], level["pred"]),
            "level_err_p90": float(level_err.abs().quantile(0.9)),
            "bias": float(g["pred"].sum() / g["task_count"].sum() - 1),
        }
    return pd.DataFrame(out).T.sort_values("monthly_wape")


def wape_by(bt: pd.DataFrame, key: str) -> pd.DataFrame:
    """모델 × key(h 또는 department_id)별 WAPE."""
    return (bt.groupby(["model", key])[["task_count", "pred"]]
              .apply(lambda g: wape(g["task_count"], g["pred"])).unstack(0))


def forecast(df: pd.DataFrame, model: Model | str = MAIN_MODEL, horizon: int = 12, as_of: str = AS_OF) -> pd.DataFrame:
    """기준시점까지 전체로 학습해 향후 horizon개월 업무량 예측."""
    model = MODELS[model] if isinstance(model, str) else model
    train = df[df["period"] <= as_of]
    fut = future_frame(train, horizon, as_of)
    return fut.assign(task_count_pred=model(train, fut))
