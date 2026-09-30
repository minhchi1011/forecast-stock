"""Chỉ số đánh giá: hiệu quả danh mục và sức dự báo của tín hiệu (IC)."""

import numpy as np
import pandas as pd

from .engine import BacktestResult
from .schedule import weekly_schedule


def performance(res: BacktestResult) -> dict[str, float | str]:
    """Chỉ số năm hóa theo số kỳ thực tế/năm (kỳ có thể dài 1 hoặc nhiều tuần)."""
    r = res.returns
    equity = (1 + r).cumprod()
    years = (res.end - r.index[0]).days / 365.25
    periods_per_year = len(r) / years
    return {
        "chiến lược": res.name,
        "CAGR": equity.iloc[-1] ** (1 / years) - 1,
        "Vol năm": r.std() * np.sqrt(periods_per_year),
        "Sharpe": r.mean() / r.std() * np.sqrt(periods_per_year),
        "MaxDD": (equity / equity.cummax() - 1).min(),
        "Turnover/năm": res.turnover.sum() / years,
        "Tổng LN": equity.iloc[-1] - 1,
    }


def rank_ic(score: pd.DataFrame, target: pd.DataFrame, eligible: pd.DataFrame, min_names: int) -> pd.Series:
    """Spearman IC theo tuần giữa điểm và mục tiêu tại mỗi ngày tín hiệu."""
    signal_dates, _ = weekly_schedule(score.index)
    out = {}
    for d in signal_dates:
        mask = eligible.loc[d].fillna(False).astype(bool)
        x, y = score.loc[d][mask], target.loc[d][mask]
        ok = x.notna() & y.notna()
        if ok.sum() >= min_names:
            out[d] = x[ok].rank().corr(y[ok].rank())
    return pd.Series(out, dtype=float)


def ic_summary(ic: pd.Series) -> dict[str, float]:
    return {
        "Rank IC TB": ic.mean(),
        "IC t-stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
        "% tuần IC>0": (ic > 0).mean(),
    }
