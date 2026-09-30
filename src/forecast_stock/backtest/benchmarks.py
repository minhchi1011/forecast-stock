"""Chiến lược tham chiếu để so sánh."""

import pandas as pd

from ..data import PriceData
from .engine import BacktestResult
from .schedule import weekly_schedule


def index_buy_and_hold(px: PriceData, name: str) -> BacktestResult:
    """Chỉ số tham chiếu, đo trên cùng các kỳ mở cửa -> mở cửa với backtest, không phí."""
    _, exec_dates = weekly_schedule(px.calendar)
    r = px.index_open.loc[exec_dates].pct_change().shift(-1).dropna()
    return BacktestResult(name, r, pd.Series(0.0, index=r.index), pd.Series(1, index=r.index), exec_dates[-1])
