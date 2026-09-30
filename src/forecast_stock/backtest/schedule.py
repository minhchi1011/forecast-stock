"""Lịch cân bằng lại danh mục."""

import pandas as pd


def weekly_schedule(calendar: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """(ngày tín hiệu = phiên cuối mỗi tuần, ngày khớp lệnh = phiên ngay sau đó)."""
    last_of_week = pd.Series(calendar, index=calendar).groupby(calendar.to_period("W")).max()
    signal = pd.DatetimeIndex(last_of_week.values)
    exec_pos = calendar.get_indexer(signal) + 1
    has_next = exec_pos < len(calendar)
    return signal[has_next], calendar[exec_pos[has_next]]


def rebalance_schedule(calendar: pd.DatetimeIndex, every_weeks: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Như weekly_schedule nhưng chỉ lấy mỗi `every_weeks` tuần một lần."""
    signal, execute = weekly_schedule(calendar)
    return signal[::every_weeks], execute[::every_weeks]
