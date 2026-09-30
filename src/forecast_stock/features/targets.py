"""Biến mục tiêu, căn theo thời điểm khớp lệnh thực tế.

Tín hiệu có sau khi đóng cửa phiên t -> sớm nhất khớp ở giá mở cửa t+1.
Mục tiêu tại t = lợi nhuận mở cửa(t+1) -> mở cửa(t+1+h).
"""

import pandas as pd

from ..data import PriceData


def forward_open_return(open_: pd.DataFrame | pd.Series, horizon: int):
    return open_.shift(-1 - horizon) / open_.shift(-1) - 1


def target_columns(horizon: int) -> tuple[str, str]:
    return f"fwd_ret_{horizon}", f"target_excess_{horizon}"


def targets(px: PriceData, horizon: int) -> dict[str, pd.DataFrame]:
    raw_col, excess_col = target_columns(horizon)
    fwd = forward_open_return(px.open, horizon)
    index_fwd = forward_open_return(px.index_open, horizon)
    return {raw_col: fwd, excess_col: fwd.sub(index_fwd, axis=0)}
