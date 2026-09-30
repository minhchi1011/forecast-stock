"""Gộp đặc trưng + mục tiêu thành một panel dạng dài, index (date, symbol)."""

import pandas as pd

from ..config import Config
from ..data import PriceData
from .liquidity import liquidity_features
from .targets import targets
from .technical import technical_features


def build_panel(px: PriceData, cfg: Config) -> pd.DataFrame:
    technical = technical_features(px, cfg.features)
    matrices = {
        **technical,
        **liquidity_features(px, cfg.universe_filter, cfg.data.price_unit),
        **targets(px, cfg.target.horizon),
    }
    panel = pd.concat({k: m.stack(future_stack=True) for k, m in matrices.items()}, axis=1)
    panel.index.names = ["date", "symbol"]
    # bỏ các dòng chưa có đặc trưng nào (mã chưa niêm yết / chưa đủ lịch sử)
    return panel.dropna(how="all", subset=list(technical))


def to_matrix(panel: pd.DataFrame, column: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Lấy một cột của panel về dạng ma trận (ngày x mã)."""
    return panel[column].unstack().reindex(dates)
