"""Đặc trưng kỹ thuật. Giá trị tại ngày t chỉ dùng dữ liệu tới hết phiên t."""

import pandas as pd

from ..config import FeatureConfig
from ..data import PriceData


def technical_features(px: PriceData, cfg: FeatureConfig) -> dict[str, pd.DataFrame]:
    close, volume = px.close, px.volume
    ret1 = close.pct_change(fill_method=None)
    skip = cfg.momentum_skip
    feats: dict[str, pd.DataFrame] = {}

    for w in cfg.momentum_windows:
        feats[f"mom_{w}"] = close.shift(skip) / close.shift(w) - 1

    feats[f"ret_{cfg.reversal_window}"] = close / close.shift(cfg.reversal_window) - 1

    for w in cfg.volatility_windows:
        feats[f"vol_{w}"] = ret1.rolling(w, min_periods=int(w * 0.75)).std()

    ma = close.rolling(cfg.ma_window, min_periods=int(cfg.ma_window * 0.8)).mean()
    feats[f"dist_ma{cfg.ma_window}"] = close / ma - 1

    vol_long = volume.rolling(cfg.volume_long, min_periods=int(cfg.volume_long * 0.75)).mean()
    feats["volume_ratio"] = volume.rolling(cfg.volume_short).mean() / vol_long
    return feats
