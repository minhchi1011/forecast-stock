"""Thanh khoản: giá trị giao dịch trung bình và cờ đủ điều kiện vào rổ."""

import numpy as np
import pandas as pd

from ..config import UniverseFilterConfig
from ..data import PriceData


def liquidity_features(px: PriceData, cfg: UniverseFilterConfig, price_unit: float) -> dict[str, pd.DataFrame]:
    traded_value = px.close * price_unit * px.volume
    w = cfg.adv_window
    adv = traded_value.rolling(w, min_periods=int(w * 0.75)).mean()
    return {
        f"log_adv{w}": np.log(adv),
        "liquid": (adv >= cfg.min_traded_value).astype(float),
    }
