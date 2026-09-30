"""Chiến lược không dùng ML: điểm = sign * một đặc trưng, định nghĩa trong config."""

import pandas as pd

from ..config import BaselineConfig
from ..features import to_matrix


def baseline_score(panel: pd.DataFrame, baseline: BaselineConfig, dates: pd.DatetimeIndex) -> pd.DataFrame:
    if baseline.feature not in panel.columns:
        raise KeyError(f"Baseline '{baseline.name}' dùng đặc trưng không tồn tại: {baseline.feature}")
    return baseline.sign * to_matrix(panel, baseline.feature, dates)


def equal_weight_score(eligible: pd.DataFrame) -> pd.DataFrame:
    """Điểm như nhau cho mọi mã -> dùng với top_k lớn để chia đều cả rổ."""
    return eligible.astype(float)
