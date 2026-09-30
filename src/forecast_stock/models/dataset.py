"""Chuyển panel thành (X, y) cho model xếp hạng cổ phiếu.

Mọi cột được đổi thành thứ hạng phần trăm theo từng ngày (cross-sectional) giữa các mã
được phép mua, trừ 0.5 để quanh 0. Nhờ vậy model học "mã nào tốt hơn mã nào" thay vì
mức tuyệt đối, vốn trôi theo trạng thái thị trường qua các năm.
"""

from dataclasses import dataclass

import pandas as pd

from ..features import target_columns

NON_FEATURES = {"liquid"}


@dataclass
class Dataset:
    X: pd.DataFrame  # index (date, symbol)
    y: pd.Series  # NaN khi mục tiêu chưa biết (cuối dữ liệu)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.X.index.get_level_values("date")


def feature_columns(panel: pd.DataFrame, horizon: int) -> list[str]:
    targets = set(target_columns(horizon))
    return [c for c in panel.columns if c not in targets and c not in NON_FEATURES]


def _cross_sectional_rank(df: pd.DataFrame | pd.Series):
    return df.groupby(level="date").rank(pct=True) - 0.5


def make_dataset(panel: pd.DataFrame, eligible: pd.DataFrame, horizon: int) -> Dataset:
    """Chỉ giữ các dòng (ngày, mã) được phép mua; xếp hạng trong đúng tập đó."""
    mask = eligible.stack(future_stack=True).reindex(panel.index, fill_value=False).astype(bool)
    rows = panel[mask.values]
    _, excess_col = target_columns(horizon)
    X = _cross_sectional_rank(rows[feature_columns(panel, horizon)])
    y = _cross_sectional_rank(rows[excess_col])
    return Dataset(X, y)
