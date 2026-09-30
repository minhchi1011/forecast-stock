"""Luật khớp lệnh của thị trường: trần/sàn, ngừng giao dịch, phí và thuế."""

from dataclasses import dataclass

import pandas as pd

from ..config import CostConfig, MarketConfig


@dataclass(frozen=True)
class Tradability:
    """Tình trạng từng mã tại giá mở cửa phiên khớp lệnh."""

    limit_up: pd.Series  # mở cửa ở giá trần -> không mua được
    limit_down: pd.Series  # mở cửa ở giá sàn -> không bán được
    halted: pd.Series  # không có giá -> không giao dịch được

    @classmethod
    def at_open(cls, prev_close: pd.Series, open_: pd.Series, market: MarketConfig) -> "Tradability":
        gap = open_ / prev_close - 1
        threshold = market.price_limit - market.limit_tolerance
        return cls(limit_up=gap >= threshold, limit_down=gap <= -threshold, halted=open_.isna())

    def can_buy(self) -> pd.Series:
        return ~self.limit_up & ~self.halted

    def cannot_sell(self) -> pd.Series:
        return self.limit_down | self.halted


def halted_sessions(open_: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Số phiên liên tiếp không có giá tính tới mỗi ngày, chỉ tính sau khi mã đã bắt đầu giao dịch."""
    missing = open_.reindex(calendar).isna()
    started = (~missing).cummax()
    total = missing.astype(int).cumsum()
    # trừ đi tổng tại phiên có giá gần nhất -> đếm lại từ 0 sau mỗi phiên có giao dịch
    at_last_trade = total.where(~missing).ffill().fillna(0)
    return (total - at_last_trade).where(started, 0)


def trading_cost(old: pd.Series, new: pd.Series, costs: CostConfig) -> tuple[float, float]:
    """(chi phí theo tỷ lệ NAV, tỷ trọng đã bán) khi chuyển danh mục old -> new."""
    diff = new.sub(old, fill_value=0.0)
    bought = diff.clip(lower=0).sum()
    sold = (-diff).clip(lower=0).sum()
    cost = bought * costs.buy_fee + sold * (costs.sell_fee + costs.sell_tax)
    return cost, sold
