"""Backtest long-only chọn top-k mã theo điểm, cân bằng lại mỗi N tuần.

- Tín hiệu tính sau khi đóng cửa phiên cuối tuần, khớp ở giá mở cửa phiên kế tiếp.
- Vùng đệm: mã đang giữ chỉ bị bán khi rơi khỏi top `hold_rank` (>= top_k), giảm vòng quay.
  Danh mục (mã giữ lại + mã mới) luôn được chia đều lại mỗi kỳ cân bằng.
- Không mua mã mở cửa trần; mã cần bán nhưng mở cửa sàn / ngừng giao dịch thì bị giữ lại.
- Mã ngừng giao dịch quá lâu (đình chỉ, hủy niêm yết) bị xóa sổ với haircut trong config.
- Nắm giữ tối thiểu 1 tuần (5 phiên) > T+2 nên ràng buộc thanh toán luôn thỏa.
"""

from dataclasses import dataclass

import pandas as pd

from ..config import CostConfig, MarketConfig
from ..data import PriceData
from .execution import Tradability, halted_sessions, trading_cost
from .schedule import rebalance_schedule


@dataclass(frozen=True)
class PortfolioRules:
    top_k: int
    hold_rank: int
    rebalance_weeks: int

    def __post_init__(self):
        if self.hold_rank < self.top_k:
            raise ValueError(f"hold_rank ({self.hold_rank}) phải >= top_k ({self.top_k})")

    def label(self) -> str:
        return f"top {self.top_k}, giữ tới {self.hold_rank}, {self.rebalance_weeks} tuần"


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series  # lợi nhuận ròng mỗi kỳ, index = ngày khớp lệnh
    turnover: pd.Series  # tỷ trọng bán ra mỗi kỳ
    n_holdings: pd.Series
    end: pd.Timestamp  # ngày kết thúc kỳ cuối cùng

    def since(self, start: pd.Timestamp) -> "BacktestResult":
        keep = self.returns.index >= start
        return BacktestResult(self.name, self.returns[keep], self.turnover[keep], self.n_holdings[keep], self.end)


def _write_off_halted(weights: pd.Series, halted: pd.Series, market: MarketConfig) -> tuple[pd.Series, float]:
    """Xóa sổ mã ngừng giao dịch quá lâu: thu hồi (1 - haircut) về tiền mặt.

    Trả về (tỷ trọng theo NAV mới, tỷ lệ NAV bị mất). Tỷ trọng còn lại có tổng < 1, phần thiếu là tiền mặt.
    """
    dead = halted.reindex(weights.index).fillna(0) >= market.writeoff_after_halted_sessions
    if not dead.any():
        return weights, 0.0
    loss = weights[dead].sum() * market.writeoff_haircut
    return weights[~dead] / (1 - loss), loss


def _target_weights(
    current: pd.Series, score: pd.Series, eligible: pd.Series, trad: Tradability, rules: PortfolioRules
) -> pd.Series:
    ranked = score.where(eligible).dropna().sort_values(ascending=False)
    in_hold_zone = set(ranked.index[: rules.hold_rank])

    keep = [s for s in current.index if s in in_hold_zone]
    stuck = [s for s in current.index if s not in in_hold_zone and trad.cannot_sell().get(s, False)]

    buyable = ranked[trad.can_buy().reindex(ranked.index, fill_value=False)]
    n_new = max(rules.top_k - len(keep), 0)
    new_names = [s for s in buyable.index if s not in current.index][:n_new]

    stuck_w = current.reindex(stuck)
    picks = keep + new_names
    free = 1.0 - stuck_w.sum()
    target = pd.Series(free / len(picks), index=picks) if picks else pd.Series(dtype=float)
    return pd.concat([target, stuck_w])


def run_backtest(
    name: str,
    score: pd.DataFrame,
    eligible: pd.DataFrame,
    px: PriceData,
    market: MarketConfig,
    costs: CostConfig,
    rules: PortfolioRules,
) -> BacktestResult:
    """score, eligible: ma trận (ngày x mã). Điểm cao hơn được ưu tiên mua."""
    open_valued = px.open.ffill()  # mã ngừng giao dịch: định giá theo giá gần nhất
    halted = halted_sessions(px.open, px.calendar)
    signal_dates, exec_dates = rebalance_schedule(px.calendar, rules.rebalance_weeks)

    weights = pd.Series(dtype=float)
    returns, turnover, n_holdings = {}, {}, {}
    for i in range(len(exec_dates) - 1):
        s, e, e_next = signal_dates[i], exec_dates[i], exec_dates[i + 1]
        weights, writeoff_loss = _write_off_halted(weights, halted.loc[e], market)

        trad = Tradability.at_open(px.close.loc[s], px.open.loc[e], market)
        new = _target_weights(weights, score.loc[s], eligible.loc[s].fillna(False).astype(bool), trad, rules)

        cost, sold = trading_cost(weights, new, costs)
        period_ret = (open_valued.loc[e_next, new.index] / open_valued.loc[e, new.index] - 1).fillna(0.0)
        gross = (new * period_ret).sum()

        returns[e] = (1 - writeoff_loss) * (1 - cost) * (1 + gross) - 1
        turnover[e] = sold
        n_holdings[e] = len(new)

        # tỷ trọng trôi theo giá; tiền mặt (nếu có) lãi 0 nên NAV cuối kỳ = 1 + gross
        grown = new * (1 + period_ret)
        weights = grown / (1 + gross) if gross > -1 else pd.Series(dtype=float)

    return BacktestResult(name, pd.Series(returns), pd.Series(turnover), pd.Series(n_holdings), exec_dates[-1])
