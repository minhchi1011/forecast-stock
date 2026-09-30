import numpy as np
import pandas as pd
import pytest

from forecast_stock.backtest.engine import PortfolioRules, _target_weights, _write_off_halted
from forecast_stock.backtest.execution import Tradability, halted_sessions, trading_cost
from forecast_stock.backtest.schedule import rebalance_schedule, weekly_schedule
from forecast_stock.config import CostConfig, MarketConfig
from forecast_stock.data.delisted import listed_mask

MARKET = MarketConfig(price_limit=0.07, limit_tolerance=0.005, writeoff_after_halted_sessions=3, writeoff_haircut=0.5)


def test_halted_sessions_counts_consecutive_gaps_after_listing():
    days = pd.bdate_range("2024-01-01", periods=8)
    open_ = pd.DataFrame(
        {
            # chưa niêm yết 2 phiên đầu -> không tính là ngừng giao dịch
            "NEW": [np.nan, np.nan, 10, 10, 10, 10, 10, 10],
            # ngừng 2 phiên, giao dịch lại, rồi ngừng hẳn
            "HALT": [10, np.nan, np.nan, 10, np.nan, np.nan, np.nan, np.nan],
        },
        index=days,
    )
    out = halted_sessions(open_, days)
    assert out["NEW"].tolist() == [0] * 8
    assert out["HALT"].tolist() == [0, 1, 2, 0, 1, 2, 3, 4]


def test_write_off_removes_long_halted_and_reports_loss():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    halted = pd.Series({"A": 0, "B": 3})
    new, loss = _write_off_halted(weights, halted, MARKET)
    assert loss == pytest.approx(0.2)  # mất 50% của 40%
    assert list(new.index) == ["A"]
    # NAV còn 0.8: A = 0.6/0.8, phần còn lại (0.2/0.8) là tiền mặt thu hồi từ B
    assert new["A"] == pytest.approx(0.75)


def test_write_off_noop_below_threshold():
    weights = pd.Series({"A": 1.0})
    new, loss = _write_off_halted(weights, pd.Series({"A": 2}), MARKET)
    assert loss == 0.0
    assert new.equals(weights)


def test_tradability_limit_up_down_and_halt():
    prev = pd.Series({"UP": 10.0, "DOWN": 10.0, "OK": 10.0, "HALT": 10.0})
    open_ = pd.Series({"UP": 10.7, "DOWN": 9.3, "OK": 10.3, "HALT": np.nan})
    t = Tradability.at_open(prev, open_, MARKET)
    assert t.can_buy().to_dict() == {"UP": False, "DOWN": True, "OK": True, "HALT": False}
    assert t.cannot_sell().to_dict() == {"UP": False, "DOWN": True, "OK": False, "HALT": True}


def test_trading_cost_charges_tax_only_on_sells():
    costs = CostConfig(buy_fee=0.0015, sell_fee=0.0015, sell_tax=0.001)
    cost, sold = trading_cost(pd.Series({"A": 1.0}), pd.Series({"B": 1.0}), costs)
    assert sold == pytest.approx(1.0)
    assert cost == pytest.approx(0.0015 + 0.0015 + 0.001)


def test_weekly_schedule_executes_on_next_session():
    days = pd.bdate_range("2024-01-01", "2024-01-19")  # 3 tuần, thứ 2 -> thứ 6
    signal, execute = weekly_schedule(days)
    assert list(signal.day_name()) == ["Friday", "Friday"]  # tuần cuối không có phiên sau
    assert list(execute.day_name()) == ["Monday", "Monday"]


def test_listed_mask_cuts_at_delist_date():
    days = pd.bdate_range("2024-01-01", periods=4)
    delisted = pd.Series({"X": pd.Timestamp("2024-01-03"), "Y": pd.NaT})
    mask = listed_mask(days, pd.Index(["X", "Y", "Z"]), delisted)
    assert mask["X"].tolist() == [True, True, False, False]
    assert mask["Y"].all() and mask["Z"].all()


def _all_tradable(symbols):
    idx = pd.Index(symbols)
    no = pd.Series(False, index=idx)
    return Tradability(limit_up=no, limit_down=no, halted=no)


def test_buffer_keeps_holding_inside_hold_rank():
    # A đang giữ, tụt xuống hạng 3: top_k=2 sẽ bán A, nhưng hold_rank=3 thì giữ A
    score = pd.Series({"B": 4.0, "C": 3.0, "A": 2.0, "D": 1.0})
    eligible = pd.Series(True, index=score.index)
    current = pd.Series({"A": 1.0})
    trad = _all_tradable(score.index)

    no_buffer = _target_weights(current, score, eligible, trad, PortfolioRules(2, 2, 1))
    assert set(no_buffer.index) == {"B", "C"}

    buffered = _target_weights(current, score, eligible, trad, PortfolioRules(2, 3, 1))
    assert set(buffered.index) == {"A", "B"}  # giữ A, chỉ mua thêm mã tốt nhất
    assert buffered.sum() == pytest.approx(1.0)


def test_portfolio_rules_rejects_hold_rank_below_top_k():
    with pytest.raises(ValueError):
        PortfolioRules(top_k=10, hold_rank=5, rebalance_weeks=1)


def test_rebalance_schedule_every_n_weeks():
    days = pd.bdate_range("2024-01-01", "2024-02-09")  # 6 tuần
    signal, execute = rebalance_schedule(days, every_weeks=2)
    assert len(signal) == 3  # tuần 1, 3, 5 (tuần 6 không có phiên sau)
    assert (execute - signal).days.tolist() == [3, 3, 3]
