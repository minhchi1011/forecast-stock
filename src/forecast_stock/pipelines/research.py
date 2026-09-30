"""Chuẩn bị dữ liệu dùng chung cho các pipeline nghiên cứu (backtest, sweep)."""

from dataclasses import dataclass

import pandas as pd

from .. import backtest as bt
from ..config import BaselineConfig, Config
from ..data import PriceData, PriceStore, listed_mask, load_delisted
from ..features import build_panel, target_columns, to_matrix
from ..strategies import baseline_score, equal_weight_score

# số mã tối đa khi chia đều cả rổ (lớn hơn mọi rổ thực tế)
ALL_NAMES = 10_000


@dataclass
class ResearchData:
    cfg: Config
    px: PriceData
    panel: pd.DataFrame
    eligible: pd.DataFrame  # ma trận (ngày x mã): được phép mua
    target: pd.DataFrame  # ma trận (ngày x mã): excess return h phiên

    def score(self, baseline: BaselineConfig) -> pd.DataFrame:
        return baseline_score(self.panel, baseline, self.px.close.index)

    def baseline(self, name: str) -> BaselineConfig:
        match = [b for b in self.cfg.baselines if b.name == name]
        if not match:
            raise KeyError(f"Không có baseline '{name}' trong config")
        return match[0]

    def backtest(self, name: str, score: pd.DataFrame, rules: bt.PortfolioRules) -> bt.BacktestResult:
        return bt.run_backtest(name, score, self.eligible, self.px, self.cfg.market, self.cfg.costs, rules)

    def equal_weight(self, rebalance_weeks: int) -> bt.BacktestResult:
        rules = bt.PortfolioRules(ALL_NAMES, ALL_NAMES, rebalance_weeks)
        return self.backtest("Chia đều rổ thanh khoản", equal_weight_score(self.eligible), rules)

    def index(self) -> bt.BacktestResult:
        return bt.index_buy_and_hold(self.px, f"{self.cfg.data.index} (mua & giữ)")


def prepare_research(cfg: Config, save_features: bool = True) -> ResearchData:
    store = PriceStore(cfg.data_dir)
    px = PriceData.from_long(store.load(store.load_universe(cfg.data.universe)), cfg.data.index)

    panel = build_panel(px, cfg)
    if save_features:
        path = cfg.data_dir / "features.parquet"
        panel.to_parquet(path)
        print(f"Đặc trưng: {panel.shape[0]:,} dòng x {panel.shape[1]} cột -> {path}")

    dates = px.close.index
    # Được mua khi: đủ thanh khoản, đủ lịch sử cho đặc trưng dài nhất (để mọi chiến lược
    # so trên cùng rổ và giai đoạn), và còn niêm yết (mã đã hủy có thể vẫn có giá ở sàn khác)
    longest = f"mom_{max(cfg.features.momentum_windows)}"
    listed = listed_mask(dates, px.close.columns, load_delisted(cfg.data.delisted_file))
    eligible = (to_matrix(panel, "liquid", dates) > 0) & to_matrix(panel, longest, dates).notna() & listed

    _, excess_col = target_columns(cfg.target.horizon)
    return ResearchData(cfg, px, panel, eligible, to_matrix(panel, excess_col, dates))


def align_start(results: list[bt.BacktestResult]) -> tuple[list[bt.BacktestResult], pd.Timestamp]:
    """Cắt mọi kết quả về kỳ đầu tiên mà tất cả chiến lược (trừ chỉ số) đều đã có danh mục."""
    start = max(r.returns[r.n_holdings > 1].index.min() for r in results if r.n_holdings.max() > 1)
    return [r.since(start) for r in results], start
