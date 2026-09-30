"""Nạp cấu hình từ TOML thành các dataclass bất biến."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path("configs/default.toml")


@dataclass(frozen=True)
class FetchConfig:
    requests_per_symbol: int
    safety_margin: float
    retries: int
    retry_wait_seconds: float
    requests_per_minute: dict[str, int]

    def pause_seconds(self, has_api_key: bool) -> float:
        """Thời gian nghỉ giữa các mã để không vượt giới hạn request/phút."""
        rpm = self.requests_per_minute["api_key" if has_api_key else "guest"]
        return 60 / rpm * self.requests_per_symbol * self.safety_margin


@dataclass(frozen=True)
class DataConfig:
    universe: str
    delisted_file: Path | None
    index: str
    start: str
    price_unit: float
    max_rows: int
    fetch: FetchConfig


@dataclass(frozen=True)
class MarketConfig:
    price_limit: float
    limit_tolerance: float
    writeoff_after_halted_sessions: int
    writeoff_haircut: float


@dataclass(frozen=True)
class CostConfig:
    buy_fee: float
    sell_fee: float
    sell_tax: float


@dataclass(frozen=True)
class FeatureConfig:
    momentum_windows: tuple[int, ...]
    momentum_skip: int
    reversal_window: int
    volatility_windows: tuple[int, ...]
    ma_window: int
    volume_short: int
    volume_long: int


@dataclass(frozen=True)
class UniverseFilterConfig:
    adv_window: int
    min_traded_value: float


@dataclass(frozen=True)
class TargetConfig:
    horizon: int


@dataclass(frozen=True)
class ModelConfig:
    min_train_years: float
    retrain_months: int
    embargo_sessions: int
    num_boost_round: int
    seed: int
    params: dict


@dataclass(frozen=True)
class BacktestConfig:
    top_k: int
    hold_rank: int
    rebalance_weeks: int
    min_ic_names: int


@dataclass(frozen=True)
class SweepConfig:
    baselines: tuple[str, ...]
    rebalance_weeks: tuple[int, ...]
    hold_rank: tuple[int, ...]


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    feature: str
    sign: int


@dataclass(frozen=True)
class Config:
    data_dir: Path
    data: DataConfig
    market: MarketConfig
    costs: CostConfig
    features: FeatureConfig
    universe_filter: UniverseFilterConfig
    target: TargetConfig
    model: ModelConfig
    backtest: BacktestConfig
    baselines: tuple[BaselineConfig, ...]
    sweep: SweepConfig


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    data = dict(raw["data"])
    data["fetch"] = FetchConfig(**data["fetch"])
    data["delisted_file"] = Path(data["delisted_file"]) if data.get("delisted_file") else None
    feats = dict(raw["features"])
    feats["momentum_windows"] = tuple(feats["momentum_windows"])
    feats["volatility_windows"] = tuple(feats["volatility_windows"])

    return Config(
        data_dir=Path(raw["paths"]["data_dir"]),
        data=DataConfig(**data),
        market=MarketConfig(**raw["market"]),
        costs=CostConfig(**raw["costs"]),
        features=FeatureConfig(**feats),
        universe_filter=UniverseFilterConfig(**raw["universe_filter"]),
        target=TargetConfig(**raw["target"]),
        model=ModelConfig(**raw["model"]),
        backtest=BacktestConfig(**raw["backtest"]),
        baselines=tuple(BaselineConfig(**b) for b in raw.get("baselines", [])),
        sweep=SweepConfig(**{k: tuple(v) for k, v in raw["sweep"].items()}),
    )
