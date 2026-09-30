import numpy as np
import pandas as pd
import pytest

from forecast_stock.config import ModelConfig
from forecast_stock.models import make_dataset, walk_forward
from forecast_stock.models.dataset import Dataset

HORIZON = 5
CFG = ModelConfig(
    min_train_years=1, retrain_months=6, embargo_sessions=HORIZON, num_boost_round=5, seed=0,
    params={"objective": "regression", "verbose": -1, "min_data_in_leaf": 5},
)


def _synthetic(n_days=500, n_symbols=20) -> tuple[Dataset, pd.DatetimeIndex]:
    rng = np.random.default_rng(0)
    calendar = pd.bdate_range("2020-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([calendar, [f"S{i}" for i in range(n_symbols)]], names=["date", "symbol"])
    X = pd.DataFrame({"f1": rng.normal(size=len(idx)), "f2": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    return Dataset(X, y), calendar


def test_walk_forward_has_no_lookahead():
    ds, calendar = _synthetic()
    wf = walk_forward(ds, calendar, CFG, HORIZON)
    assert len(wf.blocks) >= 2
    for train_end, test_start, _ in wf.blocks:
        # mục tiêu của dòng train cuối (dùng giá tới phiên t+1+h) phải biết trước phiên test đầu tiên
        last_target_pos = calendar.get_loc(train_end) + 1 + HORIZON
        assert last_target_pos <= calendar.searchsorted(test_start)


def test_walk_forward_predictions_only_out_of_sample_and_unique():
    ds, calendar = _synthetic()
    wf = walk_forward(ds, calendar, CFG, HORIZON)
    pred_dates = wf.predictions.index.get_level_values("date")
    assert pred_dates.min() >= wf.blocks[0][1]
    assert not wf.predictions.index.duplicated().any()


def test_walk_forward_rejects_short_embargo():
    ds, calendar = _synthetic(n_days=300)
    with pytest.raises(ValueError):
        walk_forward(ds, calendar, CFG, horizon=HORIZON + 1)


def test_make_dataset_ranks_only_among_eligible():
    days = pd.bdate_range("2024-01-01", periods=1)
    idx = pd.MultiIndex.from_product([days, ["A", "B", "C"]], names=["date", "symbol"])
    panel = pd.DataFrame(
        {"mom_20": [1.0, 2.0, 100.0], "liquid": [1, 1, 0], "fwd_ret_5": 0.0, "target_excess_5": [0.1, 0.2, 0.3]},
        index=idx,
    )
    eligible = pd.DataFrame({"A": [True], "B": [True], "C": [False]}, index=days)
    ds = make_dataset(panel, eligible, horizon=5)
    assert list(ds.X.columns) == ["mom_20"]  # bỏ cột liquid và cột mục tiêu
    assert ds.X.index.get_level_values("symbol").tolist() == ["A", "B"]
    assert ds.X["mom_20"].tolist() == [0.0, 0.5]  # xếp hạng 50%, 100% trừ 0.5
