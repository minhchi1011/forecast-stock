"""Walk-forward: train trên quá khứ (cửa sổ mở rộng), dự báo khối kế tiếp, lặp lại.

Embargo: mục tiêu tại ngày t dùng giá tới phiên t+1+h, nên dòng train phải thỏa
t + 1 + h < phiên đầu kỳ test. Ta bỏ `embargo_sessions + 1` phiên cuối trước mỗi kỳ test.
"""

from dataclasses import dataclass

import pandas as pd

from ..config import ModelConfig
from .dataset import Dataset
from .lgbm import LGBMRanker


@dataclass
class WalkForwardResult:
    predictions: pd.Series  # index (date, symbol), chỉ có trong kỳ ngoài mẫu
    importance: pd.DataFrame  # mỗi cột là một lần train
    blocks: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]  # (train_end, test_start, test_end)


def oos_blocks(calendar: pd.DatetimeIndex, cfg: ModelConfig) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    first_test = calendar[0] + pd.DateOffset(months=int(cfg.min_train_years * 12))
    edges = pd.date_range(first_test, calendar[-1], freq=pd.DateOffset(months=cfg.retrain_months))
    edges = edges.append(pd.DatetimeIndex([calendar[-1] + pd.Timedelta(days=1)]))
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def walk_forward(ds: Dataset, calendar: pd.DatetimeIndex, cfg: ModelConfig, horizon: int) -> WalkForwardResult:
    if cfg.embargo_sessions < horizon:
        raise ValueError(f"embargo_sessions ({cfg.embargo_sessions}) phải >= horizon ({horizon})")

    # lịch bắt đầu từ ngày đầu tiên có dữ liệu train
    calendar = calendar[calendar >= ds.dates.min()]
    dates = ds.dates
    preds, importance, blocks = [], {}, []
    for test_start, test_end in oos_blocks(calendar, cfg):
        pos = calendar.searchsorted(test_start)
        train_end = calendar[max(pos - cfg.embargo_sessions - 1, 0)]
        train = (dates <= train_end) & ds.y.notna().values
        test = (dates >= test_start) & (dates < test_end)
        if not test.any():
            continue

        model = LGBMRanker(cfg).fit(ds.X[train], ds.y[train])
        preds.append(model.predict(ds.X[test]))
        importance[f"{test_start:%Y-%m}"] = model.feature_importance()
        blocks.append((train_end, test_start, test_end))
        print(f"  train tới {train_end:%Y-%m-%d} ({train.sum():,} dòng) -> dự báo "
              f"{test_start:%Y-%m-%d} .. {min(test_end, calendar[-1]):%Y-%m-%d}", flush=True)

    return WalkForwardResult(pd.concat(preds).sort_index(), pd.DataFrame(importance), blocks)
