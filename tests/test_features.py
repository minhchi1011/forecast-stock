import numpy as np
import pandas as pd
import pytest

from forecast_stock.features.targets import forward_open_return


def test_forward_return_starts_at_next_open():
    # tín hiệu ở phiên t -> khớp mở cửa t+1, nắm giữ tới mở cửa t+1+h
    open_ = pd.Series([10.0, 11.0, 12.0, 15.0, 20.0])
    fwd = forward_open_return(open_, horizon=2)
    assert fwd.iloc[0] == pytest.approx(15.0 / 11.0 - 1)
    assert fwd.iloc[1] == pytest.approx(20.0 / 12.0 - 1)
    assert np.isnan(fwd.iloc[2])  # chưa đủ dữ liệu tương lai -> không được điền giá trị
