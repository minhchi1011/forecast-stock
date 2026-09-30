"""LightGBM qua API gốc (lgb.train), tham số lấy từ [model] trong config."""

import lightgbm as lgb
import pandas as pd

from ..config import ModelConfig


class LGBMRanker:
    """Hồi quy thứ hạng: dự báo thứ hạng excess return, điểm cao hơn = tốt hơn."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.booster: lgb.Booster | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMRanker":
        params = {**self.cfg.params, "seed": self.cfg.seed, "deterministic": True}
        self.booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=self.cfg.num_boost_round)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.booster.predict(X), index=X.index)

    def feature_importance(self) -> pd.Series:
        gain = self.booster.feature_importance(importance_type="gain")
        return pd.Series(gain, index=self.booster.feature_name()).sort_values(ascending=False)
