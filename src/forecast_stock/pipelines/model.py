"""Bước 4: LightGBM walk-forward, so với baseline trên cùng giai đoạn ngoài mẫu và cùng luật giao dịch."""

from dataclasses import dataclass

import pandas as pd

from .. import backtest as bt
from ..config import Config
from ..models import make_dataset, walk_forward
from .research import align_start, prepare_research

MODEL_NAME = "LightGBM"


@dataclass
class ModelReport:
    performance: pd.DataFrame
    yearly: pd.DataFrame
    ic: pd.DataFrame
    importance: pd.Series  # tỷ trọng gain trung bình qua các lần train
    rules: bt.PortfolioRules
    start: pd.Timestamp
    end: pd.Timestamp


def _yearly_returns(results: list[bt.BacktestResult]) -> pd.DataFrame:
    rets = pd.DataFrame({r.name: r.returns for r in results})
    return rets.groupby(rets.index.year).apply(lambda g: (1 + g.fillna(0)).prod() - 1)


def run_model(cfg: Config, rules: bt.PortfolioRules) -> ModelReport:
    data = prepare_research(cfg)
    horizon = cfg.target.horizon
    ds = make_dataset(data.panel, data.eligible, horizon)
    print(f"Dataset: {len(ds.X):,} dòng x {ds.X.shape[1]} đặc trưng, walk-forward:")

    wf = walk_forward(ds, data.px.calendar, cfg.model, horizon)
    wf.predictions.rename("score").to_frame().to_parquet(cfg.data_dir / "predictions.parquet")
    oos_start = wf.blocks[0][1]

    dates = data.px.close.index
    scores = {MODEL_NAME: wf.predictions.unstack().reindex(dates)}
    scores.update({b.name: data.score(b) for b in cfg.baselines})

    ic = {}
    for name, score in scores.items():
        s = bt.rank_ic(score, data.target, data.eligible, cfg.backtest.min_ic_names)
        ic[name] = bt.ic_summary(s[s.index >= oos_start])

    results = [data.index(), data.equal_weight(rules.rebalance_weeks)]
    results += [data.backtest(name, score, rules) for name, score in scores.items()]
    # mô hình chỉ có danh mục từ kỳ ngoài mẫu -> align_start cắt mọi chiến lược về đó
    results, start = align_start(results)

    importance = wf.importance.div(wf.importance.sum()).mean(axis=1).sort_values(ascending=False)
    return ModelReport(
        performance=pd.DataFrame([bt.performance(r) for r in results]).set_index("chiến lược"),
        yearly=_yearly_returns(results),
        ic=pd.DataFrame(ic).T,
        importance=importance,
        rules=rules,
        start=start,
        end=max(r.end for r in results),
    )
