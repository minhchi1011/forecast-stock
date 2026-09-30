"""Bước 2 + 3: tạo đặc trưng, backtest mọi baseline với tham số trong [backtest], đo IC."""

from dataclasses import dataclass

import pandas as pd

from .. import backtest as bt
from ..config import Config
from .research import align_start, prepare_research


@dataclass
class EvaluationReport:
    performance: pd.DataFrame
    ic: pd.DataFrame
    equity: pd.DataFrame
    rules: bt.PortfolioRules
    start: pd.Timestamp
    end: pd.Timestamp


def run_evaluation(cfg: Config, rules: bt.PortfolioRules) -> EvaluationReport:
    data = prepare_research(cfg)

    results = [data.index(), data.equal_weight(rules.rebalance_weeks)]
    ic = {}
    for b in cfg.baselines:
        score = data.score(b)
        results.append(data.backtest(b.name, score, rules))
        ic[b.name] = bt.ic_summary(bt.rank_ic(score, data.target, data.eligible, cfg.backtest.min_ic_names))

    results, start = align_start(results)
    equity = pd.DataFrame({r.name: (1 + r.returns).cumprod() for r in results})
    equity.to_csv(cfg.data_dir / "equity_curves.csv")

    return EvaluationReport(
        performance=pd.DataFrame([bt.performance(r) for r in results]).set_index("chiến lược"),
        ic=pd.DataFrame(ic).T,
        equity=equity,
        rules=rules,
        start=start,
        end=max(r.end for r in results),
    )
