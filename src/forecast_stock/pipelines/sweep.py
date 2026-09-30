"""Quét tham số giảm vòng quay (chu kỳ cân bằng x vùng đệm) cho các baseline trong [sweep]."""

from dataclasses import dataclass
from itertools import product

import pandas as pd

from .. import backtest as bt
from ..config import Config
from .research import align_start, prepare_research


@dataclass
class SweepReport:
    table: pd.DataFrame  # index: (chiến lược, tuần, giữ tới)
    start: pd.Timestamp
    end: pd.Timestamp


def run_sweep(cfg: Config) -> SweepReport:
    data = prepare_research(cfg, save_features=False)
    sw, top_k = cfg.sweep, cfg.backtest.top_k

    index = data.index()
    runs: list[tuple[str, int, int | None, bt.BacktestResult]] = [(index.name, 1, None, index)]
    for weeks in sw.rebalance_weeks:
        runs.append(("Chia đều rổ thanh khoản", weeks, None, data.equal_weight(weeks)))

    for name in sw.baselines:
        score = data.score(data.baseline(name))
        for weeks, hold in product(sw.rebalance_weeks, sw.hold_rank):
            if hold < top_k:
                continue
            rules = bt.PortfolioRules(top_k, hold, weeks)
            print(f"  {name}: {rules.label()}", flush=True)
            runs.append((name, weeks, hold, data.backtest(name, score, rules)))

    aligned, start = align_start([r for *_, r in runs])
    rows = []
    for (name, weeks, hold, _), res in zip(runs, aligned):
        perf = bt.performance(res)
        perf.pop("chiến lược")
        rows.append({"chiến lược": name, "tuần": weeks, "giữ tới": hold or "-", **perf})

    table = pd.DataFrame(rows).set_index(["chiến lược", "tuần", "giữ tới"])
    return SweepReport(table, start, max(r.end for r in aligned))
