"""In kết quả ra terminal."""

import pandas as pd

from .pipelines.evaluate import EvaluationReport
from .pipelines.model import ModelReport
from .pipelines.sweep import SweepReport

PCT_COLUMNS = ["CAGR", "Vol năm", "MaxDD", "Turnover/năm", "Tổng LN", "% tuần IC>0"]


def _fmt(df: pd.DataFrame) -> str:
    formatters = {c: "{:.1%}".format for c in PCT_COLUMNS if c in df.columns}
    return df.to_string(formatters=formatters, float_format="{:.3f}".format)


def print_report(report: EvaluationReport, horizon: int) -> None:
    print(f"\nBacktest {report.start:%Y-%m-%d} -> {report.end:%Y-%m-%d} ({report.rules.label()}), đã trừ phí")
    print(_fmt(report.performance))
    print(f"\nRank IC với mục tiêu excess return {horizon} phiên (so với chỉ số tham chiếu)")
    print(_fmt(report.ic))


def print_sweep(report: SweepReport) -> None:
    print(f"\nSweep {report.start:%Y-%m-%d} -> {report.end:%Y-%m-%d}, đã trừ phí")
    print(_fmt(report.table))


def print_model(report: ModelReport, horizon: int) -> None:
    print(f"\nNgoài mẫu {report.start:%Y-%m-%d} -> {report.end:%Y-%m-%d} ({report.rules.label()}), đã trừ phí")
    print(_fmt(report.performance))
    print("\nLợi nhuận theo năm")
    print(report.yearly.rename(columns=lambda c: c.split(" (")[0]).to_string(float_format="{:.1%}".format))
    print(f"\nRank IC ngoài mẫu với mục tiêu excess return {horizon} phiên")
    print(_fmt(report.ic))
    print("\nĐộ quan trọng đặc trưng (tỷ trọng gain trung bình)")
    print(report.importance.to_string(float_format="{:.1%}".format))
