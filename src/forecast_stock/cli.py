import argparse
from pathlib import Path

from dotenv import load_dotenv

from .config import Config, DEFAULT_CONFIG, load_config


def _add_rule_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--top-k", type=int, help="Ghi đè backtest.top_k")
    p.add_argument("--hold-rank", type=int, help="Ghi đè backtest.hold_rank")
    p.add_argument("--rebalance-weeks", type=int, help="Ghi đè backtest.rebalance_weeks")


def _rules(args: argparse.Namespace, cfg: Config):
    from .backtest import PortfolioRules

    top_k = args.top_k or cfg.backtest.top_k
    return PortfolioRules(
        top_k=top_k,
        hold_rank=args.hold_rank or max(cfg.backtest.hold_rank, top_k),
        rebalance_weeks=args.rebalance_weeks or cfg.backtest.rebalance_weeks,
    )


def main() -> None:
    load_dotenv()  # bí mật (VNSTOCK_API_KEY) nằm trong .env, không nằm trong config
    parser = argparse.ArgumentParser(prog="forecast-stock")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="File cấu hình TOML")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Tải giá ngày của rổ + chỉ số tham chiếu về cache")
    fetch.add_argument("--refresh", action="store_true", help="Tải lại cả các mã đã có cache")

    _add_rule_args(sub.add_parser("backtest", help="Tạo đặc trưng, backtest các baseline, đo IC"))
    sub.add_parser("sweep", help="Quét chu kỳ cân bằng x vùng đệm theo mục [sweep] trong config")
    _add_rule_args(sub.add_parser("model", help="LightGBM walk-forward, so với baseline ngoài mẫu"))

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "fetch":
        from .pipelines.fetch import run_fetch

        run_fetch(cfg, refresh=args.refresh)
    elif args.command == "backtest":
        from .pipelines.evaluate import run_evaluation
        from .reporting import print_report

        print_report(run_evaluation(cfg, _rules(args, cfg)), cfg.target.horizon)
    elif args.command == "sweep":
        from .pipelines.sweep import run_sweep
        from .reporting import print_sweep

        print_sweep(run_sweep(cfg))
    elif args.command == "model":
        from .pipelines.model import run_model
        from .reporting import print_model

        print_model(run_model(cfg, _rules(args, cfg)), cfg.target.horizon)
