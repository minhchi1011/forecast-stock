"""Bước 1: tải giá ngày của rổ + mã đã hủy niêm yết + chỉ số tham chiếu về cache."""

from ..config import Config
from ..data import PriceStore, fetch_prices, get_universe, load_delisted


def run_fetch(cfg: Config, refresh: bool = False) -> None:
    listed = get_universe(cfg.data.universe)
    delisted = [s for s in load_delisted(cfg.data.delisted_file).index if s not in listed]
    symbols = listed + delisted + [cfg.data.index]
    print(
        f"Rổ {cfg.data.universe}: {len(listed)} mã đang niêm yết + {len(delisted)} mã đã hủy niêm yết "
        f"+ {cfg.data.index}, từ {cfg.data.start}"
    )
    PriceStore(cfg.data_dir).save_universe(cfg.data.universe, symbols)
    failed = fetch_prices(cfg, symbols, refresh=refresh)
    if failed:
        print(f"Lỗi {len(failed)} mã: {failed} -> chạy lại lệnh để tải bù")
