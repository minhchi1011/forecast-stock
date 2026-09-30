# 001 · 2026-10-01 · Khởi tạo pipeline thử nghiệm model cổ phiếu HOSE

Commit: `0207b99` — *Pipeline thử nghiệm model xếp hạng cổ phiếu HOSE*

## 1. Định hướng bài toán

Đặc thù thị trường Việt Nam quyết định cách đặt bài toán:

| Đặc điểm | Hệ quả |
|---|---|
| Thanh toán T+2 | Không lướt trong ngày → dự báo khung vài tuần |
| Cá nhân gần như không bán khống | Chiến lược **long-only**: chọn mã để mua |
| Biên độ ±7% (HOSE) | Mở cửa trần thì không mua được, mở cửa sàn thì không bán được |
| Phí ~0.15%/chiều + thuế bán 0.1% | Giao dịch càng nhiều càng dễ lỗ sau phí |

**Mục tiêu đã chọn:** xếp hạng cổ phiếu (cross-sectional ranking). Dự báo **excess return 20 phiên so với VNINDEX**, mua top 10, cân bằng lại định kỳ. Không dự báo giá tuyệt đối.

## 2. Nguồn dữ liệu

- **vnstock 4.x** (nguồn KBS): giá ngày đã điều chỉnh, đơn vị nghìn đồng.
  - Không có trên PyPI, phải cài từ index riêng `https://vnstocks.com/api/simple` (đã cấu hình trong `pyproject.toml`).
  - Gói miễn phí cho tối đa **8 năm** dữ liệu ngày (từ 10/2018).
  - Giới hạn request: gói khách 20/phút, có API key miễn phí 60/phút. Key đặt trong `.env` với dòng `VNSTOCK_API_KEY=...`, không commit.
  - Tính năng gửi thống kê sử dụng (telemetry) đã được tắt trong code.
- **Rổ mã:** toàn bộ **405 mã HOSE** + **27 mã đã hủy niêm yết** + VNINDEX, tổng 433 file, khoảng 22 MB.
- Lưu local tại `data/raw/<MÃ>.parquet`, không đẩy lên git.

## 3. Cấu trúc code

```
configs/default.toml     mọi tham số: phí, biên độ, đặc trưng, target, model, luật backtest
configs/delisted.csv     mã đã hủy niêm yết + ngày hủy (duy trì thủ công)
src/forecast_stock/
  cli.py                 lệnh: fetch | backtest | sweep | model
  config.py              đọc TOML thành các dataclass
  data/                  vnstock_client, store (cache), fetcher, prices, delisted
  features/              technical, liquidity, targets, panel
  strategies/            baseline = dấu × một đặc trưng
  backtest/              schedule, execution (trần/sàn/phí), engine, benchmarks, metrics
  models/                dataset (xếp hạng theo ngày), lgbm, walk_forward
  pipelines/             fetch, research, evaluate, sweep, model
tests/                   15 unit test (uv run pytest)
```

Các lệnh:

```bash
uv run forecast-stock fetch                     # tải/bổ sung dữ liệu
uv run forecast-stock backtest                  # baseline + IC
uv run forecast-stock sweep                     # quét chu kỳ cân bằng × vùng đệm
uv run forecast-stock model                     # LightGBM walk-forward
uv run forecast-stock model --rebalance-weeks 1 --hold-rank 20   # ghi đè luật giao dịch
```

## 4. Diễn biến thử nghiệm

### 4.1 Baseline trên VN100 hiện tại → kết quả đẹp nhưng sai

| Chiến lược | CAGR |
|---|---|
| VNINDEX | 8.3% |
| Chia đều rổ | 15.0% |
| Momentum 120 | 20.9% |

Nguyên nhân là **survivorship bias**. Danh sách VN100 *năm 2026* chỉ gồm những mã đã sống sót và tăng trưởng. Các mã rớt khỏi rổ hoặc bị hủy niêm yết (FLC, HBC, BCG...) không có trong dữ liệu.

### 4.2 Sửa survivorship bias (cách 2)

- Dùng toàn bộ HOSE, lọc theo thanh khoản **tại từng thời điểm** (giá trị giao dịch trung bình 20 phiên ≥ 10 tỷ).
- Bổ sung mã đã hủy niêm yết. Chỉ cho phép mua **trước ngày hủy niêm yết**, vì nhiều mã vẫn tiếp tục có giá ở UPCoM sau khi rời HOSE.
- Ngày hủy niêm yết lấy từ báo chí. Ngày của 4 mã (FTM, RIC, PXS, LCM) được **ước lượng** từ khoảng trống trong dữ liệu; phương pháp này khớp với mọi mã đã biết ngày chính xác.
- Mã đang giữ mà ngừng giao dịch quá 20 phiên sẽ bị **xóa sổ**, chỉ thu hồi 50% giá trị.

Kết quả sau khi sửa: chia đều **4.3%**, Momentum 120 **-6.0%**, không chiến lược nào thắng VNINDEX. Năm 2022 là năm quyết định: momentum mất 67% vì dồn vào nhóm đầu cơ.

### 4.3 Giảm vòng quay vốn

Thêm 2 cơ chế: **vùng đệm** `hold_rank` (chỉ bán khi mã rơi khỏi top N) và **cân bằng mỗi N tuần**.

- Biến động thấp 20: vòng quay giảm từ 1437% xuống 289–459% mỗi năm, CAGR tăng từ khoảng 0% lên 3–4%. Như vậy phí đang ăn khoảng 4%/năm.
- Vẫn chưa thắng VNINDEX.

### 4.4 LightGBM walk-forward (bước 4)

- Đặc trưng được đổi thành thứ hạng giữa các mã trong cùng ngày. Mục tiêu là thứ hạng excess return 20 phiên.
- Cửa sổ train mở rộng dần, huấn luyện lại mỗi 6 tháng, embargo 20 phiên. Giai đoạn ngoài mẫu: 4/2021 – 9/2026.
- Luật giao dịch mặc định: top 10, giữ tới top 20, cân bằng 4 tuần.

| Chiến lược | CAGR | Sharpe | MaxDD | Rank IC |
|---|---|---|---|---|
| VNINDEX | 6.8% | 0.44 | -38.6% | |
| Chia đều rổ | 2.2% | 0.22 | -57.6% | |
| Biến động thấp 20 | -0.7% | 0.04 | -39.1% | 0.056 |
| **LightGBM** | **15.4%** | **0.62** | -54.8% | 0.053 |

Các bước kiểm tra đã làm:

- Không nhìn trước tương lai: có unit test cho walk-forward.
- Chỉ 2.5% lựa chọn của model rơi vào giai đoạn mã còn ở sàn khác, bằng tỷ lệ chung của cả rổ.
- Lợi nhuận tăng đều theo 5 nhóm điểm (-0.77% → 0.57% mỗi 20 phiên).
- Đổi luật giao dịch sang 1 tuần hoặc 2 tuần, CAGR vẫn 12–14%.

Model chủ yếu chọn **cổ phiếu vốn hóa lớn, ít biến động, đang có xu hướng tăng**, nhiều nhất là ngân hàng (VIB, BID, MSB, LPB, ACB, HDB).

## 5. Rủi ro và hạn chế còn lại

1. MaxDD -55%, sâu hơn nhiều so với VNINDEX. Năm 2022 model mất 38%.
2. Tập trung vào ngân hàng; chưa có giới hạn tỷ trọng theo ngành.
3. t-stat của IC bị thổi phồng do các mục tiêu 20 phiên chồng lên nhau; giá trị thực tế khoảng một nửa.
4. Chỉ có 5.5 năm ngoài mẫu, và đã xem kết quả nhiều lần. Chỉnh tiếp tham số trên giai đoạn này sẽ dễ overfit.
5. Chưa tính trượt giá. Danh sách mã hủy niêm yết chưa đầy đủ. Mã ROS chỉ có 34 phiên dữ liệu.

## 6. Việc tiếp theo

- [ ] Lệnh `signal`: in top 10 mã cho kỳ tới để theo dõi trên giấy (paper trading) với dữ liệu thật sự mới.
- [ ] Giới hạn tỷ trọng theo ngành, ví dụ tối đa 30% mỗi ngành.
- [ ] Thêm dữ liệu báo cáo tài chính (P/E, ROE) và dòng tiền khối ngoại.
- [ ] Vùng đệm cho bộ lọc thanh khoản, để giảm vòng quay của rổ.
