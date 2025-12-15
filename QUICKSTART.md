# Quick Start Guide

Hướng dẫn chi tiết từng bước để thiết lập và chạy hệ thống Face Access Control.

## Bước 1: Cài đặt Môi trường

Đảm bảo bạn đã cài đặt Python 3.10 hoặc 3.11.

```bash
# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## Bước 2: Tải Models

Hệ thống cần 2 file model ONNX (Detect & Recognize). Chạy script sau để tự động tải về:

```bash
python download_models.py
```

_Kiểm tra: Thư mục `models/yunet` và `models/sface` sẽ có file .onnx._

## Bước 3: Tạo Dữ Liệu (Dataset)

Có 2 cách để thêm user mới:

### Cách 1: Dùng Script (Khuyên dùng lần đầu)

```bash
python capture_dataset.py
```

1. Nhập tên user (viết liền không dấu, VD: `minhtri`).
2. Nhấn `SPACE` để chụp ảnh (cần khoảng 20-50 ảnh ở nhiều góc độ).
3. Nhấn `Q` để thoát.

### Cách 2: Dùng Giao diện Web

Sau khi chạy hệ thống, bạn có thể nhập tên và nhấn "New User" để chụp ảnh trực tiếp trên web.

## Bước 4: Training

Sau khi có ảnh trong thư mục `dataset/`, bạn cần tạo file embeddings để hệ thống nhận diện:

```bash
python train_sface.py
```

_Lưu ý: Bạn cần chạy lệnh này mỗi khi thêm user mới bằng script `capture_dataset.py`. Nếu thêm qua Web UI, hệ thống sẽ tự train._

## Bước 5: Chạy Hệ Thống

Khởi động ứng dụng:

```bash
python main.py
```

- Mở trình duyệt truy cập: **http://127.0.0.1:7860**
- Chọn phương thức nhận diện **SFace** (mặc định).
- Điều chỉnh thanh trượt **Cosine Distance** (Mặc định ~0.4).
  - Giảm xuống (0.3) nếu muốn chặt chẽ hơn (tránh nhận nhầm).
  - Tăng lên (0.5) nếu hệ thống không nhận ra bạn.
- Nhấn **START** để bắt đầu camera.

## Troubleshooting

- **Lỗi thiếu Model**: Chạy lại Bước 2.
- **Lỗi thiếu Embeddings**: Chạy lại Bước 4.
- **Accuracy thấp**:
  - Chụp thêm ảnh ở điều kiện ánh sáng giống nơi đặt máy.
  - Đảm bảo ảnh trong dataset rõ nét.
  - Điều chỉnh Threshold trên giao diện.
