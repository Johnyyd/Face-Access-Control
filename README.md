# Face Access Control System

Hệ thống kiểm soát ra vào hiện đại sử dụng công nghệ nhận diện khuôn mặt **SFace** (State-of-the-art) và detection **YuNet** của OpenCV Zoo, với giao diện web trực quan trên nền tảng **Gradio**.

## 🌟 Tính Năng Chính

- **Nhận diện chính xác**: Sử dụng mô hình SFace (ONNX) với vector đặc trưng 512 chiều.
- **Tốc độ cao**: Detection thời gian thực với YuNet.
- **Giao diện hiện đại**: Web UI (Gradio) hỗ trợ xem camera, quản lý user, và xem log trực tiếp.
- **Quản lý User**: Thêm/Xóa/Cập nhật user trực quan ngay trên giao diện.
- **Instant Update**: Xóa user có hiệu lực ngay lập tức mà không cần khởi động lại.
- **Access Logs**: Lưu lịch sử ra vào chi tiết (CSV) và hiển thị trên giao diện.

## �️ Công Nghệ

- **Language**: Python 3.11+
- **Core Vision**: OpenCV (SFace, YuNet ONNX models)
- **Interface**: Gradio (Web UI)
- **Storage**: Pickle (Embeddings), CSV (Logs)

## �🚀 Quick Start

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Tải Models

Tải các model ONNX cần thiết (YuNet và SFace):

```bash
python download_models.py
```

### 3. Tạo Dataset

Bạn có thể tạo user mới trực tiếp trên giao diện web, hoặc dùng script:

```bash
python capture_dataset.py
# Nhập tên user và làm theo hướng dẫn
```

### 4. Train Model

Tạo embeddings từ dataset ảnh:

```bash
python train_sface.py
```

### 5. Chạy Hệ Thống

```bash
python main.py
```

Truy cập giao diện tại: `http://127.0.0.1:7860`

## 📁 Cấu trúc Project

```
Face-Access-Control/
├── main.py                    # File chính để chạy hệ thống
├── config.py                  # Cấu hình hệ thống (Threshold, Paths...)
├── capture_dataset.py         # Script chụp ảnh dataset
├── train_sface.py             # Script training (tạo embeddings)
├── download_models.py         # Script tải model ONNX
├── requirements.txt           # Các thư viện cần thiết
├── modules/                   # Core logic
│   ├── detector_yunet.py      # Face Detection (YuNet)
│   ├── recognizer_sface.py    # Face Recognition (SFace)
│   ├── camera.py              # Camera handling
│   └── database.py            # Quản lý file và logs
├── gui/                       # Giao diện
│   └── main_window_gradio.py  # Gradio UI implementation
├── models/                    # Chứa model ONNX và embeddings.pkl
├── dataset/                   # Chứa ảnh training của users
└── logs/                      # Chứa file log access_log.csv
```

## ⚙️ Cấu Hình (config.py)

Bạn có thể tùy chỉnh các thông số trong `config.py`:

```python
CAMERA_ID = 0                  # 0: Webcam, 1: External Cam
SFACE_DISTANCE_THRESHOLD = 0.4 # Ngưỡng nhận diện (thấp = chặt chẽ hơn)
ACCESS_COOLDOWN = 3.0          # Thời gian chờ giữa 2 lần log
```

## 📝 License

MIT License
