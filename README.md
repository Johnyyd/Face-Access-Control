# Face Access Control

Hệ thống kiểm soát ra vào sử dụng nhận diện khuôn mặt.

## 🚀 Quick Start

### 1. Cài đặt

```bash
pip install -r requirements.txt
pip install "numpy<2.0"  # Quan trọng cho OpenFace
```

### 2. Chụp ảnh

```bash
python capture_dataset.py
# Nhập tên, chụp 15-20 ảnh
```

### 3. Train

```bash
python train_lbph.py      # Nhanh
python train_openface.py  # Chính xác
```

### 4. Chạy

```bash
python main.py
```

## 📊 So sánh Methods

| Method       | Accuracy | Speed     | Dùng khi      |
| ------------ | -------- | --------- | ------------- |
| **LBPH**     | 70-85%   | 30-40 FPS | Cần tốc độ    |
| **OpenFace** | 85-95%   | 10-15 FPS | Cần chính xác |

# Thiếu so sánh với SFace

## ⚙️ Config

Chỉnh `config.py`:

```python
LBPH_CONFIDENCE_THRESHOLD = 90.0
OPENFACE_DISTANCE_THRESHOLD = 0.6
DEFAULT_RECOGNITION_METHOD = 'lbph'  # hoặc 'openface'
```

## 🐛 Troubleshooting

**OpenFace lỗi**: `pip install "numpy<2.0"`

**LBPH không chính xác**: Chụp thêm ảnh, điều chỉnh threshold

**Camera không mở**: Đổi `CAMERA_ID` trong config.py

## 📁 Cấu trúc

```
Face-Access-Control/
├── main.py                    # Chạy app
├── config.py                  # Cấu hình
├── modules/                   # Core
│   ├── camera.py
│   ├── detector.py
│   ├── recognizer_lbph.py
│   ├── recognizer_openface.py
│   └── database.py
├── gui/                       # Giao diện
├── dataset/                   # Ảnh training
├── models/                    # Models đã train
└── logs/                      # Access logs
```

## 📝 License

MIT License
