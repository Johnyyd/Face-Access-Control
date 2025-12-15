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
python train_sface.py 
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
| **SFace**    | 
## ⚙️ Config

Chỉnh `config.py`:

```python
LBPH_CONFIDENCE_THRESHOLD = 90.0
OPENFACE_DISTANCE_THRESHOLD = 0.6
DEFAULT_RECOGNITION_METHOD = 'lbph', 'sface'  # hoặc 'openface'
```

## 🐛 Troubleshooting

**OpenFace lỗi**: `pip install "numpy<2.0"`

**LBPH không chính xác**: Chụp thêm ảnh, điều chỉnh threshold

**Camera không mở**: Đổi `CAMERA_ID` trong config.py

## 📁 Cấu trúc

```
Face-Access-Control
    ├── .gradio
    │   └── certificate.pem
    ├── dataset
    │   ├── khactrieu
    │   ├── minhtri
    │   ├── trongtri
    ├── gui
    │   ├── __init__.py
    │   ├── main_window_gradio.py
    │   └── main_window_tkinter.py
    ├── models
    │   ├── dnn
    │   │   ├── deploy.prototxt
    │   │   └── res10_300x300_ssd_iter_140000.caffemodel
    │   ├── haar
    │   │   └── haarcascade_frontalface_default.xml
    │   ├── lbph
    │   │   ├── mapping.json
    │   │   └── trainer.yml
    │   ├── openface
    │   │   └── embeddings.pickle
    │   ├── sface
    │   │   └── face_recognition_sface_2021dec.onnx
    │   └── yunet
    │       └── face_detection_yunet_2023mar.onnx
    ├── modules
    │   ├── __init__.py
    │   ├── camera.py
    │   ├── database.py
    │   ├── detector_yunet.py
    │   ├── detector.py
    │   ├── recognizer_lbph.py
    │   ├── recognizer_openface.py
    │   └── recognizer_sface.py
    ├── .gitattributes
    ├── .gitignore
    ├── capture_dataset.py
    ├── check_dataset.py
    ├── config.py
    ├── description.md
    ├── download_models.py
    ├── image.png
    ├── main.py
    ├── QUICKSTART.md
    ├── README.md
    ├── requirements.txt
    ├── TEAM_DIVISION.md
    ├── test_logs.py
    ├── train_lbph.py
    ├── train_openface.py
    └── train_sface.py

```

## 📝 License

MIT License
