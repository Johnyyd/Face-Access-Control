# Face Access Control System

Hệ thống kiểm soát ra vào bằng nhận diện khuôn mặt với 2 phương pháp: **LBPH** (nhanh) và **FaceNet** (chính xác cao).

## 🎯 Tính Năng

- ✅ **Dual Recognition Methods**:
  - LBPH (Local Binary Patterns Histograms) - Nhanh, nhẹ
  - FaceNet (Deep Learning) - Độ chính xác cao
- ✅ **Dual Detection Methods**:
  - Haar Cascade - Nhanh
  - DNN (Deep Neural Network) - Chính xác hơn
- ✅ **Real-time Recognition**: Nhận diện khuôn mặt thời gian thực
- ✅ **GUI Interface**: Giao diện đồ họa thân thiện
- ✅ **Access Logging**: Ghi nhận lịch sử ra vào
- ✅ **Adjustable Thresholds**: Điều chỉnh ngưỡng nhận diện
- ✅ **Method Switching**: Chuyển đổi phương pháp trong runtime

## 📋 Yêu Cầu Hệ Thống

### Tối Thiểu (LBPH)

- Python 3.8+
- CPU: Intel i3 hoặc tương đương
- RAM: 4GB
- Webcam: 720p, 30fps

### Khuyến Nghị (FaceNet)

- Python 3.10+
- CPU: Intel i5 hoặc tương đương
- RAM: 8GB
- GPU: NVIDIA GTX 1050+ (tùy chọn)
- Webcam: 1080p, 30fps

## 🚀 Cài Đặt

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Face-Access-Control.git
cd Face-Access-Control
```

### 2. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý**: Nếu gặp lỗi với TensorFlow trên Windows:

```bash
pip install tensorflow-cpu>=2.13.0
```

### 3. Download Pre-trained Models

#### Haar Cascade (Bắt buộc cho LBPH)

```bash
# Download từ OpenCV GitHub
# Đặt vào: models/haarcascade_frontalface_default.xml
```

Link: https://github.com/opencv/opencv/tree/master/data/haarcascades

#### DNN Face Detector (Tùy chọn)

```bash
# Download deploy.prototxt và res10_300x300_ssd_iter_140000.caffemodel
# Đặt vào: models/
```

Link: https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830

#### FaceNet Model (Bắt buộc cho FaceNet)

```bash
# Download facenet_keras.h5
# Đặt vào: models/
```

Link: https://github.com/nyoki-mtl/keras-facenet

## 📁 Cấu Trúc Dự Án

```
Face-Access-Control/
├── dataset/              # Ảnh training của users
│   ├── User1/
│   ├── User2/
│   └── ...
├── models/               # Pre-trained models
│   ├── haarcascade_frontalface_default.xml
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── facenet_keras.h5
│   ├── trainer.yml       # LBPH trained model
│   ├── mapping.json      # LBPH label mapping
│   └── embeddings.pickle # FaceNet embeddings
├── modules/              # Core modules
│   ├── camera.py
│   ├── detector.py
│   ├── database.py
│   ├── recognizer_lbph.py
│   └── recognizer_facenet.py
├── gui/                  # GUI components
│   └── main_window.py
├── logs/                 # Access logs
│   └── access_log.csv
├── config.py             # Configuration
├── main.py               # Main application
├── train_lbph.py         # LBPH training script
├── train_facenet.py      # FaceNet training script
└── requirements.txt      # Dependencies
```

## 📖 Hướng Dẫn Sử Dụng

### Bước 1: Chuẩn Bị Dataset

Tạo thư mục cho mỗi người trong `dataset/`:

```
dataset/
├── Alice/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ... (10-20 ảnh)
├── Bob/
│   └── ... (10-20 ảnh)
└── Charlie/
    └── ... (10-20 ảnh)
```

**Lưu ý**:

- Mỗi người cần ít nhất 10 ảnh
- Ảnh nên chụp ở nhiều góc độ khác nhau
- Ánh sáng tốt, khuôn mặt rõ ràng

### Bước 2: Training

#### Training LBPH (Nhanh)

```bash
python train_lbph.py
```

#### Training FaceNet (Chính xác cao)

```bash
python train_facenet.py
```

**Lưu ý**: FaceNet yêu cầu tải FaceNet model trước.

### Bước 3: Chạy Ứng Dụng

```bash
python main.py
```

### Bước 4: Sử Dụng GUI

1. **Chọn Recognition Method**: LBPH hoặc FaceNet
2. **Chọn Detection Method**: Haar Cascade hoặc DNN
3. **Điều chỉnh Threshold**: Kéo thanh trượt
4. **Click "Start"**: Bắt đầu nhận diện
5. **Click "Stop"**: Dừng nhận diện
6. **View Access Logs**: Xem lịch sử ra vào

## ⚙️ Cấu Hình

Chỉnh sửa `config.py` để thay đổi:

```python
# Camera settings
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# LBPH threshold (càng thấp càng strict)
LBPH_CONFIDENCE_THRESHOLD = 50.0

# FaceNet threshold (càng thấp càng strict)
FACENET_DISTANCE_THRESHOLD = 0.6

# Detection method
DEFAULT_DETECTION_METHOD = 'haar'  # hoặc 'dnn'

# Recognition method
DEFAULT_RECOGNITION_METHOD = 'lbph'  # hoặc 'facenet'
```

## 📊 So Sánh Phương Pháp

| Tiêu chí               | LBPH                             | FaceNet                   |
| ---------------------- | -------------------------------- | ------------------------- |
| **Tốc độ**             | ⚡⚡⚡ Rất nhanh (30-60 FPS)     | ⚡ Chậm hơn (10-20 FPS)   |
| **Độ chính xác**       | ⭐⭐ Trung bình (70-85%)         | ⭐⭐⭐⭐⭐ Cao (95-99%)   |
| **Yêu cầu tài nguyên** | Thấp (CPU only)                  | Cao hơn (khuyến nghị GPU) |
| **Training time**      | Nhanh (< 1 phút)                 | Chậm hơn (vài phút)       |
| **Phù hợp**            | Thiết bị yếu, môi trường ổn định | Yêu cầu độ chính xác cao  |

## 🔧 Troubleshooting

### Camera không mở được

- Kiểm tra camera đã kết nối chưa
- Thử thay đổi `CAMERA_ID` trong `config.py`
- Đảm bảo không có ứng dụng nào khác đang dùng camera

### Model không load được

- Kiểm tra đã chạy training script chưa
- Kiểm tra file model tồn tại trong `models/`
- Xem log lỗi để biết chi tiết

### FaceNet không hoạt động

- Kiểm tra đã cài TensorFlow chưa: `pip install tensorflow`
- Kiểm tra đã download FaceNet model chưa
- Thử dùng CPU version: `pip install tensorflow-cpu`

### Độ chính xác thấp

- Tăng số lượng ảnh training (20-30 ảnh/người)
- Chụp ảnh ở nhiều góc độ, ánh sáng khác nhau
- Điều chỉnh threshold
- Thử chuyển sang FaceNet

## 📝 Access Logs

Logs được lưu trong `logs/access_log.csv`:

```csv
timestamp,name,method,confidence,status
2024-01-01 10:30:15,Alice,LBPH,35.2,GRANTED
2024-01-01 10:31:20,Unknown,FaceNet,0.85,DENIED
2024-01-01 10:32:10,Bob,LBPH,42.1,GRANTED
```

## 🤝 Đóng Góp

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👥 Team

Dự án được phát triển bởi Face Access Control Team.

Xem [TEAM_DIVISION.md](TEAM_DIVISION.md) để biết chi tiết phân công công việc.

## 📚 Tài Liệu Tham Khảo

- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [FaceNet Paper](https://arxiv.org/abs/1503.03832)
- [LBPH Algorithm](https://docs.opencv.org/4.x/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html)

## 🎓 Học Thêm

Xem file [description.md](description.md) để hiểu chi tiết về:

- Kiến trúc hệ thống
- Thuật toán LBPH và FaceNet
- Luồng dữ liệu
- Quy trình hoạt động

---

**Made with ❤️ by Face Access Control Team**
