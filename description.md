# PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG FACE ACCESS CONTROL

## 1. Tổng Quan Hệ Thống

Hệ thống hoạt động theo cơ chế **One-to-Many Matching (1:N)**. Camera quay khuôn mặt, hệ thống so sánh với N khuôn mặt đã lưu trong database để xác định danh tính.

### Đặc điểm kỹ thuật:

- **Độ chính xác**: 70-95% (tùy phương pháp)
- **Tốc độ**: Real-time (10-40 FPS)
- **Phương pháp**: 2 kỹ thuật có thể chuyển đổi qua GUI

## 2. So Sánh 2 Kỹ Thuật

### Phương pháp 1: LBPH (Local Binary Patterns Histogram)

| Tiêu chí         | Đánh giá                         |
| ---------------- | -------------------------------- |
| **Detection**    | Haar Cascade hoặc DNN            |
| **Recognition**  | LBPH (OpenCV)                    |
| **Tốc độ**       | ⚡⚡⚡ 30-40 FPS                 |
| **Độ chính xác** | ⭐⭐⭐ 70-85%                    |
| **Tài nguyên**   | Thấp (CPU only)                  |
| **Ưu điểm**      | Nhanh, nhẹ, không cần GPU        |
| **Nhược điểm**   | Nhạy ánh sáng, góc nghiêng       |
| **Phù hợp**      | Môi trường ổn định, thiết bị yếu |

### Phương pháp 2: OpenFace (dlib-based)

| Tiêu chí         | Đánh giá                          |
| ---------------- | --------------------------------- |
| **Detection**    | DNN hoặc Haar Cascade             |
| **Recognition**  | OpenFace Embeddings (128-d)       |
| **Tốc độ**       | ⚡⚡ 10-15 FPS                    |
| **Độ chính xác** | ⭐⭐⭐⭐⭐ 85-95%                 |
| **Tài nguyên**   | Trung bình (CPU, khuyến nghị GPU) |
| **Ưu điểm**      | Chính xác cao, robust             |
| **Nhược điểm**   | Chậm hơn LBPH                     |
| **Phù hợp**      | Cần độ chính xác cao              |

### Khuyến nghị:

- **LBPH**: Ưu tiên tốc độ, thiết bị yếu
- **OpenFace**: Ưu tiên độ chính xác

## 3. Quy Trình Hoạt Động

### Giai đoạn 1: Enrollment (Đăng ký)

1. **Capture**: Chụp 15-20 ảnh từ webcam (`capture_dataset.py`)
2. **Detection**: Phát hiện khuôn mặt
   - Haar Cascade (nhanh) hoặc DNN (chính xác)
3. **Preprocessing**: Crop, resize, normalize
4. **Feature Extraction**:
   - **LBPH**: Grayscale → LBP histogram
   - **OpenFace**: RGB → 128-d embedding vector
5. **Storage**:
   - **LBPH**: `trainer.yml` + `mapping.json`
   - **OpenFace**: `embeddings.pickle`

### Giai đoạn 2: Operation (Vận hành)

1. **Capture**: Đọc frame từ webcam
2. **Detection**: Tìm khuôn mặt trong frame
3. **Recognition**: Trích xuất features và so sánh
   - **LBPH**: So sánh histogram → confidence score
   - **OpenFace**: Tính Euclidean distance
4. **Decision**:
   - **LBPH**: `confidence < threshold` (VD: < 90) → GRANTED
   - **OpenFace**: `distance < threshold` (VD: < 0.6) → GRANTED
5. **Action**: Hiển thị kết quả, log access

## 4. Input / Output

### Input

- **Video Stream**: Webcam 640x480 @ 30fps
- **Dataset**: Thư mục `dataset/[username]/` với 15-20 ảnh
- **Config**: Threshold, method selection

### Output

- **Visual**:
  - Bounding box (xanh: granted, đỏ: denied)
  - Tên user + confidence/distance
  - FPS counter
  - Method indicator (LBPH/OpenFace)
- **Logs**: CSV (`logs/access_log.csv`)

## 5. Luồng Dữ Liệu

### LBPH Pipeline:

```
Frame (BGR) → Grayscale → Resize → LBP → Histogram
→ Compare → Confidence → Mapping → Name
```

### OpenFace Pipeline:

```
Frame (BGR) → RGB → Normalize → Resize (160x160)
→ dlib HOG → 128-d vector → Euclidean Distance → Name
```

## 6. Cấu Trúc Project

```
Face-Access-Control/
├── main.py                      # Entry point
├── config.py                    # Configuration
├── requirements.txt             # Dependencies
│
├── modules/                     # Core modules
│   ├── camera.py               # Camera management
│   ├── detector.py             # Face detection (Haar/DNN)
│   ├── recognizer_lbph.py      # LBPH recognition
│   ├── recognizer_openface.py  # OpenFace recognition
│   └── database.py             # Storage management
│
├── gui/
│   └── main_window.py          # Tkinter GUI
│
├── dataset/                     # Training images
│   └── [username]/
│
├── models/                      # Trained models
│   ├── haarcascade_*.xml       # Haar Cascade
│   ├── deploy.prototxt         # DNN config
│   ├── res10_*.caffemodel      # DNN weights
│   ├── trainer.yml             # LBPH model
│   ├── mapping.json            # LBPH label mapping
│   └── embeddings.pickle       # OpenFace embeddings
│
└── logs/
    └── access.db               # SQLite database
```

## 7. Chi Tiết Modules

### A. `detector.py` - Face Detection

**Class**: `FaceDetector`

**Methods**:

- Haar Cascade: Fast (~5ms/frame), nhiều false positives
- DNN: Slower (~20ms/frame), chính xác hơn

**Switchable**: Có thể chuyển đổi trong runtime

### B. `recognizer_lbph.py` - LBPH Recognition

**Thuật toán**: Local Binary Patterns Histogram

**Key Variables**:

- `recognizer`: `cv2.face.LBPHFaceRecognizer_create()`
- `label_mapping`: Dict {ID → Name}
- `confidence_threshold`: Default 90.0

**Công thức**: Confidence càng thấp càng tốt (0 = perfect match)

### C. `recognizer_openface.py` - OpenFace Recognition

**Thuật toán**: dlib HOG + CNN

**Key Variables**:

- `model`: `face_recognition.OpenFace()`
- `known_encodings`: List of 128-d vectors
- `known_names`: Corresponding names
- `distance_threshold`: Default 0.6

**Công thức**: Euclidean Distance = $\sqrt{\sum_{i=1}^{128}(a_i - b_i)^2}$

### D. `database.py` - Storage Management

**LBPH Storage**:

```python
models/trainer.yml       # OpenCV YAML
models/mapping.json      # {"0": "User1", "1": "User2"}
```

**OpenFace Storage**:

```python
models/embeddings.pickle # {"names": [...], "encodings": [...]}
```

**Access Logs**:

```sql
-- SQLite schema
CREATE TABLE access_logs (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    name TEXT,
    method TEXT,
    confidence REAL,
    status TEXT
);
```

## 8. GUI Features

**Tkinter-based Interface**:

- **Video Feed**: Real-time display
- **Controls**:
  - Recognition Method: LBPH / OpenFace
  - Detection Method: Haar / DNN
  - Threshold Slider: Adjustable
  - Start/Stop Buttons
- **Status Bar**: FPS, method, status
- **Access Logs Viewer**: View recent logs

## 9. Chuyển Đổi Methods

**Runtime Switching**:

1. Stop current recognition
2. Unload old model
3. Load new model (LBPH ↔ OpenFace)
4. Restart recognition loop

**Note**: Cả 2 methods cần train riêng từ cùng dataset

## 10. Dependencies

**Core Libraries**:

- `opencv-python` (4.x): Computer vision
- `numpy` (1.26.4, **< 2.0**): Numerical computing
- `face-recognition` (1.2.3): OpenFace wrapper
- `dlib` (19.24.1): Face recognition backend
- `Pillow`: Image processing
- `tkinter`: GUI (built-in)

**Important**: NumPy < 2.0 required for dlib compatibility

## 11. Performance Metrics

### LBPH:

- Training: < 1 minute
- Recognition: 30-40 FPS
- Accuracy: 70-85%
- CPU Usage: Low

### OpenFace:

- Training: 2-5 minutes
- Recognition: 10-15 FPS
- Accuracy: 85-95%
- CPU Usage: Medium

## 12. System Requirements

### Minimum (LBPH):

- CPU: Intel i3
- RAM: 4GB
- Webcam: 720p @ 30fps

### Recommended (OpenFace):

- CPU: Intel i5
- RAM: 8GB
- Webcam: 1080p @ 30fps
- GPU: Optional (CUDA-capable)

## 13. Deployment Notes

**Production Ready**: ✅

**Tested With**:

- Windows 10/11
- Python 3.11
- NumPy 1.26.4
- OpenCV 4.x

**Known Limitations**:

- No anti-spoofing (can be fooled by photos)
- Single camera only
- No encryption for stored data
- Performance degrades in poor lighting

**Security Considerations**:

- Local storage only
- Embeddings stored, not raw images
- Access logs in SQLite
- Adjustable thresholds for security vs convenience

---

**Version**: 1.1.0  
**Last Updated**: 2025-11-30  
**Status**: Production Ready
