# PHÂN CHIA CÔNG VIỆC - FACE ACCESS CONTROL SYSTEM (V2.0)

Dựa trên hệ thống SFace Recognition + YuNet Detection + Gradio UI

---

## 👤 Member 1: Backend Core & Detection (35%)

### Trách nhiệm chính

**Camera Management**:

- Quản lý webcam (open, read, release)
- Tối ưu hóa FPS
- Xử lý lỗi kết nối camera

**Face Detection (YuNet)**:

- Tích hợp YuNet ONNX Model
- Xử lý 5 landmarks (mắt, mũi, miệng)
- Tối ưu hóa preprocessing (resize, input scaling)
- Lọc nhiễu (Score threshold, NMS threshold)

**Database & Storage**:

- Quản lý file Embeddings (Pickle)
- Ghi log ra vào hệ thống (CSV)
- Load/Save model checkpoints

### Files phụ trách

```
modules/
├── camera.py           # Camera processing
├── detector_yunet.py   # YuNet Implementation
└── database.py         # Storage logic

models/
└── face_detection_yunet_2023mar.onnx
```

### Dependencies

- OpenCV (Core)
- NumPy

### Deliverables

- [x] Camera Manager ổn định
- [x] YuNet wrapper class
- [x] Database I/O utilities

---

## 👤 Member 2: AI/ML Recognition (35%)

### Trách nhiệm chính

**SFace Recognition**:

- Tích hợp SFace ONNX Model
- Trích xuất Feature Vector (512 chiều)
- Tính toán Cosine Distance
- Quản lý Threshold nhận diện

**Training Pipeline**:

- Script `train_sface.py`
- Xử lý dataset ảnh đầu vào
- Tạo và lưu file `embeddings.pkl`
- Incremental learning (hỗ trợ xóa/thêm user)

**Optimization**:

- Chuẩn hóa ảnh input (112x112)
- Face Alignment dựa trên landmarks
- Tối ưu hóa tốc độ matching

### Files phụ trách

```
modules/
├── recognizer_sface.py # SFace Logic

train_sface.py          # Training Script
download_models.py      # Model Downloader

models/
├── face_recognition_sface_2021dec.onnx
└── embeddings.pkl      # Trained Database
```

### Dependencies

- OpenCV (DNN module)
- NumPy

### Deliverables

- [x] SFace Recognizer Class
- [x] Training script hoạt động
- [x] Logic so sánh vector chính xác

---

## 👤 Member 3: Frontend & Integration (30%)

### Trách nhiệm chính

**GUI Development (Gradio)**:

- Thiết kế giao diện Web App
- Hiển thị Camera Stream realtime
- Dashboard quản lý User (Thêm/Sửa/Xóa)
- Panels: Logs view, System status

**System Integration**:

- Kết nối Detection -> Recognition -> UI
- Xử lý luồng Capture dataset
- Quản lý state của ứng dụng
- Error Handling & User Feedback

**Documentation**:

- Hướng dẫn cài đặt & sử dụng
- Tài liệu kỹ thuật
- Deployment guide

### Files phụ trách

```
gui/
└── main_window_gradio.py   # Gradio Interface Implementation

main.py                     # Entry Point
config.py                   # Global Configuration

README.md
QUICKSTART.md
description.md
```

### Dependencies

- Gradio (Web UI Framework)
- OpenCV (Image conversion)

### Deliverables

- [x] Giao diện Web đầy đủ tính năng
- [x] Kết nối trơn tru với Core modules
- [x] Tính năng quản lý User (CRUD)
- [x] Documentation hoàn chỉnh

---

## 🔗 Integration Points

### Member 1 → Member 2

**Interface**: Aligned Face for Recognition

```python
# Member 1 (Detector)
results = detector.infer(frame) # return faces + landmarks

# Member 2 (Recognizer)
# Preprocess using landmarks provided by detector
embedding = recognizer.extract(frame, landmarks)
```

### Member 2 → Member 1

**Interface**: Logging Data

```python
# Member 2 returns result
name, score = recognizer.predict(face_roi)

# Member 1 logs to DB
database.log_access(name, score, status)
```

### Member 3 → All

**Orchestration**: Main Application Flow

```python
# Member 3 ties it all together in Gradio Loop
def recognition_loop():
    frame = camera.read()           # Mem 1
    faces = detector.detect(frame)  # Mem 1
    for face in faces:
        name = recognizer.predict() # Mem 2
    yield ui_update                 # Mem 3
```

---

## 📊 Workload Distribution

| Member | Focus Area       | Technologies  | Complexity |
| :----- | :--------------- | :------------ | :--------- |
| **1**  | Core & Detection | OpenCV, YuNet | Medium     |
| **2**  | AI Model         | SFace, ONNX   | High       |
| **3**  | Frontend         | Gradio, Async | Medium     |

---

## 📋 Quick Reference

| Module           | Phụ Trách          | Trạng Thái   |
| :--------------- | :----------------- | :----------- |
| **Detector**     | YuNet (OpenCV Zoo) | ✅ Completed |
| **Recognizer**   | SFace (OpenCV Zoo) | ✅ Completed |
| **UI Framework** | Gradio (Web)       | ✅ Completed |
| **Storage**      | Pickle / CSV       | ✅ Completed |

---

**Version**: 2.0.0
**Project Status**: Stable & Deployment Ready
