# Hướng Dẫn: Thêm SFace Configuration vào config.py

## ⚠️ Quan trọng

File `config.py` cần được edit **MANUAL** vì auto-edit đang gặp vấn đề.

## 📝 Các bước thực hiện

### Bước 1: Mở file config.py

Mở file: `c:\LUUDULIEU\CODE\github\Face-Access-Control\config.py`

### Bước 2: Tìm dòng sau (khoảng dòng 81-86)

```python
# FaceNet Recognition threshold
# Distance càng THẤP càng GIỐNG (0 = identical)
# Nếu distance < threshold → HỢP LỆ
FACENET_DISTANCE_THRESHOLD = 0.6

# ==================== CẤU HÌNH RECOGNITION CHUNG ====================
```

### Bước 3: Thêm SFace configuration GIỮA 2 sections trên

Thay thế đoạn code trên bằng:

```python
# FaceNet Recognition threshold
# Distance càng THẤP càng GIỐNG (0 = identical)
# Nếu distance < threshold → HỢP LỆ
FACENET_DISTANCE_THRESHOLD = 0.6

# ==================== CẤU HÌNH SFACE RECOGNITION ====================

# SFace Model paths
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")

# SFace Parameters
SFACE_EMBEDDING_SIZE = 512       # SFace tạo vector 512 chiều
SFACE_DISTANCE_THRESHOLD = 0.4   # Cosine distance threshold (lower = stricter)

# ==================== CẤU HÌNH RECOGNITION CHUNG ====================
```

### Bước 4: Update DEFAULT_RECOGNITION_METHOD comment

Tìm dòng (khoảng dòng 88-89):

```python
# Phương pháp recognition mặc định: 'lbph' hoặc 'facenet'
DEFAULT_RECOGNITION_METHOD = 'lbph'
```

Thay bằng:

```python
# Phương pháp recognition mặc định: 'lbph', 'openface', hoặc 'sface'
DEFAULT_RECOGNITION_METHOD = 'lbph'
```

### Bước 5: Kiểm tra

Chạy để kiểm tra config có lỗi syntax không:

```bash
python config.py
```

Nếu thành công, sẽ hiển thị:

```
✓ Configuration is valid
```

## ✅ Hoàn tất!

Sau khi thêm xong, config.py sẽ có đầy đủ cấu hình cho 3 methods:

- LBPH
- OpenFace
- SFace

## 🎯 Next Steps

Sau khi config.py đã OK:

1. Download models: `python download_models.py`
2. Train SFace: `python train_sface.py`
3. Test modules: `python modules/detector_yunet.py`
