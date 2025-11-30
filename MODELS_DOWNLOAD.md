# Hướng Dẫn Download Pre-trained Models

Hệ thống cần các pre-trained models sau để hoạt động. Hãy làm theo hướng dẫn dưới đây.

## 📥 Models Cần Thiết

### 1. Haar Cascade (BẮT BUỘC cho LBPH)

**File**: `haarcascade_frontalface_default.xml`

**Cách download**:

#### Option 1: Download trực tiếp

```bash
# Windows PowerShell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml" -OutFile "models/haarcascade_frontalface_default.xml"

# Linux/Mac
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -P models/
```

#### Option 2: Download thủ công

1. Truy cập: https://github.com/opencv/opencv/tree/master/data/haarcascades
2. Click vào `haarcascade_frontalface_default.xml`
3. Click nút "Raw"
4. Save file vào thư mục `models/`

---

### 2. DNN Face Detector (TÙY CHỌN - cho detection chính xác hơn)

**Files**:

- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

**Cách download**:

#### Option 1: Download trực tiếp

```bash
# deploy.prototxt
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt" -OutFile "models/deploy.prototxt"

# caffemodel (file lớn ~10MB)
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel" -OutFile "models/res10_300x300_ssd_iter_140000.caffemodel"
```

#### Option 2: Download thủ công

1. **deploy.prototxt**:

   - Link: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
   - Save vào `models/deploy.prototxt`

2. **caffemodel**:
   - Link: https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
   - Click "Download" hoặc "Raw"
   - Save vào `models/res10_300x300_ssd_iter_140000.caffemodel`

---

### 3. FaceNet Model (BẮT BUỘC cho FaceNet)

**File**: `facenet_keras.h5` (~90MB)

**Cách download**:

#### Option 1: Google Drive

1. Truy cập: https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_
2. Download file `facenet_keras.h5`
3. Di chuyển vào thư mục `models/`

#### Option 2: GitHub Release

1. Truy cập: https://github.com/nyoki-mtl/keras-facenet/releases
2. Download `facenet_keras.h5` từ Assets
3. Di chuyển vào thư mục `models/`

#### Option 3: Sử dụng script Python

```python
# download_facenet.py
import gdown

# Google Drive file ID
file_id = "1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn"
output = "models/facenet_keras.h5"

# Download
gdown.download(id=file_id, output=output, quiet=False)
```

Chạy:

```bash
pip install gdown
python download_facenet.py
```

---

## ✅ Kiểm Tra Models

Sau khi download, kiểm tra cấu trúc thư mục:

```
models/
├── haarcascade_frontalface_default.xml  ✓ (Bắt buộc)
├── deploy.prototxt                      ✓ (Tùy chọn)
├── res10_300x300_ssd_iter_140000.caffemodel  ✓ (Tùy chọn)
└── facenet_keras.h5                     ✓ (Bắt buộc cho FaceNet)
```

**Kiểm tra bằng Python**:

```python
import os

models = {
    'Haar Cascade': 'models/haarcascade_frontalface_default.xml',
    'DNN Prototxt': 'models/deploy.prototxt',
    'DNN Model': 'models/res10_300x300_ssd_iter_140000.caffemodel',
    'FaceNet': 'models/facenet_keras.h5'
}

for name, path in models.items():
    status = "✓" if os.path.exists(path) else "✗"
    print(f"{status} {name}: {path}")
```

---

## 📊 Kích Thước Files

| File                                     | Size    | Required    |
| ---------------------------------------- | ------- | ----------- |
| haarcascade_frontalface_default.xml      | ~900 KB | ✓ Yes       |
| deploy.prototxt                          | ~30 KB  | Optional    |
| res10_300x300_ssd_iter_140000.caffemodel | ~10 MB  | Optional    |
| facenet_keras.h5                         | ~90 MB  | For FaceNet |

---

## ⚠️ Lưu Ý

1. **Haar Cascade** là bắt buộc để chạy LBPH
2. **DNN models** chỉ cần nếu muốn dùng DNN detection (chính xác hơn)
3. **FaceNet model** chỉ cần nếu muốn dùng FaceNet recognition
4. Đảm bảo đặt files đúng vị trí trong thư mục `models/`
5. Không commit các model files lên Git (đã có trong .gitignore)

---

## 🔧 Troubleshooting

### Lỗi: "File not found"

- Kiểm tra đường dẫn file
- Kiểm tra tên file chính xác (case-sensitive)
- Chạy script kiểm tra ở trên

### Lỗi download: "SSL Certificate"

```bash
# Thêm --no-check-certificate
wget --no-check-certificate <URL>
```

### File bị corrupt

- Download lại file
- Kiểm tra kích thước file
- Thử download từ nguồn khác

---

## 📚 Tài Liệu Tham Khảo

- OpenCV Haar Cascades: https://github.com/opencv/opencv/tree/master/data/haarcascades
- OpenCV DNN: https://github.com/opencv/opencv_3rdparty
- FaceNet Keras: https://github.com/nyoki-mtl/keras-facenet
