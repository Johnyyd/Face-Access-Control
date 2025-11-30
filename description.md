# PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG FACE ACCESS CONTROL

## 1. Tổng Quan Hệ Thống

Hệ thống hoạt động theo cơ chế **One-to-Many Matching (1:N)**. Camera sẽ quay khuôn mặt, hệ thống so sánh khuôn mặt đó với $N$ khuôn mặt đã lưu trong cơ sở dữ liệu để xác định danh tính và quyết định mở cửa.

### Đặc điểm kỹ thuật:

- **Độ chính xác**: Phụ thuộc vào kỹ thuật được chọn
- **Tốc độ**: Yêu cầu xử lý thời gian thực (Real-time)
- **Tùy chọn kỹ thuật**: Hệ thống hỗ trợ 2 phương pháp nhận diện có thể chuyển đổi qua giao diện

## 2. So Sánh 2 Kỹ Thuật Nhận Diện

Hệ thống cho phép người dùng lựa chọn giữa 2 kỹ thuật:

### Phương pháp 1: LBPH (Local Binary Patterns Histograms)

| Tiêu chí               | Đánh giá                                     |
| ---------------------- | -------------------------------------------- |
| **Detection**          | Haar Cascade (Viola-Jones)                   |
| **Recognition**        | LBPH Face Recognizer                         |
| **Tốc độ**             | ⚡⚡⚡ Rất nhanh (~30-60 FPS)                |
| **Độ chính xác**       | ⭐⭐ Trung bình (70-85%)                     |
| **Yêu cầu tài nguyên** | Thấp (CPU only)                              |
| **Ưu điểm**            | Nhẹ, nhanh, dễ triển khai, không cần GPU     |
| **Nhược điểm**         | Nhạy cảm với ánh sáng, góc nghiêng, biểu cảm |
| **Phù hợp**            | Môi trường ổn định, thiết bị yếu             |

### Phương pháp 2: OpenFace (Deep Learning)

| Tiêu chí               | Đánh giá                                  |
| ---------------------- | ----------------------------------------- |
| **Detection**          | DNN (Caffe/TensorFlow model)              |
| **Recognition**        | OpenFace Embeddings (128-d vector)         |
| **Tốc độ**             | ⚡ Chậm hơn (~10-20 FPS)                  |
| **Độ chính xác**       | ⭐⭐⭐⭐⭐ Cao (95-99%)                   |
| **Yêu cầu tài nguyên** | Cao hơn (khuyến nghị GPU)                 |
| **Ưu điểm**            | Chính xác cao, robust với nhiều điều kiện |
| **Nhược điểm**         | Chậm hơn, cần tài nguyên mạnh hơn         |
| **Phù hợp**            | Yêu cầu độ chính xác cao, thiết bị mạnh   |

### Khuyến nghị lựa chọn:

- **Chọn LBPH** nếu: Ưu tiên tốc độ, thiết bị yếu, môi trường ánh sáng ổn định
- **Chọn OpenFace** nếu: Ưu tiên độ chính xác, có GPU, môi trường phức tạp

## 3. Sơ Đồ Hoạt Động (Flowchart)

Quy trình hoạt động được chia thành 2 giai đoạn chính: **Giai đoạn Đăng ký (Enrollment)** và **Giai đoạn Vận hành (Operation/Inference)**.

### Giai đoạn 1: Đăng ký (Admin thực hiện)

1. **Start**: Admin nhập tên thành viên và chọn kỹ thuật (LBPH hoặc OpenFace)
2. **Capture**: Webcam chụp $N$ ảnh khuôn mặt của thành viên (ở các góc độ khác nhau)
3. **Pre-process**:
   - Phát hiện khuôn mặt (Detection)
     - **LBPH**: Sử dụng Haar Cascade
     - **OpenFace**: Sử dụng DNN Detector
   - Cắt vùng mặt (Crop)
   - Chỉnh kích thước (Resize)
4. **Feature Extraction**:
   - **LBPH**: Chuyển sang Grayscale → Tính toán LBP Histogram
   - **OpenFace**: Normalize → Tạo vector embedding 128 chiều
5. **Save**: Lưu dữ liệu đặc trưng vào Database/File
   - **LBPH**: Lưu vào `trainer.yml` + `mapping.json`
   - **OpenFace**: Lưu vào `embeddings.pickle`
6. **End**

### Giai đoạn 2: Vận hành

1. **Start**: Khởi động Webcam và load model theo kỹ thuật đã chọn
2. **Loop**: Đọc frame liên tục
3. **Detect**: Tìm khuôn mặt trong khung hình
   - **LBPH**: Haar Cascade Classifier
   - **OpenFace**: DNN Face Detector (Caffe model)
   - **Có mặt**: Sang bước tiếp theo
   - **Không**: Quay lại bước 2
4. **Recognize**: Trích xuất đặc trưng khuôn mặt hiện tại và so sánh với Database
   - **LBPH**: So sánh histogram, tính confidence score
   - **OpenFace**: Tính Euclidean distance giữa embeddings
5. **Decision**:
   - **LBPH**: Nếu `Confidence < Threshold` (VD: < 50): HỢP LỆ
   - **OpenFace**: Nếu `Distance < Threshold` (VD: < 0.6): HỢP LỆ
   - **HỢP LỆ** → Gửi lệnh Mở khóa (Unlock) → Hiển thị tên
   - **KHÔNG HỢP LỆ** → Cảnh báo (Access Denied)
6. **Loop**: Tiếp tục vòng lặp

## 4. Input / Output

### Input

- **Nguồn hình ảnh**: Luồng video (Video Stream) từ Webcam (độ phân giải khuyến nghị 640x480 hoặc 720p)
- **Dữ liệu mẫu**: Thư mục ảnh khuôn mặt của các thành viên (`dataset/`)
- **Cấu hình**:
  - Lựa chọn kỹ thuật (LBPH hoặc OpenFace)
  - Ngưỡng chính xác (Threshold)
  - Đường dẫn model

### Output

- **Giao diện (Visual)**: Màn hình hiển thị video realtime với:
  - Khung hình chữ nhật (Bounding Box) bao quanh mặt
  - Tên thành viên và trạng thái (Xanh: OK, Đỏ: Denied)
  - Hiển thị kỹ thuật đang sử dụng (LBPH/OpenFace)
  - Hiển thị FPS và độ tin cậy
- **Tín hiệu điều khiển**:
  - Log ghi nhận thời gian ra vào (File `.csv` và Console log)

## 5. Luồng Dữ Liệu (Data Flow)

Dữ liệu sẽ di chuyển qua các tầng xử lý như sau:

### Luồng chung (cả 2 kỹ thuật):

1. **Raw Data (Frame)**: Webcam trả về ma trận điểm ảnh (NumPy array, BGR format)
2. **Preprocessing**: Resize ảnh để tăng tốc độ xử lý
3. **Region of Interest (ROI)**: Tọa độ `(x, y, w, h)` của khuôn mặt được cắt ra từ Frame

### Luồng riêng theo kỹ thuật:

#### LBPH Pipeline:

4. **Color Conversion**: BGR → Grayscale
5. **Feature Extraction**: Tính toán LBP patterns và histogram
6. **Storage**: Lưu vào `trainer.yml` (YAML format)
7. **Matching**: So sánh histogram, trả về `(ID, Confidence)`
8. **Mapping**: Tra cứu `mapping.json` để lấy tên từ ID

#### FaceNet Pipeline:

4. **Color Conversion**: BGR → RGB
5. **Normalization**: Chuẩn hóa pixel values (0-1 hoặc -1 to 1)
6. **Feature Extraction**: Forward pass qua FaceNet model → 128-d embedding vector
7. **Storage**: Lưu vào `embeddings.pickle` (Dictionary: `{"names": [...], "embeddings": [...]}`)
8. **Matching**: Tính Euclidean distance với tất cả embeddings, trả về `(Name, Distance)`

## 6. Cấu Trúc Dự Án (Project Structure)

Chia dự án theo hướng **Modular** để dễ dàng chuyển đổi giữa 2 kỹ thuật:

```
FaceAccessControl/
├── dataset/                    # Chứa ảnh thô của thành viên (chia theo thư mục tên)
│   ├── User_A/
│   │   ├── 001.jpg
│   │   └── 002.jpg
│   └── User_B/
│       ├── 001.jpg
│       └── 002.jpg
├── models/                     # Chứa các file model đã train hoặc pre-trained
│   ├── haarcascade_frontalface_default.xml    # Haar Cascade cho LBPH
│   ├── trainer.yml                             # LBPH trained model
│   ├── mapping.json                            # ID to Name mapping cho LBPH
│   ├── deploy.prototxt                         # DNN config cho OpenFace
│   └── embeddings.pickle                       # OpenFace embeddings database
├── modules/
│   ├── __init__.py
│   ├── camera.py               # Quản lý Webcam (Open, Read, Close)
│   ├── detector.py             # Class bọc kỹ thuật Detection (Haar/DNN)
│   ├── recognizer_lbph.py      # Class cho LBPH Recognition
│   ├── recognizer_openface.py   # Class cho OpenFace Recognition
│   └── database.py             # Quản lý lưu/đọc dữ liệu người dùng
├── gui/
│   ├── __init__.py
│   └── main_window.py          # Giao diện chính với tùy chọn kỹ thuật
├── main.py                     # Chương trình chính (Vận hành)
├── train_lbph.py               # Training script cho LBPH
├── train_openface.py            # Training script cho OpenFace
├── config.py                   # File cấu hình (thresholds, paths)
└── requirements.txt            # Các thư viện cần thiết
```

## 7. Phân Tích Chi Tiết Module

### A. Module `detector.py`

**Nhiệm vụ**: Tìm vị trí khuôn mặt (hỗ trợ cả 2 kỹ thuật)

**Class**: `FaceDetector`

```python
class FaceDetector:
    def __init__(self, method='haar'):
        """
        method: 'haar' hoặc 'dnn'
        """
        self.method = method
        if method == 'haar':
            self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        elif method == 'dnn':
            self.net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd.caffemodel')

    def detect_faces(self, frame):
        """
        Input: Một khung hình ảnh (BGR)
        Output: List các tọa độ [(x, y, w, h), ...]
        """
```

**Đặc điểm**:

- **Haar Cascade**: Nhanh (~5-10ms/frame), nhưng nhiều false positives
- **DNN**: Chậm hơn (~20-30ms/frame), nhưng chính xác hơn

### B. Module `recognizer_lbph.py`

**Nhiệm vụ**: Nhận diện khuôn mặt bằng LBPH

**Class**: `LBPHRecognizer`

**Biến quan trọng**:

- `recognizer`: Instance của `cv2.face.LBPHFaceRecognizer_create()`
- `label_map`: Dictionary mapping ID → Name
- `confidence_threshold`: Ngưỡng chấp nhận (VD: 50)

**Hàm chức năng**:

```python
def train(self, dataset_path):
    """
    Train LBPH model từ dataset
    Lưu vào trainer.yml và mapping.json
    """

def predict(self, face_image):
    """
    Input: Ảnh khuôn mặt (Grayscale)
    Output: (Name, Confidence)
    Confidence càng thấp càng chính xác
    """
```

### C. Module `recognizer_openface.py`

**Nhiệm vụ**: Nhận diện khuôn mặt bằng OpenFace

**Class**: `OpenFaceRecognizer`

**Biến quan trọng**:

- `model`: OpenFace model (Keras/TensorFlow)
- `known_embeddings`: List các vector 128-d đã lưu
- `known_names`: List tên tương ứng
- `distance_threshold`: Ngưỡng khoảng cách Euclidean (VD: 0.6)

**Hàm chức năng**:

```python
def extract_embedding(self, face_image):
    """
    Input: Ảnh khuôn mặt (RGB, normalized)
    Output: Vector 128 chiều
    """

def train(self, dataset_path):
    """
    Tạo embeddings cho tất cả ảnh trong dataset
    Lưu vào embeddings.pickle
    """

def predict(self, face_image):
    """
    Input: Ảnh khuôn mặt (RGB)
    Output: (Name, Distance)
    Distance càng thấp càng giống
    """
```

### D. Module `database.py`

**Nhiệm vụ**: Quản lý dữ liệu bền vững cho cả 2 kỹ thuật

**Hàm chức năng**:

```python
def save_lbph_model(recognizer, label_map, path):
    """
    Lưu LBPH model và mapping
    """

def load_lbph_model(path):
    """
    Load LBPH model và mapping
    Return: (recognizer, label_map)
    """

def save_openface_embeddings(names, embeddings, path):
    """
    Lưu OpenFace embeddings vào pickle
    """

def load_openface_embeddings(path):
    """
    Load FaceNet embeddings
    Return: (names, embeddings)
    """
```

## 8. Phân Tích Frontend, Backend & Cơ Sở Dữ Liệu

### Backend (Logic xử lý)

Là các file trong thư mục `modules/`. Nơi thực hiện các thuật toán:

**Thư viện sử dụng**:

- **OpenCV** (`cv2`): Xử lý ảnh, Haar Cascade, DNN, LBPH
- **TensorFlow/Keras**: OpenFace model
- **NumPy**: Tính toán khoảng cách vector (Euclidean Distance)
- **Pickle**: Serialization cho OpenFace embeddings
- **JSON**: Lưu mapping cho LBPH

**Công thức tính toán**:

- **LBPH Confidence**: Giá trị càng thấp càng tốt (0 = perfect match)
- **OpenFace Distance**: Euclidean distance = $\sqrt{\sum_{i=1}^{128}(a_i - b_i)^2}$

### Frontend (Giao diện hiển thị)

**Công nghệ**:

- **OpenCV HighGUI** (`cv2.imshow`) cho preview video
- **Tkinter/PyQt5** (tùy chọn) cho control panel

**Chức năng giao diện**:

- **Video Display**: Hiển thị realtime với bounding box và labels
- **Control Panel**:
  - Radio buttons: Chọn kỹ thuật (LBPH / FaceNet)
  - Sliders: Điều chỉnh threshold
  - Buttons: Start/Stop, Train, Add User
  - Status bar: Hiển thị FPS, kỹ thuật đang dùng

**Vẽ đồ họa**:

- `cv2.rectangle`: Vẽ khung mặt (Xanh: OK, Đỏ: Denied)
- `cv2.putText`: Ghi tên, confidence/distance, FPS
- `cv2.circle`: Indicator cho trạng thái (đèn xanh/đỏ)

### Cơ Sở Dữ Liệu (Database)

Không cần cài đặt MySQL/PostgreSQL phức tạp.

#### Cấu trúc lưu trữ:

**1. LBPH Method**:

```
models/
├── trainer.yml          # OpenCV YAML format
│   └── Contains: Histogram data for each label ID
└── mapping.json         # JSON format
    └── {"0": "Tuan", "1": "Nam", "2": "Huy"}
```

**2. FaceNet Method**:

```
models/
└── embeddings.pickle    # Python Pickle format
    └── {
          "names": ["Tuan", "Nam", "Huy"],
          "embeddings": [array(128,), array(128,), array(128,)]
        }
```

**3. Access Logs**:

```
logs/
└── access_log.csv       # CSV format
    └── timestamp,name,method,confidence,status
        2024-01-01 10:30:15,Tuan,LBPH,45.2,GRANTED
        2024-01-01 10:31:20,Unknown,OpenFace,0.85,DENIED
```

## 9. Quy Trình Chuyển Đổi Giữa 2 Kỹ Thuật

Người dùng có thể chuyển đổi kỹ thuật trong runtime:

1. **Dừng detection** hiện tại
2. **Unload model** cũ (giải phóng memory)
3. **Load model** mới theo kỹ thuật được chọn
4. **Khởi động lại** detection loop

**Lưu ý**: Mỗi kỹ thuật cần train riêng từ cùng một dataset.

## 10. Yêu Cầu Hệ Thống

### Tối thiểu (LBPH):

- CPU: Intel i3 hoặc tương đương
- RAM: 4GB
- Webcam: 720p, 30fps
- OS: Windows/Linux/MacOS

### Khuyến nghị (FaceNet):

- CPU: Intel i5 hoặc tương đương
- RAM: 8GB
- GPU: NVIDIA GTX 1050 hoặc cao hơn (tùy chọn)
- Webcam: 1080p, 30fps
- OS: Windows/Linux/MacOS
