# PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG FACE ACCESS CONTROL (V2.0)

## 1. Tổng Quan Hệ Thống

Hệ thống Face Access Control V2.0 là giải pháp kiểm soát ra vào sử dụng các mô hình học sâu (Deep Learning) tiên tiến nhất hiện nay của OpenCV Zoo, tập trung vào độ chính xác, tốc độ và khả năng triển khai dễ dàng.

### Công Nghệ Cốt Lõi:

- **Detection**: YuNet (CNN-based Face Detector)
- **Recognition**: SFace (Sigmoid-constrained Hypersphere Loss)
- **Interface**: Gradio (Web-based GUI)

## 2. Chi Tiết Kỹ Thuật

### A. Face Detection: YuNet

YuNet là một mô hình detection cực nhẹ nhưng hiệu quả cao.

- **Input**: Ảnh BGR (kích thước linh hoạt, resize về 320x320 hoặc dynamic).
- **Output**: Bounding Box (x, y, w, h) và 5 điểm landmarks (mắt, mũi, miệng).
- **Ưu điểm**:
  - Tốc độ cực nhanh trên CPU (>40 FPS).
  - Phát hiện được khuôn mặt nhỏ, nghiêng, bị che khuất một phần.
  - Thay thế hoàn toàn Haar Cascade (lỗi thời) và DNN cơ bản.

### B. Face Recognition: SFace

SFace là mô hình nhận diện khuôn mặt hiện đại, sử dụng hàm mất mát đặc biệt để tối ưu hóa khoảng cách giữa các vector đặc trưng.

- **Input**: Ảnh khuôn mặt đã được căn chỉnh (aligned) và crop (112x112).
- **Output**: Feature Vector (Embedding) 512 chiều (float32).
- **Metric**: Cosine Distance (Khoảng cách Cosine).
  - $Distance = 1 - CosineSimilarity(u, v)$
  - Giá trị càng gần 0 càng giống nhau.
  - Ngưỡng mặc định (Threshold): **0.4**.

### C. Quản Lý Dữ Liệu (Database)

Hệ thống không lưu trữ ảnh gốc của người dùng để so sánh, mà chỉ lưu trữ các vector đặc trưng (Embeddings), đảm bảo an toàn dữ liệu và tốc độ truy xuất.

- **Lưu trữ**: File `embeddings.pkl` (Pickle format).
- **Cấu trúc**: Tuple `(names, embeddings_list)`.
- **Logs**: File `access_log.csv` lưu lại lịch sử truy cập (Timestamp, Name, Score, Status).

## 3. Quy Trình Hoạt Động

### Giai đoạn 1: Đăng ký (Enrollment)

1. **Thu thập**: Chụp ảnh khuôn mặt người dùng từ Camera (`capture_dataset.py`).
2. **Detection**: YuNet tìm và crop khuôn mặt.
3. **Training**:
   - Load toàn bộ ảnh dataset.
   - Đưa qua SFace để trích xuất vector 512-d.
   - Lưu danh sách vector vào `embeddings.pkl`.

### Giai đoạn 2: Nhận diện & Kiểm soát (Inference)

1. **Capture**: Lấy frame từ Camera Stream.
2. **Detection**: YuNet phát hiện các khuôn mặt trong frame.
3. **Recognition**:
   - Với mỗi khuôn mặt, SFace trích xuất vector đặc trưng hiện tại ($v_{curr}$).
   - Tính Cosine Distance giữa $v_{curr}$ và toàn bộ vector trong Database ($V_{db}$).
   - Tìm vector có khoảng cách nhỏ nhất ($d_{min}$).
4. **Decision**:
   - Nếu $d_{min} < Threshold$ (0.4) --> **GRANTED** (Xác nhận danh tính).
   - Ngược lại --> **DENIED** (Unknown).
5. **Logging**: Ghi kết quả vào CSV và hiển thị lên Web UI.

## 4. Giao Diện Người Dùng (Gradio UI)

Hệ thống chuyển đổi từ giao diện Desktop (Tkinter) sang Web App (Gradio) để tăng tính linh hoạt và thẩm mỹ.

- **Live View**: Xem camera trực tiếp trình duyệt.
- **Real-time Status**: Hiển thị người vừa truy cập kèm hình ảnh đối chiếu.
- **Control Model**:
  - Chỉnh sửa Threshold trực tiếp (Slider).
  - Bật/Tắt Detection/Recognition.
- **User Management**:
  - Thêm user mới (Capture & Train tự động).
  - Xóa user (Cập nhật Database tức thời).
- **Log Viewer**: Xem lịch sử ra vào ngay trên web.

## 5. So Sánh V2.0 (SFace) vs V1.0 (LBPH/OpenFace)

| Tiêu chí          | V1.0 (LBPH)               | V1.0 (OpenFace)      | **V2.0 (SFace)**         |
| :---------------- | :------------------------ | :------------------- | :----------------------- |
| **Model**         | Hand-crafted features     | Deep Learning (dlib) | **Deep Learning (ONNX)** |
| **Tốc độ**        | Rất nhanh                 | Chậm (CPU)           | **Rất nhanh (CPU)**      |
| **Độ chính xác**  | Thấp                      | Cao                  | **Rất cao**              |
| **Vector Size**   | Histogram                 | 128-d                | **512-d**                |
| **Trọng số file** | ~N/A                      | ~100MB               | **~3MB (Lightweight)**   |
| **Triển khai**    | Phức tạp (Cần build dlib) | Phức tạp             | **Dễ dàng (OpenCV)**     |

## 6. Yêu Cầu Hệ Thống

- **CPU**: Intel Core i3 gen 4 trở lên (khuyến nghị i5).
- **RAM**: 4GB+.
- **Camera**: Webcam USB cơ bản (720p).
- **OS**: Windows 10/11, Linux, macOS.
- **Python**: 3.8 - 3.12.

---

**Version**: 2.0.0
**Status**: Stable
**Tech Stack**: Python, OpenCV, Gradio
