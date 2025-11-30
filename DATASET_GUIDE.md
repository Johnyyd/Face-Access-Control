# Hướng Dẫn Chuẩn Bị Dataset

Hướng dẫn chi tiết cách chuẩn bị dataset cho training.

## 📁 Cấu Trúc Dataset

```
dataset/
├── User1/
│   ├── 001.jpg
│   ├── 002.jpg
│   ├── 003.jpg
│   └── ... (10-20 ảnh)
├── User2/
│   ├── 001.jpg
│   └── ... (10-20 ảnh)
└── User3/
    └── ... (10-20 ảnh)
```

## 📸 Yêu Cầu Ảnh

### Số Lượng

- **Tối thiểu**: 10 ảnh/người
- **Khuyến nghị**: 15-20 ảnh/người
- **Tối đa**: 100 ảnh/người (để tránh overfitting)

### Chất Lượng

- ✅ Độ phân giải: Tối thiểu 640x480
- ✅ Format: JPG, JPEG, hoặc PNG
- ✅ Khuôn mặt rõ ràng, không bị che khuất
- ✅ Ánh sáng tốt, không quá tối hoặc quá sáng
- ✅ Khuôn mặt chiếm ít nhất 30% ảnh

### Đa Dạng

Chụp ảnh ở nhiều điều kiện khác nhau:

- 📐 **Góc độ**: Thẳng, nghiêng trái/phải, ngửa/cúi nhẹ
- 💡 **Ánh sáng**: Sáng, tối, ánh sáng tự nhiên, đèn
- 😊 **Biểu cảm**: Mặt bình thường, cười, nghiêm túc
- 👓 **Phụ kiện**: Có/không kính, mũ (nếu thường xuyên đeo)
- 🎨 **Background**: Nhiều background khác nhau

## 🎯 Cách Chụp Ảnh

### Option 1: Chụp Thủ Công

1. Tạo thư mục cho user: `dataset/TenNguoi/`
2. Chụp 15-20 ảnh với điều kiện đa dạng
3. Đặt tên: `001.jpg`, `002.jpg`, ...
4. Copy vào thư mục user

### Option 2: Sử Dụng Script Capture

Tạo file `capture_dataset.py`:

```python
import cv2
import os

def capture_images(name, num_images=20):
    """Chụp ảnh từ webcam"""
    # Tạo thư mục
    user_dir = f"dataset/{name}"
    os.makedirs(user_dir, exist_ok=True)

    # Mở camera
    cap = cv2.VideoCapture(0)
    count = 0

    print(f"Capturing {num_images} images for {name}")
    print("Press SPACE to capture, Q to quit")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        # Hiển thị
        cv2.putText(frame, f"Captured: {count}/{num_images}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)
        cv2.imshow("Capture Dataset", frame)

        key = cv2.waitKey(1) & 0xFF

        # SPACE để chụp
        if key == ord(' '):
            filename = f"{user_dir}/{count+1:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            count += 1

        # Q để thoát
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✓ Captured {count} images for {name}")

if __name__ == "__main__":
    name = input("Enter user name: ")
    num_images = int(input("Number of images (default 20): ") or 20)
    capture_images(name, num_images)
```

Chạy:

```bash
python capture_dataset.py
```

## ✅ Kiểm Tra Dataset

### Script Kiểm Tra

```python
import os

def check_dataset(dataset_dir="dataset"):
    """Kiểm tra dataset"""
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    if not os.path.exists(dataset_dir):
        print(f"✗ Dataset directory not found: {dataset_dir}")
        return False

    users = [d for d in os.listdir(dataset_dir)
             if os.path.isdir(os.path.join(dataset_dir, d))
             and not d.startswith('.')]

    if not users:
        print("✗ No user directories found")
        return False

    print(f"\nFound {len(users)} user(s):\n")

    total_images = 0
    valid_users = 0

    for user in users:
        user_path = os.path.join(dataset_dir, user)
        images = [f for f in os.listdir(user_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        num_images = len(images)
        total_images += num_images

        status = "✓" if num_images >= 10 else "✗"
        if num_images >= 10:
            valid_users += 1

        print(f"{status} {user}: {num_images} images")

    print("\n" + "=" * 60)
    print(f"Total users: {len(users)}")
    print(f"Valid users (≥10 images): {valid_users}")
    print(f"Total images: {total_images}")
    print("=" * 60)

    if valid_users == 0:
        print("\n✗ No valid users found!")
        print("Each user needs at least 10 images")
        return False

    print("\n✓ Dataset is ready for training!")
    return True

if __name__ == "__main__":
    check_dataset()
```

Lưu thành `check_dataset.py` và chạy:

```bash
python check_dataset.py
```

## 📝 Checklist

Trước khi training, đảm bảo:

- [ ] Mỗi user có thư mục riêng trong `dataset/`
- [ ] Mỗi user có ít nhất 10 ảnh
- [ ] Ảnh có chất lượng tốt (rõ ràng, đủ sáng)
- [ ] Ảnh đa dạng (góc độ, ánh sáng, biểu cảm)
- [ ] Tên thư mục không có ký tự đặc biệt
- [ ] Format ảnh: JPG, JPEG, hoặc PNG

## ⚠️ Lưu Ý

### Nên

- ✅ Chụp ở nhiều góc độ khác nhau
- ✅ Thay đổi ánh sáng
- ✅ Thay đổi biểu cảm
- ✅ Giữ khuôn mặt ở giữa frame
- ✅ Đảm bảo khuôn mặt rõ ràng

### Không Nên

- ❌ Ảnh mờ, không rõ
- ❌ Khuôn mặt bị che khuất nhiều
- ❌ Quá tối hoặc quá sáng
- ❌ Khuôn mặt quá nhỏ trong ảnh
- ❌ Ảnh trùng lặp

## 🎨 Ví Dụ Dataset Tốt

```
dataset/
├── Alice/
│   ├── 001.jpg  # Thẳng, ánh sáng tự nhiên
│   ├── 002.jpg  # Nghiêng trái 15°
│   ├── 003.jpg  # Nghiêng phải 15°
│   ├── 004.jpg  # Cười
│   ├── 005.jpg  # Nghiêm túc
│   ├── 006.jpg  # Đèn trong nhà
│   ├── 007.jpg  # Ánh sáng yếu
│   ├── 008.jpg  # Đeo kính
│   ├── 009.jpg  # Không kính
│   └── 010.jpg  # Background khác
└── Bob/
    └── ... (tương tự)
```

## 🚀 Sau Khi Chuẩn Bị Dataset

1. Chạy validation: `python check_dataset.py`
2. Nếu OK, chạy training:
   - LBPH: `python train_lbph.py`
   - FaceNet: `python train_facenet.py`

---

**Lưu ý**: Dataset càng tốt, độ chính xác nhận diện càng cao!
