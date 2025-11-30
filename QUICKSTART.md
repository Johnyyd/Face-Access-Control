# Quick Start Guide

## Bước 1: Cài đặt

```bash
pip install -r requirements.txt
pip install "numpy<2.0"
```

## Bước 2: Chụp Dataset

```bash
python capture_dataset.py
```

- Nhập tên user
- Chụp 15-20 ảnh (nhiều góc độ)
- Ảnh lưu vào `dataset/[username]/`

## Bước 3: Train Models

```bash
python train_lbph.py      # LBPH (nhanh)
python train_openface.py  # OpenFace (chính xác)
```

## Bước 4: Chạy App

```bash
python main.py
```

## Controls trong GUI

- Chọn Recognition Method: LBPH hoặc OpenFace
- Chọn Detection Method: Haar Cascade hoặc DNN
- Điều chỉnh Threshold bằng slider
- Click Start để bắt đầu

## Tips

- **LBPH**: Nhanh (30-40 FPS), accuracy 70-85%
- **OpenFace**: Chính xác (85-95%), chậm hơn (10-15 FPS)
- **Threshold thấp** = strict hơn, ít false positives
- **Threshold cao** = loose hơn, ít false negatives

## Troubleshooting

- **OpenFace lỗi**: Chạy `pip install "numpy<2.0"`
- **Camera không mở**: Đổi `CAMERA_ID` trong config.py
- **Accuracy thấp**: Chụp thêm ảnh, nhiều góc độ hơn
