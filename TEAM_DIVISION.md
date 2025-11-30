# PHÂN CHIA CÔNG VIỆC - FACE ACCESS CONTROL

## 👤 Member 1: Backend Core Developer (35%)

### Trách nhiệm

- Camera management
- Face detection (Haar + DNN)
- Database operations
- Logging system

### Files

```
modules/camera.py
modules/detector.py
modules/database.py
logs/access_log.csv
models/haarcascade_frontalface_default.xml
models/deploy.prototxt
models/res10_300x300_ssd_iter_140000.caffemodel
```

---

## 👤 Member 2: AI/ML Recognition Developer (35%)

### Trách nhiệm

- LBPH recognition algorithm
- FaceNet recognition algorithm
- Training scripts
- Feature extraction

### Files

```
modules/recognizer_lbph.py
modules/recognizer_facenet.py
train_lbph.py
train_facenet.py
dataset/
models/trainer.yml
models/mapping.json
models/facenet_keras.h5
models/embeddings.pickle
```

---

## 👤 Member 3: Frontend & Integration Developer (30%)

### Trách nhiệm

- GUI development
- Main application flow
- Configuration management
- System integration

### Files

```
gui/__init__.py
gui/main_window.py
main.py
config.py
requirements.txt
README.md
```

---

## 🔗 Integration Points

### Member 1 → Member 2

- Cung cấp detected faces
- Database save/load functions

### Member 1 → Member 3

- Camera frames
- Detection results
- Logging functions

### Member 2 → Member 3

- Recognition results
- Training interface

### Member 3 → All

- Tích hợp toàn bộ modules
- Main application workflow

---

## 📋 Quick Reference

| Member | Main Focus   | Key Modules                         | Dependencies      |
| ------ | ------------ | ----------------------------------- | ----------------- |
| **1**  | Backend Core | camera, detector, database          | OpenCV, NumPy     |
| **2**  | AI/ML        | recognizer_lbph, recognizer_facenet | TensorFlow, Keras |
| **3**  | Frontend     | GUI, main, config                   | PyQt5/Tkinter     |

---

**Chi tiết đầy đủ**: Xem file `team_division.md` trong thư mục artifacts
