# PHÂN CHIA CÔNG VIỆC - FACE ACCESS CONTROL SYSTEM

Dựa trên hệ thống LBPH + OpenFace Recognition

---

## 👤 Member 1: Backend Core & Detection (35%)

### Trách nhiệm chính

**Camera Management**:

- Quản lý webcam (open, read, release)
- Context manager support
- FPS control

**Face Detection**:

- SFace Regconize implementation
- YuNet detection implementation
- Switchable detection methods
- Bounding box extraction

**Database & Storage**:

- Model persistence (LBPH, OpenFace)
- Access logging (CSV)
- File I/O operations

### Files phụ trách

```
modules/
├── camera.py           # Camera management
├── detector.py         # Face detection (Haar/DNN)
└── database.py         # Storage & logging

models/
├── sface
│   └── face_recognition_sface_2021dec.onnx
└── yunet
    └── face_detection_yunet_2023mar.onnx


logs/
└── access_log.csv
```

### Dependencies

- OpenCV (cv2)
- NumPy
- CSV

### Deliverables

- [x] Camera class với context manager
- [x] Dual detection methods (Haar + DNN)
- [x] Database operations (save/load models)
- [x] Access logging system

---

## 👤 Member 2: AI/ML Recognition (35%)

### Trách nhiệm chính

**SFace Recognition**:

- SFace algorithm implementation
- Training từ dataset
- Prediction với confidence score
- Threshold management

**Training Pipeline**:

- Dataset validation
- Feature extraction
- Model training
- Performance optimization

### Files phụ trách

```
modules/
└── recognizer_sface.py  # OpenFace recognition

train_sface.py               # SFace training script

models/
├── sface
│   └── face_recognition_sface_2021dec.onnx
└── yunet
    └── face_detection_yunet_2023mar.onnx

dataset/
└── [username]/             # Training images
```

### Dependencies

- OpenCV (LBPH)
- dlib
- NumPy (< 2.0)
- Pickle

### Deliverables

- [x] SFace recognizer class
- [x] Training scripts
- [x] Threshold tuning support

---

## 👤 Member 3: Frontend & Integration (30%)

### Trách nhiệm chính

**GUI Development**:

- Gradio interface
- Video display
- Control panel (method selection, threshold slider)
- Access logs viewer

**System Integration**:

- Main application flow
- Module integration
- Threading for smooth GUI
- Error handling

**Configuration & Documentation**:

- Config management
- System documentation
- User guides

### Files phụ trách

```
gui/
├── __init__.py
└── main_window_gradio.py          # Main GUI

main.py                     # Entry point
config.py                   # Configuration

README.md                   # Main documentation
QUICKSTART.md              # Quick start guide
description.md             # Technical specs
requirements.txt           # Dependencies
```

### Dependencies

- Gradio (GUI)
- Pillow (Image display)
- Threading

### Deliverables

- [x] Gradio GUI với dual method support
- [x] Real-time video display
- [x] Control panel (method switching, threshold)
- [x] Main application integration
- [x] Documentation

---

## 🔗 Integration Points

### Member 1 → Member 2

**Interface**: Detection results

```python
# Member 1 provides
faces = detector.detect_faces(frame)  # [(x,y,w,h), ...]

# Member 2 receives
for (x,y,w,h) in faces:
    face_roi = frame[y:y+h, x:x+w]
    name, score = recognizer.predict(face_roi)
```

### Member 1 → Member 3

**Interface**: Camera frames & logging

```python
# Member 1 provides
ret, frame = camera.read()
database.log_access(name, method, score, status)

# Member 3 receives
# Display frame in GUI
# Show access logs
```

### Member 2 → Member 3

**Interface**: Recognition results

```python
# Member 2 provides
name, score = recognizer.predict(face_roi)
users = recognizer.get_user_list()

# Member 3 receives
# Display name, score in GUI
# Show trained users list
```

### Member 3 → All

**Orchestration**: Main workflow

```python
# Member 3 integrates
1. Initialize camera (Member 1)
2. Load recognizer (Member 2)
3. Start recognition loop
4. Display results in GUI
5. Log access (Member 1)
```

---

## 📊 Workload Distribution

| Member | Focus Area   | Lines of Code | Complexity | Time Estimate |
| ------ | ------------ | ------------- | ---------- | ------------- |
| **1**  | Backend Core | ~800 lines    | Medium     | 35%           |
| **2**  | AI/ML        | ~900 lines    | High       | 35%           |
| **3**  | Frontend     | ~700 lines    | Medium     | 30%           |

---

## 🎯 Milestones

### Phase 1: Core Development (Week 1-2)

- **Member 1**: Camera + Detection modules
- **Member 2**: SFace recognizer
- **Member 3**: Basic GUI structure

### Phase 2: Advanced Features (Week 3)

- **Member 1**: Database + Logging
- **Member 2**: SFace recognizer
- **Member 3**: GUI controls + integration

### Phase 3: Testing & Documentation (Week 4)

- **All**: Integration testing
- **Member 3**: Documentation
- **All**: Bug fixes & optimization

---

## 📋 Quick Reference

| Member | Main Modules                         | Key Technologies               | Output              |
| ------ | ------------------------------------ | ------------------------------ | ------------------- |
| **1**  | camera, detector, database           | OpenCV, csv                    | Detection + Storage |
| **2**  | recognizer_sface                     | OpenCV, dlib                   | Recognition         |
| **3**  | gui, main, config                    | Tkinter, Threading             | UI + Integration    |

---

## ✅ Completion Checklist

### Member 1

- [x] Camera management with context manager
- [x] YuNet Cascade detection
- [x] Database operations
- [x] CSV logging

### Member 2

- [x] SFace recognition
- [x] Training scripts
- [x] Threshold management
- [x] Error handling for corrupted images

### Member 3

- [x] Gradio GUI
- [x] Threshold slider
- [x] Access logs viewer
- [x] Documentation

---

**Status**: ✅ **ALL TASKS COMPLETED**

**Version**: 1.1.0  
**Last Updated**: 2025-11-30
