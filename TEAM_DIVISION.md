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

- Haar Cascade implementation
- DNN detection implementation
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
├── haarcascade_frontalface_default.xml
├── deploy.prototxt
└── res10_300x300_ssd_iter_140000.caffemodel

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

**LBPH Recognition**:

- LBPH algorithm implementation
- Training từ dataset
- Prediction với confidence score
- Threshold management

**OpenFace Recognition**:

- OpenFace/dlib integration
- 128-d embedding extraction
- Euclidean distance calculation
- Encoding storage

**Training Pipeline**:

- Dataset validation
- Feature extraction
- Model training
- Performance optimization

### Files phụ trách

```
modules/
├── recognizer_lbph.py      # LBPH recognition
└── recognizer_openface.py  # OpenFace recognition

train_lbph.py               # LBPH training script
train_openface.py           # OpenFace training script

models/
├── trainer.yml             # LBPH model
├── mapping.json            # LBPH label mapping
└── embeddings.pickle       # OpenFace encodings

dataset/
└── [username]/             # Training images
```

### Dependencies

- OpenCV (LBPH)
- face_recognition (OpenFace)
- dlib
- NumPy (< 2.0)
- Pickle

### Deliverables

- [x] LBPH recognizer class
- [x] OpenFace recognizer class
- [x] Training scripts cho cả 2 methods
- [x] Threshold tuning support

---

## 👤 Member 3: Frontend & Integration (30%)

### Trách nhiệm chính

**GUI Development**:

- Tkinter interface
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
└── main_window.py          # Main GUI

main.py                     # Entry point
config.py                   # Configuration

README.md                   # Main documentation
QUICKSTART.md              # Quick start guide
description.md             # Technical specs
requirements.txt           # Dependencies
```

### Dependencies

- Tkinter (GUI)
- Pillow (Image display)
- Threading

### Deliverables

- [x] Tkinter GUI với dual method support
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
- **Member 2**: LBPH recognizer
- **Member 3**: Basic GUI structure

### Phase 2: Advanced Features (Week 3)

- **Member 1**: Database + Logging
- **Member 2**: OpenFace recognizer
- **Member 3**: GUI controls + integration

### Phase 3: Testing & Documentation (Week 4)

- **All**: Integration testing
- **Member 3**: Documentation
- **All**: Bug fixes & optimization

---

## 📋 Quick Reference

| Member | Main Modules                         | Key Technologies               | Output              |
| ------ | ------------------------------------ | ------------------------------ | ------------------- |
| **1**  | camera, detector, database           | OpenCV, SQLite                 | Detection + Storage |
| **2**  | recognizer_lbph, recognizer_openface | OpenCV, dlib, face_recognition | Recognition         |
| **3**  | gui, main, config                    | Tkinter, Threading             | UI + Integration    |

---

## ✅ Completion Checklist

### Member 1

- [x] Camera management with context manager
- [x] Haar Cascade detection
- [x] DNN detection
- [x] Database operations
- [x] CSV logging

### Member 2

- [x] LBPH recognition
- [x] OpenFace recognition
- [x] Training scripts
- [x] Threshold management
- [x] Error handling for corrupted images

### Member 3

- [x] Tkinter GUI
- [x] Method switching (LBPH ↔ OpenFace)
- [x] Threshold slider
- [x] Access logs viewer
- [x] Documentation

---

**Status**: ✅ **ALL TASKS COMPLETED**

**Version**: 1.1.0  
**Last Updated**: 2025-11-30
