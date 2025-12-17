# Face detection module (Haar/DNN)
import cv2
import numpy as np

class FaceDetector:
    def __init__(self, method='dnn'):
        self.HAAR_MODEL = 'models/haarcascade_frontalface_default.xml'
        self.DNN_PROTOTXT = 'models/deploy.prototxt'
        self.DNN_MODEL = 'models/res10_300x300_ssd_iter_140000.caffemodel'
        self.CONFIDENCE_THRESHOLD = 0.5

        self.method = method.lower()
        self.model = self._load_model()

    def _load_model(self):
        if self.method == 'haar':
            model = cv2.CascadeClassifier(self.HAAR_MODEL)
            if model.empty():
                raise IOError(f"Lỗi: Không thể tải Haar model từ {self.HAAR_MODEL}")
            return model
        elif self.method == 'dnn':
            model = cv2.dnn.readNetFromCaffe(self.DNN_PROTOTXT, self.DNN_MODEL)
            if model.empty():
                raise IOError("Lỗi: Không thể tải DNN model. Kiểm tra tệp .prototxt và .caffemodel.")
            return model
        else:
            raise ValueError("Phương thức phát hiện không hợp lệ. Chọn 'haar' hoặc 'dnn'.")

    def detect(self, frame):
        if self.method == 'haar':
            return self._detect_haar(frame)
        elif self.method == 'dnn':
            return self._detect_dnn(frame)
        return []

    def _detect_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = self.model.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30), 
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return list(rects) 

    def _detect_dnn(self, frame):
        (h, w) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                     (300, 300), (104.0, 177.0, 123.0))
        self.model.setInput(blob)
        detections = self.model.forward()

        boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                w_box = endX - startX
                h_box = endY - startY
                boxes.append((startX, startY, w_box, h_box))
        
        return boxes