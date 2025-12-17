"""
Face Access Control - Face Detection Module
Phát hiện khuôn mặt bằng DNN hoặc Haar Cascade
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import sys
import os

# Thêm thư mục gốc vào sys.path để import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
except ImportError:
    # Fallback config nếu không tìm thấy file config.py
    class ConfigFallback:
        DNN_PROTOTXT = "models/deploy.prototxt"
        DNN_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
        HAAR_CASCADE = "models/haarcascade_frontalface_default.xml"
        CONFIDENCE_THRESHOLD = 0.7
        MIN_FACE_SIZE = (30, 30)
        DEBUG = True
    
    config = ConfigFallback()
    print("[FaceDetector] Warning: Using fallback config")


class FaceDetector:
    """
    Class phát hiện khuôn mặt sử dụng DNN hoặc Haar Cascade
    
    Attributes:
        method (str): 'dnn' hoặc 'haar'
        net (cv2.dnn_Net): DNN model (nếu dùng DNN)
        haar_cascade (cv2.CascadeClassifier): Haar cascade classifier (nếu dùng Haar)
    """
    
    def __init__(self, method: str = "dnn"):
        """
        Khởi tạo Face Detector
        
        Args:
            method: 'dnn' (mặc định) hoặc 'haar'
        """
        self.method = method.lower()
        
        if self.method == "dnn":
            # Khởi tạo DNN model
            model_path = config.DNN_MODEL
            prototxt_path = config.DNN_PROTOTXT
            
            if not os.path.exists(model_path) or not os.path.exists(prototxt_path):
                print(f"[FaceDetector] WARNING: DNN model files not found!")
                print(f"  Model: {model_path}")
                print(f"  Prototxt: {prototxt_path}")
                print("  Falling back to Haar Cascade...")
                self.method = "haar"
            else:
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                if config.DEBUG:
                    print(f"[FaceDetector] DNN model loaded: {model_path}")
        
        if self.method == "haar":
            # Khởi tạo Haar Cascade
            cascade_path = config.HAAR_CASCADE
            
            if not os.path.exists(cascade_path):
                print(f"[FaceDetector] ERROR: Haar cascade file not found: {cascade_path}")
                raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")
            
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            if config.DEBUG:
                print(f"[FaceDetector] Haar Cascade loaded: {cascade_path}")
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Phát hiện khuôn mặt trong frame
        
        Args:
            frame: Frame ảnh BGR
            
        Returns:
            List[Tuple[int, int, int, int]]: Danh sách bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.method == "dnn":
            return self._detect_dnn(frame, gray)
        else:  # haar
            return self._detect_haar(frame, gray)
    
    def _detect_dnn(self, frame: np.ndarray, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Phát hiện khuôn mặt bằng DNN
        """
        (h, w) = frame.shape[:2]
        
        # Tạo blob từ ảnh
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        # Detect faces
        self.net.setInput(blob)
        detections = self.net.forward()
        
        boxes = []
        confidence_threshold = getattr(config, 'CONFIDENCE_THRESHOLD', 0.7)
        
        # Lặp qua các detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Lọc các detection với confidence cao hơn threshold
            if confidence > confidence_threshold:
                # Tính toán tọa độ bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Đảm bảo bounding box nằm trong kích thước frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Chuyển đổi sang định dạng (x, y, w, h)
                boxes.append((startX, startY, endX - startX, endY - startY))
        
        if config.DEBUG and boxes:
            print(f"[FaceDetector] DNN detected {len(boxes)} face(s)")
        
        return boxes
    
    def _detect_haar(self, frame: np.ndarray, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Phát hiện khuôn mặt bằng Haar Cascade
        """
        min_face_size = getattr(config, 'MIN_FACE_SIZE', (30, 30))
        
        # Detect faces
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if config.DEBUG and len(faces) > 0:
            print(f"[FaceDetector] Haar detected {len(faces)} face(s)")
        
        return faces
    
    def detect_with_confidence(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Phát hiện khuôn mặt với confidence scores (chỉ DNN)
        
        Args:
            frame: Frame ảnh BGR
            
        Returns:
            List[Tuple[Tuple[int, int, int, int], float]]: Bounding boxes với confidence
        """
        if self.method != "dnn":
            boxes = self.detect(frame)
            return [(box, 1.0) for box in boxes]  # Haar không có confidence
        
        (h, w) = frame.shape[:2]
        
        # Tạo blob từ ảnh
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        # Detect faces
        self.net.setInput(blob)
        detections = self.net.forward()
        
        results = []
        confidence_threshold = getattr(config, 'CONFIDENCE_THRESHOLD', 0.7)
        
        # Lặp qua các detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Lọc các detection với confidence cao hơn threshold
            if confidence > confidence_threshold:
                # Tính toán tọa độ bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Đảm bảo bounding box nằm trong kích thước frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Chuyển đổi sang định dạng (x, y, w, h)
                box_tuple = (startX, startY, endX - startX, endY - startY)
                results.append((box_tuple, confidence))
        
        return results
    
    def draw_boxes(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                   color: Tuple[int, int, int] = (0, 255, 0), 
                   thickness: int = 2) -> np.ndarray:
        """
        Vẽ bounding boxes lên frame
        
        Args:
            frame: Frame gốc
            boxes: Danh sách bounding boxes
            color: Màu bounding box (BGR)
            thickness: Độ dày đường vẽ
            
        Returns:
            np.ndarray: Frame với bounding boxes
        """
        frame_copy = frame.copy()
        
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
            
            # Vẽ label nếu có confidence
            label = "Face"
            cv2.putText(frame_copy, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_copy


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing Face Detector...")
    print("=" * 50)
    
    # Test camera và detector
    from modules.camera import CameraManager
    
    # Test DNN
    print("\n1. Testing DNN detector...")
    detector_dnn = FaceDetector(method="dnn")
    
    with CameraManager() as camera:
        if camera.is_opened():
            ret, frame = camera.read()
            if ret:
                boxes = detector_dnn.detect(frame)
                print(f"✓ DNN detected {len(boxes)} faces")
                
                # Vẽ và hiển thị
                frame_with_boxes = detector_dnn.draw_boxes(frame, boxes)
                cv2.imshow("DNN Detection", frame_with_boxes)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
    
    # Test Haar
    print("\n2. Testing Haar detector...")
    detector_haar = FaceDetector(method="haar")
    
    with CameraManager() as camera:
        if camera.is_opened():
            ret, frame = camera.read()
            if ret:
                boxes = detector_haar.detect(frame)
                print(f"✓ Haar detected {len(boxes)} faces")
                
                # Vẽ và hiển thị
                frame_with_boxes = detector_haar.draw_boxes(frame, boxes, color=(255, 0, 0))
                cv2.imshow("Haar Detection", frame_with_boxes)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
    
    print("=" * 50)
    print("Face Detector test completed")