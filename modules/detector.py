"""
Face Access Control - Face Detection Module
Phát hiện khuôn mặt sử dụng Haar Cascade hoặc DNN
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
import config


class FaceDetector:
    """
    Class phát hiện khuôn mặt hỗ trợ 2 phương pháp:
    - Haar Cascade (nhanh, nhẹ)
    - DNN (chính xác hơn)
    
    Attributes:
        method (str): Phương pháp detection ('haar' hoặc 'dnn')
        detector: Haar Cascade classifier hoặc DNN network
    """
    
    def __init__(self, method: str = None):
        """
        Khởi tạo Face Detector
        
        Args:
            method: 'haar' hoặc 'dnn' (mặc định từ config)
        """
        self.method = method if method is not None else config.DEFAULT_DETECTION_METHOD
        self.detector = None
        self.net = None
        
        # Load detector theo method
        self._load_detector()
        
        if config.DEBUG:
            print(f"[FaceDetector] Initialized with method: {self.method}")
    
    def _load_detector(self) -> bool:
        """
        Load detector model theo method được chọn
        
        Returns:
            bool: True nếu load thành công
        """
        try:
            if self.method == 'haar':
                return self._load_haar_cascade()
            elif self.method == 'dnn':
                return self._load_dnn()
            else:
                print(f"[FaceDetector] ERROR: Invalid method '{self.method}'")
                return False
        except Exception as e:
            print(f"[FaceDetector] ERROR loading detector: {e}")
            return False
    
    def _load_haar_cascade(self) -> bool:
        """Load Haar Cascade classifier"""
        if not os.path.exists(config.HAAR_CASCADE_PATH):
            print(f"[FaceDetector] ERROR: Haar Cascade file not found: {config.HAAR_CASCADE_PATH}")
            print("[FaceDetector] Please download from: https://github.com/opencv/opencv/tree/master/data/haarcascades")
            return False
        
        self.detector = cv2.CascadeClassifier(config.HAAR_CASCADE_PATH)
        
        if self.detector.empty():
            print("[FaceDetector] ERROR: Failed to load Haar Cascade")
            return False
        
        if config.DEBUG:
            print("[FaceDetector] Haar Cascade loaded successfully")
        
        return True
    
    def _load_dnn(self) -> bool:
        """Load DNN face detector"""
        if not os.path.exists(config.DNN_PROTOTXT_PATH):
            print(f"[FaceDetector] ERROR: DNN prototxt not found: {config.DNN_PROTOTXT_PATH}")
            print("[FaceDetector] Please download from: https://github.com/opencv/opencv_3rdparty")
            return False
        
        if not os.path.exists(config.DNN_MODEL_PATH):
            print(f"[FaceDetector] ERROR: DNN model not found: {config.DNN_MODEL_PATH}")
            print("[FaceDetector] Please download from: https://github.com/opencv/opencv_3rdparty")
            return False
        
        self.net = cv2.dnn.readNetFromCaffe(config.DNN_PROTOTXT_PATH, config.DNN_MODEL_PATH)
        
        if self.net is None:
            print("[FaceDetector] ERROR: Failed to load DNN model")
            return False
        
        if config.DEBUG:
            print("[FaceDetector] DNN model loaded successfully")
        
        return True
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Phát hiện khuôn mặt trong frame
        
        Args:
            frame: Frame ảnh (BGR format)
            
        Returns:
            List[Tuple[int, int, int, int]]: Danh sách bounding boxes (x, y, w, h)
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            if self.method == 'haar':
                return self._detect_haar(frame)
            elif self.method == 'dnn':
                return self._detect_dnn(frame)
            else:
                return []
        except Exception as e:
            print(f"[FaceDetector] ERROR detecting faces: {e}")
            return []
    
    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces sử dụng Haar Cascade
        
        Args:
            frame: Frame ảnh (BGR)
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        if self.detector is None:
            return []
        
        # Convert to grayscale (Haar Cascade yêu cầu grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=config.HAAR_SCALE_FACTOR,
            minNeighbors=config.HAAR_MIN_NEIGHBORS,
            minSize=config.HAAR_MIN_SIZE
        )
        
        # Convert từ numpy array sang list of tuples
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    
    def _detect_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces sử dụng DNN
        
        Args:
            frame: Frame ảnh (BGR)
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        if self.net is None:
            return []
        
        h, w = frame.shape[:2]
        
        # Tạo blob từ frame
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, config.DNN_INPUT_SIZE),
            1.0,
            config.DNN_INPUT_SIZE,
            (104.0, 177.0, 123.0)  # Mean subtraction values
        )
        
        # Forward pass
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Parse detections
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Lọc theo confidence threshold
            if confidence > config.DNN_CONFIDENCE_THRESHOLD:
                # Tính toán bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Convert sang format (x, y, w, h)
                x = max(0, x1)
                y = max(0, y1)
                width = min(w - x, x2 - x1)
                height = min(h - y, y2 - y1)
                
                if width > 0 and height > 0:
                    faces.append((x, y, width, height))
        
        return faces
    
    def switch_method(self, new_method: str) -> bool:
        """
        Chuyển đổi phương pháp detection
        
        Args:
            new_method: 'haar' hoặc 'dnn'
            
        Returns:
            bool: True nếu chuyển đổi thành công
        """
        if new_method not in ['haar', 'dnn']:
            print(f"[FaceDetector] ERROR: Invalid method '{new_method}'")
            return False
        
        if new_method == self.method:
            if config.DEBUG:
                print(f"[FaceDetector] Already using method: {new_method}")
            return True
        
        old_method = self.method
        self.method = new_method
        
        if self._load_detector():
            if config.DEBUG:
                print(f"[FaceDetector] Switched from {old_method} to {new_method}")
            return True
        else:
            # Rollback nếu load thất bại
            self.method = old_method
            print(f"[FaceDetector] ERROR: Failed to switch to {new_method}, keeping {old_method}")
            return False
    
    def draw_faces(self, 
                   frame: np.ndarray, 
                   faces: List[Tuple[int, int, int, int]],
                   color: Tuple[int, int, int] = None,
                   thickness: int = None,
                   label: str = None) -> np.ndarray:
        """
        Vẽ bounding boxes lên frame
        
        Args:
            frame: Frame ảnh
            faces: List of (x, y, w, h) bounding boxes
            color: Màu BGR (mặc định từ config)
            thickness: Độ dày viền (mặc định từ config)
            label: Text hiển thị (optional)
            
        Returns:
            Frame đã vẽ bounding boxes
        """
        if color is None:
            color = config.COLOR_SUCCESS
        if thickness is None:
            thickness = config.BBOX_THICKNESS
        
        frame_copy = frame.copy()
        
        for (x, y, w, h) in faces:
            # Vẽ rectangle
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
            
            # Vẽ label nếu có
            if label:
                # Background cho text
                text_size = cv2.getTextSize(label, config.FONT_FACE, config.FONT_SCALE, config.FONT_THICKNESS)[0]
                cv2.rectangle(frame_copy, 
                            (x, y - text_size[1] - 10), 
                            (x + text_size[0], y), 
                            color, -1)
                
                # Text
                cv2.putText(frame_copy, label, (x, y - 5),
                           config.FONT_FACE, config.FONT_SCALE, 
                           config.COLOR_TEXT, config.FONT_THICKNESS)
        
        return frame_copy
    
    def get_face_roi(self, 
                     frame: np.ndarray, 
                     face: Tuple[int, int, int, int],
                     padding: int = 0) -> Optional[np.ndarray]:
        """
        Cắt vùng khuôn mặt từ frame
        
        Args:
            frame: Frame ảnh
            face: Bounding box (x, y, w, h)
            padding: Padding thêm xung quanh face (pixels)
            
        Returns:
            Face ROI (Region of Interest)
        """
        if frame is None or frame.size == 0:
            return None
        
        x, y, w, h = face
        h_frame, w_frame = frame.shape[:2]
        
        # Apply padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_frame, x + w + padding)
        y2 = min(h_frame, y + h + padding)
        
        # Crop face ROI
        face_roi = frame[y1:y2, x1:x2]
        
        return face_roi if face_roi.size > 0 else None
    
    def get_method(self) -> str:
        """Lấy phương pháp detection hiện tại"""
        return self.method


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing Face Detector...")
    print("=" * 50)
    
    from camera import CameraManager
    
    # Test Haar Cascade
    print("\n1. Testing Haar Cascade...")
    detector_haar = FaceDetector(method='haar')
    
    # Test DNN
    print("\n2. Testing DNN...")
    detector_dnn = FaceDetector(method='dnn')
    
    # Test với camera
    print("\n3. Testing with camera...")
    with CameraManager() as camera:
        if camera.is_opened():
            detector = detector_haar  # Bắt đầu với Haar
            
            print("Press 'h' for Haar, 'd' for DNN, 'q' to quit")
            
            while True:
                ret, frame = camera.read()
                if not ret:
                    break
                
                # Detect faces
                faces = detector.detect_faces(frame)
                
                # Draw faces
                frame_with_faces = detector.draw_faces(
                    frame, faces, 
                    label=f"Faces: {len(faces)} ({detector.get_method().upper()})"
                )
                
                # Display
                cv2.imshow("Face Detection Test", frame_with_faces)
                
                # Keyboard control
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    detector.switch_method('haar')
                elif key == ord('d'):
                    detector.switch_method('dnn')
            
            cv2.destroyAllWindows()
            print("✓ Face detection test completed")
        else:
            print("✗ Failed to open camera")
    
    print("=" * 50)
