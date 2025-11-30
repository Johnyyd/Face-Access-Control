"""
Face Access Control - LBPH Recognition Module
Nhận diện khuôn mặt sử dụng Local Binary Patterns Histograms
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import os
import config
from .database import Database


class LBPHRecognizer:
    """
    Class nhận diện khuôn mặt sử dụng LBPH algorithm
    
    Attributes:
        recognizer: LBPH Face Recognizer từ OpenCV
        label_mapping: Dictionary {label_id: name}
        confidence_threshold: Ngưỡng confidence (càng thấp càng tốt)
        database: Database manager
    """
    
    def __init__(self, confidence_threshold: float = None):
        """
        Khởi tạo LBPH Recognizer
        
        Args:
            confidence_threshold: Ngưỡng confidence (mặc định từ config)
        """
        self.confidence_threshold = (confidence_threshold if confidence_threshold is not None 
                                    else config.LBPH_CONFIDENCE_THRESHOLD)
        
        # Tạo LBPH recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=config.LBPH_RADIUS,
            neighbors=config.LBPH_NEIGHBORS,
            grid_x=config.LBPH_GRID_X,
            grid_y=config.LBPH_GRID_Y
        )
        
        self.label_mapping: Dict[int, str] = {}
        self.database = Database()
        self.is_trained = False
        
        if config.DEBUG:
            print(f"[LBPHRecognizer] Initialized with threshold: {self.confidence_threshold}")
    
    def train(self, dataset_path: str = None) -> bool:
        """
        Train LBPH model từ dataset
        
        Args:
            dataset_path: Đường dẫn thư mục dataset (mặc định từ config)
            
        Returns:
            bool: True nếu train thành công
        """
        try:
            dataset_path = dataset_path or config.DATASET_DIR
            
            if not os.path.exists(dataset_path):
                print(f"[LBPHRecognizer] ERROR: Dataset not found: {dataset_path}")
                return False
            
            print(f"[LBPHRecognizer] Training from dataset: {dataset_path}")
            
            # Collect faces và labels
            faces = []
            labels = []
            label_mapping = {}
            current_label = 0
            
            # Duyệt qua các thư mục người dùng
            user_dirs = [d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))]
            
            if not user_dirs:
                print("[LBPHRecognizer] ERROR: No user directories found in dataset")
                return False
            
            for user_name in user_dirs:
                user_path = os.path.join(dataset_path, user_name)
                
                # Lấy tất cả ảnh trong thư mục user
                image_files = [f for f in os.listdir(user_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if len(image_files) < config.MIN_IMAGES_PER_PERSON:
                    print(f"[LBPHRecognizer] WARNING: {user_name} has only {len(image_files)} images "
                          f"(minimum: {config.MIN_IMAGES_PER_PERSON})")
                    continue
                
                print(f"[LBPHRecognizer] Processing {user_name}: {len(image_files)} images")
                
                # Assign label cho user này
                label_mapping[current_label] = user_name
                
                # Process mỗi ảnh
                for image_file in image_files[:config.MAX_IMAGES_PER_PERSON]:
                    image_path = os.path.join(user_path, image_file)
                    
                    # Đọc ảnh
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Convert sang grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Resize về kích thước chuẩn
                    gray = cv2.resize(gray, config.LBPH_FACE_SIZE)
                    
                    # Thêm vào dataset
                    faces.append(gray)
                    labels.append(current_label)
                
                current_label += 1
            
            if not faces:
                print("[LBPHRecognizer] ERROR: No valid faces found in dataset")
                return False
            
            print(f"[LBPHRecognizer] Total faces: {len(faces)}, Total users: {len(label_mapping)}")
            
            # Train recognizer
            self.recognizer.train(faces, np.array(labels))
            self.label_mapping = label_mapping
            self.is_trained = True
            
            # Lưu model
            if self.database.save_lbph_model(self.recognizer, self.label_mapping):
                print("[LBPHRecognizer] ✓ Training completed and model saved")
                return True
            else:
                print("[LBPHRecognizer] WARNING: Training completed but failed to save model")
                return False
            
        except Exception as e:
            print(f"[LBPHRecognizer] ERROR during training: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        Load trained model từ file
        
        Returns:
            bool: True nếu load thành công
        """
        try:
            recognizer, label_mapping = self.database.load_lbph_model()
            
            if recognizer is None or label_mapping is None:
                print("[LBPHRecognizer] ERROR: Failed to load model")
                return False
            
            self.recognizer = recognizer
            self.label_mapping = label_mapping
            self.is_trained = True
            
            print(f"[LBPHRecognizer] ✓ Model loaded with {len(label_mapping)} users")
            return True
            
        except Exception as e:
            print(f"[LBPHRecognizer] ERROR loading model: {e}")
            return False
    
    def predict(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """
        Nhận diện khuôn mặt
        
        Args:
            face_roi: Vùng khuôn mặt (BGR hoặc Grayscale)
            
        Returns:
            Tuple[str, float]: (name, confidence)
                - name: Tên người (hoặc "Unknown")
                - confidence: Confidence score (càng thấp càng tốt)
        """
        if not self.is_trained:
            return config.UNKNOWN_PERSON_NAME, 100.0
        
        try:
            # Convert sang grayscale nếu cần
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi
            
            # Resize về kích thước chuẩn
            gray = cv2.resize(gray, config.LBPH_FACE_SIZE)
            
            # Predict
            label, confidence = self.recognizer.predict(gray)
            
            # Kiểm tra threshold
            if confidence < self.confidence_threshold:
                name = self.label_mapping.get(label, config.UNKNOWN_PERSON_NAME)
            else:
                name = config.UNKNOWN_PERSON_NAME
            
            return name, confidence
            
        except Exception as e:
            print(f"[LBPHRecognizer] ERROR during prediction: {e}")
            return config.UNKNOWN_PERSON_NAME, 100.0
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        Cập nhật confidence threshold
        
        Args:
            new_threshold: Ngưỡng mới
        """
        self.confidence_threshold = new_threshold
        
        if config.DEBUG:
            print(f"[LBPHRecognizer] Threshold updated to: {new_threshold}")
    
    def get_threshold(self) -> float:
        """Lấy confidence threshold hiện tại"""
        return self.confidence_threshold
    
    def get_user_list(self) -> list:
        """Lấy danh sách users đã train"""
        return list(self.label_mapping.values())
    
    def is_model_trained(self) -> bool:
        """Kiểm tra model đã được train chưa"""
        return self.is_trained


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing LBPH Recognizer...")
    print("=" * 50)
    
    recognizer = LBPHRecognizer()
    
    # Test 1: Check if model exists
    print("\n1. Checking for existing model...")
    if recognizer.database.model_exists('lbph'):
        print("✓ Model exists, loading...")
        if recognizer.load_model():
            print(f"✓ Model loaded successfully")
            print(f"  Users: {recognizer.get_user_list()}")
        else:
            print("✗ Failed to load model")
    else:
        print("✗ No existing model found")
        print("  Run train_lbph.py to train a new model")
    
    # Test 2: Test prediction (nếu có model)
    if recognizer.is_model_trained():
        print("\n2. Testing prediction with camera...")
        
        from camera import CameraManager
        from detector import FaceDetector
        
        with CameraManager() as camera:
            if camera.is_opened():
                detector = FaceDetector(method='haar')
                
                print("Press 'q' to quit")
                
                while True:
                    ret, frame = camera.read()
                    if not ret:
                        break
                    
                    # Detect faces
                    faces = detector.detect_faces(frame)
                    
                    # Recognize each face
                    for (x, y, w, h) in faces:
                        face_roi = frame[y:y+h, x:x+w]
                        name, confidence = recognizer.predict(face_roi)
                        
                        # Draw result
                        color = config.COLOR_SUCCESS if name != config.UNKNOWN_PERSON_NAME else config.COLOR_DENIED
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        label = f"{name} ({confidence:.1f})"
                        cv2.putText(frame, label, (x, y-10),
                                  config.FONT_FACE, config.FONT_SCALE,
                                  color, config.FONT_THICKNESS)
                    
                    cv2.imshow("LBPH Recognition Test", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cv2.destroyAllWindows()
                print("✓ Recognition test completed")
    
    print("=" * 50)
