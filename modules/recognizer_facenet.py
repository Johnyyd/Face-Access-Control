"""
Face Access Control - FaceNet Recognition Module
Nhận diện khuôn mặt sử dụng FaceNet embeddings (Deep Learning)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import os
from scipy.spatial.distance import euclidean
import config
from .database import Database

# Import keras-facenet (wrapper đã build sẵn)
try:
    from keras_facenet import FaceNet
    KERAS_AVAILABLE = True
except ImportError:
    print("[FaceNetRecognizer] WARNING: keras-facenet not available")
    print("[FaceNetRecognizer] Please install: pip install keras-facenet")
    KERAS_AVAILABLE = False


class FaceNetRecognizer:
    """
    Class nhận diện khuôn mặt sử dụng FaceNet embeddings
    
    Attributes:
        model: FaceNet model (Keras)
        known_names: List tên người đã đăng ký
        known_embeddings: List embeddings tương ứng
        distance_threshold: Ngưỡng khoảng cách Euclidean
        database: Database manager
    """
    
    def __init__(self, distance_threshold: float = None):
        """
        Khởi tạo FaceNet Recognizer
        
        Args:
            distance_threshold: Ngưỡng distance (mặc định từ config)
        """
        self.distance_threshold = (distance_threshold if distance_threshold is not None 
                                  else config.FACENET_DISTANCE_THRESHOLD)
        
        self.model = None
        self.known_names: List[str] = []
        self.known_embeddings: List[np.ndarray] = []
        self.database = Database()
        self.is_trained = False
        
        if not KERAS_AVAILABLE:
            print("[FaceNetRecognizer] ERROR: Cannot initialize without TensorFlow/Keras")
        
        if config.DEBUG:
            print(f"[FaceNetRecognizer] Initialized with threshold: {self.distance_threshold}")
    
    def load_facenet_model(self, model_path: str = None) -> bool:
        """
        Load pre-trained FaceNet model using keras-facenet
        
        Args:
            model_path: Not used (keras-facenet handles model internally)
            
        Returns:
            bool: True nếu load thành công
        """
        if not KERAS_AVAILABLE:
            print("[FaceNetRecognizer] ERROR: keras-facenet not available")
            return False
        
        try:
            print("[FaceNetRecognizer] Loading FaceNet model...")
            
            # Use keras-facenet FaceNet class (auto-downloads model if needed)
            self.model = FaceNet()
            
            if config.DEBUG:
                print("[FaceNetRecognizer] ✓ FaceNet model loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"[FaceNetRecognizer] ERROR loading FaceNet model: {e}")
            return False
    
    def extract_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Trích xuất embedding vector từ khuôn mặt
        
        Args:
            face_roi: Vùng khuôn mặt (BGR format)
            
        Returns:
            np.ndarray: Embedding vector 128 chiều (hoặc None nếu lỗi)
        """
        if self.model is None:
            print("[FaceNetRecognizer] ERROR: FaceNet model not loaded")
            return None
        
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            # Resize về kích thước FaceNet yêu cầu (160x160)
            resized = cv2.resize(rgb, config.FACENET_INPUT_SIZE)
            
            # keras-facenet expects images in [0, 255] range
            # Extract embedding using keras-facenet
            embedding = self.model.embeddings([resized])
            
            return embedding[0]
            
        except Exception as e:
            print(f"[FaceNetRecognizer] ERROR extracting embedding: {e}")
            return None
    
    def train(self, dataset_path: str = None) -> bool:
        """
        Train FaceNet model từ dataset (tạo embeddings database)
        
        Args:
            dataset_path: Đường dẫn thư mục dataset (mặc định từ config)
            
        Returns:
            bool: True nếu train thành công
        """
        if not KERAS_AVAILABLE:
            print("[FaceNetRecognizer] ERROR: TensorFlow/Keras not available")
            return False
        
        if self.model is None:
            print("[FaceNetRecognizer] ERROR: FaceNet model not loaded")
            print("[FaceNetRecognizer] Please call load_facenet_model() first")
            return False
        
        try:
            dataset_path = dataset_path or config.DATASET_DIR
            
            if not os.path.exists(dataset_path):
                print(f"[FaceNetRecognizer] ERROR: Dataset not found: {dataset_path}")
                return False
            
            print(f"[FaceNetRecognizer] Training from dataset: {dataset_path}")
            
            # Collect embeddings và names
            names = []
            embeddings = []
            
            # Duyệt qua các thư mục người dùng
            user_dirs = [d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))]
            
            if not user_dirs:
                print("[FaceNetRecognizer] ERROR: No user directories found in dataset")
                return False
            
            for user_name in user_dirs:
                user_path = os.path.join(dataset_path, user_name)
                
                # Lấy tất cả ảnh trong thư mục user
                image_files = [f for f in os.listdir(user_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if len(image_files) < config.MIN_IMAGES_PER_PERSON:
                    print(f"[FaceNetRecognizer] WARNING: {user_name} has only {len(image_files)} images "
                          f"(minimum: {config.MIN_IMAGES_PER_PERSON})")
                    continue
                
                print(f"[FaceNetRecognizer] Processing {user_name}: {len(image_files)} images")
                
                # Process mỗi ảnh
                for image_file in image_files[:config.MAX_IMAGES_PER_PERSON]:
                    image_path = os.path.join(user_path, image_file)
                    
                    # Đọc ảnh
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Extract embedding
                    embedding = self.extract_embedding(image)
                    if embedding is not None:
                        names.append(user_name)
                        embeddings.append(embedding)
            
            if not embeddings:
                print("[FaceNetRecognizer] ERROR: No valid embeddings extracted")
                return False
            
            print(f"[FaceNetRecognizer] Total embeddings: {len(embeddings)}, "
                  f"Unique users: {len(set(names))}")
            
            # Lưu embeddings
            self.known_names = names
            self.known_embeddings = embeddings
            self.is_trained = True
            
            if self.database.save_facenet_embeddings(names, embeddings):
                print("[FaceNetRecognizer] ✓ Training completed and embeddings saved")
                return True
            else:
                print("[FaceNetRecognizer] WARNING: Training completed but failed to save embeddings")
                return False
            
        except Exception as e:
            print(f"[FaceNetRecognizer] ERROR during training: {e}")
            return False
    
    def load_embeddings(self) -> bool:
        """
        Load embeddings database từ file
        
        Returns:
            bool: True nếu load thành công
        """
        try:
            names, embeddings = self.database.load_facenet_embeddings()
            
            if names is None or embeddings is None:
                print("[FaceNetRecognizer] ERROR: Failed to load embeddings")
                return False
            
            self.known_names = names
            self.known_embeddings = embeddings
            self.is_trained = True
            
            print(f"[FaceNetRecognizer] ✓ Embeddings loaded: {len(names)} embeddings, "
                  f"{len(set(names))} unique users")
            return True
            
        except Exception as e:
            print(f"[FaceNetRecognizer] ERROR loading embeddings: {e}")
            return False
    
    def predict(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """
        Nhận diện khuôn mặt
        
        Args:
            face_roi: Vùng khuôn mặt (BGR format)
            
        Returns:
            Tuple[str, float]: (name, distance)
                - name: Tên người (hoặc "Unknown")
                - distance: Khoảng cách Euclidean (càng thấp càng giống)
        """
        if not self.is_trained or not self.known_embeddings:
            return config.UNKNOWN_PERSON_NAME, 1.0
        
        if self.model is None:
            return config.UNKNOWN_PERSON_NAME, 1.0
        
        try:
            # Extract embedding từ face
            embedding = self.extract_embedding(face_roi)
            if embedding is None:
                return config.UNKNOWN_PERSON_NAME, 1.0
            
            # Tính khoảng cách với tất cả embeddings đã biết
            min_distance = float('inf')
            best_match_name = config.UNKNOWN_PERSON_NAME
            
            for known_name, known_embedding in zip(self.known_names, self.known_embeddings):
                distance = euclidean(embedding, known_embedding)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match_name = known_name
            
            # Kiểm tra threshold
            if min_distance < self.distance_threshold:
                return best_match_name, min_distance
            else:
                return config.UNKNOWN_PERSON_NAME, min_distance
            
        except Exception as e:
            print(f"[FaceNetRecognizer] ERROR during prediction: {e}")
            return config.UNKNOWN_PERSON_NAME, 1.0
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        Cập nhật distance threshold
        
        Args:
            new_threshold: Ngưỡng mới
        """
        self.distance_threshold = new_threshold
        
        if config.DEBUG:
            print(f"[FaceNetRecognizer] Threshold updated to: {new_threshold}")
    
    def get_threshold(self) -> float:
        """Lấy distance threshold hiện tại"""
        return self.distance_threshold
    
    def get_user_list(self) -> list:
        """Lấy danh sách users đã train (unique)"""
        return list(set(self.known_names))
    
    def is_model_loaded(self) -> bool:
        """Kiểm tra FaceNet model đã được load chưa"""
        return self.model is not None
    
    def is_embeddings_loaded(self) -> bool:
        """Kiểm tra embeddings đã được load chưa"""
        return self.is_trained


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing FaceNet Recognizer...")
    print("=" * 50)
    
    if not KERAS_AVAILABLE:
        print("✗ TensorFlow/Keras not available")
        print("  Please install: pip install tensorflow keras")
        exit(1)
    
    recognizer = FaceNetRecognizer()
    
    # Test 1: Load FaceNet model
    print("\n1. Loading FaceNet model...")
    if recognizer.load_facenet_model():
        print("✓ FaceNet model loaded")
    else:
        print("✗ Failed to load FaceNet model")
        print("  Run train_facenet.py to download and setup the model")
        exit(1)
    
    # Test 2: Check if embeddings exist
    print("\n2. Checking for existing embeddings...")
    if recognizer.database.model_exists('facenet'):
        print("✓ Embeddings exist, loading...")
        if recognizer.load_embeddings():
            print(f"✓ Embeddings loaded successfully")
            print(f"  Users: {recognizer.get_user_list()}")
        else:
            print("✗ Failed to load embeddings")
    else:
        print("✗ No existing embeddings found")
        print("  Run train_facenet.py to create embeddings")
    
    # Test 3: Test prediction (nếu có embeddings)
    if recognizer.is_embeddings_loaded():
        print("\n3. Testing prediction with camera...")
        
        from camera import CameraManager
        from detector import FaceDetector
        
        with CameraManager() as camera:
            if camera.is_opened():
                detector = FaceDetector(method='dnn')  # DNN tốt hơn cho FaceNet
                
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
                        name, distance = recognizer.predict(face_roi)
                        
                        # Draw result
                        color = config.COLOR_SUCCESS if name != config.UNKNOWN_PERSON_NAME else config.COLOR_DENIED
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        label = f"{name} ({distance:.3f})"
                        cv2.putText(frame, label, (x, y-10),
                                  config.FONT_FACE, config.FONT_SCALE,
                                  color, config.FONT_THICKNESS)
                    
                    cv2.imshow("FaceNet Recognition Test", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cv2.destroyAllWindows()
                print("✓ Recognition test completed")
    
    print("=" * 50)
