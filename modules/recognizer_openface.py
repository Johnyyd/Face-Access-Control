"""
Face Access Control - OpenFace Recognition Module
Nhận diện khuôn mặt sử dụng OpenFace (via face_recognition library)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import os
import config
from .database import Database

# Import face_recognition (OpenFace wrapper)
try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("[OpenFaceRecognizer] WARNING: face_recognition not available")
    print("[OpenFaceRecognizer] Please install: pip install face-recognition")
    FACE_RECOGNITION_AVAILABLE = False


class OpenFaceRecognizer:
    """
    Class nhận diện khuôn mặt sử dụng OpenFace (dlib-based)

    Attributes:
        known_names: List tên người đã đăng ký
        known_encodings: List face encodings (128-d vectors)
        distance_threshold: Ngưỡng khoảng cách (mặc định 0.6)
        database: Database manager
    """

    def __init__(self, distance_threshold: float = None):
        """
        Khởi tạo OpenFace Recognizer

        Args:
            distance_threshold: Ngưỡng distance (mặc định 0.6)
        """
        self.distance_threshold = (
            distance_threshold if distance_threshold is not None else 0.6
        )

        self.known_names: List[str] = []
        self.known_encodings: List[np.ndarray] = []
        self.database = Database()
        self.is_trained = False

        if not FACE_RECOGNITION_AVAILABLE:
            print(
                "[OpenFaceRecognizer] ERROR: Cannot initialize without face_recognition"
            )

        if config.DEBUG:
            print(
                f"[OpenFaceRecognizer] Initialized with threshold: {self.distance_threshold}"
            )

    def train(self, dataset_path: str = None) -> bool:
        """
        Train OpenFace từ dataset (tạo face encodings)

        Args:
            dataset_path: Đường dẫn thư mục dataset (mặc định từ config)

        Returns:
            bool: True nếu train thành công
        """
        if not FACE_RECOGNITION_AVAILABLE:
            print("[OpenFaceRecognizer] ERROR: face_recognition not available")
            return False

        try:
            dataset_path = dataset_path or config.DATASET_DIR

            if not os.path.exists(dataset_path):
                print(f"[OpenFaceRecognizer] ERROR: Dataset not found: {dataset_path}")
                return False

            print(f"[OpenFaceRecognizer] Training from dataset: {dataset_path}")

            # Collect encodings và names
            names = []
            encodings = []

            # Duyệt qua các thư mục người dùng
            user_dirs = [
                d
                for d in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, d))
                and not d.startswith(".")
            ]

            if not user_dirs:
                print(
                    "[OpenFaceRecognizer] ERROR: No user directories found in dataset"
                )
                return False

            for user_name in user_dirs:
                user_path = os.path.join(dataset_path, user_name)

                # Lấy tất cả ảnh trong thư mục user
                image_files = [
                    f
                    for f in os.listdir(user_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                if len(image_files) < config.MIN_IMAGES_PER_PERSON:
                    print(
                        f"[OpenFaceRecognizer] WARNING: {user_name} has only {len(image_files)} images "
                        f"(minimum: {config.MIN_IMAGES_PER_PERSON})"
                    )
                    continue

                print(
                    f"[OpenFaceRecognizer] Processing {user_name}: {len(image_files)} images"
                )

                # Process mỗi ảnh
                for image_file in image_files[: config.MAX_IMAGES_PER_PERSON]:
                    image_path = os.path.join(user_path, image_file)

                    try:
                        # Load ảnh
                        image = face_recognition.load_image_file(image_path)

                        # Get face encodings
                        face_encodings = face_recognition.face_encodings(image)

                        if len(face_encodings) > 0:
                            # Lấy encoding đầu tiên
                            encoding = face_encodings[0]
                            names.append(user_name)
                            encodings.append(encoding)
                    except Exception as img_error:
                        print(
                            f"[OpenFaceRecognizer] WARNING: Skipping {image_file}: {img_error}"
                        )

            if not encodings:
                print("[OpenFaceRecognizer] ERROR: No valid encodings extracted")
                return False

            print(
                f"[OpenFaceRecognizer] Total encodings: {len(encodings)}, "
                f"Unique users: {len(set(names))}"
            )

            # Lưu encodings
            self.known_names = names
            self.known_encodings = encodings
            self.is_trained = True

            # Lưu vào database (sử dụng format tương tự OpenFace)
            if self.database.save_openface_embeddings(names, encodings):
                print(
                    "[OpenFaceRecognizer] [OK] Training completed and encodings saved"
                )
                return True
            else:
                print(
                    "[OpenFaceRecognizer] WARNING: Training completed but failed to save encodings"
                )
                return False

        except Exception as e:
            print(f"[OpenFaceRecognizer] ERROR during training: {e}")
            return False

    def load_encodings(self) -> bool:
        """
        Load encodings từ file

        Returns:
            bool: True nếu load thành công
        """
        try:
            names, encodings = self.database.load_openface_embeddings()

            if names is None or encodings is None:
                print("[OpenFaceRecognizer] ERROR: Failed to load encodings")
                return False

            self.known_names = names
            self.known_encodings = encodings
            self.is_trained = True

            print(
                f"[OpenFaceRecognizer] [OK] Encodings loaded: {len(names)} encodings, "
                f"{len(set(names))} unique users"
            )
            return True

        except Exception as e:
            print(f"[OpenFaceRecognizer] ERROR loading encodings: {e}")
            return False

    def predict(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """
        Nhận diện khuôn mặt

        Args:
            face_roi: Vùng khuôn mặt (BGR format)

        Returns:
            Tuple[str, float]: (name, distance)
                - name: Tên người (hoặc "Unknown")
                - distance: Khoảng cách (càng thấp càng giống)
        """
        if not self.is_trained or not self.known_encodings:
            return config.UNKNOWN_PERSON_NAME, 1.0

        if not FACE_RECOGNITION_AVAILABLE:
            return config.UNKNOWN_PERSON_NAME, 1.0

        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb)

            if len(face_encodings) == 0:
                return config.UNKNOWN_PERSON_NAME, 1.0

            face_encoding = face_encodings[0]

            # Compare với known encodings
            distances = face_recognition.face_distance(
                self.known_encodings, face_encoding
            )

            # Tìm match tốt nhất
            min_distance = float(np.min(distances))
            best_match_idx = int(np.argmin(distances))
            best_match_name = self.known_names[best_match_idx]

            # Kiểm tra threshold
            if min_distance < self.distance_threshold:
                return best_match_name, min_distance
            else:
                return config.UNKNOWN_PERSON_NAME, min_distance

        except Exception as e:
            print(f"[OpenFaceRecognizer] ERROR during prediction: {e}")
            return config.UNKNOWN_PERSON_NAME, 1.0

    def update_threshold(self, new_threshold: float) -> None:
        """
        Cập nhật distance threshold

        Args:
            new_threshold: Ngưỡng mới
        """
        self.distance_threshold = new_threshold

        if config.DEBUG:
            print(f"[OpenFaceRecognizer] Threshold updated to: {new_threshold}")

    def get_threshold(self) -> float:
        """Lấy distance threshold hiện tại"""
        return self.distance_threshold

    def get_user_list(self) -> list:
        """Lấy danh sách users đã train (unique)"""
        return list(set(self.known_names))

    def is_encodings_loaded(self) -> bool:
        """Kiểm tra encodings đã được load chưa"""
        return self.is_trained


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing OpenFace Recognizer...")
    print("=" * 50)

    if not FACE_RECOGNITION_AVAILABLE:
        print("✗ face_recognition not available")
        print("  Please install: pip install face-recognition")
        exit(1)

    recognizer = OpenFaceRecognizer()

    # Test: Check if encodings exist
    print("\nChecking for existing encodings...")
    if recognizer.database.model_exists("openface"):  # Reuse same storage
        print("✓ Encodings exist, loading...")
        if recognizer.load_encodings():
            print(f"✓ Encodings loaded successfully")
            print(f"  Users: {recognizer.get_user_list()}")
        else:
            print("✗ Failed to load encodings")
    else:
        print("✗ No existing encodings found")
        print("  Run train_openface.py to create encodings")

    print("=" * 50)
