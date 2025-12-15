"""
SFace Face Recognition Module
Sử dụng SFace ONNX model từ OpenCV Zoo
"""

import cv2
import numpy as np
from typing import Tuple, List
import os
import config
from .database import Database


class SFaceRecognizer:
    """
    SFace face recognizer using ONNX model

    Attributes:
        model: SFace recognition model
        known_names: List of registered names
        known_embeddings: List of 512-d embeddings
        distance_threshold: Cosine distance threshold (default 0.4)
    """

    def __init__(self, distance_threshold: float = None):
        """
        Initialize SFace recognizer

        Args:
            distance_threshold: Cosine distance threshold (default 0.4)
        """
        self.distance_threshold = (
            distance_threshold if distance_threshold is not None else 0.4
        )
        self.model_path = config.SFACE_MODEL_PATH
        self.model = None
        self.known_names: List[str] = []
        self.known_embeddings: List[np.ndarray] = []
        self.database = Database()
        self.is_trained = False

        # Load model
        self.load_model()

        if config.DEBUG:
            print(
                f"[SFaceRecognizer] Initialized with threshold: {self.distance_threshold}"
            )

    def load_model(self) -> bool:
        """
        Load SFace ONNX model

        Returns:
            bool: True if loaded successfully
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"[SFaceRecognizer] ERROR: Model not found: {self.model_path}")
                print("[SFaceRecognizer] Run: python download_models.py")
                return False

            # Load SFace model
            self.model = cv2.FaceRecognizerSF.create(self.model_path, "")

            if config.DEBUG:
                print(f"[SFaceRecognizer] [OK] Model loaded: {self.model_path}")

            return True

        except Exception as e:
            print(f"[SFaceRecognizer] ERROR loading model: {e}")
            return False

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract 512-d embedding from face image

        Args:
            face_image: Face ROI (BGR)

        Returns:
            512-d embedding vector
        """
        if self.model is None:
            return None

        try:
            # Align and resize face
            aligned_face = cv2.resize(face_image, (112, 112))

            # Extract feature
            embedding = self.model.feature(aligned_face)

            return embedding.flatten()

        except Exception as e:
            if config.DEBUG:
                print(f"[SFaceRecognizer] ERROR extracting embedding: {e}")
            return None

    def train(self, dataset_path: str = None) -> bool:
        """
        Train SFace from dataset

        Args:
            dataset_path: Path to dataset directory

        Returns:
            bool: True if training successful
        """
        if self.model is None:
            print("[SFaceRecognizer] ERROR: Model not loaded")
            return False

        try:
            dataset_path = dataset_path or config.DATASET_DIR

            if not os.path.exists(dataset_path):
                print(f"[SFaceRecognizer] ERROR: Dataset not found: {dataset_path}")
                return False

            print(f"[SFaceRecognizer] Training from dataset: {dataset_path}")

            names = []
            embeddings = []

            # Get user directories
            user_dirs = [
                d
                for d in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, d))
                and not d.startswith(".")
            ]

            if not user_dirs:
                print("[SFaceRecognizer] ERROR: No user directories found")
                return False

            # Process each user
            for user_name in user_dirs:
                user_path = os.path.join(dataset_path, user_name)

                # Get image files
                image_files = [
                    f
                    for f in os.listdir(user_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                if len(image_files) < config.MIN_IMAGES_PER_PERSON:
                    print(
                        f"[SFaceRecognizer] WARNING: {user_name} has only {len(image_files)} images "
                        f"(minimum: {config.MIN_IMAGES_PER_PERSON})"
                    )
                    continue

                print(
                    f"[SFaceRecognizer] Processing {user_name}: {len(image_files)} images"
                )

                # Process each image
                for image_file in image_files[: config.MAX_IMAGES_PER_PERSON]:
                    image_path = os.path.join(user_path, image_file)

                    try:
                        # Load image
                        img = cv2.imread(image_path)
                        if img is None:
                            continue

                        # Extract embedding
                        embedding = self.extract_embedding(img)

                        if embedding is not None:
                            names.append(user_name)
                            embeddings.append(embedding)

                    except Exception as img_error:
                        print(
                            f"[SFaceRecognizer] WARNING: Skipping {image_file}: {img_error}"
                        )

            if not embeddings:
                print("[SFaceRecognizer] ERROR: No valid embeddings extracted")
                return False

            print(
                f"[SFaceRecognizer] Total embeddings: {len(embeddings)}, "
                f"Unique users: {len(set(names))}"
            )

            # Save embeddings
            self.known_names = names
            self.known_embeddings = embeddings
            self.is_trained = True

            # Save to database (reuse openface storage)
            if self.database.save_embeddings(names, embeddings):
                print("[SFaceRecognizer] [OK] Training completed and embeddings saved")
                return True
            else:
                print(
                    "[SFaceRecognizer] WARNING: Training completed but failed to save"
                )
                return False

        except Exception as e:
            print(f"[SFaceRecognizer] ERROR during training: {e}")
            return False

    def load_embeddings(self) -> bool:
        """
        Load embeddings from file

        Returns:
            bool: True if loaded successfully
        """
        try:
            names, embeddings = self.database.load_embeddings()

            if names is None or embeddings is None:
                print("[SFaceRecognizer] ERROR: Failed to load embeddings")
                return False

            self.known_names = names
            self.known_embeddings = embeddings
            self.is_trained = True

            print(
                f"[SFaceRecognizer] [OK] Embeddings loaded: {len(names)} encodings, "
                f"{len(set(names))} unique users"
            )
            return True

        except Exception as e:
            print(f"[SFaceRecognizer] ERROR loading embeddings: {e}")
            return False

    def predict(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """
        Recognize face

        Args:
            face_roi: Face ROI (BGR)

        Returns:
            Tuple[str, float]: (name, distance)
        """
        if not self.is_trained or not self.known_embeddings:
            return config.UNKNOWN_PERSON_NAME, 1.0

        if self.model is None:
            return config.UNKNOWN_PERSON_NAME, 1.0

        try:
            # Extract embedding
            embedding = self.extract_embedding(face_roi)

            if embedding is None:
                return config.UNKNOWN_PERSON_NAME, 1.0

            # Calculate cosine distances
            distances = []
            for known_embedding in self.known_embeddings:
                # Cosine similarity
                distance = self.model.match(
                    embedding.astype(np.float32).reshape(1, -1),
                    known_embedding.astype(np.float32).reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
                # if config.DEBUG:
                #     print(
                #         f"[DEBUG] SFace match: dist={distance}, const={cv2.FaceRecognizerSF_FR_COSINE}"
                #     )
                distances.append(distance)

            # Find best match
            min_distance = float(np.min(distances))
            best_match_idx = int(np.argmin(distances))
            best_match_name = self.known_names[best_match_idx]

            # Check threshold
            if min_distance < self.distance_threshold:
                return best_match_name, min_distance
            else:
                return config.UNKNOWN_PERSON_NAME, min_distance

        except Exception as e:
            print(f"[SFaceRecognizer] ERROR during prediction: {e}")
            return config.UNKNOWN_PERSON_NAME, 1.0

    def update_threshold(self, new_threshold: float) -> None:
        """Update distance threshold"""
        self.distance_threshold = new_threshold
        if config.DEBUG:
            print(f"[SFaceRecognizer] Threshold updated to: {new_threshold}")

    def get_threshold(self) -> float:
        """Get current threshold"""
        return self.distance_threshold

    def get_user_list(self) -> list:
        """Get list of trained users"""
        return list(set(self.known_names))

    def is_embeddings_loaded(self) -> bool:
        """Check if embeddings are loaded"""
        return self.is_trained

    def delete_user(self, name: str) -> bool:
        """
        Xóa user khỏi bộ nhớ và database

        Args:
            name: Tên user cần xóa

        Returns:
            bool: True nếu xóa thành công
        """
        try:
            if not self.known_names:
                return False

            # Filter out indices needed to keep
            indices_to_keep = [i for i, n in enumerate(self.known_names) if n != name]

            if len(indices_to_keep) == len(self.known_names):
                print(f"[SFaceRecognizer] User '{name}' not found in embeddings")
                return False

            # Update lists
            new_names = [self.known_names[i] for i in indices_to_keep]
            new_embeddings = [self.known_embeddings[i] for i in indices_to_keep]

            self.known_names = new_names
            self.known_embeddings = new_embeddings

            # Save to disk
            if self.database.save_embeddings(new_names, new_embeddings):
                print(f"[SFaceRecognizer] User '{name}' deleted from embeddings")
                return True
            else:
                print(
                    f"[SFaceRecognizer] ERROR: Failed to save embeddings after deletion"
                )
                return False

        except Exception as e:
            print(f"[SFaceRecognizer] ERROR deleting user: {e}")
            return False


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing SFace Recognizer...")
    print("=" * 50)

    recognizer = SFaceRecognizer()

    if recognizer.model is None:
        print("[X] Failed to load model")
        print("Run: python download_models.py")
        exit(1)

    print("[OK] Model loaded successfully")
    print(f"  Threshold: {recognizer.distance_threshold}")

    # Check for existing embeddings
    print("\nChecking for existing embeddings...")
    if recognizer.database.model_exists("sface"):
        print("[OK] Embeddings exist, loading...")
        if recognizer.load_embeddings():
            print(f"[OK] Loaded successfully")
            print(f"  Users: {recognizer.get_user_list()}")
        else:
            print("[X] Failed to load embeddings")
    else:
        print("[X] No existing embeddings found")
        print("  Run: python train_sface.py")

    print("=" * 50)
