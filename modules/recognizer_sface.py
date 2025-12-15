import cv2
import numpy as np
from typing import Tuple, List, Optional
import os
import config
from .database import Database


class SFaceRecognizer:
    """
    SFace face recognizer using ONNX model
    """

    def __init__(self, distance_threshold: Optional[float] = None):
        self.distance_threshold = distance_threshold or 0.4
        self.model_path = config.SFACE_MODEL_PATH
        self.model = None
        self.known_names: List[str] = []
        self.known_embeddings: List[np.ndarray] = []
        self.database = Database()
        self.is_trained = False

        self.load_model()

        if config.DEBUG:
            self._log(f"Initialized with threshold: {self.distance_threshold}")

    def _log(self, msg: str) -> None:
        """Internal logging helper"""
        print(f"[SFaceRecognizer] {msg}")

    def load_model(self) -> bool:
        """Load SFace ONNX model"""
        if not os.path.exists(self.model_path):
            self._log(f"ERROR: Model not found: {self.model_path}")
            self._log("Run: python download_models.py")
            return False

        try:
            self.model = cv2.FaceRecognizerSF.create(self.model_path, "")
            if config.DEBUG:
                self._log(f"[OK] Model loaded: {self.model_path}")
            return True
        except Exception as e:
            self._log(f"ERROR loading model: {e}")
            return False

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512-d embedding from face image"""
        if self.model is None:
            return None

        try:
            aligned_face = cv2.resize(face_image, (112, 112))
            embedding = self.model.feature(aligned_face)
            return embedding.flatten()
        except Exception as e:
            if config.DEBUG:
                self._log(f"ERROR extracting embedding: {e}")
            return None

    def train(self, dataset_path: Optional[str] = None) -> bool:
        """Train SFace from dataset"""
        if self.model is None:
            self._log("ERROR: Model not loaded")
            return False

        dataset_path = dataset_path or config.DATASET_DIR
        if not os.path.exists(dataset_path):
            self._log(f"ERROR: Dataset not found: {dataset_path}")
            return False

        self._log(f"Training from dataset: {dataset_path}")

        names, embeddings = [], []
        user_dirs = [
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith(".")
        ]

        if not user_dirs:
            self._log("ERROR: No user directories found")
            return False

        for user_name in user_dirs:
            user_path = os.path.join(dataset_path, user_name)
            image_files = [
                f for f in os.listdir(user_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if len(image_files) < config.MIN_IMAGES_PER_PERSON:
                self._log(
                    f"WARNING: {user_name} has only {len(image_files)} images "
                    f"(minimum: {config.MIN_IMAGES_PER_PERSON})"
                )
                continue

            self._log(f"Processing {user_name}: {len(image_files)} images")

            for image_file in image_files[: config.MAX_IMAGES_PER_PERSON]:
                image_path = os.path.join(user_path, image_file)
                img = cv2.imread(image_path)
                if img is None:
                    continue

                embedding = self.extract_embedding(img)
                if embedding is not None:
                    names.append(user_name)
                    embeddings.append(embedding)

        if not embeddings:
            self._log("ERROR: No valid embeddings extracted")
            return False

        self._log(
            f"Total embeddings: {len(embeddings)}, Unique users: {len(set(names))}"
        )

        self.known_names = names
        self.known_embeddings = embeddings
        self.is_trained = True

        if self.database.save_embeddings(names, embeddings):
            self._log("[OK] Training completed and embeddings saved")
            return True

        self._log("WARNING: Training completed but failed to save")
        return False

    def load_embeddings(self) -> bool:
        """Load embeddings from file"""
        try:
            names, embeddings = self.database.load_embeddings()
            if not names or not embeddings:
                self._log("ERROR: Failed to load embeddings")
                return False

            self.known_names = names
            self.known_embeddings = embeddings
            self.is_trained = True

            self._log(
                f"[OK] Embeddings loaded: {len(names)} encodings, "
                f"{len(set(names))} unique users"
            )
            return True
        except Exception as e:
            self._log(f"ERROR loading embeddings: {e}")
            return False

    def predict(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Recognize face"""
        if not self.is_trained or not self.known_embeddings or self.model is None:
            return config.UNKNOWN_PERSON_NAME, 1.0

        embedding = self.extract_embedding(face_roi)
        if embedding is None:
            return config.UNKNOWN_PERSON_NAME, 1.0

        try:
            distances = [
                self.model.match(
                    embedding.astype(np.float32).reshape(1, -1),
                    known.astype(np.float32).reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
                for known in self.known_embeddings
            ]

            min_distance = float(np.min(distances))
            best_match_idx = int(np.argmin(distances))
            best_match_name = self.known_names[best_match_idx]

            return (
                best_match_name if min_distance < self.distance_threshold
                else config.UNKNOWN_PERSON_NAME,
                min_distance,
            )
        except Exception as e:
            self._log(f"ERROR during prediction: {e}")
            return config.UNKNOWN_PERSON_NAME, 1.0

    def update_threshold(self, new_threshold: float) -> None:
        self.distance_threshold = new_threshold
        if config.DEBUG:
            self._log(f"Threshold updated to: {new_threshold}")

    def get_threshold(self) -> float:
        return self.distance_threshold

    def get_user_list(self) -> List[str]:
        return list(set(self.known_names))

    def is_embeddings_loaded(self) -> bool:
        return self.is_trained

    def delete_user(self, name: str) -> bool:
        """Xóa user khỏi bộ nhớ và database"""
        if not self.known_names:
            return False

        indices_to_keep = [i for i, n in enumerate(self.known_names) if n != name]
        if len(indices_to_keep) == len(self.known_names):
            self._log(f"User '{name}' not found in embeddings")
            return False

        self.known_names = [self.known_names[i] for i in indices_to_keep]
        self.known_embeddings = [self.known_embeddings[i] for i in indices_to_keep]

        if self.database.save_embeddings(self.known_names, self.known_embeddings):
            self._log(f"User '{name}' deleted from embeddings")
            return True

        self._log("ERROR: Failed to save embeddings after deletion")
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

    print("\nChecking for existing embeddings...")
    if recognizer.database.model_exists("sface"):
        print("[OK] Embeddings exist, loading...")
        if recognizer.load_embeddings():
            print("[OK] Loaded successfully")
            print(f"  Users: {recognizer.get_user_list()}")
        else:
            print("[X] Failed to load embeddings")
    else:
        print("[X] No existing embeddings found")
        print("  Run: python train_sface.py")

    print("=" * 50)
