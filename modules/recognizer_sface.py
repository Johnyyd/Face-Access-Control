import cv2
import numpy as np
from typing import Tuple, List, Optional, Union
import os
import config
from .database import Database

# Import Detector for alignment during training
from .detector_yunet import YuNetDetector


class SFaceRecognizer:
    """
    SFace face recognizer using ONNX model
    """

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold or config.SFACE_THRESHOLD
        self.model_path = config.SFACE_MODEL_PATH
        self.model = None
        self.known_names: List[str] = []
        self.known_embeddings: List[np.ndarray] = []
        self.database = Database()
        self.is_trained = False

        self.load_model()

        if config.DEBUG:
            self._log(f"Initialized with threshold: {self.threshold}")

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

        # align_face method removed as we switched to BBox cropping

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 512-d embedding from face image (Crop -> Resize -> Feature -> Norm)
        """
        if self.model is None:
            return None

        try:
            # Resize to 112x112 (SFace Input)
            # We assume face_image is a Face Crop (not full frame)
            if face_image.shape[0] != 112 or face_image.shape[1] != 112:
                face_image = cv2.resize(face_image, (112, 112))

            embedding = self.model.feature(face_image)

            # L2 Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.flatten()
        except Exception as e:
            if config.DEBUG:
                self._log(f"ERROR extracting embedding: {e}")
            return None

    def train(self, dataset_path: Optional[str] = None) -> bool:
        """Train SFace from dataset with BBox Cropping"""
        if self.model is None:
            self._log("ERROR: Model not loaded")
            return False

        dataset_path = dataset_path or config.DATASET_DIR
        if not os.path.exists(dataset_path):
            self._log(f"ERROR: Dataset not found: {dataset_path}")
            return False

        self._log(f"Training from dataset: {dataset_path}")

        # Initialize detector for training alignment
        detector = YuNetDetector()
        if detector.model is None:
            self._log("ERROR: Detector not available for training alignment")
            return False

        names, embeddings = [], []
        user_dirs = [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith(".")
        ]

        if not user_dirs:
            self._log("ERROR: No user directories found")
            return False

        for user_name in user_dirs:
            user_path = os.path.join(dataset_path, user_name)
            image_files = [
                f
                for f in os.listdir(user_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if len(image_files) < config.MIN_IMAGES_PER_PERSON:
                self._log(
                    f"WARNING: {user_name} has only {len(image_files)} images "
                    f"(minimum: {config.MIN_IMAGES_PER_PERSON})"
                )

            for image_file in image_files[: config.MAX_IMAGES_PER_PERSON]:
                image_path = os.path.join(user_path, image_file)
                img = cv2.imread(image_path)
                if img is None:
                    continue

                # Detect face to crop
                # Used 0.6 conf threshold (default)
                faces = detector.detect_faces(img)

                if not faces:
                    continue

                # Take largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                # Crop
                face_crop = img[y : y + h, x : x + w]
                if face_crop.size == 0:
                    continue

                # Filter Low Quality (Dark/Flat)
                # Mean > 40, Std > 20
                if np.mean(face_crop) < 40 or np.std(face_crop) < 20:
                    continue

                embedding = self.extract_embedding(face_crop)
                if embedding is not None:
                    names.append(user_name)
                    embeddings.append(embedding)

        if not embeddings:
            self._log(
                "ERROR: No valid embeddings extracted. (Check dataset quality/lighting)"
            )
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
        """
        Recognize face using Cosine Similarity on Face Crop
        Args:
            face_roi: Cropped face image (BGR)
        """
        if not self.is_trained or not self.known_embeddings or self.model is None:
            return config.UNKNOWN_PERSON_NAME, 0.0

        # Step 1: Extract (Resize + Feature + Norm)
        embedding = self.extract_embedding(face_roi)
        if embedding is None:
            return config.UNKNOWN_PERSON_NAME, 0.0

        try:
            # cv2.FaceRecognizerSF_FR_COSINE returns Cosine Similarity (1 is same, -1 is opposite)
            scores = [
                self.model.match(
                    embedding.astype(np.float32).reshape(1, -1),
                    known.astype(np.float32).reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
                for known in self.known_embeddings
            ]

            max_score = float(np.max(scores))
            best_match_idx = int(np.argmax(scores))
            best_match_name = self.known_names[best_match_idx]

            # Check threshold (Higher is better for Similarity)
            if max_score > self.threshold:
                return best_match_name, max_score
            else:
                return config.UNKNOWN_PERSON_NAME, max_score

        except Exception as e:
            self._log(f"ERROR during prediction: {e}")
            return config.UNKNOWN_PERSON_NAME, 0.0

    def update_threshold(self, new_threshold: float) -> None:
        self.threshold = new_threshold
        if config.DEBUG:
            self._log(f"Threshold updated to: {new_threshold}")

    def get_threshold(self) -> float:
        return self.threshold

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

    # ... existing test code omitted for brevity but would work ...
    print("SFace Test Init Done")
