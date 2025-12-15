"""
YuNet Face Detector Module
Sử dụng YuNet ONNX model từ OpenCV Zoo
"""

import cv2
import numpy as np
from typing import List, Tuple
import os
import config


class YuNetDetector:
    """
    YuNet face detector using ONNX model

    Attributes:
        model: YuNet detector model
        input_size: Input size for the model (320x320)
        conf_threshold: Confidence threshold (default 0.6)
        nms_threshold: NMS threshold (default 0.3)
    """

    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ):
        """
        Initialize YuNet detector

        Args:
            model_path: Path to YuNet ONNX model
            conf_threshold: Confidence threshold for detection
            nms_threshold: NMS threshold for filtering
        """
        self.model_path = model_path or config.YUNET_MODEL_PATH
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (320, 320)
        self.model = None

        # Load model
        self.load_model()

    def load_model(self) -> bool:
        """
        Load YuNet ONNX model

        Returns:
            bool: True if loaded successfully
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"[YuNetDetector] ERROR: Model not found: {self.model_path}")
                print("[YuNetDetector] Run: python download_models.py")
                return False

            # Create YuNet detector
            self.model = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                self.input_size,
                self.conf_threshold,
                self.nms_threshold,
            )

            if config.DEBUG:
                print(f"[YuNetDetector] [OK] Model loaded: {self.model_path}")

            return True

        except Exception as e:
            print(f"[YuNetDetector] ERROR loading model: {e}")
            return False

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame

        Args:
            frame: Input frame (BGR)

        Returns:
            List of (x, y, w, h) tuples
        """
        if self.model is None:
            return []

        try:
            # Set input size based on frame size
            h, w = frame.shape[:2]
            self.model.setInputSize((w, h))

            # Detect faces
            _, faces = self.model.detect(frame)

            if faces is None:
                return []

            # Convert to (x, y, w, h) format
            results = []
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                # Ensure coordinates are within frame
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                results.append((x, y, w, h))

            return results

        except Exception as e:
            if config.DEBUG:
                print(f"[YuNetDetector] ERROR during detection: {e}")
            return []

    def detect_with_landmarks(self, frame: np.ndarray) -> List[dict]:
        """
        Detect faces with 5 facial landmarks

        Args:
            frame: Input frame (BGR)

        Returns:
            List of dicts with 'bbox' and 'landmarks'
        """
        if self.model is None:
            return []

        try:
            h, w = frame.shape[:2]
            self.model.setInputSize((w, h))

            _, faces = self.model.detect(frame)

            if faces is None:
                return []

            results = []
            for face in faces:
                # Bounding box
                x, y, w, h = face[:4].astype(int)

                # 5 landmarks: right eye, left eye, nose, right mouth, left mouth
                landmarks = face[4:14].reshape(5, 2).astype(int)

                results.append(
                    {
                        "bbox": (x, y, w, h),
                        "landmarks": landmarks,
                        "confidence": float(face[14]),
                    }
                )

            return results

        except Exception as e:
            if config.DEBUG:
                print(f"[YuNetDetector] ERROR during landmark detection: {e}")
            return []


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing YuNet Detector...")
    print("=" * 50)

    detector = YuNetDetector()

    if detector.model is None:
        print("[X] Failed to load model")
        print("Run: python download_models.py")
        exit(1)

    print("[OK] Model loaded successfully")
    print(f"  Confidence threshold: {detector.conf_threshold}")
    print(f"  NMS threshold: {detector.nms_threshold}")

    # Test with camera
    print("\nTesting with camera...")
    print("Press 'q' to quit")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detector.detect_faces(frame)

        # Draw bounding boxes
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display count
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YuNet Detector Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[OK] Test completed")
    print("=" * 50)
