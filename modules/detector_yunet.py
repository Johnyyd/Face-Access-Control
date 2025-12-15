import cv2
import os

class YuNetDetector:
    def __init__(self, model_path, input_size=(320, 320),
                 score_threshold=0.9, nms_threshold=0.3):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            input_size,
            score_threshold,
            nms_threshold
        )
        self.input_size = input_size

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces
