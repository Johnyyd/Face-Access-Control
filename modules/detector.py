# Face detection module (Haar/DNN)
import cv2
from config import MODEL_DETECTOR

class FaceDetector:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            MODEL_DETECTOR,
            "",
            (320, 320)
        )

    def detect(self, img):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        faces = self.detector.detect(img)
        if faces[1] is None:
            return []
        boxes = []
        for f in faces[1]:
            x, y, w, h = f[:4].astype(int)
            boxes.append([x, y, w, h])
        return boxes
