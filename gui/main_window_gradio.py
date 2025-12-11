import gradio as gr
import cv2
from PIL import Image, ImageTk
import time
import threading
from typing import Optional

from modules.camera import CameraManager
from modules.detector import FaceDetector
from modules.recognizer_lbph import LBPHRecognizer
from modules.recognizer_openface import OpenFaceRecognizer
from modules.recognizer_sface import SFaceRecognizer
from modules.database import Database
import config

# Check if face_recognition is available
try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

# Check if sface is available
SFACE_RECOGNITION_AVAILABLE = True  # Assume available since we have the module


class MainWindow:
    """Main GUI window for Face Access Control"""

    def __init__(self, root):
        pass


    def _create_gui(self):
        pass