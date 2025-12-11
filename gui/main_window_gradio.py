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
        """
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title(config.WINDOW_TITLE)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Components
        self.camera: Optional[CameraManager] = None
        self.detector: Optional[FaceDetector] = None
        self.recognizer_lbph: Optional[LBPHRecognizer] = None
        self.recognizer_openface: Optional[OpenFaceRecognizer] = None
        self.recognizer_sface: Optional[SFaceRecognizer] = None
        self.database = Database()

        # State
        self.is_running = False
        self.current_method = tk.StringVar(value=config.DEFAULT_RECOGNITION_METHOD)
        self.current_detection = tk.StringVar(value=config.DEFAULT_DETECTION_METHOD)
        self.threshold_var = tk.DoubleVar(value=config.LBPH_CONFIDENCE_THRESHOLD)

        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # Last access tracking (để tránh log liên tục)
        self.last_access_time = {}

        # Create GUI
        self._create_gui()

        # Initialize components
        self._initialize_components()


    def _create_gui(self):
        # Main window
        gr.Blocks(
            # 
            gr.Row(
                [
                    gr.Column(
                        [
                            
                        ]
                    )
                ]
            )
        )
