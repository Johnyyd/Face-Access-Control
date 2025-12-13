import gradio as gr
import cv2
from PIL import Image
import time
from typing import Optional
import numpy as np

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


class GradioMainWindow:
    """Main GUI window for Face Access Control (Gradio Version)"""

    def __init__(self):
        # Components
        self.camera: Optional[CameraManager] = None
        self.detector: Optional[FaceDetector] = None
        self.recognizer_lbph: Optional[LBPHRecognizer] = None
        self.recognizer_openface: Optional[OpenFaceRecognizer] = None
        self.recognizer_sface: Optional[SFaceRecognizer] = None
        self.database = Database()

        # State
        self.is_running = False

        # We store these as simple values or standard python types,
        # but in Gradio they will be driven by component values.
        self.current_method = config.DEFAULT_RECOGNITION_METHOD
        self.current_detection = config.DEFAULT_DETECTION_METHOD
        self.threshold_value = config.LBPH_CONFIDENCE_THRESHOLD

        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # Last access tracking
        self.last_access_time = {}

        # Initialize components
        self._initialize_components()

        # Create GUI
        self.demo = self._create_gui()

    def _initialize_components(self):
        """Initialize components similar to Tkinter version"""
        # Initialize camera
        self.camera = CameraManager()

        # Initialize detector
        self.detector = FaceDetector(method=self.current_detection)

        # Initialize recognizers
        self.recognizer_lbph = LBPHRecognizer()

        # Load LBPH model if exists
        if self.database.model_exists("lbph"):
            self.recognizer_lbph.load_model()

        # Initialize OpenFace if available
        if FACE_RECOGNITION_AVAILABLE:
            self.recognizer_openface = OpenFaceRecognizer()
            if self.database.model_exists("openface"):
                self.recognizer_openface.load_encodings()

        # Initialize SFace if available
        if SFACE_RECOGNITION_AVAILABLE:
            self.recognizer_sface = SFaceRecognizer()
            if self.database.model_exists("sface"):
                self.recognizer_sface.load_embeddings()

    def _create_gui(self):
        """Create Gradio GUI layout"""
        with gr.Blocks(title=config.WINDOW_TITLE) as demo:
            gr.Markdown(f"# {config.WINDOW_TITLE}")

            with gr.Row():
                # --- Column 1: Video Feed ---
                with gr.Column(scale=3):
                    self.video_feed = gr.Image(label="Video Feed", streaming=True)

                # --- Column 2: User Info ---
                with gr.Column(scale=1):
                    gr.Markdown("### User Information")
                    self.user_image_cam = gr.Image(
                        label="User from Camera", interactive=False
                    )
                    self.user_name_text = gr.Textbox(label="Name", value="Name: ")
                    self.user_last_access = gr.Textbox(
                        label="Last Access Time", value="Last Access Time: "
                    )
                    self.user_image_db = gr.Image(
                        label="User from Database", interactive=False
                    )

                # --- Column 3: Controls ---
                with gr.Column(scale=1):
                    gr.Markdown("### Controls")

                    # Recognition Method
                    self.method_radio = gr.Radio(
                        choices=["lbph", "openface", "sface"],
                        value=self.current_method,
                        label="Recognition Method",
                        info="Select recognition algorithm",
                    )

                    # Detection Method
                    self.detection_radio = gr.Radio(
                        choices=["haar", "dnn", "yunet"],
                        value=self.current_detection,
                        label="Detection Method",
                    )

                    # Threshold
                    self.threshold_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=self.threshold_value,
                        label="Threshold",
                    )

                    # Start/Stop Buttons
                    with gr.Row():
                        self.start_btn = gr.Button("Start", variant="primary")
                        self.stop_btn = gr.Button("Stop", variant="stop")

                    # User Management (Placeholders)
                    gr.Markdown("#### User Management")
                    with gr.Row():
                        self.new_user_btn = gr.Button("New User")
                        self.delete_user_btn = gr.Button("Delete User")
                    self.update_user_btn = gr.Button("Update User")

                    self.logs_btn = gr.Button("View Access Logs")

            # --- Status Panel ---
            with gr.Row():
                self.status_text = gr.Textbox(
                    label="Status", value="Ready", interactive=False
                )
                self.fps_text = gr.Textbox(
                    label="FPS", value="FPS: 0", interactive=False
                )

            # --- Logs Panel (Hidden by default, shown on click) ---
            self.logs_output = gr.TextArea(label="Access Logs", visible=False)

            # --- Event Bindings ---

            # Start/Stop
            # Start triggers the generator loop
            # Output targets: video_feed, status_text, fps_text, user info fields
            self.start_btn.click(
                fn=self._start_recognition,
                inputs=[self.method_radio, self.detection_radio, self.threshold_slider],
                outputs=[
                    self.video_feed,
                    self.status_text,
                    self.fps_text,
                    self.user_name_text,
                    self.user_last_access,
                ],
            )

            self.stop_btn.click(fn=self._stop_recognition, outputs=[self.status_text])

            # Method change updates
            self.method_radio.change(
                fn=self._on_method_change,
                inputs=[self.method_radio],
                outputs=[self.threshold_slider, self.status_text],
            )

            # Detection change updates
            self.detection_radio.change(
                fn=self._on_detection_change,
                inputs=[self.detection_radio],
                outputs=[self.status_text],
            )

            # Threshold change
            self.threshold_slider.change(
                fn=self._on_threshold_change,
                inputs=[self.threshold_slider, self.method_radio],
            )

            # View Logs
            self.logs_btn.click(fn=self._view_logs, outputs=[self.logs_output])

        return demo

    def _start_recognition(self, method_name, detection_name, threshold):
        """Start the recognition loop generator"""
        # Validations similar to Tkinter
        self.current_method = method_name
        self.current_detection = detection_name

        # Check models
        if method_name == "lbph" and not self.recognizer_lbph.is_model_trained():
            raise gr.Error("LBPH model not trained! Please run train_lbph.py first.")

        if method_name == "openface":
            if not FACE_RECOGNITION_AVAILABLE:
                raise gr.Error("face_recognition library not available!")
            if not self.recognizer_openface.is_encodings_loaded():
                raise gr.Error("OpenFace encodings not trained!")

        if method_name == "sface":
            if not SFACE_RECOGNITION_AVAILABLE:
                raise gr.Error("sface library not available!")
            if not self.recognizer_sface.is_embeddings_loaded():
                raise gr.Error("SFace embeddings not trained!")

        # Open Camera
        if not self.camera.open():
            raise gr.Error("Failed to open camera!")

        self.is_running = True

        # Update detector and threshold before loop
        if self.detector:
            self.detector.switch_method(detection_name)

        # Initial threshold update
        self._on_threshold_change(threshold, method_name)

        # Loop
        return self._recognition_loop()

    def _stop_recognition(self):
        """Stop the loop"""
        self.is_running = False
        self.camera.release()
        return "Stopped"

    def _recognition_loop(self):
        """Main loop yielding frames"""
        self.frame_count = 0
        self.fps_start_time = time.time()

        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                continue

            # Detect faces
            try:
                faces = self.detector.detect_faces(frame)
            except Exception as e:
                # If detection fails, just show frame
                faces = []

            current_name = "Unknown"
            current_status = ""

            for x, y, w, h in faces:
                face_roi = frame[y : y + h, x : x + w]

                # Recognize
                name = config.UNKNOWN_PERSON_NAME
                score = 0.0

                if self.current_method == "lbph":
                    name, score = self.recognizer_lbph.predict(face_roi)
                elif self.current_method == "openface":
                    name, score = self.recognizer_openface.predict(face_roi)
                else:  # sface
                    name, score = self.recognizer_sface.predict(face_roi)

                # Determine access status
                is_granted = name != config.UNKNOWN_PERSON_NAME
                color = config.COLOR_SUCCESS if is_granted else config.COLOR_DENIED
                status_str = "GRANTED" if is_granted else "DENIED"

                # Visualization on frame
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), color, config.BBOX_THICKNESS
                )
                label = f"{name} ({score:.2f})"
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    config.FONT_FACE,
                    config.FONT_SCALE,
                    color,
                    config.FONT_THICKNESS,
                )

                # Log access
                current_time = time.time()
                if (
                    name not in self.last_access_time
                    or (current_time - self.last_access_time[name])
                    > config.ACCESS_COOLDOWN
                ):
                    self.database.log_access(
                        name, self.current_method.upper(), score, status_str
                    )
                    self.last_access_time[name] = current_time

                # Update current info for GUI side panel (just taking the last face found)
                current_name = name
                current_status = f"Last Access: {time.strftime('%H:%M:%S')}"

            # FPS
            self.frame_count += 1
            if self.frame_count >= config.FPS_UPDATE_INTERVAL:
                elapsed = time.time() - self.fps_start_time
                self.fps = int(self.frame_count / elapsed) if elapsed > 0 else 0
                self.frame_count = 0
                self.fps_start_time = time.time()

            cv2.putText(
                frame,
                f"FPS: {self.fps}",
                (10, 60),
                config.FONT_FACE,
                config.FONT_SCALE,
                config.COLOR_TEXT,
                config.FONT_THICKNESS,
            )

            # Info Text on Frame
            info_text = (
                f"{self.current_method.upper()} | {self.current_detection.upper()}"
            )
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                config.FONT_FACE,
                config.FONT_SCALE,
                config.COLOR_TEXT,
                config.FONT_THICKNESS,
            )

            # Convert to RGB for Gradio
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Yield updates
            # Outputs: video_feed, status_text, fps_text, user_name_text, user_last_access
            yield (
                frame_rgb,
                "Running...",
                f"FPS: {self.fps}",
                f"Name: {current_name}",
                current_status,
            )

            # Small sleep to prevent tight loop if needed, but grab() usually blocks enough
            time.sleep(0.01)


            # outputs=[
            #         self.video_feed,
            #         self.status_text,
            #         self.fps_text,
            #         self.user_name_text,
            #         self.user_last_access,
            #     ],
            # return frame, status_str, fps_str, current_name, current_status

    def _on_method_change(self, method):
        """Handle method change"""
        self.current_method = method
        val = config.LBPH_CONFIDENCE_THRESHOLD if method == "lbph" else 0.6
        return val, f"Method switched to {method}"

    def _on_detection_change(self, detection):
        """Handle detection change"""
        self.current_detection = detection
        if self.detector:
            self.detector.switch_method(detection)
        return f"Detection switched to {detection}"

    def _on_threshold_change(self, value, method):
        """Handle threshold change"""
        threshold = float(value)
        if method == "lbph" and self.recognizer_lbph:
            self.recognizer_lbph.update_threshold(threshold)
        elif method == "openface" and self.recognizer_openface:
            self.recognizer_openface.update_threshold(threshold)
        elif method == "sface" and self.recognizer_sface:
            self.recognizer_sface.update_threshold(threshold)

    def _view_logs(self):
        """Fetch logs and show"""
        logs = self.database.read_access_logs(limit=50)
        if not logs:
            return gr.update(value="No logs found.", visible=True)

        log_text = "Recent Access Logs (Latest 50):\n" + "=" * 50 + "\n"
        for log in reversed(logs):
            log_text += f"{log['timestamp']} | {log['name']} | {log['method']} | {log['status']}\n"

        return gr.update(value=log_text, visible=True)

    def launch(self):
        self.demo.launch()


if __name__ == "__main__":
    app = GradioMainWindow()
    app.launch()
