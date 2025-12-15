import gradio as gr
import cv2
from PIL import Image
import time
from typing import Optional
import numpy as np
import os
import shutil
import datetime

from modules.camera import CameraManager
from modules.detector import FaceDetector
from modules.recognizer_lbph import LBPHRecognizer
from modules.recognizer_openface import OpenFaceRecognizer
from modules.recognizer_sface import SFaceRecognizer
from modules.database import Database
import config
import capture_dataset  # Import external capture logic

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
        self.latest_frame = None  # To store the current frame for capture
        self.latest_face_roi = None  # To store the current face for capture
        self.reload_recognition = (
            False  # Flag to force recognition loop to reset its cache
        )

        # We store these as simple values or standard python types,
        # but in Gradio they will be driven by component values.
        self.current_method = config.DEFAULT_RECOGNITION_METHOD
        self.current_detection = config.DEFAULT_DETECTION_METHOD

        # Initialize separate thresholds
        self.threshold_lbph = config.LBPH_CONFIDENCE_THRESHOLD
        self.threshold_openface = 0.6
        self.threshold_sface = 0.4

        # Set initial value based on default method
        if self.current_method == "lbph":
            self.threshold_value = self.threshold_lbph
        elif self.current_method == "openface":
            self.threshold_value = self.threshold_openface
        else:
            self.threshold_value = self.threshold_sface

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
            # Sync threshold
            self.recognizer_lbph.update_threshold(self.threshold_lbph)

        # Initialize OpenFace if available
        if FACE_RECOGNITION_AVAILABLE:
            self.recognizer_openface = OpenFaceRecognizer()
            if self.database.model_exists("openface"):
                self.recognizer_openface.load_encodings()
                # Sync threshold
                self.recognizer_openface.update_threshold(self.threshold_openface)

        # Initialize SFace if available
        if SFACE_RECOGNITION_AVAILABLE:
            self.recognizer_sface = SFaceRecognizer()
            if self.database.model_exists("sface"):
                self.recognizer_sface.load_embeddings()
                # Sync threshold
                self.recognizer_sface.update_threshold(self.threshold_sface)

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
                    # Initial Config based on Default Method (LBPH)
                    self.threshold_slider = gr.Slider(
                        minimum=0,
                        maximum=200,
                        value=self.threshold_lbph,
                        step=1,
                        label="LBPH Distance (Lower is Stricter)",
                    )

                    # Start/Stop Buttons
                    with gr.Row():
                        self.start_btn = gr.Button("Start", variant="primary")
                        self.stop_btn = gr.Button("Stop", variant="stop")

                    # User Management
                    gr.Markdown("#### User Management")
                    self.mgmt_name_input = gr.Textbox(
                        label="User Name",
                        placeholder="Enter name BEFORE clicking New/Update",
                    )
                    with gr.Row():
                        self.new_user_btn = gr.Button("New User")
                        self.delete_user_btn = gr.Button("Delete User")

                    with gr.Row():
                        self.update_user_btn = gr.Button("Update User")
                        self.train_btn = gr.Button("Train Models", variant="secondary")

            # --- Status Panel ---
            with gr.Row():
                self.status_text = gr.Textbox(
                    label="Status", value="Ready", interactive=False
                )
                self.fps_text = gr.Textbox(
                    label="FPS", value="FPS: 0", interactive=False
                )

            # --- Logs Panel ---
            with gr.Row():
                with gr.Accordion("Access Logs", open=False):
                    self.logs_refresh_btn = gr.Button("Refresh Logs")
                    # Using gr.Code for better monospace formatting of logs, or Textbox
                    self.logs_output = gr.Code(
                        label="Log Data", language=None, lines=20
                    )

            # --- Event Bindings ---

            # Start/Stop
            self.start_btn.click(
                fn=self._start_recognition,
                inputs=[self.method_radio, self.detection_radio, self.threshold_slider],
                outputs=[
                    self.video_feed,
                    self.status_text,
                    self.fps_text,
                    self.user_name_text,
                    self.user_last_access,
                    self.user_image_cam,
                    self.user_image_db,
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

            # User Management Actions
            # Launch Capture Window for New/Update
            self.new_user_btn.click(
                fn=self._launch_capture_window,
                inputs=[self.mgmt_name_input],
                outputs=[self.status_text, self.user_image_cam, self.user_image_db],
            )

            self.update_user_btn.click(
                fn=self._launch_capture_window,
                inputs=[self.mgmt_name_input],
                outputs=[self.status_text, self.user_image_cam, self.user_image_db],
            )

            self.delete_user_btn.click(
                fn=self._delete_user,
                inputs=[self.mgmt_name_input],
                outputs=[self.status_text],
            )

            # Train Models
            self.train_btn.click(fn=self._train_models, outputs=[self.status_text])

            # View Logs
            self.logs_refresh_btn.click(fn=self._view_logs, outputs=[self.logs_output])

        return demo

    def _start_recognition(self, method_name, detection_name, threshold):
        """Start the recognition loop generator"""
        # Validations
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

        # Update detector and threshold
        if self.detector:
            self.detector.switch_method(detection_name)

        # Initial threshold update
        self._on_threshold_change(threshold, method_name)

        # Loop
        yield from self._recognition_loop()

    def _stop_recognition(self):
        """Stop the loop"""
        self.is_running = False
        self.camera.release()
        return "Stopped"

    def _train_models(self):
        """Train all 3 models"""
        if self.is_running:
            self._stop_recognition()
            time.sleep(1.0)

        status_msg = "Training Results:\n"

        # 1. LBPH
        try:
            print("Training LBPH...")
            if self.recognizer_lbph.train(config.DATASET_DIR):
                self.recognizer_lbph.load_model()  # Reload
                status_msg += "✓ LBPH: Success\n"
            else:
                status_msg += "✗ LBPH: Failed\n"
        except Exception as e:
            status_msg += f"✗ LBPH Error: {e}\n"

        # 2. OpenFace
        if FACE_RECOGNITION_AVAILABLE:
            try:
                print("Training OpenFace...")
                if self.recognizer_openface.train(config.DATASET_DIR):
                    self.recognizer_openface.load_encodings()  # Reload
                    status_msg += "✓ OpenFace: Success\n"
                else:
                    status_msg += "✗ OpenFace: Failed\n"
            except Exception as e:
                status_msg += f"✗ OpenFace Error: {e}\n"
        else:
            status_msg += "- OpenFace: Not available\n"

        # 3. SFace
        if SFACE_RECOGNITION_AVAILABLE:
            try:
                print("Training SFace...")
                if self.recognizer_sface.train(config.DATASET_DIR):
                    self.recognizer_sface.load_embeddings()  # Reload
                    status_msg += "✓ SFace: Success\n"
                else:
                    status_msg += "✗ SFace: Failed\n"
            except Exception as e:
                status_msg += f"✗ SFace Error: {e}\n"
        else:
            status_msg += "- SFace: Not available\n"

        self.reload_recognition = True
        return status_msg

    def _launch_capture_window(self, name):
        """
        Launch the standalone capture window (capture_dataset.py logic)
        This stops the Main Loop first to free the camera.
        """
        # Debug print
        print(f"DEBUG: Launching capture window for {name}")

        if not name or name.strip() == "":
            return (
                "Error: Name cannot be empty. Please enter a name first.",
                gr.update(),
                gr.update(),
            )

        # 1. Stop current recognition loop logic if running
        if self.is_running:
            self.is_running = False
            self.camera.release()
            time.sleep(
                1.0
            )  # Wait for camera to be fully released by the generator thread
            # Note: The Gradio loop thread will see self.is_running=False and exit.

        try:
            # 2. Call the capture function from capture_dataset.py
            # This will open a CV2 window on the server/local machine.
            # Assuming defaults: 20 images, auto_detect=True
            success = capture_dataset.capture_images(
                name, num_images=50, auto_detect=True
            )

            if success:
                self.reload_recognition = True
                return (
                    f"Success: Captured images for '{name}'. Please Click START to resume recognition.",
                    None,
                    None,
                )
            else:
                return "Capture failed or cancelled.", None, None

        except Exception as e:
            print(f"DEBUG: Error in capture process: {e}")
            return f"Error executing capture: {e}", gr.update(), gr.update()

    def _delete_user(self, name):
        """Delete user directory"""
        if not name or name.strip() == "":
            return "Error: Name cannot be empty"

        user_dir = os.path.join(config.DATASET_DIR, name)
        if os.path.exists(user_dir):
            try:
                shutil.rmtree(user_dir)
                self.reload_recognition = True  # Also reload logic
                return f"Success: User '{name}' deleted."
            except Exception as e:
                return f"Error deleting user: {e}"
        else:
            return f"Error: User '{name}' does not exist."

    def _get_user_db_image(self, name):
        """Retrieve a representative image from the database for the user"""
        if name == config.UNKNOWN_PERSON_NAME:
            return None

        user_dir = os.path.join(config.DATASET_DIR, name)

        if os.path.exists(user_dir):
            images = [
                f
                for f in os.listdir(user_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            if images:
                # Pick the first one
                img_path = os.path.join(user_dir, images[0])
                # Read and convert to RGB
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    pass
        return None

    def _recognition_loop(self):
        """Main loop yielding frames"""
        self.frame_count = 0
        self.fps_start_time = time.time()

        # State tracking for optimization
        last_recognized_name = None
        cached_db_image = None

        while self.is_running:
            # Ensure camera is open (e.g. if restarted loop)
            if not self.camera.is_opened():
                try:
                    self.camera.open()
                except:
                    time.sleep(1)
                    continue

            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Store raw frame and reset roi
            self.latest_frame = frame.copy()
            self.latest_face_roi = None

            # Check forced reload (e.g. from New/Delete user)
            if self.reload_recognition:
                last_recognized_name = None
                cached_db_image = None
                self.reload_recognition = False

            # Detect faces
            try:
                faces = self.detector.detect_faces(frame)
            except Exception as e:
                faces = []

            # Defaults for this frame
            current_name = "Unknown"
            current_status = ""
            current_face_crop_rgb = None

            # Clear cache if no faces found
            if not faces:
                last_recognized_name = None
                cached_db_image = None

            for x, y, w, h in faces:
                # Extract Face ROI (BGR)
                face_roi = frame[y : y + h, x : x + w]
                self.latest_face_roi = face_roi.copy()  # Store for snapshot

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

                # Visualization matches Config
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

                # Update current info for GUI
                current_name = name
                current_status = f"Last Access: {time.strftime('%H:%M:%S')}"

                # Prepare LIVE face crop
                # Important: Gradio Image expects arrays. We must ensure this is a valid array.
                try:
                    current_face_crop_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                except:
                    current_face_crop_rgb = None

                # Update DB Image with Caching
                if name != last_recognized_name:
                    last_recognized_name = name
                    if name != config.UNKNOWN_PERSON_NAME:
                        cached_db_image = self._get_user_db_image(name)
                    else:
                        cached_db_image = None

                # Break after first face
                break

            # FPS
            self.frame_count += 1
            if self.frame_count >= config.FPS_UPDATE_INTERVAL:
                elapsed = time.time() - self.fps_start_time
                self.fps = int(self.frame_count / elapsed) if elapsed > 0 else 0
                self.frame_count = 0
                self.fps_start_time = time.time()

            # OSD
            cv2.putText(
                frame,
                f"FPS: {self.fps}",
                (10, 60),
                config.FONT_FACE,
                config.FONT_SCALE,
                config.COLOR_TEXT,
                config.FONT_THICKNESS,
            )
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

            # Convert main frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Yield outcomes
            yield (
                frame_rgb,
                (
                    "User recognized"
                    if current_name != config.UNKNOWN_PERSON_NAME
                    and current_name != "Unknown"
                    else "Scanning..."
                ),
                f"FPS: {self.fps}",
                f"Name: {current_name}",
                current_status,
                current_face_crop_rgb,
                cached_db_image,
            )

            # Sleep
            time.sleep(0.02)

    def _on_method_change(self, method):
        """Handle method change - Update Threshold Slider"""
        self.current_method = method

        status_msg = f"Method switched to {method}"

        # Return gr.update to change slider properties
        if method == "lbph":
            return (
                gr.update(
                    minimum=0,
                    maximum=200,
                    step=1,
                    value=self.threshold_lbph,
                    label="LBPH Distance (Lower is Stricter: ~50-80 Good)",
                ),
                status_msg,
            )

        elif method == "openface":
            return (
                gr.update(
                    minimum=0.1,
                    maximum=2.0,
                    step=0.05,
                    value=self.threshold_openface,
                    label="Euclidean Distance (Lower is Stricter: ~0.6 Default)",
                ),
                status_msg,
            )

        elif method == "sface":
            return (
                gr.update(
                    minimum=0.1,
                    maximum=2.0,
                    step=0.05,
                    value=self.threshold_sface,
                    label="Cosine Distance (Lower is Stricter: ~0.4 Default)",
                ),
                status_msg,
            )

        return gr.update(), status_msg

    def _on_detection_change(self, detection):
        """Handle detection change"""
        self.current_detection = detection
        if self.detector:
            self.detector.switch_method(detection)
        return f"Detection switched to {detection}"

    def _on_threshold_change(self, value, method):
        """Handle threshold change - Save to separate state variables"""
        threshold = float(value)

        # Update internal state based on current method
        if method == "lbph":
            self.threshold_lbph = threshold
            if self.recognizer_lbph:
                self.recognizer_lbph.update_threshold(threshold)

        elif method == "openface":
            self.threshold_openface = threshold
            if self.recognizer_openface:
                self.recognizer_openface.update_threshold(threshold)

        elif method == "sface":
            self.threshold_sface = threshold
            if self.recognizer_sface:
                self.recognizer_sface.update_threshold(threshold)

    def _view_logs(self):
        """Fetch logs and show"""
        logs = self.database.read_access_logs(limit=50)
        if not logs:
            return gr.update(value="No logs found.")

        log_text = "Recent Access Logs (Latest 50):\n" + "=" * 50 + "\n"
        for log in reversed(logs):
            log_text += f"{log['timestamp']} | {log['name']} | {log['method']} | {log['status']}\n"

        return gr.update(value=log_text)

    def launch(self):
        self.demo.launch()


if __name__ == "__main__":
    app = GradioMainWindow()
    app.launch()
