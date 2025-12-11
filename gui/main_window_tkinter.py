"""
Face Access Control - Main Window GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox
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
        """Tạo giao diện GUI"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Video display panel
        self._create_video_panel(main_frame)

        # Control panel
        self._create_control_panel(main_frame)


        # View user panel
        self._view_assess_user_(main_frame)

        # Status panel
        self._create_status_panel(main_frame)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)

    def _create_video_panel(self, parent):
        """Tạo video display panel"""
        video_frame = ttk.LabelFrame(parent, text="Video Feed", padding="10")
        video_frame.grid(
            row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S)
        )

        # Video label
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()

        # Placeholder image
        self._show_placeholder()

    def _create_new_user_(self):
        """Tạo người dùng mới"""
        pass
    
    def _delete_user_(self):
        """Xóa người dùng"""
        pass
    
    def _update_user_(self):
        """Cập nhật người dùng"""
        pass

    def _create_control_panel(self, parent):
        """Tạo control panel"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(
            row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S)
        )

        # Recognition method selection
        ttk.Label(control_frame, text="Recognition Method:").pack(pady=5)

        lbph_radio = ttk.Radiobutton(
            control_frame,
            text="LBPH (Fast)",
            variable=self.current_method,
            value="lbph",
            command=self._on_method_change,
        )
        lbph_radio.pack(anchor=tk.W)

        openface_radio = ttk.Radiobutton(
            control_frame,
            text="OpenFace (Accurate)",
            variable=self.current_method,
            value="openface",
            command=self._on_method_change,
        )
        openface_radio.pack(anchor=tk.W)

        sface_radio = ttk.Radiobutton(
            control_frame,
            text="SFace (Accurate)",
            variable=self.current_method,
            value="sface",
            command=self._on_method_change,
        )
        sface_radio.pack(anchor=tk.W)

        if not FACE_RECOGNITION_AVAILABLE:
            openface_radio.config(state="disabled")

        if not SFACE_RECOGNITION_AVAILABLE:
            sface_radio.config(state="disabled")

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)

        # Detection method selection
        ttk.Label(control_frame, text="Detection Method:").pack(pady=5)

        haar_radio = ttk.Radiobutton(
            control_frame,
            text="Haar Cascade",
            variable=self.current_detection,
            value="haar",
            command=self._on_detection_change,
        )
        haar_radio.pack(anchor=tk.W)

        dnn_radio = ttk.Radiobutton(
            control_frame,
            text="DNN",
            variable=self.current_detection,
            value="dnn",
            command=self._on_detection_change,
        )
        dnn_radio.pack(anchor=tk.W)

        yunet_radio = ttk.Radiobutton(
            control_frame,
            text="YuNet (Fast & Accurate)",
            variable=self.current_detection,
            value="yunet",
            command=self._on_detection_change,
        )
        yunet_radio.pack(anchor=tk.W)

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)

        # Threshold adjustment
        ttk.Label(control_frame, text="Threshold:").pack(pady=5)

        self.threshold_label = ttk.Label(
            control_frame, text=f"{self.threshold_var.get():.1f}"
        )
        self.threshold_label.pack()

        threshold_scale = ttk.Scale(
            control_frame,
            from_=0,
            to=100,
            variable=self.threshold_var,
            orient="horizontal",
            command=self._on_threshold_change,
        )
        threshold_scale.pack(fill="x", pady=5)

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)

        # Start/Stop buttons
        self.start_button = ttk.Button(
            control_frame, text="Start", command=self._start_recognition
        )
        self.start_button.pack(fill="x", pady=5)

        self.stop_button = ttk.Button(
            control_frame, text="Stop", command=self._stop_recognition, state="disabled"
        )
        self.stop_button.pack(fill="x", pady=5)

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)

        # below control panel
        user_frame = ttk.LabelFrame(control_frame, text="User", padding="10")
        user_frame.pack(fill="x", pady=5)
        """Tạo người dùng mới"""
        self.new_user_frame = ttk.Button(user_frame, text="New User", padding="10", command=self._create_new_user_)
        self.new_user_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        """Xóa người dùng"""
        self.delete_user_frame = ttk.Button(user_frame, text="Delete User", padding="10", command=self._delete_user_)
        self.delete_user_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        """Cập nhật người dùng"""
        self.update_user_frame = ttk.Button(user_frame, text="Update User", padding="10", command=self._update_user_)
        self.update_user_frame.grid(row=0, column=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # View logs button
        self.logs_button = ttk.Button(control_frame, text="View Access Logs", command=self._view_logs)
        self.logs_button.pack(fill="x", pady=5)

    # left window    
    def _view_assess_user_(self, parent):
        """Hiển thị thông tin người dùng"""
        user_frame = ttk.LabelFrame(parent, text="User", padding="10")
        user_frame.grid(
            row=0, column=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S)
        )

    def _create_status_panel(self, parent):
        """Tạo status panel"""
        status_frame = ttk.LabelFrame(parent, text="Status", padding="10")
        status_frame.grid(
            row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E)
        )

        self.status_label = ttk.Label(status_frame, text="Ready", foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.fps_label = ttk.Label(status_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.RIGHT, padx=10)

    def _show_placeholder(self):
        """Hiển thị placeholder image"""
        # Tạo placeholder image
        placeholder = Image.new("RGB", (640, 480), color=(50, 50, 50))
        photo = ImageTk.PhotoImage(placeholder)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def _initialize_components(self):
        """Khởi tạo các components"""
        # Initialize camera
        self.camera = CameraManager()

        # Initialize detector
        self.detector = FaceDetector(method=self.current_detection.get())

        # Initialize recognizers
        self.recognizer_lbph = LBPHRecognizer()

        # Load LBPH model nếu có
        if self.database.model_exists("lbph"):
            if self.recognizer_lbph.load_model():
                self._update_status("LBPH model loaded", "green")

        # Initialize OpenFace nếu có
        if FACE_RECOGNITION_AVAILABLE:
            self.recognizer_openface = OpenFaceRecognizer()
            if self.database.model_exists("openface"):
                # Load OpenFace encodings
                if self.recognizer_openface.load_encodings():
                    self._update_status("OpenFace encodings loaded", "green")

        # Initialize SFace nếu có
        if SFACE_RECOGNITION_AVAILABLE:
            self.recognizer_sface = SFaceRecognizer()
            # SFace uses sface embeddings storage
            if self.database.model_exists("sface"):
                if self.recognizer_sface.load_embeddings():
                    self._update_status("SFace embeddings loaded", "green")

    def _start_recognition(self):
        """Bắt đầu recognition"""
        # Kiểm tra model đã train chưa
        method = self.current_method.get()

        if method == "lbph":
            if not self.recognizer_lbph.is_model_trained():
                messagebox.showerror(
                    "Error", "LBPH model not trained!\nPlease run train_lbph.py first."
                )
                return

        elif method == "openface":
            if not FACE_RECOGNITION_AVAILABLE:
                messagebox.showerror("Error", "face_recognition library not available!")
                return
            if not self.recognizer_openface.is_encodings_loaded():
                messagebox.showerror(
                    "Error",
                    "OpenFace encodings not trained!\nPlease run train_openface.py first.",
                )
                return

        elif method == "sface":
            if not SFACE_RECOGNITION_AVAILABLE:
                messagebox.showerror("Error", "sface library not available!")
                return
            if not self.recognizer_sface.is_embeddings_loaded():
                messagebox.showerror(
                    "Error",
                    "SFace embeddings not trained!\nPlease run train_sface.py first.",
                )
                return

        # Mở camera
        if not self.camera.open():
            messagebox.showerror("Error", "Failed to open camera!")
            return

        # Start recognition thread
        self.is_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self._update_status("Running...", "blue")

        # Start recognition loop trong thread riêng
        recognition_thread = threading.Thread(
            target=self._recognition_loop, daemon=True
        )
        recognition_thread.start()

    def _stop_recognition(self):
        """Dừng recognition"""
        self.is_running = False
        self.camera.release()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self._update_status("Stopped", "orange")
        self._show_placeholder()

    def _recognition_loop(self):
        """Main recognition loop"""
        while self.is_running:
            # Read frame
            ret, frame = self.camera.read()
            if not ret:
                continue

            # Detect faces
            faces = self.detector.detect_faces(frame)

            # Recognize each face
            method = self.current_method.get()

            for x, y, w, h in faces:
                face_roi = frame[y : y + h, x : x + w]

                # Recognize
                if method == "lbph":
                    name, score = self.recognizer_lbph.predict(face_roi)
                elif method == "openface":
                    name, score = self.recognizer_openface.predict(face_roi)
                else:  # sface
                    name, score = self.recognizer_sface.predict(face_roi)

                # Determine access status
                is_granted = name != config.UNKNOWN_PERSON_NAME
                color = config.COLOR_SUCCESS if is_granted else config.COLOR_DENIED
                status = "GRANTED" if is_granted else "DENIED"

                # Draw bounding box
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), color, config.BBOX_THICKNESS
                )

                # Draw label
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

                # Log access (với cooldown)
                current_time = time.time()
                if (
                    name not in self.last_access_time
                    or (current_time - self.last_access_time[name])
                    > config.ACCESS_COOLDOWN
                ):
                    self.database.log_access(name, method.upper(), score, status)
                    self.last_access_time[name] = current_time

            # Draw info
            info_text = f"{method.upper()} | {self.current_detection.get().upper()}"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                config.FONT_FACE,
                config.FONT_SCALE,
                config.COLOR_TEXT,
                config.FONT_THICKNESS,
            )

            # Calculate and draw FPS
            self._update_fps()
            cv2.putText(
                frame,
                f"FPS: {self.fps}",
                (10, 60),
                config.FONT_FACE,
                config.FONT_SCALE,
                config.COLOR_TEXT,
                config.FONT_THICKNESS,
            )

            # Convert frame to PhotoImage và display
            self._display_frame(frame)

    def _display_frame(self, frame):
        """Display frame lên GUI"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)

        # Update label
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1

        if self.frame_count >= config.FPS_UPDATE_INTERVAL:
            elapsed = time.time() - self.fps_start_time
            self.fps = int(self.frame_count / elapsed)

            # Update FPS label
            self.fps_label.config(text=f"FPS: {self.fps}")

            # Reset counter
            self.frame_count = 0
            self.fps_start_time = time.time()

    def _update_status(self, message: str, color: str = "black"):
        """Update status label"""
        self.status_label.config(text=message, foreground=color)

    def _on_method_change(self):
        """Callback khi thay đổi recognition method"""
        method = self.current_method.get()

        # Update threshold range
        if method == "lbph":
            self.threshold_var.set(config.LBPH_CONFIDENCE_THRESHOLD)
            # LBPH: 0-100
        else:  # openface or sface
            self.threshold_var.set(0.6)  # Default threshold
            # OpenFace/SFace: 0-1

        self._update_status(f"Method: {method.upper()}", "blue")

    def _on_detection_change(self):
        """Callback khi thay đổi detection method"""
        if self.detector:
            self.detector.switch_method(self.current_detection.get())
        self._update_status(
            f"Detection: {self.current_detection.get().upper()}", "blue"
        )

    def _on_threshold_change(self, value):
        """Callback khi thay đổi threshold"""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.1f}")

        # Update recognizer threshold
        method = self.current_method.get()
        if method == "lbph" and self.recognizer_lbph:
            self.recognizer_lbph.update_threshold(threshold)
        elif method == "openface" and self.recognizer_openface:
            self.recognizer_openface.update_threshold(threshold)
        elif method == "sface" and self.recognizer_sface:
            self.recognizer_sface.update_threshold(threshold)

    def _view_logs(self):
        """Hiển thị access logs"""
        logs = self.database.read_access_logs(limit=50)

        # Tạo window mới
        log_window = tk.Toplevel(self.root)
        log_window.title("Access Logs")
        log_window.geometry("800x400")

        # Tạo text widget với scrollbar
        frame = ttk.Frame(log_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text = tk.Text(frame, yscrollcommand=scrollbar.set, wrap=tk.WORD)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=text.yview)

        # Display logs
        if logs:
            text.insert(tk.END, "Recent Access Logs (Latest 50):\n")
            text.insert(tk.END, "=" * 80 + "\n\n")

            for log in reversed(logs):  # Hiển thị mới nhất trước
                text.insert(tk.END, f"Time: {log['timestamp']}\n")
                text.insert(tk.END, f"Name: {log['name']}\n")
                text.insert(tk.END, f"Method: {log['method']}\n")
                text.insert(tk.END, f"Confidence: {log['confidence']}\n")
                text.insert(tk.END, f"Status: {log['status']}\n")
                text.insert(tk.END, "-" * 80 + "\n\n")
        else:
            text.insert(tk.END, "No access logs found.")

        text.config(state="disabled")  # Read-only

    def on_closing(self):
        """Callback khi đóng window"""
        if self.is_running:
            self._stop_recognition()

        self.root.destroy()


# ==================== MAIN ====================


def main():
    """Main function"""
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
