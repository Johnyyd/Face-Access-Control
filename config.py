"""
Face Access Control - Configuration File
Chứa tất cả các cấu hình cho hệ thống nhận diện khuôn mặt
"""

import os

# ==================== ĐƯỜNG DẪN CƠ BẢN ====================

# Thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thư mục dataset chứa ảnh training
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Thư mục models chứa các file model
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Thư mục logs chứa access logs
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ==================== CẤU HÌNH CAMERA ====================

# Camera ID (0 = webcam mặc định, 1 = camera ngoài)
CAMERA_ID = 0

# Độ phân giải camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# FPS mong muốn
CAMERA_FPS = 30

# ==================== CẤU HÌNH FACE DETECTION ====================

# Phương pháp detection mặc định: 'yunet'
DEFAULT_DETECTION_METHOD = "yunet"

# SFace Model paths
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "sface/face_recognition_sface_2021dec.onnx")
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "yunet/face_detection_yunet_2023mar.onnx")
SFACE_EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "sface/embeddings.pkl")

# SFace Parameters
SFACE_EMBEDDING_SIZE = 512  # SFace tạo vector 512 chiều
SFACE_DISTANCE_THRESHOLD = 0.4  # Cosine distance threshold (lower = stricter)

# ==================== CẤU HÌNH RECOGNITION CHUNG ====================

# Phương pháp recognition mặc định: 'sface'
DEFAULT_RECOGNITION_METHOD = "sface"

# Tên hiển thị cho unknown person
UNKNOWN_PERSON_NAME = "Unknown"

# ==================== CẤU HÌNH TRAINING ====================

# Số lượng ảnh tối thiểu mỗi người khi training
MIN_IMAGES_PER_PERSON = 10

# Số lượng ảnh tối đa mỗi người (để tránh overfitting)
MAX_IMAGES_PER_PERSON = 100

# ==================== CẤU HÌNH LOGGING ====================

# Access log file path
ACCESS_LOG_PATH = os.path.join(LOGS_DIR, "access_log.csv")

# Log format
LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Có log mỗi frame hay không (False = chỉ log khi có access granted/denied)
LOG_EVERY_FRAME = False

# ==================== CẤU HÌNH GUI ====================

# Window title
WINDOW_TITLE = "Face Access Control System"

# Video display size
VIDEO_DISPLAY_WIDTH = 800
VIDEO_DISPLAY_HEIGHT = 600

# Colors (BGR format)
COLOR_SUCCESS = (0, 255, 0)  # Xanh lá - Access granted
COLOR_DENIED = (0, 0, 255)  # Đỏ - Access denied
COLOR_UNKNOWN = (0, 165, 255)  # Cam - Unknown person
COLOR_TEXT = (255, 255, 255)  # Trắng - Text

# Font settings
FONT_FACE = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Bounding box thickness
BBOX_THICKNESS = 2

# FPS display
SHOW_FPS = True
FPS_UPDATE_INTERVAL = 30  # Update FPS mỗi 30 frames

# ==================== CẤU HÌNH ACCESS CONTROL ====================

# Thời gian cooldown giữa các lần access (giây)
# Tránh log liên tục cho cùng 1 người
ACCESS_COOLDOWN = 3.0

# Có tự động mở cửa không (simulation)
AUTO_UNLOCK = True

# Thời gian mở cửa (giây)
UNLOCK_DURATION = 2.0

# ==================== CẤU HÌNH DEBUG ====================

# Debug mode
DEBUG = True

# Hiển thị confidence/distance score
SHOW_CONFIDENCE = True

# Hiển thị detection method
SHOW_METHOD = True

# Lưu ảnh detected faces
SAVE_DETECTED_FACES = False
DETECTED_FACES_DIR = os.path.join(BASE_DIR, "detected_faces")

# ==================== HELPER FUNCTIONS ====================


def create_directories():
    """Tạo các thư mục cần thiết nếu chưa tồn tại"""
    directories = [
        DATASET_DIR,
        MODELS_DIR,
        LOGS_DIR,
    ]

    if SAVE_DETECTED_FACES:
        directories.append(DETECTED_FACES_DIR)

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def validate_config():
    """Kiểm tra tính hợp lệ của config"""
    errors = []

    # Kiểm tra detection method
    if DEFAULT_DETECTION_METHOD not in ["yunet"]:
        errors.append("DEFAULT_DETECTION_METHOD must be 'yunet'")

    # Kiểm tra recognition method
    if DEFAULT_RECOGNITION_METHOD not in ["sface"]:
        errors.append("DEFAULT_RECOGNITION_METHOD must be 'sface'")

    if not (0 < SFACE_DISTANCE_THRESHOLD < 2):
        errors.append("SFACE_DISTANCE_THRESHOLD should be between 0 and 2")

    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


# ==================== INITIALIZATION ====================

if __name__ == "__main__":
    print("Face Access Control - Configuration")
    print("=" * 50)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    print("=" * 50)
    print(f"Default Detection Method: {DEFAULT_DETECTION_METHOD}")
    print(f"Default Recognition Method: {DEFAULT_RECOGNITION_METHOD}")

    print(f"SFace Threshold: {SFACE_DISTANCE_THRESHOLD}")
    print("=" * 50)

    # Tạo thư mục
    create_directories()

    # Validate config
    if validate_config():
        print("Configuration is valid")
    else:
        print("Configuration has errors")
