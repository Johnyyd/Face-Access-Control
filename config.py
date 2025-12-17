# config.py
# Cấu hình chung cho hệ thống Face Access Control

# ==================== CAMERA SETTINGS ====================
CAMERA_ID = 0               # 0 cho camera mặc định, 1 cho camera ngoài
CAMERA_WIDTH = 640          # Chiều rộng frame
CAMERA_HEIGHT = 480         # Chiều cao frame
CAMERA_FPS = 30             # Frames per second

# ==================== DEBUG SETTINGS ====================
DEBUG = True                # Chế độ debug (True/False)

# ==================== DETECTION SETTINGS ====================
# DNN Model paths
DNN_PROTOTXT = "models/deploy.prototxt"
DNN_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

# Haar Cascade path
HAAR_CASCADE = "models/haarcascade_frontalface_default.xml"

# Detection settings
CONFIDENCE_THRESHOLD = 0.7  # Ngưỡng tin cậy cho DNN (0-1)
MIN_FACE_SIZE = (30, 30)    # Kích thước khuôn mặt tối thiểu cho Haar

# ==================== DATABASE SETTINGS ====================
DB_PATH = "database/faces.db"

# ==================== DISPLAY SETTINGS ====================
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# ==================== SYSTEM PATHS ====================
# Tự động tạo các thư mục cần thiết
import os

def create_directories():
    """Tạo các thư mục cần thiết nếu chưa tồn tại"""
    directories = [
        "models",
        "database", 
        "captures",
        "logs"
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        if DEBUG:
            print(f"[Config] Created/verified directory: {dir_name}")

# Gọi hàm tạo thư mục
create_directories()

# In thông báo khi import
if DEBUG:
    print(f"[Config] Module imported successfully")
    print(f"[Config] CAMERA_WIDTH = {CAMERA_WIDTH}")
    print(f"[Config] CAMERA_HEIGHT = {CAMERA_HEIGHT}")
    print(f"[Config] CAMERA_FPS = {CAMERA_FPS}")