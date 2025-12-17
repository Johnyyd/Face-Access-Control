# config.py
# Cấu hình chung cho hệ thống

# Camera settings
CAMERA_ID = 0  # 0 cho camera mặc định, 1 cho camera ngoài

# DNN Model paths
DNN_PROTOTXT = "models/deploy.prototxt"
DNN_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

# Haar Cascade path
HAAR_CASCADE = "models/haarcascade_frontalface_default.xml"

# Database settings
DB_PATH = "database/faces.db"

# Detection settings
CONFIDENCE_THRESHOLD = 0.7  # Ngưỡng tin cậy cho DNN
MIN_FACE_SIZE = (30, 30)    # Kích thước khuôn mặt tối thiểu

# Display settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600