# check_config.py
# Script kiểm tra cấu hình hệ thống

import os
import sys
import importlib

def check_system():
    """Kiểm tra toàn bộ hệ thống"""
    print("=" * 60)
    print("KIỂM TRA HỆ THỐNG FACE ACCESS CONTROL")
    print("=" * 60)
    
    # 1. Kiểm tra thư mục hiện tại
    print("\n📁 1. THÔNG TIN THƯ MỤC:")
    current_dir = os.getcwd()
    print(f"   Thư mục hiện tại: {current_dir}")
    print(f"   Files trong thư mục:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"     📂 {item}/")
        else:
            print(f"     📄 {item}")
    
    # 2. Kiểm tra Python path
    print("\n🐍 2. PYTHON PATH:")
    for i, path in enumerate(sys.path[:5]):  # Hiển thị 5 path đầu
        print(f"   [{i}] {path}")
    
    # 3. Kiểm tra file config.py
    print("\n⚙️ 3. KIỂM TRA FILE CONFIG.PY:")
    config_path = os.path.join(current_dir, 'config.py')
    
    if os.path.exists(config_path):
        print(f"   ✓ File config.py tồn tại tại: {config_path}")
        
        # Đọc và hiển thị nội dung
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"   📄 Nội dung ({len(lines)} dòng):")
                print("   " + "-" * 40)
                for i, line in enumerate(lines[:20], 1):  # Hiển thị 20 dòng đầu
                    print(f"   {i:3}: {line.rstrip()}")
                if len(lines) > 20:
                    print(f"   ... và {len(lines)-20} dòng nữa")
        except Exception as e:
            print(f"   ✗ Lỗi đọc file: {e}")
    else:
        print(f"   ✗ File config.py KHÔNG tồn tại!")
        print(f"   Tạo file mới tại: {config_path}")
    
    # 4. Thử import config
    print("\n🔧 4. THỬ IMPORT CONFIG:")
    try:
        # Thêm thư mục hiện tại vào path nếu cần
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        import config
        print(f"   ✓ Import config thành công từ: {config.__file__}")
        
        # Kiểm tra các thuộc tính trong config
        print("\n   📋 CÁC THUỘC TÍNH TRONG CONFIG:")
        required_attrs = [
            'CAMERA_ID', 'CAMERA_WIDTH', 'CAMERA_HEIGHT', 'CAMERA_FPS',
            'DEBUG', 'DNN_PROTOTXT', 'DNN_MODEL', 'HAAR_CASCADE',
            'CONFIDENCE_THRESHOLD', 'MIN_FACE_SIZE', 'DB_PATH',
            'WINDOW_WIDTH', 'WINDOW_HEIGHT'
        ]
        
        missing_attrs = []
        for attr in required_attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"   ✓ {attr:25} = {value}")
            else:
                print(f"   ✗ {attr:25} = KHÔNG CÓ")
                missing_attrs.append(attr)
        
        if missing_attrs:
            print(f"\n   ⚠️  Thiếu {len(missing_attrs)} thuộc tính: {', '.join(missing_attrs)}")
        else:
            print(f"\n   ✅ Tất cả thuộc tính đã đầy đủ!")
            
    except ImportError as e:
        print(f"   ✗ Lỗi import config: {e}")
        print(f"   Nguyên nhân có thể:")
        print(f"   1. File config.py không tồn tại")
        print(f"   2. File config.py có lỗi cú pháp")
        print(f"   3. Python không tìm thấy file")
    
    # 5. Kiểm tra thư mục models
    print("\n🤖 5. KIỂM TRA MODELS:")
    models_dir = os.path.join(current_dir, 'models')
    if os.path.exists(models_dir):
        print(f"   ✓ Thư mục models tồn tại")
        model_files = os.listdir(models_dir)
        if model_files:
            print(f"   📁 Các file trong models/:")
            for file in model_files:
                file_path = os.path.join(models_dir, file)
                size = os.path.getsize(file_path)
                print(f"     - {file} ({size:,} bytes)")
        else:
            print(f"   ⚠️  Thư mục models trống!")
            print(f"   Cần các file:")
            print(f"     • haarcascade_frontalface_default.xml")
            print(f"     • deploy.prototxt")
            print(f"     • res10_300x300_ssd_iter_140000.caffemodel")
    else:
        print(f"   ✗ Thư mục models không tồn tại!")
    
    # 6. Kiểm tra OpenCV
    print("\n👁️ 6. KIỂM TRA OPENCV:")
    try:
        import cv2
        version = cv2.__version__
        print(f"   ✓ OpenCV version: {version}")
        
        # Kiểm tra camera
        print("\n   📷 Kiểm tra camera:")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(f"   ✓ Camera (ID=0) có thể mở")
            cap.release()
        else:
            print(f"   ⚠️  Camera (ID=0) không thể mở")
    except ImportError:
        print(f"   ✗ OpenCV chưa cài đặt!")
        print(f"   Cài đặt: pip install opencv-python")
    
    # 7. Kiểm tra các modules
    print("\n🧩 7. KIỂM TRA MODULES:")
    modules_to_check = ['numpy', 'pickle', 'csv', 'json', 'datetime']
    
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"   ✓ {module:15} đã cài đặt")
        except ImportError:
            print(f"   ✗ {module:15} chưa cài đặt")
    
    print("\n" + "=" * 60)
    print("KẾT QUẢ KIỂM TRA HOÀN TẤT")
    print("=" * 60)
    
    # Đề xuất
    print("\n💡 ĐỀ XUẤT:")
    print("1. Chạy 'python main.py' để khởi động hệ thống")
    print("2. Nhấn 'D' để chuyển đổi giữa DNN và Haar")
    print("3. Nhấn 'Q' để thoát chương trình")
    print("4. Nhấn 'S' để chụp ảnh màn hình")

if __name__ == "__main__":
    check_system()