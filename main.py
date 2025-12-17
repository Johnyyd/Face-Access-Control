"""
Face Access Control - Main Application
Hệ thống kiểm soát ra vào bằng nhận diện khuôn mặt
"""

import cv2
import numpy as np
from modules.camera import CameraManager
from modules.detector import FaceDetector
from modules.database import Database
import sys
import os

# Thêm config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import config
except ImportError:
    # Fallback config
    class ConfigFallback:
        WINDOW_WIDTH = 800
        WINDOW_HEIGHT = 600
        DEBUG = True
    
    config = ConfigFallback()
    print("[Main] Warning: Using fallback config")


def main():
    print("=" * 50)
    print("Face Access Control System")
    print("=" * 50)
    
    # Khởi tạo camera
    print("\n[1] Initializing camera...")
    cam = CameraManager()
    if not cam.open():
        print("❌ Không mở được camera!")
        return
    
    # Khởi tạo detector
    print("\n[2] Initializing face detector...")
    detector_method = "dnn"  # hoặc "haar"
    detector = FaceDetector(method=detector_method)
    
    # Khởi tạo database
    print("\n[3] Initializing database...")
    db = Database()
    
    print("\n[4] System ready!")
    print("🔥 Camera started! (Nhấn Q để thoát)")
    print("-" * 50)
    
    frame_count = 0
    detection_count = 0
    
    while True:
        # Đọc frame từ camera
        ok, frame = cam.read()
        if not ok:
            print("❌ Lỗi đọc frame!")
            break
        
        frame_count += 1
        
        # Phát hiện khuôn mặt
        boxes = detector.detect(frame)
        
        # FIX: Chuyển numpy array sang list nếu cần
        if hasattr(boxes, 'size'):  # Nếu là numpy array
            if boxes.size > 0:
                boxes_list = boxes.tolist()  # Chuyển sang list
                detection_count += 1
                
                # Log vào database (tạm thời chỉ ghi thông tin đơn giản)
                if frame_count % 30 == 0:  # Mỗi 30 frames log 1 lần
                    for i, box in enumerate(boxes):
                        db.log_access(
                            name=f"Detected_Face_{i}",
                            method=detector_method.upper(),
                            confidence=0.95,
                            status="DETECTED"
                        )
            else:
                boxes_list = []
        else:
            # Nếu đã là list
            if len(boxes) > 0:
                boxes_list = boxes
                detection_count += 1
                
                # Log vào database
                if frame_count % 30 == 0:
                    for i, box in enumerate(boxes):
                        db.log_access(
                            name=f"Detected_Face_{i}",
                            method=detector_method.upper(),
                            confidence=0.95,
                            status="DETECTED"
                        )
            else:
                boxes_list = []
        
        # Vẽ bounding boxes
        for box in boxes_list:
            # Haar Cascade trả về (x, y, w, h)
            if len(box) == 4:
                x, y, w, h = box
            else:
                continue  # Bỏ qua nếu không đúng định dạng
            
            # Vẽ rectangle
            cv2.rectangle(frame, 
                         (x, y), 
                         (x + w, y + h), 
                         (0, 255, 0), 
                         2)
            
            # Vẽ label
            label = "Face"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Hiển thị thông tin thống kê
        stats_text = f"Frames: {frame_count} | Detections: {detection_count}"
        cv2.putText(frame, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Hiển thị tên phương pháp
        method_text = f"Method: {detector_method.upper()}"
        cv2.putText(frame, method_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Thông tin faces detected
        faces_text = f"Faces: {len(boxes_list)}"
        cv2.putText(frame, faces_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Resize cửa sổ nếu cần
        display_frame = cv2.resize(frame, 
                                  (getattr(config, 'WINDOW_WIDTH', 800), 
                                   getattr(config, 'WINDOW_HEIGHT', 600)))
        
        # Hiển thị frame
        cv2.imshow("Face Access Control - Detector", display_frame)
        
        # Nhấn Q để thoát
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n👋 Thoát chương trình...")
            break
        elif key == ord('d'):
            # Chuyển đổi giữa DNN và Haar
            detector_method = "dnn" if detector_method == "haar" else "haar"
            detector = FaceDetector(method=detector_method)
            print(f"\n🔄 Chuyển sang phương pháp: {detector_method.upper()}")
        elif key == ord('s'):
            # Chụp ảnh hiện tại
            timestamp = cv2.getTickCount()
            filename = f"captures/capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n📸 Đã lưu ảnh: {filename}")
        elif key == ord('c'):
            # Clear screen
            print("\n" * 3)
            print("-" * 50)
            print("Đã xóa màn hình console")
            print("-" * 50)
    
    # Giải phóng tài nguyên
    cam.release()
    cv2.destroyAllWindows()
    
    # Hiển thị thống kê cuối cùng
    print("\n" + "=" * 50)
    print("THỐNG KÊ")
    print("=" * 50)
    print(f"Tổng số frames: {frame_count}")
    print(f"Số lần phát hiện khuôn mặt: {detection_count}")
    if frame_count > 0:
        detection_rate = (detection_count / frame_count) * 100
        print(f"Tỷ lệ phát hiện: {detection_rate:.1f}%")
    
    # Hiển thị logs gần đây
    print("\nLOGS GẦN ĐÂY:")
    logs = db.read_access_logs(limit=5)
    for log in logs:
        print(f"  {log['timestamp']} - {log['name']}: {log['status']}")
    
    print("\n✅ Hoàn thành!")


if __name__ == "__main__":
    main()