"""
Face Access Control - Camera Management Module
Quản lý webcam: mở, đọc frame, đóng camera
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import config


class CameraManager:
    """
    Class quản lý camera cho hệ thống Face Access Control
    
    Attributes:
        camera_id (int): ID của camera (0 = webcam mặc định)
        width (int): Chiều rộng frame
        height (int): Chiều cao frame
        fps (int): Frames per second
        cap (cv2.VideoCapture): OpenCV VideoCapture object
    """
    
    def __init__(self, 
                 camera_id: int = None,
                 width: int = None,
                 height: int = None,
                 fps: int = None):
        """
        Khởi tạo Camera Manager
        
        Args:
            camera_id: ID của camera (mặc định từ config)
            width: Chiều rộng frame (mặc định từ config)
            height: Chiều cao frame (mặc định từ config)
            fps: Frames per second (mặc định từ config)
        """
        self.camera_id = camera_id if camera_id is not None else config.CAMERA_ID
        self.width = width if width is not None else config.CAMERA_WIDTH
        self.height = height if height is not None else config.CAMERA_HEIGHT
        self.fps = fps if fps is not None else config.CAMERA_FPS
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened_flag = False
        
        if config.DEBUG:
            print(f"[CameraManager] Initialized with camera_id={self.camera_id}, "
                  f"resolution={self.width}x{self.height}, fps={self.fps}")
    
    def open(self) -> bool:
        """
        Mở camera
        
        Returns:
            bool: True nếu mở thành công, False nếu thất bại
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print(f"[CameraManager] ERROR: Cannot open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.is_opened_flag = True
            
            if config.DEBUG:
                print(f"[CameraManager] Camera opened successfully")
                print(f"[CameraManager] Actual resolution: {actual_width}x{actual_height}")
                print(f"[CameraManager] Actual FPS: {actual_fps}")
            
            return True
            
        except Exception as e:
            print(f"[CameraManager] ERROR opening camera: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Đọc một frame từ camera
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
                - success: True nếu đọc thành công
                - frame: Frame ảnh dạng NumPy array (BGR format)
        """
        if not self.is_opened_flag or self.cap is None:
            print("[CameraManager] ERROR: Camera is not opened")
            return False, None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                print("[CameraManager] ERROR: Failed to read frame")
                return False, None
            
            return True, frame
            
        except Exception as e:
            print(f"[CameraManager] ERROR reading frame: {e}")
            return False, None
    
    def release(self) -> None:
        """
        Đóng camera và giải phóng tài nguyên
        """
        if self.cap is not None:
            self.cap.release()
            self.is_opened_flag = False
            
            if config.DEBUG:
                print("[CameraManager] Camera released")
    
    def is_opened(self) -> bool:
        """
        Kiểm tra camera có đang mở không
        
        Returns:
            bool: True nếu camera đang mở
        """
        return self.is_opened_flag and self.cap is not None and self.cap.isOpened()
    
    def get_properties(self) -> dict:
        """
        Lấy thông tin properties của camera
        
        Returns:
            dict: Dictionary chứa các properties
        """
        if not self.is_opened():
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
        }
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Thay đổi độ phân giải camera
        
        Args:
            width: Chiều rộng mới
            height: Chiều cao mới
            
        Returns:
            bool: True nếu thành công
        """
        if not self.is_opened():
            print("[CameraManager] ERROR: Camera is not opened")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if config.DEBUG:
            print(f"[CameraManager] Resolution changed to {self.width}x{self.height}")
        
        return True
    
    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Destructor"""
        self.release()


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing Camera Manager...")
    print("=" * 50)
    
    # Test với context manager
    with CameraManager() as camera:
        if camera.is_opened():
            print("✓ Camera opened successfully")
            print(f"Properties: {camera.get_properties()}")
            
            # Đọc và hiển thị 100 frames
            for i in range(100):
                ret, frame = camera.read()
                if ret:
                    cv2.imshow("Camera Test", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            cv2.destroyAllWindows()
            print("✓ Camera test completed")
        else:
            print("✗ Failed to open camera")
    
    print("=" * 50)
    print("Camera released")
