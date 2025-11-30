"""
Capture Dataset Script
Script để chụp ảnh từ webcam cho dataset training
"""

import cv2
import os
import sys
import config
from modules.camera import CameraManager
from modules.detector import FaceDetector


def capture_images(name: str, num_images: int = 20, auto_detect: bool = True):
    """
    Chụp ảnh từ webcam cho user
    
    Args:
        name: Tên user
        num_images: Số lượng ảnh cần chụp
        auto_detect: Tự động detect face và chỉ lưu khi có face
    """
    # Tạo thư mục user
    user_dir = os.path.join(config.DATASET_DIR, name)
    os.makedirs(user_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"CAPTURING DATASET FOR: {name}")
    print("=" * 60)
    print(f"Target: {num_images} images")
    print(f"Output: {user_dir}")
    print(f"Auto-detect: {'ON' if auto_detect else 'OFF'}")
    print("\nControls:")
    print("  SPACE - Capture image")
    print("  Q     - Quit")
    print("=" * 60)
    
    # Initialize camera và detector
    camera = CameraManager()
    if not camera.open():
        print("✗ Failed to open camera")
        return False
    
    detector = None
    if auto_detect:
        detector = FaceDetector(method='haar')  # Haar nhanh hơn cho capture
    
    count = 0
    
    try:
        while count < num_images:
            ret, frame = camera.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            display_frame = frame.copy()
            can_capture = True
            
            # Detect face nếu auto_detect
            if auto_detect and detector:
                faces = detector.detect_faces(frame)
                
                if len(faces) == 0:
                    can_capture = False
                    cv2.putText(display_frame, "No face detected", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif len(faces) > 1:
                    can_capture = False
                    cv2.putText(display_frame, "Multiple faces detected", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    # Vẽ bounding box
                    x, y, w, h = faces[0]
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Face detected - Ready", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hiển thị progress
            progress_text = f"Captured: {count}/{num_images}"
            cv2.putText(display_frame, progress_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Hiển thị hướng dẫn
            if can_capture:
                cv2.putText(display_frame, "Press SPACE to capture", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Capture Dataset", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # SPACE để chụp
            if key == ord(' '):
                if can_capture:
                    filename = os.path.join(user_dir, f"{count+1:03d}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"✓ Saved: {filename}")
                    count += 1
                else:
                    print("✗ Cannot capture - face detection failed")
            
            # Q để thoát
            elif key == ord('q'):
                print("\nCapture cancelled by user")
                break
        
        cv2.destroyAllWindows()
        camera.release()
        
        print("\n" + "=" * 60)
        print(f"✓ Captured {count} images for {name}")
        print(f"✓ Saved to: {user_dir}")
        print("=" * 60)
        
        return count >= config.MIN_IMAGES_PER_PERSON
        
    except Exception as e:
        print(f"\n✗ Error during capture: {e}")
        cv2.destroyAllWindows()
        camera.release()
        return False


def main():
    """Main function"""
    print("=" * 60)
    print("FACE ACCESS CONTROL - DATASET CAPTURE")
    print("=" * 60)
    
    # Input user name
    name = input("\nEnter user name: ").strip()
    if not name:
        print("✗ User name cannot be empty")
        return
    
    # Kiểm tra user đã tồn tại chưa
    user_dir = os.path.join(config.DATASET_DIR, name)
    if os.path.exists(user_dir):
        existing_images = len([f for f in os.listdir(user_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"\nWARNING: User '{name}' already exists with {existing_images} images")
        response = input("Continue and add more images? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    # Input số lượng ảnh
    num_images_input = input(f"\nNumber of images (default {config.MIN_IMAGES_PER_PERSON}): ").strip()
    num_images = int(num_images_input) if num_images_input else config.MIN_IMAGES_PER_PERSON
    
    # Auto-detect option
    auto_detect_input = input("\nAuto-detect face? (y/n, default y): ").strip()
    auto_detect = auto_detect_input.lower() != 'n'
    
    # Capture
    print("\n" + "=" * 60)
    print("STARTING CAPTURE...")
    print("=" * 60)
    
    if capture_images(name, num_images, auto_detect):
        print("\n✓ Dataset capture completed successfully!")
        print("\nNext steps:")
        print("  1. Review captured images")
        print("  2. Run: python check_dataset.py")
        print("  3. Run: python train_lbph.py or python train_facenet.py")
    else:
        print("\n✗ Dataset capture incomplete")
        print(f"Minimum {config.MIN_IMAGES_PER_PERSON} images required per user")


if __name__ == "__main__":
    main()
