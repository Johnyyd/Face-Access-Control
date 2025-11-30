"""
Face Access Control - LBPH Training Script
Script để train LBPH model từ dataset
"""

import cv2
import os
import sys
from modules.recognizer_lbph import LBPHRecognizer
from modules.detector import FaceDetector
import config


def main():
    """Main training function"""
    print("=" * 60)
    print("FACE ACCESS CONTROL - LBPH TRAINING")
    print("=" * 60)
    
    # Kiểm tra dataset tồn tại
    if not os.path.exists(config.DATASET_DIR):
        print(f"\n✗ ERROR: Dataset directory not found: {config.DATASET_DIR}")
        print("\nPlease create dataset directory and add user images:")
        print(f"  {config.DATASET_DIR}/")
        print(f"    ├── User1/")
        print(f"    │   ├── 001.jpg")
        print(f"    │   ├── 002.jpg")
        print(f"    │   └── ...")
        print(f"    ├── User2/")
        print(f"    │   └── ...")
        return False
    
    # Kiểm tra có user directories không
    user_dirs = [d for d in os.listdir(config.DATASET_DIR) 
                if os.path.isdir(os.path.join(config.DATASET_DIR, d)) 
                and not d.startswith('.')]
    
    if not user_dirs:
        print(f"\n✗ ERROR: No user directories found in {config.DATASET_DIR}")
        print("\nPlease add user directories with images")
        return False
    
    print(f"\nDataset directory: {config.DATASET_DIR}")
    print(f"Found {len(user_dirs)} user(s): {', '.join(user_dirs)}")
    
    # Hiển thị thống kê dataset
    print("\nDataset statistics:")
    print("-" * 60)
    total_images = 0
    for user_name in user_dirs:
        user_path = os.path.join(config.DATASET_DIR, user_name)
        image_files = [f for f in os.listdir(user_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_images = len(image_files)
        total_images += num_images
        
        status = "✓" if num_images >= config.MIN_IMAGES_PER_PERSON else "✗"
        print(f"  {status} {user_name}: {num_images} images")
    
    print("-" * 60)
    print(f"Total images: {total_images}")
    
    if total_images == 0:
        print("\n✗ ERROR: No images found in dataset")
        return False
    
    # Xác nhận training
    print(f"\nLBPH Configuration:")
    print(f"  - Radius: {config.LBPH_RADIUS}")
    print(f"  - Neighbors: {config.LBPH_NEIGHBORS}")
    print(f"  - Grid: {config.LBPH_GRID_X}x{config.LBPH_GRID_Y}")
    print(f"  - Confidence Threshold: {config.LBPH_CONFIDENCE_THRESHOLD}")
    print(f"  - Face Size: {config.LBPH_FACE_SIZE}")
    
    response = input("\nStart training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled")
        return False
    
    # Tạo recognizer và train
    print("\n" + "=" * 60)
    print("TRAINING IN PROGRESS...")
    print("=" * 60)
    
    recognizer = LBPHRecognizer()
    
    if recognizer.train(config.DATASET_DIR):
        print("\n" + "=" * 60)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nModel saved to:")
        print(f"  - {config.LBPH_MODEL_PATH}")
        print(f"  - {config.LBPH_MAPPING_PATH}")
        print(f"\nTrained users: {recognizer.get_user_list()}")
        print("\nYou can now run the main application:")
        print("  python main.py")
        return True
    else:
        print("\n" + "=" * 60)
        print("✗ TRAINING FAILED")
        print("=" * 60)
        print("\nPlease check the error messages above")
        return False


def test_trained_model():
    """Test trained model với camera"""
    print("\n" + "=" * 60)
    print("TESTING TRAINED MODEL")
    print("=" * 60)
    
    from modules.camera import CameraManager
    
    recognizer = LBPHRecognizer()
    
    if not recognizer.load_model():
        print("✗ Failed to load model")
        return
    
    detector = FaceDetector(method='haar')
    
    print("\nPress 'q' to quit")
    print("Testing with camera...")
    
    with CameraManager() as camera:
        if not camera.is_opened():
            print("✗ Failed to open camera")
            return
        
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Recognize each face
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                name, confidence = recognizer.predict(face_roi)
                
                # Draw result
                color = config.COLOR_SUCCESS if name != config.UNKNOWN_PERSON_NAME else config.COLOR_DENIED
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label = f"{name} ({confidence:.1f})"
                cv2.putText(frame, label, (x, y-10),
                          config.FONT_FACE, config.FONT_SCALE,
                          color, config.FONT_THICKNESS)
            
            # Display info
            info_text = f"LBPH | Threshold: {recognizer.get_threshold()}"
            cv2.putText(frame, info_text, (10, 30),
                       config.FONT_FACE, config.FONT_SCALE,
                       config.COLOR_TEXT, config.FONT_THICKNESS)
            
            cv2.imshow("LBPH Model Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    print("✓ Test completed")


if __name__ == "__main__":
    # Train model
    success = main()
    
    # Nếu training thành công, hỏi có muốn test không
    if success:
        response = input("\nTest trained model with camera? (y/n): ")
        if response.lower() == 'y':
            test_trained_model()
    
    print("\nDone!")
