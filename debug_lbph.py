"""
Debug LBPH Recognition
Script để debug và kiểm tra chi tiết nhận diện LBPH
"""

import cv2
import os
import numpy as np
from modules.recognizer_lbph import LBPHRecognizer
from modules.detector import FaceDetector
from modules.camera import CameraManager
import config


def test_with_training_images():
    """Test model với chính ảnh training"""
    print("=" * 60)
    print("TESTING WITH TRAINING IMAGES")
    print("=" * 60)
    
    recognizer = LBPHRecognizer()
    if not recognizer.load_model():
        print("✗ Failed to load model")
        return
    
    detector = FaceDetector(method='haar')
    
    print(f"\nTrained users: {recognizer.get_user_list()}")
    print(f"Label mapping: {recognizer.label_mapping}")
    print(f"Threshold: {recognizer.confidence_threshold}")
    
    # Test với từng user
    for label_id, user_name in recognizer.label_mapping.items():
        user_path = os.path.join(config.DATASET_DIR, user_name)
        
        if not os.path.exists(user_path):
            print(f"\n✗ User directory not found: {user_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing user: {user_name} (Label: {label_id})")
        print(f"{'='*60}")
        
        # Lấy ảnh đầu tiên
        image_files = [f for f in os.listdir(user_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"✗ No images found for {user_name}")
            continue
        
        # Test với 3 ảnh đầu
        for img_file in image_files[:3]:
            img_path = os.path.join(user_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Detect face
            faces = detector.detect_faces(img)
            
            if not faces:
                print(f"  ✗ {img_file}: No face detected")
                continue
            
            # Get first face
            x, y, w, h = faces[0]
            face_roi = img[y:y+h, x:x+w]
            
            # Predict
            name, confidence = recognizer.predict(face_roi)
            
            status = "✓" if name == user_name else "✗"
            print(f"  {status} {img_file}: Predicted={name}, Confidence={confidence:.2f}, Expected={user_name}")


def test_live_with_details():
    """Test live với chi tiết confidence scores"""
    print("\n" + "=" * 60)
    print("LIVE TESTING WITH DETAILED SCORES")
    print("=" * 60)
    
    recognizer = LBPHRecognizer()
    if not recognizer.load_model():
        print("✗ Failed to load model")
        return
    
    detector = FaceDetector(method='haar')
    
    print(f"\nTrained users: {recognizer.get_user_list()}")
    print(f"Threshold: {recognizer.confidence_threshold}")
    print("\nPress 'q' to quit, '+' to increase threshold, '-' to decrease threshold")
    
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
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                # Get raw prediction (without threshold check)
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, config.LBPH_FACE_SIZE)
                
                # Predict với recognizer
                label, confidence = recognizer.recognizer.predict(gray)
                predicted_name = recognizer.label_mapping.get(label, "Unknown")
                
                # Kiểm tra threshold
                is_valid = confidence < recognizer.confidence_threshold
                final_name = predicted_name if is_valid else "Unknown"
                
                # Determine color
                if is_valid:
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw detailed info
                info_lines = [
                    f"Name: {final_name}",
                    f"Raw: {predicted_name}",
                    f"Conf: {confidence:.2f}",
                    f"Thresh: {recognizer.confidence_threshold:.1f}",
                    f"Valid: {is_valid}"
                ]
                
                y_offset = y - 10
                for line in reversed(info_lines):
                    cv2.putText(frame, line, (x, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset -= 20
            
            # Display threshold info
            cv2.putText(frame, f"Threshold: {recognizer.confidence_threshold:.1f} (Press +/- to adjust)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("LBPH Debug", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                recognizer.confidence_threshold += 5
                print(f"Threshold increased to: {recognizer.confidence_threshold}")
            elif key == ord('-') or key == ord('_'):
                recognizer.confidence_threshold = max(0, recognizer.confidence_threshold - 5)
                print(f"Threshold decreased to: {recognizer.confidence_threshold}")
        
        cv2.destroyAllWindows()


def analyze_dataset():
    """Phân tích dataset"""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    detector = FaceDetector(method='haar')
    
    for user_dir in os.listdir(config.DATASET_DIR):
        user_path = os.path.join(config.DATASET_DIR, user_dir)
        
        if not os.path.isdir(user_path) or user_dir.startswith('.'):
            continue
        
        print(f"\n{user_dir}:")
        
        image_files = [f for f in os.listdir(user_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        faces_detected = 0
        no_face = []
        multiple_faces = []
        
        for img_file in image_files:
            img_path = os.path.join(user_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            faces = detector.detect_faces(img)
            
            if len(faces) == 0:
                no_face.append(img_file)
            elif len(faces) > 1:
                multiple_faces.append(img_file)
            else:
                faces_detected += 1
        
        print(f"  Total images: {len(image_files)}")
        print(f"  Faces detected: {faces_detected}")
        
        if no_face:
            print(f"  ⚠ No face: {', '.join(no_face)}")
        if multiple_faces:
            print(f"  ⚠ Multiple faces: {', '.join(multiple_faces)}")


if __name__ == "__main__":
    print("LBPH RECOGNITION DEBUG TOOL")
    print("=" * 60)
    
    print("\nOptions:")
    print("1. Test with training images")
    print("2. Live test with detailed scores")
    print("3. Analyze dataset")
    print("4. All of the above")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        test_with_training_images()
    elif choice == '2':
        test_live_with_details()
    elif choice == '3':
        analyze_dataset()
    elif choice == '4':
        analyze_dataset()
        test_with_training_images()
        test_live_with_details()
    else:
        print("Invalid choice")
