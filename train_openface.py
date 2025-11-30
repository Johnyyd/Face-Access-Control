"""
Face Access Control - OpenFace Training Script
Script để train OpenFace model từ dataset (tạo face encodings)
"""

import os
import sys
from modules.recognizer_openface import OpenFaceRecognizer, FACE_RECOGNITION_AVAILABLE
import config


def main():
    """Main training function"""
    print("=" * 60)
    print("FACE ACCESS CONTROL - OPENFACE TRAINING")
    print("=" * 60)
    
    # Kiểm tra face_recognition
    if not FACE_RECOGNITION_AVAILABLE:
        print("\n✗ ERROR: face_recognition not available")
        print("\nPlease install required package:")
        print("  pip install face-recognition")
        return False
    
    # Kiểm tra dataset tồn tại
    if not os.path.exists(config.DATASET_DIR):
        print(f"\n✗ ERROR: Dataset directory not found: {config.DATASET_DIR}")
        print("\nPlease create dataset directory and add user images")
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
    
    # Hiển thị OpenFace configuration
    print(f"\nOpenFace Configuration:")
    print(f"  - Model: dlib HOG + CNN")
    print(f"  - Encoding size: 128-d vector")
    print(f"  - Distance threshold: 0.6 (default)")
    
    # Xác nhận training
    response = input("\nStart training (creating encodings)? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled")
        return False
    
    # Tạo recognizer
    print("\n" + "=" * 60)
    print("TRAINING IN PROGRESS...")
    print("=" * 60)
    
    recognizer = OpenFaceRecognizer()
    
    # Train (create encodings)
    print("\nCreating face encodings from dataset...")
    print("(This may take a while depending on dataset size...)")
    
    if recognizer.train(config.DATASET_DIR):
        print("\n" + "=" * 60)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nEncodings saved to:")
        print(f"  - {config.FACENET_EMBEDDINGS_PATH}")
        print(f"\nTrained users: {recognizer.get_user_list()}")
        print(f"Total encodings: {len(recognizer.known_encodings)}")
        print("\nYou can now run the main application:")
        print("  python main.py")
        return True
    else:
        print("\n" + "=" * 60)
        print("✗ TRAINING FAILED")
        print("=" * 60)
        print("\nPlease check the error messages above")
        return False


if __name__ == "__main__":
    # Train model
    success = main()
    
    print("\nDone!")
    sys.exit(0 if success else 1)
