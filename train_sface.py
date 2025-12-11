"""
Train SFace Recognition Model
Script để train SFace model từ dataset
"""

import os
import sys
import config
from modules.recognizer_sface import SFaceRecognizer
from modules.database import Database


def main():
    """Main training function"""
    print("=" * 60)
    print("FACE ACCESS CONTROL - SFACE TRAINING")
    print("=" * 60)
    
    # Check dataset
    dataset_dir = config.DATASET_DIR
    print(f"\nDataset directory: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"\n✗ ERROR: Dataset directory not found: {dataset_dir}")
        print("\nPlease create dataset and add user images:")
        print("  python capture_dataset.py")
        return
    
    # Get user list
    users = [d for d in os.listdir(dataset_dir)
             if os.path.isdir(os.path.join(dataset_dir, d))
             and not d.startswith('.')]
    
    if not users:
        print("\n✗ ERROR: No users found in dataset")
        print("\nPlease add user images:")
        print("  python capture_dataset.py")
        return
    
    print(f"Found {len(users)} user(s): {', '.join(users)}")
    
    # Show dataset statistics
    print("\nDataset statistics:")
    print("-" * 60)
    for user in users:
        user_path = os.path.join(dataset_dir, user)
        num_images = len([f for f in os.listdir(user_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        status = "✓" if num_images >= config.MIN_IMAGES_PER_PERSON else "✗"
        print(f"  {status} {user}: {num_images} images")
    print("-" * 60)
    
    # Show configuration
    print("\nSFace Configuration:")
    print(f"  - Model: OpenCV SFace ONNX")
    print(f"  - Embedding size: 512-d vector")
    print(f"  - Distance metric: Cosine similarity")
    print(f"  - Distance threshold: 0.4 (default)")
    
    # Confirm training
    response = input("\nStart training (creating embeddings)? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled")
        return
    
    # Train
    print("\n" + "=" * 60)
    print("TRAINING IN PROGRESS...")
    print("=" * 60)
    
    recognizer = SFaceRecognizer()
    
    if recognizer.model is None:
        print("\n✗ ERROR: SFace model not loaded")
        print("\nPlease download the model:")
        print("  python download_models.py")
        return
    
    print("\nCreating face embeddings from dataset...")
    print("(This may take a while depending on dataset size...)")
    
    if recognizer.train(dataset_dir):
        print("\n" + "=" * 60)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nEmbeddings saved to:")
        print(f"  - {config.MODELS_DIR}/sface/face_recognition_sface_2021dec.onnx")
        
        print(f"\nTrained users: {recognizer.get_user_list()}")
        print(f"Total embeddings: {len(recognizer.known_embeddings)}")
        
        print("\nYou can now run the main application:")
        print("  python main.py")
    else:
        print("\n" + "=" * 60)
        print("✗ TRAINING FAILED")
        print("=" * 60)
        print("\nPlease check the error messages above")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
