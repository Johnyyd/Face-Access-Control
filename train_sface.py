"""
Train SFace Recognition Model
Script để train SFace model từ dataset
"""

import os
import config
from modules.recognizer_sface import SFaceRecognizer
from modules.database import Database


def print_header(title: str):
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_dataset(dataset_dir):
    """Kiểm tra dataset và trả về danh sách user"""
    print(f"\nDataset directory: {dataset_dir}")

    if not os.path.exists(dataset_dir):
        print(f"\n[X] ERROR: Dataset directory not found: {dataset_dir}")
        print("\nPlease create dataset and add user images:")
        print("  python capture_dataset.py")
        return None

    users = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ]

    if not users:
        print("\n[X] ERROR: No users found in dataset")
        print("\nPlease add user images:")
        print("  python capture_dataset.py")
        return None

    print(f"Found {len(users)} user(s): {', '.join(users)}")
    return users


def show_dataset_statistics(dataset_dir, users):
    """Hiển thị thống kê dataset"""
    print("\nDataset statistics:")
    print("-" * 60)
    for user in users:
        user_path = os.path.join(dataset_dir, user)
        num_images = len(
            [
                f
                for f in os.listdir(user_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        status = "[OK]" if num_images >= config.MIN_IMAGES_PER_PERSON else "[X]"
        print(f"  {status} {user}: {num_images} images")
    print("-" * 60)


def show_configuration():
    """Hiển thị cấu hình SFace"""
    print("\nSFace Configuration:")
    print(f"  - Model: OpenCV SFace ONNX")
    print(f"  - Embedding size: 512-d vector")
    print(f"  - Distance metric: Cosine Similarity (Higher is better)")
    print(f"  - Threshold: 0.6 (default)")


def train_model(dataset_dir):
    """Thực hiện training"""
    recognizer = SFaceRecognizer()

    if recognizer.model is None:
        print("\n[X] ERROR: SFace model not loaded")
        print("\nPlease download the model:")
        print("  python download_models.py")
        return False

    print("\nCreating face embeddings from dataset...")
    print("(This may take a while depending on dataset size...)")

    if recognizer.train(dataset_dir):
        print_header("[OK] TRAINING COMPLETED SUCCESSFULLY!")
        print("\nEmbeddings saved to:")
        print(f"  - {config.SFACE_EMBEDDINGS_PATH}")
        print(f"\nTrained users: {recognizer.get_user_list()}")
        print(f"Total embeddings: {len(recognizer.known_embeddings)}")
        print("\nYou can now run the main application:")
        print("  python main.py")
        return True
    else:
        print_header("[X] TRAINING FAILED")
        print("\nPlease check the error messages above")
        return False


def main():
    """Main training function"""
    print_header("FACE ACCESS CONTROL - SFACE TRAINING")

    dataset_dir = config.DATASET_DIR
    users = check_dataset(dataset_dir)
    if not users:
        return

    show_dataset_statistics(dataset_dir, users)
    show_configuration()

    response = input("\nStart training (creating embeddings)? (y/n): ")
    if response.lower() != "y":
        print("Training cancelled")
        return

    print_header("TRAINING IN PROGRESS...")
    train_model(dataset_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
