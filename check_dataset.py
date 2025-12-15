"""
Check Dataset Script
Script kiểm tra tính hợp lệ của dataset
"""

import os
import cv2
import config


def check_dataset(dataset_dir: str = None):
    """
    Kiểm tra dataset
    
    Args:
        dataset_dir: Đường dẫn dataset (mặc định từ config)
    """
    dataset_dir = dataset_dir or config.DATASET_DIR
    
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    print(f"Dataset directory: {dataset_dir}\n")
    
    # Kiểm tra thư mục tồn tại
    if not os.path.exists(dataset_dir):
        print(f"✗ Dataset directory not found: {dataset_dir}")
        print("\nPlease create dataset directory and add user images")
        return False
    
    # Lấy danh sách users
    users = [d for d in os.listdir(dataset_dir) 
             if os.path.isdir(os.path.join(dataset_dir, d))
             and not d.startswith('.')]
    
    if not users:
        print("✗ No user directories found")
        print("\nPlease add user directories with images:")
        print(f"  {dataset_dir}/User1/")
        print(f"  {dataset_dir}/User2/")
        return False
    
    print(f"Found {len(users)} user(s):\n")
    print("-" * 60)
    
    total_images = 0
    valid_users = 0
    warnings = []
    
    for user in sorted(users):
        user_path = os.path.join(dataset_dir, user)
        
        # Lấy danh sách ảnh
        image_files = [f for f in os.listdir(user_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        num_images = len(image_files)
        total_images += num_images
        
        # Kiểm tra số lượng ảnh
        if num_images >= config.MIN_IMAGES_PER_PERSON:
            status = "✓"
            valid_users += 1
        else:
            status = "✗"
            warnings.append(f"{user} has only {num_images} images (minimum: {config.MIN_IMAGES_PER_PERSON})")
        
        # Kiểm tra kích thước ảnh
        if num_images > 0:
            sample_image = os.path.join(user_path, image_files[0])
            img = cv2.imread(sample_image)
            if img is not None:
                h, w = img.shape[:2]
                size_info = f"({w}x{h})"
            else:
                size_info = "(invalid)"
                warnings.append(f"{user}/{image_files[0]} cannot be read")
        else:
            size_info = ""
        
        print(f"{status} {user:20s}: {num_images:3d} images {size_info}")
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Total users: {len(users)}")
    print(f"  Valid users (≥{config.MIN_IMAGES_PER_PERSON} images): {valid_users}")
    print(f"  Total images: {total_images}")
    
    # Hiển thị warnings
    if warnings:
        print("\n" + "=" * 60)
        print("WARNINGS:")
        print("=" * 60)
        for warning in warnings:
            print(f"  ⚠ {warning}")
    
    # Kết luận
    print("\n" + "=" * 60)
    if valid_users == 0:
        print("✗ DATASET NOT READY")
        print("=" * 60)
        print("\nNo valid users found!")
        print(f"Each user needs at least {config.MIN_IMAGES_PER_PERSON} images")
        print("\nTo capture images, run:")
        print("  python capture_dataset.py")
        return False
    elif valid_users < len(users):
        print("⚠ DATASET PARTIALLY READY")
        print("=" * 60)
        print(f"\n{valid_users}/{len(users)} users are ready for training")
        print("Some users don't have enough images")
        print("\nYou can proceed with training, but consider adding more images")
        return True
    
    if not os.path.exists(dataset_dir):
        return
    
    users = [d for d in os.listdir(dataset_dir) 
             if os.path.isdir(os.path.join(dataset_dir, d))
             and not d.startswith('.')]
    
    if not users:
        return
    
    print("\n" + "=" * 60)
    print("DETAILED STATISTICS")
    print("=" * 60)
    
    image_counts = []
    for user in users:
        user_path = os.path.join(dataset_dir, user)
        num_images = len([f for f in os.listdir(user_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        image_counts.append(num_images)
    
    if image_counts:
        print(f"\nImages per user:")
        print(f"  Minimum: {min(image_counts)}")
        print(f"  Maximum: {max(image_counts)}")
        print(f"  Average: {sum(image_counts) / len(image_counts):.1f}")
        print(f"  Total: {sum(image_counts)}")


if __name__ == "__main__":
    import sys
    
    # Cho phép pass dataset path qua argument
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Check dataset
    is_valid = check_dataset(dataset_path)
    
    # Show detailed stats
    if is_valid:
        show_dataset_stats(dataset_path)
    
    print("\n" + "=" * 60)
    
    # Exit code
    sys.exit(0 if is_valid else 1)
