"""
Face Access Control - Main Application
Ứng dụng chính để chạy hệ thống Face Access Control

Usage:
    python main.py
"""

from gui.main_window_gradio import GradioMainWindow
import config
import sys
import os


def check_requirements():
    """Kiểm tra các yêu cầu cơ bản"""
    errors = []

    # Kiểm tra thư mục tồn tại
    if not os.path.exists(config.DATASET_DIR):
        errors.append(f"Dataset directory not found: {config.DATASET_DIR}")

    if not os.path.exists(config.MODELS_DIR):
        errors.append(f"Models directory not found: {config.MODELS_DIR}")

    if not os.path.exists(config.LOGS_DIR):
        errors.append(f"Logs directory not found: {config.LOGS_DIR}")

    # Kiểm tra có model nào đã train chưa
    from modules.database import Database

    db = Database()

    has_sface = db.model_exists("sface")

    if not has_sface:
        errors.append(
            "No trained models found! Please run train_sface.py first."
        )

    return errors


def print_banner():
    """In banner chào mừng"""
    banner = """
    ==========================================================
                                                          
        FACE ACCESS CONTROL SYSTEM v1.0                  
                                                          
        Recognition: SFace        
                                                          
    ==========================================================
    """
    print(banner)


def print_system_info():
    """In thông tin hệ thống"""
    from modules.database import Database
    from modules.recognizer_sface import SFaceRecognizer



    db = Database()

    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  - Dataset: {config.DATASET_DIR}")
    print(f"  - Models: {config.MODELS_DIR}")
    print(f"  - Logs: {config.LOGS_DIR}")

    print(f"\nRecognition Methods:")
    print(
        f"  - SFace: {'[OK] Available' if db.model_exists('sface') else '[X] Not trained'}"
    )

    print(f"\nDefault Settings:")
    print(f"  - Recognition Method: {config.DEFAULT_RECOGNITION_METHOD.upper()}")
    print(f"  - Detection Method: {config.DEFAULT_DETECTION_METHOD.upper()}")
    print(f"  - SFace Distance Threshold: {config.SFACE_DISTANCE_THRESHOLD}")

    # Hiển thị danh sách users
    if db.model_exists("sface"):
        users_lbph = db.get_user_list("sface")
        print(f"\nSFace Registered Users ({len(users_lbph)}):")
        print(f"  {', '.join(users_lbph)}")


    print("=" * 60)


def main():
    """Main function"""
    # Print banner
    print_banner()

    # Validate config
    print("\nValidating configuration...")
    if not config.validate_config():
        print("\n[X] Configuration validation failed!")
        return 1
    print("[OK] Configuration valid")

    # Create directories
    print("\nCreating directories...")
    config.create_directories()
    print("[OK] Directories ready")

    # Check requirements
    print("\nChecking requirements...")
    errors = check_requirements()

    if errors:
        print("\n[X] Requirements check failed:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix the errors above before running the application.")
        return 1
    print("[OK] Requirements satisfied")

    # Print system info
    print_system_info()

    # Start GUI
    print("\n" + "=" * 60)
    print("STARTING APPLICATION...")
    print("=" * 60)
    print("\nLaunching GUI...")
    try:
        app = GradioMainWindow()
        app.demo.launch(share=True)
        print("[OK] GUI launched successfully")
        print("\nApplication is running. Close the window to exit.")

    except Exception as e:
        print(f"\n[X] ERROR: Failed to start application: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
