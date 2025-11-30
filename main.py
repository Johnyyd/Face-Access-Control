"""
Face Access Control - Main Application
Ứng dụng chính để chạy hệ thống Face Access Control

Usage:
    python main.py
"""

import tkinter as tk
from gui.main_window import MainWindow
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
    
    has_lbph = db.model_exists('lbph')
    has_openface = db.model_exists('facenet')  # Reuse same storage
    
    if not has_lbph and not has_openface:
        errors.append("No trained models found! Please run train_lbph.py or train_openface.py first.")
    
    return errors


def print_banner():
    """In banner chào mừng"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║         FACE ACCESS CONTROL SYSTEM v1.0                  ║
    ║                                                          ║
    ║         Dual Recognition: LBPH + OpenFace                ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_system_info():
    """In thông tin hệ thống"""
    from modules.database import Database
    from modules.recognizer_openface import FACE_RECOGNITION_AVAILABLE
    
    db = Database()
    
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  - Dataset: {config.DATASET_DIR}")
    print(f"  - Models: {config.MODELS_DIR}")
    print(f"  - Logs: {config.LOGS_DIR}")
    
    print(f"\nRecognition Methods:")
    print(f"  - LBPH: {'✓ Available' if db.model_exists('lbph') else '✗ Not trained'}")
    print(f"  - OpenFace: {'✓ Available' if db.model_exists('facenet') and FACE_RECOGNITION_AVAILABLE else '✗ Not available'}")
    
    print(f"\nDefault Settings:")
    print(f"  - Recognition Method: {config.DEFAULT_RECOGNITION_METHOD.upper()}")
    print(f"  - Detection Method: {config.DEFAULT_DETECTION_METHOD.upper()}")
    print(f"  - LBPH Threshold: {config.LBPH_CONFIDENCE_THRESHOLD}")
    print(f"  - OpenFace Threshold: 0.6")
    
    # Hiển thị danh sách users
    if db.model_exists('lbph'):
        users_lbph = db.get_user_list('lbph')
        print(f"\nLBPH Registered Users ({len(users_lbph)}):")
        print(f"  {', '.join(users_lbph)}")
    
    if db.model_exists('facenet') and FACE_RECOGNITION_AVAILABLE:
        users_openface = db.get_user_list('facenet')  # Reuse same storage
        print(f"\nOpenFace Registered Users ({len(users_openface)}):")
        print(f"  {', '.join(users_openface)}")
    
    print("=" * 60)


def main():
    """Main function"""
    # Print banner
    print_banner()
    
    # Validate config
    print("\nValidating configuration...")
    if not config.validate_config():
        print("\n✗ Configuration validation failed!")
        return 1
    print("✓ Configuration valid")
    
    # Create directories
    print("\nCreating directories...")
    config.create_directories()
    print("✓ Directories ready")
    
    # Check requirements
    print("\nChecking requirements...")
    errors = check_requirements()
    
    if errors:
        print("\n✗ Requirements check failed:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix the errors above before running the application.")
        return 1
    print("✓ Requirements satisfied")
    
    # Print system info
    print_system_info()
    
    # Start GUI
    print("\n" + "=" * 60)
    print("STARTING APPLICATION...")
    print("=" * 60)
    print("\nLaunching GUI...")
    
    try:
        root = tk.Tk()
        app = MainWindow(root)
        
        print("✓ GUI launched successfully")
        print("\nApplication is running. Close the window to exit.")
        
        root.mainloop()
        
        print("\n" + "=" * 60)
        print("APPLICATION CLOSED")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
