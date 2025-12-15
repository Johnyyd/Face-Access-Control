"""
Face Access Control - Database Management Module
Quản lý lưu/đọc models, embeddings và access logs
"""

import cv2
import json
import pickle
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import config


class Database:
    """
    Class quản lý database cho Face Access Control
    Hỗ trợ:
    - LBPH model (YAML + JSON mapping)
    - Access logs (CSV)
    """

    def __init__(self):
        """Khởi tạo Database manager"""
        # Tạo các thư mục cần thiết
        config.create_directories()

        if config.DEBUG:
            print("[Database] Initialized")

    # ==================== ACCESS LOGS ====================

    def log_access(
        self,
        name: str,
        method: str,
        confidence: float,
        status: str,
        log_path: str = None,
    ) -> bool:
        """
        Ghi log truy cập

        Args:
            name: Tên người
            method: Phương pháp nhận diện ('LBPH', 'OpenFace' hoặc 'sFace)
            confidence: Confidence score hoặc distance
            status: Trạng thái ('GRANTED' hoặc 'DENIED')
            log_path: Đường dẫn log file (mặc định từ config)

        Returns:
            bool: True nếu ghi log thành công
        """
        try:
            log_path = log_path or config.ACCESS_LOG_PATH

            # Tạo timestamp
            timestamp = datetime.now().strftime(config.LOG_TIMESTAMP_FORMAT)

            # Kiểm tra file có tồn tại không
            file_exists = os.path.exists(log_path)

            # Ghi vào CSV
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Ghi header nếu file mới
                if not file_exists:
                    writer.writerow(
                        ["timestamp", "name", "method", "confidence", "status"]
                    )

                # Ghi data
                writer.writerow([timestamp, name, method, f"{confidence:.2f}", status])

            if config.DEBUG:
                print(
                    f"[Database] Access logged: {timestamp} | {name} | {method} | {confidence:.2f} | {status}"
                )

            return True

        except Exception as e:
            print(f"[Database] ERROR logging access: {e}")
            return False

    def read_access_logs(
        self, log_path: str = None, limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Đọc access logs

        Args:
            log_path: Đường dẫn log file (mặc định từ config)
            limit: Số lượng records tối đa (None = tất cả)

        Returns:
            List[Dict]: Danh sách access records
        """
        try:
            log_path = log_path or config.ACCESS_LOG_PATH

            if not os.path.exists(log_path):
                print(f"[Database] WARNING: Log file not found: {log_path}")
                return []

            logs = []
            with open(log_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    logs.append(row)

            # Limit nếu cần
            if limit is not None and limit > 0:
                logs = logs[-limit:]

            if config.DEBUG:
                print(f"[Database] Read {len(logs)} access logs")

            return logs

        except Exception as e:
            print(f"[Database] ERROR reading access logs: {e}")
            return []

    def clear_access_logs(self, log_path: str = None) -> bool:
        """
        Xóa tất cả access logs

        Args:
            log_path: Đường dẫn log file (mặc định từ config)

        Returns:
            bool: True nếu xóa thành công
        """
        try:
            log_path = log_path or config.ACCESS_LOG_PATH

            if os.path.exists(log_path):
                os.remove(log_path)

                if config.DEBUG:
                    print(f"[Database] Access logs cleared: {log_path}")

            return True

        except Exception as e:
            print(f"[Database] ERROR clearing access logs: {e}")
            return False

            return False

    # ==================== EMBEDDINGS MANAGEMENT ====================

    def save_embeddings(self, names: List[str], embeddings: List[np.ndarray]) -> bool:
        """
        Lưu embeddings vào file

        Args:
            names: Danh sách tên
            embeddings: Danh sách embeddings

        Returns:
            bool: True nếu lưu thành công
        """
        try:
            with open(config.SFACE_EMBEDDINGS_PATH, "wb") as f:
                pickle.dump((names, embeddings), f)

            if config.DEBUG:
                print(f"[Database] Embeddings saved to: {config.SFACE_EMBEDDINGS_PATH}")

            return True

        except Exception as e:
            print(f"[Database] ERROR saving embeddings: {e}")
            return False

    def load_embeddings(self) -> Tuple[List[str], List[np.ndarray]]:
        """
        Đọc embeddings từ file

        Returns:
            Tuple[List[str], List[np.ndarray]]: (names, embeddings)
        """
        try:
            if not os.path.exists(config.SFACE_EMBEDDINGS_PATH):
                if config.DEBUG:
                    print(
                        f"[Database] Embeddings file not found: {config.SFACE_EMBEDDINGS_PATH}"
                    )
                return [], []

            with open(config.SFACE_EMBEDDINGS_PATH, "rb") as f:
                names, embeddings = pickle.load(f)

            if config.DEBUG:
                print(f"[Database] Embeddings loaded: {len(names)} users")

            return names, embeddings

        except Exception as e:
            print(f"[Database] ERROR loading embeddings: {e}")
            return [], []

    # ==================== UTILITY FUNCTIONS ====================

    def get_user_list(self, method: str = "lbph") -> List[str]:
        """
        Lấy danh sách users đã đăng ký

        Args:
            method: 'lbph', 'openface', 'sface'

        Returns:
            List[str]: Danh sách tên users
        """
        try:
            if method == "sface":
                names, _ = self.load_embeddings()  # SFace uses same format
                if names:
                    return list(set(names))  # Unique names

            return []

        except Exception as e:
            print(f"[Database] ERROR getting user list: {e}")
            return []

    def model_exists(self, method: str = "sface") -> bool:
        """
        Kiểm tra model đã tồn tại chưa

        Args:
            method: 'sface'

        Returns:
            bool: True nếu model tồn tại
        """
        if method == "sface":
            return os.path.exists(config.SFACE_EMBEDDINGS_PATH)

        return False


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing Database Manager...")
    print("=" * 50)

    db = Database()

    # Test 1: LBPH model (mock data)
    print("\n1. Testing SFace model...")

    # Test 2: Access logs
    print("\n4. Testing access logs...")
    db.log_access("Alice", "SFace", 35.5, "GRANTED")
    db.log_access("Unknown", "SFace", 0.85, "DENIED")
    db.log_access("Bob", "SFace", 42.1, "GRANTED")

    logs = db.read_access_logs(limit=10)
    print(f"✓ Access logs: {len(logs)} entries")
    for log in logs:
        print(f"  {log}")

    print(f"✓ SFace model exists: {db.model_exists('sface')}")
    print("=" * 50)
    print("Database test completed")
