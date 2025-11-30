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
    - FaceNet embeddings (Pickle)
    - Access logs (CSV)
    """
    
    def __init__(self):
        """Khởi tạo Database manager"""
        # Tạo các thư mục cần thiết
        config.create_directories()
        
        if config.DEBUG:
            print("[Database] Initialized")
    
    # ==================== LBPH MODEL ====================
    
    def save_lbph_model(self, 
                       recognizer: cv2.face.LBPHFaceRecognizer,
                       label_mapping: Dict[int, str],
                       model_path: str = None,
                       mapping_path: str = None) -> bool:
        """
        Lưu LBPH model và label mapping
        
        Args:
            recognizer: LBPH recognizer đã train
            label_mapping: Dictionary {label_id: name}
            model_path: Đường dẫn lưu model (mặc định từ config)
            mapping_path: Đường dẫn lưu mapping (mặc định từ config)
            
        Returns:
            bool: True nếu lưu thành công
        """
        try:
            model_path = model_path or config.LBPH_MODEL_PATH
            mapping_path = mapping_path or config.LBPH_MAPPING_PATH
            
            # Lưu LBPH model (YAML format)
            recognizer.save(model_path)
            
            # Lưu label mapping (JSON format)
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(label_mapping, f, indent=4, ensure_ascii=False)
            
            if config.DEBUG:
                print(f"[Database] LBPH model saved to: {model_path}")
                print(f"[Database] Label mapping saved to: {mapping_path}")
                print(f"[Database] Total users: {len(label_mapping)}")
            
            return True
            
        except Exception as e:
            print(f"[Database] ERROR saving LBPH model: {e}")
            return False
    
    def load_lbph_model(self,
                       model_path: str = None,
                       mapping_path: str = None) -> Tuple[Optional[cv2.face.LBPHFaceRecognizer], Optional[Dict[int, str]]]:
        """
        Load LBPH model và label mapping
        
        Args:
            model_path: Đường dẫn model (mặc định từ config)
            mapping_path: Đường dẫn mapping (mặc định từ config)
            
        Returns:
            Tuple[recognizer, label_mapping]: (None, None) nếu thất bại
        """
        try:
            model_path = model_path or config.LBPH_MODEL_PATH
            mapping_path = mapping_path or config.LBPH_MAPPING_PATH
            
            # Kiểm tra file tồn tại
            if not os.path.exists(model_path):
                print(f"[Database] ERROR: LBPH model not found: {model_path}")
                return None, None
            
            if not os.path.exists(mapping_path):
                print(f"[Database] ERROR: Label mapping not found: {mapping_path}")
                return None, None
            
            # Load LBPH model
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(model_path)
            
            # Load label mapping
            with open(mapping_path, 'r', encoding='utf-8') as f:
                label_mapping_str = json.load(f)
            
            # Convert keys từ string sang int
            label_mapping = {int(k): v for k, v in label_mapping_str.items()}
            
            if config.DEBUG:
                print(f"[Database] LBPH model loaded from: {model_path}")
                print(f"[Database] Label mapping loaded from: {mapping_path}")
                print(f"[Database] Total users: {len(label_mapping)}")
            
            return recognizer, label_mapping
            
        except Exception as e:
            print(f"[Database] ERROR loading LBPH model: {e}")
            return None, None
    
    # ==================== FACENET EMBEDDINGS ====================
    
    def save_facenet_embeddings(self,
                               names: List[str],
                               embeddings: List[np.ndarray],
                               embeddings_path: str = None) -> bool:
        """
        Lưu FaceNet embeddings
        
        Args:
            names: List tên người
            embeddings: List embeddings (128-d vectors)
            embeddings_path: Đường dẫn lưu (mặc định từ config)
            
        Returns:
            bool: True nếu lưu thành công
        """
        try:
            embeddings_path = embeddings_path or config.FACENET_EMBEDDINGS_PATH
            
            # Tạo dictionary
            data = {
                'names': names,
                'embeddings': embeddings
            }
            
            # Lưu vào pickle file
            with open(embeddings_path, 'wb') as f:
                pickle.dump(data, f)
            
            if config.DEBUG:
                print(f"[Database] FaceNet embeddings saved to: {embeddings_path}")
                print(f"[Database] Total embeddings: {len(names)}")
            
            return True
            
        except Exception as e:
            print(f"[Database] ERROR saving FaceNet embeddings: {e}")
            return False
    
    def load_facenet_embeddings(self,
                               embeddings_path: str = None) -> Tuple[Optional[List[str]], Optional[List[np.ndarray]]]:
        """
        Load FaceNet embeddings
        
        Args:
            embeddings_path: Đường dẫn embeddings (mặc định từ config)
            
        Returns:
            Tuple[names, embeddings]: (None, None) nếu thất bại
        """
        try:
            embeddings_path = embeddings_path or config.FACENET_EMBEDDINGS_PATH
            
            # Kiểm tra file tồn tại
            if not os.path.exists(embeddings_path):
                print(f"[Database] ERROR: FaceNet embeddings not found: {embeddings_path}")
                return None, None
            
            # Load từ pickle file
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
            
            names = data['names']
            embeddings = data['embeddings']
            
            if config.DEBUG:
                print(f"[Database] FaceNet embeddings loaded from: {embeddings_path}")
                print(f"[Database] Total embeddings: {len(names)}")
            
            return names, embeddings
            
        except Exception as e:
            print(f"[Database] ERROR loading FaceNet embeddings: {e}")
            return None, None
    
    # ==================== ACCESS LOGS ====================
    
    def log_access(self,
                  name: str,
                  method: str,
                  confidence: float,
                  status: str,
                  log_path: str = None) -> bool:
        """
        Ghi log truy cập
        
        Args:
            name: Tên người
            method: Phương pháp nhận diện ('LBPH' hoặc 'FaceNet')
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
            with open(log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Ghi header nếu file mới
                if not file_exists:
                    writer.writerow(['timestamp', 'name', 'method', 'confidence', 'status'])
                
                # Ghi data
                writer.writerow([timestamp, name, method, f"{confidence:.2f}", status])
            
            if config.DEBUG:
                print(f"[Database] Access logged: {timestamp} | {name} | {method} | {confidence:.2f} | {status}")
            
            return True
            
        except Exception as e:
            print(f"[Database] ERROR logging access: {e}")
            return False
    
    def read_access_logs(self,
                        log_path: str = None,
                        limit: int = None) -> List[Dict[str, Any]]:
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
            with open(log_path, 'r', encoding='utf-8') as f:
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
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def get_user_list(self, method: str = 'lbph') -> List[str]:
        """
        Lấy danh sách users đã đăng ký
        
        Args:
            method: 'lbph' hoặc 'facenet'
            
        Returns:
            List[str]: Danh sách tên users
        """
        try:
            if method == 'lbph':
                _, label_mapping = self.load_lbph_model()
                if label_mapping:
                    return list(label_mapping.values())
            elif method == 'facenet':
                names, _ = self.load_facenet_embeddings()
                if names:
                    return list(set(names))  # Unique names
            
            return []
            
        except Exception as e:
            print(f"[Database] ERROR getting user list: {e}")
            return []
    
    def model_exists(self, method: str = 'lbph') -> bool:
        """
        Kiểm tra model đã tồn tại chưa
        
        Args:
            method: 'lbph' hoặc 'facenet'
            
        Returns:
            bool: True nếu model tồn tại
        """
        if method == 'lbph':
            return (os.path.exists(config.LBPH_MODEL_PATH) and 
                   os.path.exists(config.LBPH_MAPPING_PATH))
        elif method == 'facenet':
            return os.path.exists(config.FACENET_EMBEDDINGS_PATH)
        
        return False


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing Database Manager...")
    print("=" * 50)
    
    db = Database()
    
    # Test 1: LBPH model (mock data)
    print("\n1. Testing LBPH model save/load...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    label_mapping = {0: "Alice", 1: "Bob", 2: "Charlie"}
    
    if db.save_lbph_model(recognizer, label_mapping):
        print("✓ LBPH model saved")
        
        loaded_recognizer, loaded_mapping = db.load_lbph_model()
        if loaded_recognizer and loaded_mapping:
            print(f"✓ LBPH model loaded: {loaded_mapping}")
    
    # Test 2: FaceNet embeddings (mock data)
    print("\n2. Testing FaceNet embeddings save/load...")
    names = ["Alice", "Alice", "Bob", "Bob", "Charlie"]
    embeddings = [np.random.rand(128) for _ in range(5)]
    
    if db.save_facenet_embeddings(names, embeddings):
        print("✓ FaceNet embeddings saved")
        
        loaded_names, loaded_embeddings = db.load_facenet_embeddings()
        if loaded_names and loaded_embeddings:
            print(f"✓ FaceNet embeddings loaded: {len(loaded_names)} embeddings")
    
    # Test 3: Access logs
    print("\n3. Testing access logs...")
    db.log_access("Alice", "LBPH", 35.5, "GRANTED")
    db.log_access("Unknown", "FaceNet", 0.85, "DENIED")
    db.log_access("Bob", "LBPH", 42.1, "GRANTED")
    
    logs = db.read_access_logs(limit=10)
    print(f"✓ Access logs: {len(logs)} entries")
    for log in logs:
        print(f"  {log}")
    
    # Test 4: Utility functions
    print("\n4. Testing utility functions...")
    users_lbph = db.get_user_list('lbph')
    print(f"✓ LBPH users: {users_lbph}")
    
    users_facenet = db.get_user_list('facenet')
    print(f"✓ FaceNet users: {users_facenet}")
    
    print(f"✓ LBPH model exists: {db.model_exists('lbph')}")
    print(f"✓ FaceNet model exists: {db.model_exists('facenet')}")
    
    print("=" * 50)
    print("Database test completed")
