# download_models.py
import os
import urllib.request
import zipfile

def download_models():
    """Tải các file model cần thiết"""
    
    # Tạo thư mục models nếu chưa có
    os.makedirs("models", exist_ok=True)
    
    # URLs để tải
    urls = {
        "haarcascade": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        "deploy_prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    }
    
    print("Đang tải các file model...")
    
    # Tải từng file
    for name, url in urls.items():
        filename = url.split("/")[-1]
        filepath = os.path.join("models", filename)
        
        if not os.path.exists(filepath):
            print(f"  Đang tải {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"  ✓ Đã tải: {filename}")
            except Exception as e:
                print(f"  ✗ Lỗi tải {filename}: {e}")
        else:
            print(f"  ✓ Đã có: {filename}")
    
    print("\n⚠️ LƯU Ý: File DNN model (.caffemodel) cần tải thủ công:")
    print("  Tải từ: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
    print("  Lưu vào: models/res10_300x300_ssd_iter_140000.caffemodel")
    
    print("\nHoặc sử dụng Haar Cascade (không cần DNN model):")
    print("  Trong main.py, thay đổi: detector_method = 'haar'")

if __name__ == "__main__":
    download_models()