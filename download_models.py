"""
Download Pre-trained Models Script
Script tự động download các pre-trained models cần thiết
"""

import os
import urllib.request
import sys


def download_file(url, output_path, description):
    """Download file với progress bar"""
    print(f"\nDownloading {description}...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    
    try:
        def reporthook(count, block_size, total_size):
            """Progress callback"""
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent}% ")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, reporthook)
        print(f"\n✓ Downloaded successfully: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading: {e}")
        return False


def main():
    """Main download function"""
    print("=" * 60)
    print("FACE ACCESS CONTROL - MODEL DOWNLOADER")
    print("=" * 60)
    
    # Tạo thư mục models nếu chưa có
    os.makedirs("models", exist_ok=True)
    
    models = []
    
    # 1. Haar Cascade (Required)
    models.append({
        'name': 'Haar Cascade',
        'url': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
        'output': 'models/haarcascade_frontalface_default.xml',
        'required': True
    })
    
    # 2. DNN Prototxt (Optional)
    models.append({
        'name': 'DNN Prototxt',
        'url': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
        'output': 'models/deploy.prototxt',
        'required': False
    })
    
    # 3. DNN Caffemodel (Optional)
    models.append({
        'name': 'DNN Caffemodel',
        'url': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
        'output': 'models/res10_300x300_ssd_iter_140000.caffemodel',
        'required': False
    })
    
    print("\nModels to download:")
    for i, model in enumerate(models, 1):
        status = "REQUIRED" if model['required'] else "OPTIONAL"
        exists = "✓ EXISTS" if os.path.exists(model['output']) else "✗ MISSING"
        print(f"{i}. {model['name']} ({status}) - {exists}")
    
    print("\n" + "=" * 60)
    print("NOTE: FaceNet model (~90MB) must be downloaded manually")
    print("See MODELS_DOWNLOAD.md for instructions")
    print("=" * 60)
    
    # Hỏi user có muốn download không
    response = input("\nDownload missing models? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled")
        return
    
    # Download từng model
    print("\n" + "=" * 60)
    print("DOWNLOADING MODELS...")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for model in models:
        # Skip nếu đã tồn tại
        if os.path.exists(model['output']):
            print(f"\n✓ {model['name']} already exists, skipping...")
            success_count += 1
            continue
        
        # Download
        if download_file(model['url'], model['output'], model['name']):
            success_count += 1
        else:
            fail_count += 1
            if model['required']:
                print(f"WARNING: {model['name']} is REQUIRED but failed to download!")
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✓ Success: {success_count}")
    print(f"✗ Failed: {fail_count}")
    
    # Kiểm tra models
    print("\n" + "=" * 60)
    print("MODEL STATUS")
    print("=" * 60)
    
    all_models = {
        'Haar Cascade': 'models/haarcascade_frontalface_default.xml',
        'DNN Prototxt': 'models/deploy.prototxt',
        'DNN Caffemodel': 'models/res10_300x300_ssd_iter_140000.caffemodel',
        'FaceNet': 'models/facenet_keras.h5'
    }
    
    for name, path in all_models.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"✓ {name}: {path} ({size:.2f} MB)")
        else:
            print(f"✗ {name}: {path} (MISSING)")
    
    print("\n" + "=" * 60)
    
    # Hướng dẫn tiếp theo
    if not os.path.exists('models/facenet_keras.h5'):
        print("\nNEXT STEPS:")
        print("1. Download FaceNet model manually (see MODELS_DOWNLOAD.md)")
        print("2. Place facenet_keras.h5 in models/ directory")
        print("3. Run: python train_lbph.py or python train_facenet.py")
    else:
        print("\n✓ All models ready!")
        print("Next: python train_lbph.py or python train_facenet.py")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
