"""
Download Pre-trained Models Script
Script tự động download các pre-trained models cần thiết
"""

import os
import sys
import urllib.request


def download_file(url, output_path, description):
    """Download file với progress bar"""
    print(f"\nDownloading {description}...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")

    def reporthook(count, block_size, total_size):
        """Progress callback"""
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rProgress: {percent}% ")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, output_path, reporthook)
        print(f"\n✓ Downloaded successfully: {output_path}")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading: {e}")
        return False


def print_models_status(models):
    """In trạng thái models"""
    print("\nModels to download:")
    for i, model in enumerate(models, 1):
        status = "REQUIRED" if model['required'] else "OPTIONAL"
        exists = "✓ EXISTS" if os.path.exists(model['output']) else "✗ MISSING"
        print(f"{i}. {model['name']} ({status}) - {exists}")


def summary(success_count, fail_count):
    """In kết quả tổng kết"""
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✓ Success: {success_count}")
    print(f"✗ Failed: {fail_count}")


def check_all_models():
    """Kiểm tra trạng thái models cuối cùng"""
    print("\n" + "=" * 60)
    print("MODEL STATUS")
    print("=" * 60)

    all_models = {
        'YuNet Detector': 'models/yunet/face_detection_yunet_2023mar.onnx',
        'SFace Recognizer': 'models/sface/face_recognition_sface_2021dec.onnx'
    }

    for name, path in all_models.items():
        if os.path.exists(path):
            print(f"✓ {name} - {path}")
        else:
            print(f"✗ {name} - {path} (MISSING)")
    print("=" * 60)


def main():
    """Main download function"""
    print("=" * 60)
    print("FACE ACCESS CONTROL - MODEL DOWNLOADER")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)

    models = [
        {
            'name': 'YuNet Face Detector',
            'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
            'output': 'models/face_detection_yunet_2023mar.onnx',
            'required': False
        },
        {
            'name': 'SFace Face Recognizer',
            'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
            'output': 'models/face_recognition_sface_2021dec.onnx',
            'required': False
        }
    ]

    print_models_status(models)

    print("\n" + "=" * 60)
    print("See MODELS_DOWNLOAD.md for instructions")
    print("=" * 60)

    response = input("\nDownload missing models? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled")
        return

    print("\n" + "=" * 60)
    print("DOWNLOADING MODELS...")
    print("=" * 60)

    success_count, fail_count = 0, 0

    for model in models:
        if os.path.exists(model['output']):
            print(f"\n✓ {model['name']} already exists, skipping...")
            success_count += 1
            continue

        if download_file(model['url'], model['output'], model['name']):
            success_count += 1
        else:
            fail_count += 1
            if model['required']:
                print(f"WARNING: {model['name']} is REQUIRED but failed to download!")

    summary(success_count, fail_count)
    check_all_models()


if __name__ == "__main__":
    main()
