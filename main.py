import cv2
from modules.camera import CameraManager
from modules.detector import FaceDetector
from modules.database import Database

def main():
    # Khởi tạo camera
    cam = CameraManager()
    if not cam.open():
        print("❌ Không mở được camera!")
        return

    # Khởi tạo detector (DNN hoặc haar)
    detector = FaceDetector(method="dnn")

    # Khởi tạo database (tạm chưa dùng)
    db = Database()

    print("🔥 Camera started! (Nhấn Q để thoát)")

    while True:
        ok, frame = cam.read()
        if not ok:
            print("❌ Lỗi đọc frame!")
            break

        # Detect faces
        boxes = detector.detect(frame)

        # Vẽ bounding box
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, 
                          (x, y), 
                          (x + w, y + h), 
                          (0, 255, 0), 
                          2)

        # Hiển thị
        cv2.imshow("Face Access Control - Detector", frame)

        # Nhấn Q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
