"""
Face Access Control - MAIN
Chạy camera + detect face trên realtime.
"""

import cv2
from modules.camera import CameraManager
from modules.detector import FaceDetector
from modules.database import Database
import time

def main():

    print("⭐ SYSTEM STARTING ⭐")

    # ====== Khởi tạo camera ======
    camera = CameraManager()

    if not camera.open():
        print("❌ Camera không mở được")
        return

    print("✔ Camera opened!")

    # ====== Detector ======
    detector = FaceDetector(method="dnn")
    print("✔ Detector loaded!")

    # ====== Database ======
    db = Database()
    print("✔ Database loaded!")

    # ====== Main loop realtime ======
    while True:

        ret, frame = camera.read()
        if not ret:
            print("Camera mất frame")
            break

        # Detect face
        boxes = detector.detect(frame)

        # Vẽ bounding box
        for (x, y, w, h) in boxes:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Hiển thị số mặt tìm thấy
        cv2.putText(
            frame,
            f"Detected: {len(boxes)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Show image
        cv2.imshow("FACE ACCESS CONTROL", frame)

        # Nhấn Q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

    # Shutdown
    camera.release()
    cv2.destroyAllWindows()
    print("⭐ SYSTEM CLOSED ⭐")


# ======================================================
if __name__ == "__main__":
    main()
