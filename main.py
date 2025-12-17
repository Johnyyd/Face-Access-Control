# Main application entry point
import cv2
from detector import FaceDetector
from recognizer_lbph import LBPHRecognizer
from recognizer_facenet import FaceNetRecognizer
from config import CAMERA_ID

det = FaceDetector()
lbph = LBPHRecognizer()
facenet = FaceNetRecognizer()

cam = cv2.VideoCapture(CAMERA_ID)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    boxes = det.detect(frame)

    for (x,y,w,h) in boxes:
        face = frame[y:y+h, x:x+w]

        nameF, dist = facenet.predict(face)
        nameL, conf = lbph.predict(face)

        text = f"{nameF}"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, text,(x,y-5),
            cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)

    cv2.imshow("FACE ACCESS", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
